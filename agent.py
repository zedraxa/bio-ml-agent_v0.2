import argparse
import json
import logging
import logging.handlers
import os
import re
import subprocess
import sys
import textwrap
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple


from llm_backend import (
    LLMBackend, OllamaBackend, auto_create_backend, detect_backend_name,
)

from utils.config import load_config, get_config, AppConfig
from exceptions import (
    ToolExecutionError,
    FileOperationError,
    SecurityViolationError,
    AgentError,
    ToolTimeoutError,
    ValidationError,
    LLMConnectionError,
)
from progress import Spinner
from llm_backend import LLMBackend, OllamaBackend, auto_create_backend
from plugin_manager import PluginManager
from dataset_catalog import format_catalog_for_prompt
from rag_engine import RAGEngine
from mlflow_tracker import get_shared_tracker
from utils.metrics import telemetry
from ultra_agent.observability.metrics import metrics as otel_metrics
from ultra_agent.observability.audit_trail import AuditTrailLogger

# ── Yapılandırma üzerinden okunan sabitler ──
# Bu değerler config.yaml / env / CLI'dan yüklenir.
# İlk erişimde varsayılanlar kullanılır, main() içinde güncellenir.
_app_cfg = None  # load_config() sonrası set edilir
_swarm_orchestrator = None  # Swarm modunda lazy-init edilir

def _cfg() -> "AppConfig":
    """Mevcut config'i döndürür (lazy init)."""
    global _app_cfg
    if _app_cfg is None:
        _app_cfg = get_config()
    return _app_cfg

# Geriye dönük uyumluluk sabitleri (testler ve diğer modüller için)
DEFAULT_PROJECT = "scratch_project"
HISTORY_DIR_NAME = "conversation_history"
LOG_DIR_NAME = "logs"
from utils.logger import setup_logger, LOG_FILE_NAME


# Global logger — main() içinde setup_logger() ile yapılandırılacak
log = logging.getLogger("bio_ml_agent")

# ─────────────────────────────────────────────
#  System Prompt + AgentConfig — core/config.py'den import
# ─────────────────────────────────────────────
from core.config import AgentConfig, SYSTEM_PROMPT

# ─────────────────────────────────────────────
#  Refaktör: Tool ve Konuşma fonksiyonları core/ altından import
# ─────────────────────────────────────────────
from core.tools import (
    # Sabitler
    TOOL_TAGS, TOOL_RE, FENCED_BASH_RE, FENCED_PY_RE,
    DENY_PATTERNS,
    # Güvenlik
    is_dangerous_bash, safe_relpath, current_project,
    _get_deny_patterns,
    # Kod çalıştırma
    run_python, run_bash,
    # Web araçları
    web_search, web_open, browser_open, browser_action,
    # Dosya işlemleri
    read_file, write_file, append_todo, version_dataset,
    sanitize_content, _clean_file_payload, _strip_redundant_prefixes,
    # Tool parsing
    extract_tools, extract_tool, normalize_user_message,
    autosave_web_outputs,
)

from core.conversation import (
    _ensure_history_dir,
    generate_session_id,
    save_conversation,
    load_conversation,
    list_conversations,
    delete_conversation,
    print_history_help,
)


# ─────────────────────────────────────────────
#  Global LLM Backend & Plugin Manager
# ─────────────────────────────────────────────

# Global LLM backend — main() içinde oluşturulur, varsayılan Ollama
_llm_backend: Optional[LLMBackend] = None

# Global plugin manager
_plugin_manager: Optional[PluginManager] = None


def get_llm_backend(model: str = "") -> LLMBackend:
    """Aktif LLM backend'i döndür (yoksa Ollama oluştur)."""
    global _llm_backend
    if _llm_backend is None:
        _llm_backend = OllamaBackend(model=model or "qwen2.5:latest")
    return _llm_backend


def set_llm_backend(backend: LLMBackend) -> None:
    """LLM backend'i değiştir."""
    global _llm_backend
    _llm_backend = backend
    log.info("🧠 LLM backend değiştirildi: %s", backend)


def get_plugin_manager() -> PluginManager:
    """Plugin manager'ı döndür (yoksa oluştur)."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager


def llm_chat(model: str, messages: List[Dict[str, str]], session_id: str = "default") -> str:
    log.info("🧠 LLM isteği gönderiliyor | model=%s | mesaj_sayısı=%d", model, len(messages))
    start_time = time.time()
    try:
        backend = get_llm_backend(model)
        raw_content = backend.chat(messages, session_id=session_id)
        content = str(raw_content).strip()
        elapsed = time.time() - start_time
        log.info("🧠 LLM yanıt alındı | süre=%.2fs | yanıt_uzunluk=%d karakter", elapsed, len(content))
        log.debug("🧠 LLM yanıt (ilk 300 karakter): %s", content[:300])
        return content
    except LLMConnectionError:
        raise
    except Exception as e:
        elapsed = time.time() - start_time
        log.error("🧠 LLM HATA | süre=%.2fs | model=%s | hata=%s", elapsed, model, e, exc_info=True)
        raise LLMConnectionError(model, str(e))


# ─────────────────────────────────────────────
#  Placeholder tool dispatcher (main loop uses inline dispatch)
# ─────────────────────────────────────────────
def _run_tool(tool: str, payload: str, cfg: AgentConfig) -> str:
    # This is a placeholder. In a real scenario, this would dispatch to the actual tool function.
    # For the purpose of this edit, we'll just return a mock output.
    return f"[MOCK_TOOL_OUTPUT] {tool} executed with payload: {payload[:50]}..."

def _format_tool_output(tool: str, payload: str, output: str) -> str:
    # This is a placeholder. In a real scenario, this would format the output nicely.
    return f"\n🛠️ {tool} output:\n{output}\n"

def main():
    # ── 1. config.yaml'ı yükle (varsayılanlar + yaml + env) ──
    global _app_cfg
    _app_cfg = load_config()
    app = _app_cfg

    # ── 2. CLI argümanları (en yüksek öncelik) ──
    parser = argparse.ArgumentParser(
        description="Bio-ML Agent — Yerel LLM destekli ML proje asistanı",
        epilog="Yapılandırma: config.yaml > ortam değişkenleri > CLI argümanları",
    )
    parser.add_argument("--model", default=app.agent.model,
                        help=f"Ollama model adı (varsayılan: {app.agent.model})")
    parser.add_argument("--workspace", default=app.workspace.base_dir,
                        help=f"Çalışma alanı (varsayılan: {app.workspace.base_dir})")
    parser.add_argument("--timeout", type=int, default=app.agent.timeout,
                        help=f"Komut zaman aşımı saniye (varsayılan: {app.agent.timeout})")
    parser.add_argument("--max-steps", type=int, default=app.agent.max_steps,
                        help=f"Maks. adım sayısı (varsayılan: {app.agent.max_steps})")
    parser.add_argument("--history-dir", default=app.history.directory,
                        help=f"Konuşma geçmişi klasörü (varsayılan: {app.history.directory})")
    parser.add_argument("--load-session", default=None,
                        help="Başlangıçta yüklenecek oturum ID'si")
    parser.add_argument("--log-level", default=app.logging.level,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help=f"Log seviyesi (varsayılan: {app.logging.level})")
    parser.add_argument("--log-dir", default=app.logging.directory,
                        help=f"Log dosyaları klasörü (varsayılan: {app.logging.directory})")
    parser.add_argument("--config", dest="config_file", default="config.yaml",
                        help="Yapılandırma dosyası yolu (varsayılan: config.yaml)")
    parser.add_argument("--backend", dest="backend_mode", choices=["auto", "local", "remote"], default="auto",
                        help="LLM backend modu: local (Ollama), remote (model adına göre OpenAI/Anthropic/Gemini), auto (otomatik algıla) (varsayılan: auto)")
    parser.add_argument("--swarm", dest="swarm", action="store_true",
                        help="V6 Multi-Agent Swarm mimarisini taklit eden modu etkinleştir")
    args = parser.parse_args()

    # ── 3. CLI ile config farklıysa config'i güncelle ──
    if args.config_file: # Changed from args.config to args.config_file
        _app_cfg = load_config(config_path=args.config_file)
        app = _app_cfg

    # ── 4. Logger'ı kur ──
    log_dir = Path(args.log_dir).expanduser().resolve()
    global log
    log = setup_logger(log_dir, args.log_level)

    log.info("📋 Yapılandırma yüklendi:\n%s", app.summary())

    cfg = AgentConfig(
        model=args.model,
        workspace=Path(args.workspace).expanduser().resolve(),
        timeout=args.timeout,
        max_steps=args.max_steps,
        history_dir=Path(args.history_dir).expanduser().resolve(),
        load_session=args.load_session,
        log_level=args.log_level,
        log_dir=log_dir,
        config_file=args.config_file,
        backend_mode=args.backend_mode,
        swarm=args.swarm,
    )
    cfg.workspace.mkdir(parents=True, exist_ok=True)
    _ensure_history_dir(cfg.history_dir)

    log.info("Agent başlatıldı | model=%s | workspace=%s | timeout=%d | max_steps=%d | swarm=%s",
             cfg.model, cfg.workspace, cfg.timeout, cfg.max_steps, cfg.swarm)

    # ── LLM backend ve plugin sistemi ──
    backend_mode = cfg.backend_mode # Changed from args.backend to cfg.backend_mode
    backend = auto_create_backend(cfg.model, mode=backend_mode)
    set_llm_backend(backend)
    log.info("🧠 LLM backend oluşturuldu | backend=%s | model=%s | mod=%s",
             type(backend).__name__, cfg.model, backend_mode)
    pm = get_plugin_manager()
    plugins_dir = Path(__file__).resolve().parent / "plugins"
    loaded = pm.discover(plugins_dir)
    if loaded:
        log.info("🔌 %d plugin yüklendi", loaded)
        print(f"🔌 {loaded} plugin yüklendi: {', '.join(pm.tool_names)}")

    # ── RAG Motoru başlat ──
    global rag
    rag = RAGEngine(workspace_dir=cfg.workspace)
    log.info("🔍 RAG Motoru başlatıldı | db_dir=%s", rag.db_dir)

    # ── Oturum başlat veya yükle ──
    session_id = generate_session_id()
    session_metadata = {"created_at": datetime.now().isoformat()}
    system_prompt = SYSTEM_PROMPT + format_catalog_for_prompt() + pm.get_prompt_additions()
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    if cfg.load_session: # Changed from args.load_session to cfg.load_session
        try:
            messages, session_metadata = load_conversation(cfg.history_dir, cfg.load_session)
            session_id = session_metadata.get("session_id", session_id)
            log.info("Oturum yüklendi | session=%s | mesaj_sayısı=%d", session_id, len(messages))
            print(f"📂 Oturum yüklendi: {session_id}")
            print(f"   Mesaj sayısı: {len(messages)}")
        except FileNotFoundError as e:
            log.warning("Oturum yüklenemedi | session=%s | hata=%s", cfg.load_session, e)
            print(f"❌ {e}")
            print("   Yeni oturum başlatılıyor...\n")

    log.info("Yeni oturum başlatıldı | session=%s", session_id)

    backend_label = type(backend).__name__.replace("Backend", "")
    print(f"🧠 Bio-ML Agent ready | model={cfg.model} | backend={backend_label} | workspace={cfg.workspace}")
    print(f"📜 Oturum ID: {session_id}")
    print(f"💾 Geçmiş klasörü: {cfg.history_dir}")
    print(f"📋 Log klasörü: {log_dir}")
    print(f"🔌 Backend modu: {backend_label}")
    print("Çıkmak için: exit / quit | Komutlar: /history /load /new /save /delete /info /logs /rag /ragindex\n")

    while True:
        try:
            user = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            log.info("Kullanıcı Ctrl+C/EOF ile çıkış yaptı | session=%s", session_id)
            print("\n💾 Oturum kaydediliyor...")
            save_conversation(cfg.history_dir, session_id, messages, session_metadata)
            print(f"✅ Kaydedildi: {session_id}")
            print("Çıkılıyor.")
            break

        if not user:
            continue

        # ── Çıkış komutları ──
        if user.lower() in {"exit", "quit"}:
            log.info("Kullanıcı çıkış yaptı | session=%s | komut=%s", session_id, user.lower())
            print("💾 Oturum kaydediliyor...")
            save_conversation(cfg.history_dir, session_id, messages, session_metadata)
            print(f"✅ Kaydedildi: {session_id}")
            break

        # ── Geçmiş yönetimi komutları ──
        if user.lower() == "/history":
            sessions = list_conversations(cfg.history_dir)
            if not sessions:
                print("\n📭 Kayıtlı oturum bulunamadı.\n")
            else:
                print(f"\n📜 Kayıtlı Oturumlar ({len(sessions)} adet):")
                print("─" * 90)
                for i, s in enumerate(sessions, 1):
                    marker = " 👈 (aktif)" if s["session_id"] == session_id else ""
                    print(f"  {i:2}. 🆔 {s['session_id']}{marker}")
                    print(f"      📅 {s['created_at'][:19]}  |  💬 {s['message_count']} mesaj")
                    print(f"      📝 {s['summary'][:70]}")
                    print()
                print("─" * 90)
                print("  Yüklemek için: /load <session_id>\n")
            continue

        if user.lower().startswith("/load "):
            target_id = user.split(" ", 1)[1].strip()
            try:
                # Mevcut oturumu önce kaydet
                save_conversation(cfg.history_dir, session_id, messages, session_metadata)
                print(f"💾 Mevcut oturum kaydedildi: {session_id}")

                messages, session_metadata = load_conversation(cfg.history_dir, target_id)
                session_id = session_metadata.get("session_id", target_id)
                print(f"✅ Oturum yüklendi: {session_id}")
                print(f"   Mesaj sayısı: {len(messages)}")

                # Son birkaç mesajı göster
                user_msgs = [m for m in messages if m["role"] == "user"
                             and not m["content"].startswith("TOOL_OUTPUT")]
                if user_msgs:
                    print(f"\n   📝 Son kullanıcı mesajı:")
                    print(f"      \"{user_msgs[-1]['content'][:100]}...\"\n")
            except FileNotFoundError as e:
                print(f"❌ {e}\n")
            continue

        if user.lower().startswith("/delete "):
            target_id = user.split(" ", 1)[1].strip()
            if target_id == session_id:
                print("❌ Aktif oturumu silemezsiniz! Önce /new ile yeni oturum başlatın.\n")
            elif delete_conversation(cfg.history_dir, target_id):
                print(f"🗑️  Oturum silindi: {target_id}\n")
            else:
                print(f"❌ Oturum bulunamadı: {target_id}\n")
            continue

        if user.lower() == "/new":
            # Mevcut oturumu kaydet, yenisini başlat
            log.info("Yeni oturum başlatılıyor | eski_session=%s", session_id)
            save_conversation(cfg.history_dir, session_id, messages, session_metadata)
            print(f"💾 Mevcut oturum kaydedildi: {session_id}")

            session_id = generate_session_id()
            session_metadata = {"created_at": datetime.now().isoformat()}
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            log.info("Yeni oturum oluşturuldu | yeni_session=%s", session_id)
            print(f"🆕 Yeni oturum başlatıldı: {session_id}\n")
            continue

        if user.lower() == "/save":
            path = save_conversation(cfg.history_dir, session_id, messages, session_metadata)
            print(f"💾 Oturum kaydedildi: {path}\n")
            continue

        if user.lower() == "/info":
            user_msg_count = sum(1 for m in messages if m["role"] == "user"
                                and not m["content"].startswith("TOOL_OUTPUT"))
            asst_msg_count = sum(1 for m in messages if m["role"] == "assistant")
            print(f"\n📊 Oturum Bilgileri:")
            print(f"   🆔 Oturum ID  : {session_id}")
            print(f"   📅 Oluşturulma: {session_metadata.get('created_at', '?')[:19]}")
            print(f"   💬 Toplam mesaj: {len(messages)}")
            print(f"   👤 Kullanıcı   : {user_msg_count} mesaj")
            print(f"   🤖 Asistan     : {asst_msg_count} mesaj")
            print(f"   💾 Geçmiş yolu : {cfg.history_dir / f'{session_id}.json'}")
            print()
            continue

        if user.lower() in {"/help", "/h"}:
            print_history_help()
            continue

        if user.lower().startswith("/logs"):
            # Son logları göster
            parts = user.split()
            tail_lines = 30
            if len(parts) > 1:
                try:
                    tail_lines = int(parts[1])
                except ValueError:
                    pass
            log_file = log_dir / LOG_FILE_NAME
            if log_file.exists():
                lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
                show = lines[-tail_lines:] if len(lines) > tail_lines else lines
                print(f"\n📋 Son {len(show)} log satırı ({log_file}):")
                print("─" * 90)
                for line in show:
                    print(f"  {line}")
                print("─" * 90)
                print(f"  Toplam: {len(lines)} satır | Gösterilen: son {len(show)} satır")
                print(f"  Daha fazla görmek için: /logs <satır_sayısı>\n")
            else:
                print(f"\n📭 Log dosyası henüz oluşturulmamış: {log_file}\n")
            continue

        if user.lower() == "/ragindex":
            print("🔍 Workspace indeksleniyor. Lütfen bekleyin...")
            count = rag.index_workspace()
            print(f"✅ İndeksleme tamamlandı. {count} dosya işlendi.\n")
            continue

        if user.lower().startswith("/rag "):
            query = user.split(" ", 1)[1].strip()
            print(f"🔍 RAG araması yapılıyor: '{query}'")
            results = rag.search(query)
            if not results:
                print("📭 Eşleşen sonuç bulunamadı.\n")
            else:
                for i, r in enumerate(results, 1):
                    print(f"\n[{i}] 📄 {r['source']} (Mesafe: {r['distance']:.4f})")
                    print("─" * 40)
                    print(r['document'])
                    print("─" * 40)
                print()
            continue

        if user.lower() == "/stats":
            stats = telemetry.get_session(session_id).get_stats()
            print(f"\n📊 Oturum İstatistikleri ({session_id}):")
            print(f"   ⏱️  Süre         : {stats['duration_s']} s")
            print(f"   🤖 LLM Çağrısı   : {stats['total_llm_calls']}")
            print(f"   🎟️  Toplam Token : {stats['total_tokens']} (P: {stats['total_prompt_tokens']}, C: {stats['total_completion_tokens']})")
            print(f"   ⚡ LLM Gecikme   : {stats['avg_llm_latency_ms']} ms (ort)")
            print(f"   🛠️  Tool Çağrısı  : {stats['total_tool_calls']} ({stats['successful_tool_calls']} başarılı)")
            print(f"   ⚡ Tool Gecikme  : {stats['avg_tool_latency_ms']} ms (ort)")
            print()
            continue

        # ── Normal agent akışı ──
        otel_metrics.increment_counter("agent_runs_total", {"session": session_id})
        log.info("👤 Kullanıcı mesajı alındı | uzunluk=%d | session=%s", len(user), session_id)
        log.debug("👤 Kullanıcı mesajı: %s", user[:300])
        user = normalize_user_message(user)
        mproj = re.search(r"(?i)\bPROJECT\s*:\s*([a-z0-9_\-]+)", user)
        project = mproj.group(1) if mproj else DEFAULT_PROJECT
        os.environ["AGENT_PROJECT"] = project
        (cfg.workspace / project).mkdir(parents=True, exist_ok=True)
        log.info("📁 Aktif proje: %s", project)

        try:
            from memory_manager import memory
            mem_context = memory.get_context_string(user, n_results=2)
            if mem_context:
                enriched_user = f"{mem_context}\n\n[Mevcut Görev/Soru]:\n{user}"
                messages.append({"role": "user", "content": enriched_user})
                log.info("🧠 RAG Hafızası (%d sonuç) mesaja eklendi", 2)
            else:
                messages.append({"role": "user", "content": user})
        except Exception as e:
            log.warning("Hafıza yöneticisi hatası: %s", e)
            messages.append({"role": "user", "content": user})

        # Her kullanıcı mesajından sonra otomatik kaydet
        save_conversation(cfg.history_dir, session_id, messages, session_metadata)

        # Swarm Orchestrator Entegrasyonu (V6 Model) veya Monolitik (V5 Model)
        if cfg.swarm:
            from swarm.orchestrator import SwarmOrchestrator
            try:
                global _swarm_orchestrator
                if _swarm_orchestrator is None:
                    _swarm_orchestrator = SwarmOrchestrator(cfg)
                
                with Spinner("🧠 Swarm Orchestrator Devrede (Görev Dağıtılıyor)"):
                    assistant = _swarm_orchestrator.process(messages)
                
                # Sonucu ekrana ve hafızaya ekle
                print("\n🤖 Swarm Sonucu:\n", assistant)
                messages.append({"role": "assistant", "content": assistant})
                
                # Yeni etkileşimi RAG DB'ye kaydet
                try:
                    from memory_manager import memory
                    memory.store_interaction(session_id, user, assistant)
                    log.info("🧠 Etkileşim kalıcı hafızaya (RAG) kaydedildi")
                except Exception as e:
                    log.warning("Hafıza kaydetme hatası: %s", e)
                    
                # Asistan cevabından sonra otomatik kaydet
                save_conversation(cfg.history_dir, session_id, messages, session_metadata)
                
            except Exception as e:
                log.error(f"[Swarm Error] {str(e)}", exc_info=True)
                print(f"\n❌ Swarm yöneticisinde kritik hata: {str(e)}")
        else:
            # Geleneksel Monolitik V5 Döngüsü
            break_loop = False
            _consecutive_errors = 0
            _last_error_sig = ""
            for step in range(cfg.max_steps):
                log.info("🔄 Adım %d/%d başlıyor", step + 1, cfg.max_steps)
                
                try:
                    from llm_backend import summarize_memory
                    backend_for_mem = auto_create_backend(cfg.model)
                    messages = summarize_memory(messages, backend_for_mem, threshold=40)
                except Exception as e:
                    log.warning("Bellek özetleme adımı atlatıldı: %s", e)
                
                with Spinner(f"🧠 LLM düşünüyor (adım {step + 1}/{cfg.max_steps})"):
                    assistant = llm_chat(cfg.model, messages, session_id=session_id)

                tools_to_run, outside = extract_tools(assistant)

                if not tools_to_run:
                    otel_metrics.increment_counter("llm_fallbacks_total", {"reason": "no_tool_found"})
                    py_m = FENCED_PY_RE.search(assistant)
                    bash_m = FENCED_BASH_RE.search(assistant)
                    if py_m and (not bash_m or len(py_m.group(1)) >= len(bash_m.group(1))):
                        tools_to_run = [("PYTHON", py_m.group(1))]
                        outside = FENCED_PY_RE.sub("", assistant).strip()
                        log.info("🔧 Fenced code block'tan PYTHON tool algılandı")
                    elif bash_m:
                        tools_to_run = [("BASH", bash_m.group(1))]
                        outside = FENCED_BASH_RE.sub("", assistant).strip()
                        log.info("🔧 Fenced code block'tan BASH tool algılandı")
                    else:
                        # Tool-First Policy: İlk 2 adımda aksiyon isteğiyse retry gönder
                        from services.agent_service import _is_action_request, TOOL_ENFORCEMENT_PROMPT
                        if _is_action_request(user) and step < 2:
                            log.info("🔄 Tool-First Policy: Aksiyon isteği ama tool yok, retry (adım %d)", step + 1)
                            messages.append({"role": "assistant", "content": assistant})
                            messages.append({"role": "user", "content": TOOL_ENFORCEMENT_PROMPT})
                            print("\n🔄 Tool zorunluluğu uygulanıyor — agent tekrar deneyecek...\n")
                            continue
                        
                        log.info("💬 Agent düz metin yanıtı verdi (tool yok) | adım=%d", step + 1)
                        print("\n🤖 Agent:\n", assistant)
                        messages.append({"role": "assistant", "content": assistant})
                        
                        try:
                            from memory_manager import memory
                            memory.store_interaction(session_id, user, assistant)
                            log.info("🧠 Etkileşim kalıcı hafızaya (RAG) kaydedildi")
                        except Exception as e:
                            log.warning("Hafıza kaydetme hatası: %s", e)
                            
                        # Asistan cevabından sonra otomatik kaydet
                        save_conversation(cfg.history_dir, session_id, messages, session_metadata)
                        break

                if outside:
                    log.warning("⚠️ Tool bloğu dışında metin vardı | dış_metin_uzunluk=%d", len(outside))
                    print("\n⚠️ Uyarı: Tool bloğu dışında metin vardı; yine de tool çalıştırılıyor.\n")

                messages.append({"role": "assistant", "content": assistant})
                
                all_outputs = []
                break_loop = False
                
                audit_logger = AuditTrailLogger(cfg.workspace)

                for tool, payload in tools_to_run:
                    log.info("🔧 Tool algılandı: %s | payload_uzunluk=%d", tool, len(payload or ""))
                    
                    # S8-1 & S8-2 OTel Metrics
                    otel_metrics.increment_counter("tool_calls_total", {"tool": tool})
                    
                    # S8-3 & S8-4 Governance and Audit Trails for 5 critical tools
                    if tool in ["BASH", "PYTHON", "WRITE_FILE", "BROWSER_ACTION", "BROWSER_AGENT"]:
                        audit_logger.log_critical_action(
                            agent_id=session_id,
                            action=tool,
                            details={"payload": payload[:500]},
                            approval_status="AUTO_APPROVED_BY_POLICY_HITL"
                        )
                    
                    tool_start = time.time()
                    try:
                        if tool == "PYTHON":
                            with Spinner("🐍 Python çalıştırılıyor"):
                                out = run_python(payload, cfg.workspace, timeout_s=cfg.timeout)
                        elif tool == "BASH":
                            with Spinner("💻 Bash çalıştırılıyor"):
                                out = run_bash(payload, cfg.workspace, timeout_s=cfg.timeout)
                        elif tool == "WEB_SEARCH":
                            if not _cfg().security.allow_web_search:
                                out = "[BLOCKED] WEB_SEARCH is disabled in config."
                            else:
                                with Spinner("🌐 Web'de aranıyor"):
                                    out = web_search(payload)
                        elif tool == "WEB_OPEN":
                            with Spinner("📖 Sayfa okunuyor"):
                                out = web_open(payload)
                        elif tool == "BROWSER_OPEN":
                            with Spinner("🌐 Headless browser ile sayfa açılıyor"):
                                out = browser_open(payload, session_id=session_id, workspace=cfg.workspace)
                        elif tool == "BROWSER_ACTION":
                            with Spinner("🌐 Browser Action çalıştırılıyor"):
                                out = browser_action(payload, cfg.workspace)
                        elif tool == "BROWSER_AGENT":
                            with Spinner("🤖 Browser Sub-Agent görev üzerinde çalışıyor"):
                                from ultra_agent.runtime.browser.browser_agent import run_browser_agent
                                out = run_browser_agent(payload, model=cfg.model, workspace=cfg.workspace)
                        elif tool == "READ_FILE":
                            out = read_file(payload, cfg.workspace)
                        elif tool == "WRITE_FILE":
                            out = write_file(payload, cfg.workspace)
                        elif tool == "VERSION_DATASET":
                            out = version_dataset(payload, cfg.workspace)
                        elif tool == "TODO":
                            out = append_todo(payload, cfg.workspace)
                        elif tool == "RAG_SEARCH":
                            with Spinner("🔍 RAG'da aranıyor"):
                                results = rag.search(payload)
                                if not results:
                                    out = "[RAG_SEARCH] Sonuç bulunamadı."
                                else:
                                    out = "[RAG_SEARCH] Bulunan metinler:\n\n"
                                    for i, r in enumerate(results, 1):
                                        out += f"--- Kaynak: {r['source']} (Mesafe: {r['distance']:.4f}) ---\n"
                                        out += f"{r['document']}\n\n"
                        elif pm.get(tool):
                            with Spinner(f"🔌 Plugin çalıştırılıyor: {tool}"):
                                out = pm.execute(tool, payload, cfg.workspace)
                        else:
                            out = f"[ERROR] Bilinmeyen tool: {tool}"
                            
                        # Başarıyı ve Histogramı kaydet
                        elapsed_ms = (time.time() - tool_start) * 1000
                        telemetry.get_session(session_id).record_tool_call(tool, elapsed_ms, True)
                        otel_metrics.record_histogram("tool_duration_ms", elapsed_ms, {"tool": tool, "status": "success"})

                    except LLMConnectionError as e:
                        telemetry.get_session(session_id).record_tool_call(tool, (time.time() - tool_start) * 1000, False)
                        log.error("🧠 LLM bağlantı hatası | %s", e)
                        print(f"\n{e.user_message()}")
                        print("\n⏳ 5 saniye sonra tekrar denenecek...\n")
                        time.sleep(5)
                        try:
                            with Spinner("🧠 LLM tekrar deneniyor"):
                                assistant = llm_chat(cfg.model, messages, session_id=session_id)
                            messages.append({"role": "assistant", "content": assistant})
                            save_conversation(cfg.history_dir, session_id, messages, session_metadata)
                        except LLMConnectionError as e2:
                            log.error("🧠 LLM tekrar deneme başarısız | %s", e2)
                            print(f"\n{e2.user_message()}")
                            print("\n⚠️ LLM'e bağlanılamıyor. Lütfen Ollama servisini kontrol edin.\n")
                            save_conversation(cfg.history_dir, session_id, messages, session_metadata)
                        break_loop = True
                        break

                    except SecurityViolationError as e:
                        telemetry.get_session(session_id).record_tool_call(tool, (time.time() - tool_start) * 1000, False)
                        log.warning("🔒 Güvenlik ihlali | %s", e)
                        print(f"\n{e.user_message()}")
                        out = e.tool_output()

                    except ToolTimeoutError as e:
                        telemetry.get_session(session_id).record_tool_call(tool, (time.time() - tool_start) * 1000, False)
                        log.error("⏰ Zaman aşımı | %s", e)
                        print(f"\n{e.user_message()}")
                        out = f"[TIMEOUT] {tool} timed out after {cfg.timeout}s"

                    except (ToolExecutionError, FileOperationError, ValidationError) as e:
                        telemetry.get_session(session_id).record_tool_call(tool, (time.time() - tool_start) * 1000, False)
                        log.error("🛠️ Tool hatası | %s", e)
                        print(f"\n{e.user_message()}")
                        out = e.tool_output()

                    except AgentError as e:
                        telemetry.get_session(session_id).record_tool_call(tool, (time.time() - tool_start) * 1000, False)
                        log.error("❌ Agent hatası | %s", e)
                        print(f"\n{e.user_message()}")
                        out = e.tool_output()

                    except Exception as e:
                        telemetry.get_session(session_id).record_tool_call(tool, (time.time() - tool_start) * 1000, False)
                        log.error("💥 Beklenmeyen hata | tool=%s | %s", tool, e, exc_info=True)
                        print(f"\n❌ Beklenmeyen hata: {e}")
                        print(f"   💡 Öneri: Bu hatayı /logs komutuyla inceleyebilirsiniz.\n")
                        out = f"[UNEXPECTED_ERROR] {type(e).__name__}: {e}"

                if tool in {"WEB_SEARCH", "WEB_OPEN"} and not out.startswith("["):
                    autosave_web_outputs(cfg, tool, out)

                log.info("🛠️ Tool tamamlandı | tool=%s | çıktı_uzunluk=%d", tool, len(out))
                print(f"\n🛠️ {tool} output:\n{out}\n")
                all_outputs.append((tool, out))

                # ── Ardışık hata algılama (Fix: sonsuz retry döngüsünü kır) ──
                _is_err = any([
                    out.startswith("[") and any(k in out[:80] for k in ("ERROR", "TIMEOUT", "UNEXPECTED", "BASH_ERROR")),
                    "Dosya bulunamadı" in out[:200],
                    "hata" in out[:200].lower(),
                    "[python exit code:" in out[:80] and "exit code: 0" not in out[:80],
                    "Traceback" in out[:200],
                ])
                if _is_err:
                    # Tool + çıktının ilk kısmı → imza
                    _err_sig = f"{tool}:{out[:80]}"
                    if _err_sig == _last_error_sig:
                        _consecutive_errors += 1
                    else:
                        _consecutive_errors = 1
                        _last_error_sig = _err_sig
                    if _consecutive_errors >= 3:
                        log.warning("🔄 Ardışık %d aynı hata tespit edildi — strateji değişikliği isteniyor", _consecutive_errors)
                        all_outputs.append(("SYSTEM", "⚠️ UYARI: Aynı hata 3 kez tekrarlandı. DURMALSIN ve FARKLI bir yaklaşım denemelisin. "
                                           "Aynı komutu tekrar çalıştırma. Hatanın kök nedenini analiz et. "
                                           "Eksik kütüphane varsa pip install yap, dosya yoksa oluştur, farklı bir yol dene."))
                        _consecutive_errors = 0
                else:
                    _consecutive_errors = 0
                    _last_error_sig = ""

                if break_loop:
                    break
                
                user_msg = ""
                for t, o in all_outputs:
                    user_msg += f"TOOL_OUTPUT ({t}):\n{o[:2000]}\n\n"
                user_msg += (
                    "---\n"
                    "Yukarıdaki tool çıktısını aldın. Planındaki bir SONRAKİ adıma geç.\n"
                    "BİR SONRAKİ dosyayı oluştur veya bir sonraki komutu çalıştır.\n"
                    "Her yanıtında MUTLAKA bir tool çağrısı (<WRITE_FILE>, <PYTHON>, <BASH>, <WEB_SEARCH>) olmalı.\n"
                    "Tüm adımlar tamamlandıysa ve tüm dosyalar disk'e yazıldıysa, SON ÖZET'i yaz (tool olmadan).\n"
                    "AMA henüz eksik dosya varsa — DEVAM ET, tool kullan!"
                )
                
                messages.append({
                    "role": "user",
                    "content": user_msg
                })

                # Her tool adımından sonra otomatik kaydet
                save_conversation(cfg.history_dir, session_id, messages, session_metadata)
            
            log.warning("⚠️ Maksimum adım sayısına ulaşıldı (%d) | session=%s", cfg.max_steps, session_id)
            print("\n⚠️ Max steps reached. Task may be incomplete.\n")
            save_conversation(cfg.history_dir, session_id, messages, session_metadata)


if __name__ == "__main__":
    main()
