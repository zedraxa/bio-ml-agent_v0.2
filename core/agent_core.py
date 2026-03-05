# core/agent_core.py
# ═══════════════════════════════════════════════════════════
#  AgentCore — Akıllı Orkestratör
#  Faz 3: agent.py main() döngüsünden çıkarıldı.
#
#  Görev: Kullanıcı isteğini analiz et → doğru stratejiye yönlendir:
#    - CHAT:        Genel konuşma (tool yok)
#    - TOOL_LOOP:   Basit tool-tabanlı görevler
#    - ML_PIPELINE: Makine öğrenmesi iş akışları
#    - SWARM:       Çok ajanlı karmaşık analizler
# ═══════════════════════════════════════════════════════════

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from core.config import AgentConfig, SYSTEM_PROMPT
from core.tools import (
    extract_tools, extract_tool, normalize_user_message,
    run_python, run_bash, web_search, web_open,
    browser_open, browser_action, read_file, write_file,
    append_todo, version_dataset, autosave_web_outputs,
    FENCED_BASH_RE, FENCED_PY_RE,
)
from core.conversation import save_conversation, generate_session_id
from exceptions import (
    AgentError, LLMConnectionError, SecurityViolationError,
    ToolTimeoutError, ToolExecutionError, FileOperationError, ValidationError,
)

log = logging.getLogger("bio_ml_agent")

# Intent anahtar kelimeleri
_ML_KEYWORDS = {
    "model eğit", "train", "predict", "classification", "regression",
    "cross validation", "karşılaştır", "compare", "hiperparametre",
    "hyperparameter", "feature importance", "shap", "lime", "xai",
    "deep learning", "cnn", "neural network", "sklearn", "preprocessing",
    "preprocess", "veri hazırla", "dataset", "pipeline",
}

_SWARM_KEYWORDS = {
    "kanser analizi", "genome", "protein analiz", "dna analiz",
    "biyoinformatik", "tam analiz", "detaylı analiz", "kapsamlı analiz",
    "multi agent", "multi-agent", "swarm", "end-to-end", "uçtan uca",
    "tam pipeline", "full pipeline",
}


class AgentCore:
    """Merkezi orkestratör — görevleri akıllıca yönlendirir."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self._llm_backend = None
        self._plugin_manager = None
        self._rag = None
        self._swarm = None

    # ─────────────────────────────────────────────
    #  Lazy initializers
    # ─────────────────────────────────────────────

    @property
    def llm(self):
        if self._llm_backend is None:
            from llm_backend import auto_create_backend
            self._llm_backend = auto_create_backend(
                self.config.model, mode=self.config.backend_mode
            )
        return self._llm_backend

    @property
    def plugins(self):
        if self._plugin_manager is None:
            from plugin_manager import PluginManager
            self._plugin_manager = PluginManager()
            plugins_dir = Path(__file__).resolve().parent.parent / "plugins"
            self._plugin_manager.discover(plugins_dir)
        return self._plugin_manager

    @property
    def rag(self):
        if self._rag is None:
            from rag_engine import RAGEngine
            self._rag = RAGEngine(workspace_dir=self.config.workspace)
        return self._rag

    # ─────────────────────────────────────────────
    #  LLM Chat
    # ─────────────────────────────────────────────

    def llm_chat(self, messages: List[Dict[str, str]], session_id: str = "default") -> str:
        """LLM'e mesaj gönder ve yanıt al."""
        log.info("🧠 LLM isteği | model=%s | mesaj_sayısı=%d", self.config.model, len(messages))
        start = time.time()
        try:
            raw = self.llm.chat(messages, session_id=session_id)
            content = str(raw).strip()
            elapsed = time.time() - start
            log.info("🧠 LLM yanıt | süre=%.2fs | uzunluk=%d", elapsed, len(content))
            return content
        except Exception as e:
            elapsed = time.time() - start
            log.error("🧠 LLM HATA | süre=%.2fs | %s", elapsed, e, exc_info=True)
            raise LLMConnectionError(self.config.model, str(e))

    # ─────────────────────────────────────────────
    #  Intent Sınıflandırma
    # ─────────────────────────────────────────────

    def classify_intent(self, user_msg: str) -> str:
        """Kullanıcı mesajının amacını sınıflandır.

        Returns:
            "CHAT" | "TOOL_LOOP" | "ML_PIPELINE" | "SWARM"
        """
        lower = user_msg.lower()

        # Swarm ilk kontrol — en spesifik
        if self.config.swarm:
            return "SWARM"
        if any(kw in lower for kw in _SWARM_KEYWORDS):
            return "SWARM"

        # ML pipeline
        if any(kw in lower for kw in _ML_KEYWORDS):
            return "ML_PIPELINE"

        # Genel bir aksiyon isteği mi?
        action_indicators = [
            "yap", "oluştur", "yaz", "çalıştır", "ekle", "sil", "güncelle",
            "create", "write", "run", "execute", "build", "generate", "analyze",
            "dosya", "file", "code", "kod", "script", "install", "kur",
        ]
        if any(kw in lower for kw in action_indicators):
            return "TOOL_LOOP"

        return "CHAT"

    # ─────────────────────────────────────────────
    #  route_task — ana yönlendirme
    # ─────────────────────────────────────────────

    def route_task(
        self,
        user_msg: str,
        messages: List[Dict[str, str]],
        session_id: str = "default",
        session_metadata: Optional[Dict] = None,
        intent_override: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Kullanıcı mesajını işle ve olayları yayınla (streaming).
        Faz 6: LangGraph Topolojisi Entegrasyonu (Plan -> Execute -> Verify -> Artifact)
        """
        intent = intent_override or self.classify_intent(user_msg)
        log.info("🎯 Intent: %s | mesaj: %s", intent, user_msg[:80])
        yield {"type": "intent", "intent": intent}

        if intent in ("CHAT", "SWARM"):
            # Swarm veya basit chat için eski mekanizmayı koru (şimdilik)
            if intent == "SWARM":
                yield from self._swarm_pipeline(user_msg, messages, session_id, session_metadata)
            else:
                yield from self._chat(user_msg, messages, session_id, session_metadata)
            return

        # Faz 6: LangGraph Entegrasyonu (TOOL_LOOP ve ML_PIPELINE niyetleri için)
        from ultra_agent.orchestration.langgraph.state import AgentState
        from ultra_agent.orchestration.langgraph.graph import build_graph

        state_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
        state_messages.append({"role": "user", "content": user_msg})

        initial_state: AgentState = {
            "messages": state_messages,
            "current_step": "PLAN",
            "requires_approval": False,
            "approval_result": None,
            "error_counter": 0,
            "feedback": None
        }

        graph = build_graph()

        try:
            for chunk in graph.stream(initial_state):
                for node_name, node_state in chunk.items():
                    step_name = node_state.get("current_step", "UNKNOWN")
                    
                    yield {"type": "status", "content": f"LangGraph: [{node_name.upper()}] Düğümü çalışıyor... (Sıradaki: {step_name})"}

                    # Eğer grafik Artifact adımındaysa ve sonuç çıktıysa (başarılı bitiş)
                    if node_name == "artifact":
                        # LangGraph stub akışı tamamlandı, ASIL motoru (tool_loop) tetikliyoruz:
                        yield from self._tool_loop(user_msg, messages, session_id, session_metadata)
                        return
                            
                    # Kullanıcı onayı gerekirse (HITL - Human In The Loop)
                    if node_state.get("requires_approval") and not node_state.get("approval_result"):
                        yield {"type": "status", "content": "Kritik bir adım için onay bekleniyor."}
                        yield {"type": "assistant", "content": "Sistemin bu işlemi yapabilmesi için GUI üzerinden _DEVAM_ET_ onayı vermeniz gerekiyor."}
                        return
                        
        except Exception as e:
            log.error(f"LangGraph execution error: {str(e)}", exc_info=True)
            yield {"type": "error", "content": f"LangGraph Akış Hatası: {str(e)}"}
            yield {"type": "assistant", "content": f"Ajan hatası: {str(e)}"}

    # ─────────────────────────────────────────────
    #  _chat — Basit konuşma (tool yok)
    # ─────────────────────────────────────────────

    def _chat(
        self,
        user_msg: str,
        messages: List[Dict[str, str]],
        session_id: str,
        session_metadata: Optional[Dict],
    ) -> Generator[Dict[str, Any], None, None]:
        """Tool kullanmadan düz LLM yanıtı."""
        yield {"type": "status", "content": "💬 Yanıt hazırlanıyor..."}

        try:
            assistant = ""
            yield {"type": "assistant_start"}
            for chunk in self.llm.chat_stream(messages, session_id=session_id):
                assistant += chunk
                yield {"type": "chunk", "content": chunk}
            messages.append({"role": "assistant", "content": assistant})
            yield {"type": "assistant", "content": assistant}
        except Exception as e:
            yield {"type": "error", "content": str(e)}

        self._auto_save(messages, session_id, session_metadata)
        yield {"type": "done"}

    # ─────────────────────────────────────────────
    #  _tool_loop — Adım adım tool yürütme
    # ─────────────────────────────────────────────

    def _tool_loop(
        self,
        user_msg: str,
        messages: List[Dict[str, str]],
        session_id: str,
        session_metadata: Optional[Dict],
    ) -> Generator[Dict[str, Any], None, None]:
        """Ana tool yürütme döngüsü — agent.py main()'den taşındı."""

        consecutive_errors = 0
        last_error_sig = ""

        for step in range(self.config.max_steps):
            log.info("🔄 Adım %d/%d", step + 1, self.config.max_steps)
            yield {"type": "status", "content": f"🧠 Düşünüyor (adım {step + 1}/{self.config.max_steps})"}

            # ── Bellek özetleme ──
            try:
                from llm_backend import auto_create_backend, summarize_memory
                backend_for_mem = auto_create_backend(self.config.model)
                messages = summarize_memory(messages, backend_for_mem, threshold=40)
            except Exception as e:
                log.warning("Bellek özetleme atlandı: %s", e)

            # ── LLM çağrısı ──
            try:
                assistant = ""
                yield {"type": "assistant_start"}
                for chunk in self.llm.chat_stream(messages, session_id=session_id):
                    assistant += chunk
                    yield {"type": "chunk", "content": chunk}
            except Exception as e:
                yield {"type": "error", "content": str(e)}
                yield {"type": "done"}
                return

            # ── Tool ayrıştırma ──
            tools_to_run, outside = extract_tools(assistant)

            if not tools_to_run:
                # Fenced code block deneyelim
                py_m = FENCED_PY_RE.search(assistant)
                bash_m = FENCED_BASH_RE.search(assistant)
                if py_m and (not bash_m or len(py_m.group(1)) >= len(bash_m.group(1))):
                    tools_to_run = [("PYTHON", py_m.group(1))]
                    outside = FENCED_PY_RE.sub("", assistant).strip()
                elif bash_m:
                    tools_to_run = [("BASH", bash_m.group(1))]
                    outside = FENCED_BASH_RE.sub("", assistant).strip()
                else:
                    # Tool-First Policy: ilk 2 adımda aksiyon bekleniyor ama tool yok → retry
                    from services.agent_service import _is_action_request, TOOL_ENFORCEMENT_PROMPT
                    if _is_action_request(user_msg) and step < 2:
                        log.info("🔄 Tool-First retry (adım %d)", step + 1)
                        messages.append({"role": "assistant", "content": assistant})
                        messages.append({"role": "user", "content": TOOL_ENFORCEMENT_PROMPT})
                        continue

                    # Düz metin yanıt
                    messages.append({"role": "assistant", "content": assistant})
                    yield {"type": "assistant", "content": assistant}
                    self._store_memory(session_id, user_msg, assistant)
                    self._auto_save(messages, session_id, session_metadata)
                    yield {"type": "done"}
                    return

            if outside:
                log.warning("⚠️ Tool bloğu dışında metin: %d karakter", len(outside))

            messages.append({"role": "assistant", "content": assistant})

            # ── Tool'ları çalıştır ──
            all_outputs: List[Tuple[str, str]] = []
            break_loop = False

            for tool, payload in tools_to_run:
                log.info("🔧 Tool: %s | payload=%d", tool, len(payload or ""))
                yield {"type": "tool_start", "tool": tool}

                tool_start = time.time()
                try:
                    out = self._execute_tool(tool, payload, session_id)
                    elapsed_ms = (time.time() - tool_start) * 1000
                    log.info("✅ %s tamamlandı (%.0fms, %d karakter)", tool, elapsed_ms, len(out))

                except LLMConnectionError as e:
                    log.error("🧠 LLM bağlantı hatası: %s", e)
                    yield {"type": "error", "content": str(e)}
                    self._auto_save(messages, session_id, session_metadata)
                    break_loop = True
                    break

                except SecurityViolationError as e:
                    log.warning("🔒 Güvenlik: %s", e)
                    out = e.tool_output()

                except ToolTimeoutError as e:
                    log.error("⏰ Timeout: %s", e)
                    out = f"[TIMEOUT] {tool} timed out after {self.config.timeout}s"

                except (ToolExecutionError, FileOperationError, ValidationError) as e:
                    log.error("🛠️ Tool hatası: %s", e)
                    out = e.tool_output()

                except AgentError as e:
                    log.error("❌ Agent hatası: %s", e)
                    out = e.tool_output()

                except Exception as e:
                    log.error("💥 Beklenmeyen: %s", e, exc_info=True)
                    out = f"[UNEXPECTED_ERROR] {type(e).__name__}: {e}"

                # Web çıktılarını kaydet
                if tool in {"WEB_SEARCH", "WEB_OPEN"} and not out.startswith("["):
                    try:
                        autosave_web_outputs(self.config, tool, out)
                    except Exception:
                        pass

                yield {"type": "tool_output", "tool": tool, "output": out}
                all_outputs.append((tool, out))

                # Ardışık hata algılama
                is_err = self._is_error_output(out)
                if is_err:
                    err_sig = f"{tool}:{out[:80]}"
                    if err_sig == last_error_sig:
                        consecutive_errors += 1
                    else:
                        consecutive_errors = 1
                        last_error_sig = err_sig
                    if consecutive_errors >= 3:
                        log.warning("🔄 3 ardışık aynı hata — strateji değişikliği isteniyor")
                        all_outputs.append(("SYSTEM",
                            "⚠️ UYARI: Aynı hata 3 kez tekrarlandı. FARKLI yaklaşım dene. "
                            "Aynı komutu tekrar çalıştırma."
                        ))
                        consecutive_errors = 0
                else:
                    consecutive_errors = 0
                    last_error_sig = ""

            if break_loop:
                yield {"type": "done"}
                return

            # Sonraki adım mesajı
            user_feedback = ""
            for t, o in all_outputs:
                user_feedback += f"TOOL_OUTPUT ({t}):\n{o[:2000]}\n\n"
            user_feedback += (
                "---\n"
                "Tool çıktısını aldın. Planındaki SONRAKİ adıma geç.\n"
                "Her yanıtında MUTLAKA bir tool çağrısı olmalı.\n"
                "Tüm adımlar tamamlandıysa SON ÖZET'i yaz (tool olmadan).\n"
            )
            messages.append({"role": "user", "content": user_feedback})
            self._auto_save(messages, session_id, session_metadata)

        log.warning("⚠️ Max step (%d) | session=%s", self.config.max_steps, session_id)
        yield {"type": "status", "content": "⚠️ Maksimum adım sayısına ulaşıldı."}
        self._auto_save(messages, session_id, session_metadata)
        yield {"type": "done"}

    # ─────────────────────────────────────────────
    #  _swarm_pipeline — Multi-agent pipeline
    # ─────────────────────────────────────────────

    def _swarm_pipeline(
        self,
        user_msg: str,
        messages: List[Dict[str, str]],
        session_id: str,
        session_metadata: Optional[Dict],
    ) -> Generator[Dict[str, Any], None, None]:
        """SwarmOrchestrator'ı çağır."""
        yield {"type": "status", "content": "🐝 Swarm Orchestrator devrede..."}

        try:
            from swarm.orchestrator import SwarmOrchestrator
            if self._swarm is None:
                self._swarm = SwarmOrchestrator(self.config)
            assistant = self._swarm.process(messages)
            messages.append({"role": "assistant", "content": assistant})
            yield {"type": "assistant", "content": assistant}
            self._store_memory(session_id, user_msg, assistant)
        except Exception as e:
            log.error("Swarm hatası: %s", e, exc_info=True)
            yield {"type": "error", "content": f"❌ Swarm hatası: {e}"}

        self._auto_save(messages, session_id, session_metadata)
        yield {"type": "done"}

    # ─────────────────────────────────────────────
    #  Tool Executor
    # ─────────────────────────────────────────────

    def _execute_tool(self, tool: str, payload: str, session_id: str) -> str:
        """Tek bir tool'u çalıştır."""
        ws = self.config.workspace
        timeout = self.config.timeout

        if tool == "PYTHON":
            return run_python(payload, ws, timeout_s=timeout)
        elif tool == "BASH":
            return run_bash(payload, ws, timeout_s=timeout)
        elif tool == "WEB_SEARCH":
            from utils.config import get_config
            if not get_config().security.allow_web_search:
                return "[BLOCKED] WEB_SEARCH devre dışı."
            return web_search(payload)
        elif tool == "WEB_OPEN":
            return web_open(payload)
        elif tool == "BROWSER_OPEN":
            return browser_open(payload, session_id=session_id, workspace=ws)
        elif tool == "BROWSER_ACTION":
            return browser_action(payload, ws)
        elif tool == "BROWSER_AGENT":
            from ultra_agent.runtime.browser.browser_agent import run_browser_agent
            return run_browser_agent(payload, model=self.config.model, workspace=ws)
        elif tool == "READ_FILE":
            return read_file(payload, ws)
        elif tool == "WRITE_FILE":
            return write_file(payload, ws)
        elif tool == "VERSION_DATASET":
            return version_dataset(payload, ws)
        elif tool == "CLINICAL_VISION":
            from core.tools import clinical_vision
            return clinical_vision(payload, ws)
        elif tool == "TODO":
            return append_todo(payload, ws)
        elif tool == "RAG_SEARCH":
            results = self.rag.search(payload)
            if not results:
                return "[RAG_SEARCH] Sonuç bulunamadı."
            out = "[RAG_SEARCH] Bulunan metinler:\n\n"
            for i, r in enumerate(results, 1):
                out += f"--- Kaynak: {r['source']} (Mesafe: {r['distance']:.4f}) ---\n"
                out += f"{r['document']}\n\n"
            return out
        elif self.plugins.get(tool):
            return self.plugins.execute(tool, payload, ws)
        else:
            return f"[ERROR] Bilinmeyen tool: {tool}"

    # ─────────────────────────────────────────────
    #  Yardımcılar
    # ─────────────────────────────────────────────

    def _auto_save(
        self,
        messages: List[Dict[str, str]],
        session_id: str,
        metadata: Optional[Dict],
    ) -> None:
        """Konuşmayı otomatik kaydet."""
        try:
            save_conversation(
                self.config.history_dir, session_id,
                messages, metadata or {},
            )
        except Exception as e:
            log.warning("Otomatik kaydetme hatası: %s", e)

    def _store_memory(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        """Faz 6: Kalıcı hafızaya (Qdrant Semantic Memory) etkileşimi kaydet."""
        try:
            from ultra_agent.memory.qdrant_store import QdrantMemoryStore
            memory = QdrantMemoryStore()
            if memory.enabled:
                text_to_embed = f"User: {user_msg}\nAssistant: {assistant_msg}"
                memory.store_memory(
                    text=text_to_embed, 
                    source_file=f"session_{session_id}", 
                    project=str(self.config.workspace)
                )
                log.info("🧠 Etkileşim Qdrant Semantic Memory'e kaydedildi")
            else:
                # Fallback mekanizması
                from memory_manager import memory as legacy_memory
                legacy_memory.store_interaction(session_id, user_msg, assistant_msg)
                log.info("🧠 Etkileşim Legacy JSON hafızaya kaydedildi (Qdrant pasif)")
        except Exception as e:
            log.warning("Hafıza kaydetme hatası: %s", e)

    @staticmethod
    def _is_error_output(out: str) -> bool:
        """Tool çıktısının hata olup olmadığını kontrol et."""
        return any([
            out.startswith("[") and any(k in out[:80] for k in ("ERROR", "TIMEOUT", "UNEXPECTED", "BASH_ERROR")),
            "Dosya bulunamadı" in out[:200],
            "hata" in out[:200].lower(),
            "[python exit code:" in out[:80] and "exit code: 0" not in out[:80],
            "Traceback" in out[:200],
        ])
