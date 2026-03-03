# services/agent_service.py
import os
import re as _re
import sys
import json
import logging
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Generator, Tuple

# Proje kökü importları
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from utils.config import load_config
from exceptions import AgentError
from agent import (
    SYSTEM_PROMPT,
    FENCED_PY_RE,
    FENCED_BASH_RE,
    extract_tool,
    run_python,
    run_bash,
    web_search,
    web_open,
    browser_open,
    browser_action,
    read_file,
    write_file,
    append_todo,
    save_conversation,
    generate_session_id,
    AgentConfig,
)
from memory_manager import memory
from llm_backend import auto_create_backend, summarize_memory

log = logging.getLogger("bio_ml_agent")


def _slugify_project_text(text: str, max_words: int = 6, max_len: int = 48) -> str:
    """Kullanıcı mesajından proje klasörü adı için ASCII slug üretir."""
    text = text or "untitled-project"
    # ALLOW_WEB_SEARCH gibi direktifleri temizle
    text = _re.sub(r"ALLOW_WEB_SEARCH", "", text, flags=_re.IGNORECASE).strip()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    words = _re.findall(r"[a-z0-9]+", text)
    if not words:
        return "untitled-project"
    slug = "-".join(words[:max_words]).strip("-")
    return slug[:max_len] or "untitled-project"

# ─────────────────────────────────────────────
#  Tool-First Policy: Action Request Detection
# ─────────────────────────────────────────────
_ACTION_KEYWORDS = [
    "oluştur", "olustur", "yaz", "kur", "indir", "kaydet", "eğit", "egit",
    "temizle", "analiz", "karşılaştır", "karsilastir", "grafik", "rapor",
    "create", "build", "download", "save", "train", "write", "generate",
    "proje", "project", "dosya", "file", "model", "plot", "report",
    "pipeline", "csv", "dataset", "veri set",
]

def _is_action_request(msg: str) -> bool:
    """Mesajın dosya/proje oluşturma gibi aksiyon gerektiren bir istek olup olmadığını algılar."""
    msg_lower = msg.lower()
    hits = sum(1 for kw in _ACTION_KEYWORDS if kw in msg_lower)
    return hits >= 2

TOOL_ENFORCEMENT_PROMPT = (
    "UYARI: Kullanıcı dosya oluşturma, veri indirme veya kod çalıştırma istedi "
    "ama sen hiç tool çağrısı yapmadın. Bu KABUL EDİLEMEZ.\n\n"
    "ŞİMDİ şu tool'lardan birini MUTLAKA kullan:\n"
    "- <WRITE_FILE> ile dosya oluştur (plan.md veya proje dosyası)\n"
    "- <PYTHON> ile kod çalıştır\n"
    "- <BASH> ile komut çalıştır (curl/wget ile veri indir)\n"
    "- <WEB_SEARCH> ile veri kaynağı ara\n\n"
    "İlk adım olarak bir plan.md dosyası yaz, sonra her adımı sırayla tool ile uygula.\n"
    "ASLA düz metin açıklama yapma — tool çağrısı ZORUNLUDUR."
)

CONTINUE_PROMPT_TEMPLATE = (
    "TOOL_OUTPUT ({tool}):\n{output}\n\n"
    "---\n"
    "Yukarıdaki tool çıktısını aldın. Planındaki bir SONRAKİ adıma geç.\n"
    "BİR SONRAKİ dosyayı oluştur veya bir sonraki komutu çalıştır.\n"
    "Her yanıtında MUTLAKA bir tool çağrısı (<WRITE_FILE>, <PYTHON>, <BASH>, <WEB_SEARCH>) olmalı.\n"
    "Tüm adımlar tamamlandıysa ve tüm dosyalar disk'e yazıldıysa, SON ÖZET'i yaz (tool olmadan).\n"
    "AMA henüz eksik dosya varsa — DEVAM ET, tool kullan!"
)

class AgentService:
    """Ajanın UI'den bağımsız (headless) olarak çalışmasını sağlayan core servis katmanı.
    Gradio, FastAPI, CLI ve WhatsApp bu katmanı ortak kullanacaktır."""
    
    def __init__(self, model: str = "", workspace: str = "", timeout: int = 0, max_steps: int = 0):
        app_config = load_config()
        self.config = AgentConfig(
            model=model or app_config.agent.model,
            workspace=Path(workspace or app_config.workspace.base_dir).expanduser().resolve(),
            timeout=timeout or app_config.agent.timeout,
            max_steps=max_steps or app_config.agent.max_steps,
            history_dir=Path(app_config.history.directory).expanduser().resolve(),
        )
        self.config.workspace.mkdir(parents=True, exist_ok=True)
        self.session_id = generate_session_id()
        self.session_metadata = {"created_at": datetime.now().isoformat()}
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Proje bağlamı — ilk kullanıcı mesajında doldurulur
        self.project_name: Optional[str] = None
        self.project_root: Optional[Path] = None

        # Çalışma Modları:
        # 1 = Tam Otomatik (onay yok)
        # 2 = Adım bazlı onay (her N adımda dur)
        # 3 = Akıllı onay (kritik tool'larda dur)
        # 4 = Plan onayı + checkpoint (planda dur + N. adımda dur)
        self.approval_mode: int = 1
        self.approval_interval: int = 5  # Mod 2: her N adımda dur
        self.checkpoint_step: int = 50    # Mod 4: bu adımda dur
        self._current_step: int = 0       # Mevcut adım sayacı (oturumlar arası korunur)
        self._paused: bool = False         # Duraklatılmış mı?
        self._plan_approved: bool = False  # Mod 4: plan onaylandı mı?

        # Swarm (Alt Ajan) Modu
        self.swarm_enabled: bool = False
        self._swarm_orchestrator = None

    def set_session(self, session_id: str, messages: List[Dict], metadata: Dict = None):
        """Mevcut bir oturumu geri yükle."""
        self.session_id = session_id
        self.messages = messages
        if metadata:
            self.session_metadata = metadata
            # Proje bağlamını geri yükle
            proj_name = metadata.get("project_name")
            proj_path = metadata.get("project_path")
            if proj_name:
                self.project_name = proj_name
                self.project_root = Path(proj_path) if proj_path else self.config.workspace / proj_name
                self.project_root.mkdir(parents=True, exist_ok=True)
                os.environ["AGENT_PROJECT"] = proj_name
                log.info("📁 Proje bağlamı geri yüklendi: %s", proj_name)

    def reset_session(self):
        """Oturumu sıfırla — yeni proje bağlamı da temizlenir."""
        self.session_id = generate_session_id()
        self.session_metadata = {"created_at": datetime.now().isoformat()}
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.project_name = None
        self.project_root = None
        os.environ.pop("AGENT_PROJECT", None)

    def _ensure_project_context(self, user_msg: str) -> None:
        """İlk kullanıcı mesajında otomatik proje klasörü oluşturur.
        Mevcut write_file() zaten AGENT_PROJECT env'ini okuduğu için
        burası set edilince tüm dosyalar doğru yere yazılır."""
        if self.project_name:
            return  # Zaten oluşturulmuş

        date_prefix = datetime.now().strftime("%Y-%m-%d")
        short_sid = self.session_id.split("_")[-1][:8]
        slug = _slugify_project_text(user_msg)
        self.project_name = f"{date_prefix}_{slug}_{short_sid}"
        self.project_root = self.config.workspace / self.project_name
        self.project_root.mkdir(parents=True, exist_ok=True)

        os.environ["AGENT_PROJECT"] = self.project_name
        log.info("📁 Proje oluşturuldu: %s", self.project_name)

        self.session_metadata.update({
            "project_name": self.project_name,
            "project_path": str(self.project_root),
            "session_id": self.session_id,
            "first_user_prompt": user_msg[:200],
        })

        # project.json index dosyası
        project_meta = {
            "project_name": self.project_name,
            "session_id": self.session_id,
            "created_at": datetime.now().isoformat(),
            "first_prompt": user_msg[:300],
            "status": "active",
        }
        try:
            (self.project_root / "project.json").write_text(
                json.dumps(project_meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

    def _run_tool(self, tool: str, payload: str, allow_web: bool) -> str:
        """Belirtilen tool'u güvenlik ve time limit çerçevesinde çalıştırır."""
        if tool == "PYTHON":
            return run_python(payload, self.config.workspace, timeout_s=self.config.timeout)
        elif tool == "BASH":
            return run_bash(payload, self.config.workspace, timeout_s=self.config.timeout)
        elif tool == "WEB_SEARCH":
            if not allow_web:
                return "[BLOCKED] WEB_SEARCH devre dışı. Etkinleştirmek için mesajınıza ALLOW_WEB_SEARCH ekleyin."
            return web_search(payload)
        elif tool == "WEB_OPEN":
            return web_open(payload)
        elif tool == "BROWSER_OPEN":
            return browser_open(payload)
        elif tool == "BROWSER_ACTION":
            return browser_action(payload, self.config.workspace)
        elif tool == "BROWSER_AGENT":
            from browser_agent import run_browser_agent
            return run_browser_agent(payload, model=self.config.model, workspace=self.config.workspace)
        elif tool == "READ_FILE":
            return read_file(payload, self.config.workspace)
        elif tool == "WRITE_FILE":
            return write_file(payload, self.config.workspace)
        elif tool == "TODO":
            return append_todo(payload, self.config.workspace)
        else:
            return f"[ERROR] Bilinmeyen tool: {tool}"

    def _format_tool_output(self, tool: str, output: str) -> str:
        """Arayüz dökümleri (Markdown) için aracı çıktılarını şekillendirir."""
        icon_map = {
            "PYTHON": "🐍", "BASH": "💻", "WEB_SEARCH": "🌐", 
            "WEB_OPEN": "📖", "READ_FILE": "📄", "WRITE_FILE": "✍️", "TODO": "📝"
        }
        icon = icon_map.get(tool, "🛠️")
        
        if tool in {"PYTHON", "BASH"}:
            return f"**{icon} {tool} Çıktısı:**\n```\n{output}\n```"
        elif tool == "WEB_SEARCH":
            try:
                results = json.loads(output)
                lines = [f"**{icon} Web Arama Sonuçları:**\n"]
                for r in results[:5]:
                    lines.append(f"- [{r.get('title', 'N/A')}]({r.get('href', '#')})")
                    lines.append(f"  _{r.get('body', '')[:120]}_\n")
                return "\n".join(lines)
            except:
                return f"**{icon} Web Arama:**\n```\n{output}\n```"
        else:
            return f"**{icon} {tool}:**\n```\n{output}\n```"

    def _needs_approval(self, step: int, tool: str = None) -> bool:
        """Mevcut adımda kullanıcı onayı gerekip gerekmediğini kontrol eder."""
        mode = self.approval_mode
        if mode == 1:  # Tam Otomatik
            return False
        elif mode == 2:  # Adım bazlı onay
            return step > 0 and step % self.approval_interval == 0
        elif mode == 3:  # Akıllı onay (sadece kritik tool'larda)
            critical_tools = {"BASH", "WRITE_FILE", "WEB_SEARCH", "WEB_OPEN"}
            return tool in critical_tools
        elif mode == 4:  # Plan onayı + checkpoint
            if step == 1 and not self._plan_approved:
                return True  # İlk adımda plan onayı iste
            if step > 0 and step == self.checkpoint_step:
                return True  # Checkpoint adımında dur
            return False
        return False

    def process_message(self, user_msg: str, files: List[str] = None) -> Generator[Dict[str, Any], None, None]:
        """Ajanın mesaj mantığını işler ve olayları dışarı stream eder.
        Frontend (web_ui.py veya WhatsApp) sadece bu olayları dinleyerek update alır."""

        # Devam sinyali kontrolü — duraklatılmış ajan devam ediyor
        if user_msg == "_DEVAM_ET_":
            self._paused = False
            if self.approval_mode == 4 and not self._plan_approved:
                self._plan_approved = True
            # Devam mesajını LLM'e gönder
            self.messages.append({"role": "user", "content": "Kullanıcı onay verdi. Devam et, bir sonraki adıma geç."})
            # Step counter korunuyor, for loop devam edecek
        else:
            self._paused = False
        
        allow_web = "ALLOW_WEB_SEARCH" in user_msg.upper()

        # Session'a özgü proje klasörü oluştur
        self._ensure_project_context(user_msg)
        
        try:
            mem_context = memory.get_context_string(user_msg, n_results=2)
            base_text = f"{mem_context}\n\n[Mevcut Görev/Soru]:\n{user_msg}" if mem_context else user_msg
            
            if files:
                content = [{"type": "text", "text": base_text}]
                for f in files:
                    content.append({"type": "file", "path": f})
                self.messages.append({"role": "user", "content": content})
            else:
                self.messages.append({"role": "user", "content": base_text})
        except Exception:
            if files:
                content = [{"type": "text", "text": user_msg}]
                for f in files:
                    content.append({"type": "file", "path": f})
                self.messages.append({"role": "user", "content": content})
            else:
                self.messages.append({"role": "user", "content": user_msg})

        # ═══ SWARM MODU ═══
        if self.swarm_enabled and user_msg != "_DEVAM_ET_":
            yield {"type": "status", "content": "🐝 Swarm modu aktif — alt ajanlara yönlendiriliyor..."}
            try:
                from swarm.orchestrator import SwarmOrchestrator
                if self._swarm_orchestrator is None:
                    self._swarm_orchestrator = SwarmOrchestrator(self.config)
                
                yield {"type": "assistant_start"}
                yield {"type": "status", "content": "🐝 Swarm Orchestrator görev dağıtıyor..."}
                
                result = self._swarm_orchestrator.process(self.messages)
                
                self.messages.append({"role": "assistant", "content": result})
                yield {"type": "chunk", "content": result}
                
                try:
                    memory.store_interaction(self.session_id, user_msg, result)
                except Exception:
                    pass
                save_conversation(self.config.history_dir, self.session_id, self.messages, self.session_metadata)
                yield {"type": "status", "content": "✅ Swarm tamamlandı"}
                yield {"type": "done"}
                return
            except Exception as e:
                log.warning("Swarm hatası, monolitik moda geri dönülüyor: %s", e)
                yield {"type": "status", "content": f"⚠️ Swarm hatası: {e} — standart moda geçiliyor..."}
                # Hata durumunda standart döngüye devam et

        for step in range(self.config.max_steps):
            yield {"type": "status", "content": f"Düşünüyor... (adım {step + 1})"}
            
            try:
                backend = auto_create_backend(self.config.model)
                self.messages = summarize_memory(self.messages, backend, threshold=15)
                
                assistant = ""
                yield {"type": "assistant_start"}
                
                for chunk in backend.chat_stream(self.messages):
                    assistant += chunk
                    yield {"type": "chunk", "content": chunk}
                
            except Exception as e:
                yield {"type": "error", "content": f"LLM Hatası: {e}"}
                return

            # Tool ayrıştırma
            tool, payload, outside = extract_tool(assistant)
            
            if tool is None:
                py_m = FENCED_PY_RE.search(assistant)
                bash_m = FENCED_BASH_RE.search(assistant)
                if py_m and (not bash_m or len(py_m.group(1)) >= len(bash_m.group(1))):
                    tool, payload = "PYTHON", py_m.group(1)
                    outside = FENCED_PY_RE.sub("", assistant).strip()
                elif bash_m:
                    tool, payload = "BASH", bash_m.group(1)
                    outside = FENCED_BASH_RE.sub("", assistant).strip()
                else:
                    # Tool-First Policy: Aksiyon isteğiyse retry gönder
                    if _is_action_request(user_msg) and step < self.config.max_steps - 1:
                        log.info("🔄 Tool-First Policy: Aksiyon isteği ama tool yok, retry gönderiliyor (adım %d)", step + 1)
                        self.messages.append({"role": "assistant", "content": assistant})
                        enforce_msg = (
                            "UYARI: Planında henüz tamamlanmamış adımlar var ama tool çağrısı yapmadın.\n"
                            "Bir sonraki adıma geç ve MUTLAKA bir tool kullan:\n"
                            "<WRITE_FILE>, <PYTHON>, <BASH>, veya <WEB_SEARCH>\n"
                            "Düz metin açıklama YASAK — tool çağrısı ZORUNLU!"
                        )
                        self.messages.append({
                            "role": "user",
                            "content": enforce_msg,
                        })
                        yield {"type": "status", "content": f"🔄 Tool zorunluluğu (adım {step + 1})"}
                        continue
                    else:
                        # Son adım veya basit soru-cevap — kapat
                        self.messages.append({"role": "assistant", "content": assistant})
                        try:
                            memory.store_interaction(self.session_id, user_msg, assistant)
                        except Exception:
                            pass
                        save_conversation(self.config.history_dir, self.session_id, self.messages, self.session_metadata)
                        yield {"type": "status", "content": f"✅ Tamamlandı (adım {step + 1})"}
                        yield {"type": "done"}
                        return

            if outside:
                yield {"type": "chunk", "content": f"\n\n{outside}"}
                
            yield {"type": "status", "content": f"Çalıştırılıyor: {tool} (adım {step + 1})"}
            yield {"type": "tool_start", "tool": tool, "payload": payload}
            
            try:
                out = self._run_tool(tool, payload, allow_web)
                formatted_out = self._format_tool_output(tool, out)
            except AgentError as e:
                out = e.tool_output()
                formatted_out = f"⚠️ **Hata ({type(e).__name__}):**\n```\n{e.user_message()}\n```"
            except Exception as e:
                out = f"[UNEXPECTED_ERROR] {type(e).__name__}: {e}"
                formatted_out = f"❌ **Beklenmeyen Hata:**\n```\n{e}\n```"

            yield {"type": "tool_output", "tool": tool, "output": out, "formatted": formatted_out}

            self.messages.append({"role": "assistant", "content": assistant})
            
            # Agresif devam prompt'u — plan adımlarını takip ettir
            continue_msg = CONTINUE_PROMPT_TEMPLATE.format(tool=tool, output=out[:2000])
            self.messages.append({
                "role": "user",
                "content": continue_msg,
            })
            save_conversation(self.config.history_dir, self.session_id, self.messages, self.session_metadata)

            # Onay kontrolü — gerekiyorsa duraklat
            self._current_step = step + 1
            if self._needs_approval(step + 1, tool):
                self._paused = True
                reason = {
                    2: f"Her {self.approval_interval} adımda onay gerekiyor (adım {step + 1})",
                    3: f"Kritik araç ({tool}) çalıştırıldı — onay bekleniyor",
                    4: f"Checkpoint adımına ulaşıldı (adım {step + 1})" if self._plan_approved else f"Plan hazır — onay bekleniyor (adım {step + 1})",
                }.get(self.approval_mode, f"Onay bekleniyor (adım {step + 1})")
                yield {
                    "type": "approval_required",
                    "step": step + 1,
                    "reason": reason,
                    "content": f"⏸️ {reason}\n\nDevam etmek için 'Devam Et' butonuna basın.",
                }
                return  # Durakla — kullanıcı _DEVAM_ET_ gönderince devam edecek

        yield {"type": "status", "content": f"⚠️ Maksimum adım ({self.config.max_steps}) aşıldı"}
        yield {"type": "done"}
