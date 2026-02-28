# services/agent_service.py
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Generator

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

    def set_session(self, session_id: str, messages: List[Dict], metadata: Dict = None):
        """Mevcut bir oturumu geri yükle."""
        self.session_id = session_id
        self.messages = messages
        if metadata:
            self.session_metadata = metadata

    def reset_session(self):
        """Oturumu sıfırla."""
        self.session_id = generate_session_id()
        self.session_metadata = {"created_at": datetime.now().isoformat()}
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

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

    def process_message(self, user_msg: str, files: List[str] = None) -> Generator[Dict[str, Any], None, None]:
        """Ajanın mesaj mantığını işler ve olayları dışarı stream eder.
        Frontend (web_ui.py veya WhatsApp) sadece bu olayları dinleyerek update alır."""
        
        allow_web = "ALLOW_WEB_SEARCH" in user_msg.upper()
        
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
            self.messages.append({
                "role": "user",
                "content": f"TOOL_OUTPUT ({tool}):\n{out}\n\nContinue. If done, answer normally (no tool).",
            })
            save_conversation(self.config.history_dir, self.session_id, self.messages, self.session_metadata)

        yield {"type": "status", "content": f"⚠️ Maksimum adım ({self.config.max_steps}) aşıldı"}
        yield {"type": "done"}
