import logging
import time
from typing import List, Dict, Optional, Any
from pathlib import Path
import gradio as gr
from bio_ml_agent.services.agent_service import AgentService

log = logging.getLogger("bio_ml_agent")

_agent_service: Optional[AgentService] = None

def get_agent_service(model: str = None, timeout: int = 300, max_steps: int = 50) -> AgentService:
    global _agent_service
    if _agent_service is None:
        _agent_service = AgentService(model=model or "gemini-2.0-flash", timeout=timeout, max_steps=max_steps)
    elif model and _agent_service.config.model != model:
         _agent_service = AgentService(model=model, timeout=timeout, max_steps=max_steps)
    return _agent_service

def get_active_project_name() -> Optional[str]:
    global _agent_service
    if _agent_service:
        return _agent_service.project_name
    return None

def process_message(
    user_msg: str,
    chat_history: List[Dict[str, str]],
    model: str,
    timeout: int,
    max_steps: int,
    files: List[str] = None,
    local_mode: bool = True,
):
    if not local_mode:
        chat_history.append({"role": "assistant", "content": "🌐 Cloud mode not fully implemented."})
        yield chat_history, "Hata"
        return

    service = get_agent_service(model, timeout, max_steps)
    if not chat_history:
        service.reset_session()

    for event in service.process_message(user_msg, files):
        ev_type = event.get("type")
        if ev_type == "status":
            yield chat_history, event.get("content", "")
        elif ev_type == "assistant_start":
            chat_history.append({"role": "assistant", "content": ""})
        elif ev_type in ["chunk", "assistant"]:
            if chat_history and chat_history[-1]["role"] == "assistant":
                chat_history[-1]["content"] += event.get("content", "")
            yield chat_history, "Düşünüyor..."
        elif ev_type == "tool_start":
            yield chat_history, f"Araç: {event.get('tool')}"
        elif ev_type == "tool_output":
            chat_history.append({"role": "assistant", "content": event.get("formatted", "")})
            yield chat_history, "Tamamlandı"
        elif ev_type in ["approval_needed", "approval_required"]:
            msg = f"⏸️ **Onay Gerekiyor**\n\n{event.get('content', 'Devam?')}"
            chat_history.append({"role": "assistant", "content": msg})
            yield chat_history, "Onay bekleniyor"
        elif ev_type == "error":
            msg = f"❌ {event.get('content', '')}"
            chat_history.append({"role": "assistant", "content": msg})
            yield chat_history, "Hata"
        elif ev_type == "done":
            break

MEDIA_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.svg',
                    '.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac',
                    '.mp4', '.webm', '.mov', '.avi'}

def on_send(user_data, audio_path, history, model, timeout, max_steps, mode, interval, checkpoint, swarm, is_local_mode):
    if isinstance(user_data, dict):
        user_msg = user_data.get("text", "")
        files = user_data.get("files", [])
    else:
        user_msg = user_data if user_data else ""
        files = []

    if audio_path:
        files.append(audio_path)

    if not user_msg.strip() and not files:
        yield history, gr.update(), gr.update(), "Boş mesaj gönderilemez.", gr.update(visible=False)
        return

    history = history or []
    service = get_agent_service(model, int(timeout), int(max_steps))
    service.approval_mode = int(mode)
    service.approval_interval = int(interval)
    service.checkpoint_step = int(checkpoint)
    service.swarm_enabled = bool(swarm)

    for f_path in files:
        fp = f_path if isinstance(f_path, str) else str(f_path)
        ext = Path(fp).suffix.lower()
        if ext in MEDIA_EXTENSIONS:
            history.append({"role": "user", "content": {"path": fp}})
        else:
            fname = Path(fp).name
            history.append({"role": "user", "content": f"📎 **{fname}** (Dosya eklendi)"})

    if user_msg.strip():
        history.append({"role": "user", "content": user_msg.strip()})
    
    yield history, gr.update(value=None), gr.update(value=None), "Başlatılıyor...", gr.update(visible=False)
    
    show_continue = False
    for updated_history, status in process_message(
        user_msg, history, model, int(timeout), int(max_steps), files=files, local_mode=bool(is_local_mode)
    ):
        show_continue = "⏸️" in status
        yield updated_history, gr.update(), gr.update(), status, gr.update(visible=show_continue)

def on_continue(history, model, timeout, max_steps, mode, interval, checkpoint, swarm):
    service = get_agent_service(model, int(timeout), int(max_steps))
    service.approval_mode = int(mode)
    service.approval_interval = int(interval)
    service.checkpoint_step = int(checkpoint)
    service.swarm_enabled = bool(swarm)
    
    history = history or []
    yield history, "▶️ Devam ediliyor...", gr.update(visible=False)
    
    for updated_history, status in process_message(
        "_DEVAM_ET_", history, model, int(timeout), int(max_steps)
    ):
        show_continue = "⏸️" in status
        yield updated_history, status, gr.update(visible=show_continue)

def try_read_as_text(filepath: str) -> Optional[str]:
    try:
        p = Path(filepath)
        raw = p.read_bytes()
        if b'\x00' in raw[:8192]: return None
        content = raw.decode('utf-8', errors='replace')
        return content[:50000]
    except Exception: return None
