import logging
from pathlib import Path
from typing import List, Dict, Optional
import gradio as gr
from bio_ml_agent.core.conversation import generate_session_id, list_conversations, load_conversation
from bio_ml_agent.utils.config import get_config

log = logging.getLogger("bio_ml_agent")
config = get_config()

def on_new_session(get_agent_service_func, model, timeout, max_steps):
    service = get_agent_service_func(model, int(timeout), int(max_steps))
    service.reset_session()
    sid = service.session_id[:12]
    return [], "Hazır — Yeni oturum", f"**Oturum:** `{sid}...`"

def on_refresh_sessions():
    history_dir = Path(config.history.directory).expanduser().resolve()
    sessions = list_conversations(history_dir, limit=50)
    if not sessions:
        return gr.update(choices=[], value=None)
    choices = []
    for s in sessions:
        label = f"{s['session_id'][:20]}  |  💬{s['message_count']}  |  {s['summary'][:40]}"
        choices.append((label, s['session_id']))
    return gr.update(choices=choices, value=None)

def on_load_session(session_id, get_agent_service_func, model, timeout, max_steps):
    if not session_id:
        return gr.update(), "Oturum seçilmedi.", gr.update()
    history_dir = Path(config.history.directory).expanduser().resolve()
    try:
        messages, metadata = load_conversation(history_dir, session_id)
        service = get_agent_service_func(model, int(timeout), int(max_steps))
        service.set_session(session_id, messages, metadata)
        chat_history = []
        for m in messages:
            if m["role"] == "system": continue
            chat_history.append({"role": m["role"], "content": m["content"][:2000]})
        sid = session_id[:8]
        proj = metadata.get("project_name", "—")
        return chat_history, f"✅ Oturum yüklendi", f"**Oturum:** `{sid}...`\n\n**Proje:** `{proj}`"
    except Exception as e:
        return gr.update(), f"❌ Hata: {str(e)}", gr.update()
