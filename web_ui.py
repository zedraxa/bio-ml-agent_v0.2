# web_ui.py
# ═══════════════════════════════════════════════════════════
#  Bio-ML Agent — Gradio Web Arayüzü
#  Çalıştırma: python web_ui.py
#  Tarayıcı: http://localhost:7860
# ═══════════════════════════════════════════════════════════

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Proje kökünü path'e ekle
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import load_config
from exceptions import (
    AgentError,
    LLMConnectionError,
    SecurityViolationError,
    ToolTimeoutError,
    ToolExecutionError,
    FileOperationError,
    ValidationError,
)
from agent import (
    SYSTEM_PROMPT,
    TOOL_RE,
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
    setup_logger,
    AgentConfig,
)

log = logging.getLogger("bio_ml_agent")

# ─────────────────────────────────────────────
#  Global Durum
# ─────────────────────────────────────────────
_cfg: Optional[AgentConfig] = None
_session_id: str = ""
_messages: List[Dict[str, str]] = []
_session_metadata: Dict[str, Any] = {}


def _init_config(
    model: str = "",
    workspace: str = "",
    timeout: int = 0,
    max_steps: int = 0,
) -> AgentConfig:
    """Web UI için agent config başlat."""
    app = load_config()

    return AgentConfig(
        model=model or app.agent.model,
        workspace=Path(workspace or app.workspace.base_dir).expanduser().resolve(),
        timeout=timeout or app.agent.timeout,
        max_steps=max_steps or app.agent.max_steps,
        history_dir=Path(app.history.directory).expanduser().resolve(),
    )


def _reset_session() -> None:
    """Yeni oturum başlat."""
    global _session_id, _messages, _session_metadata
    _session_id = generate_session_id()
    _session_metadata = {"created_at": datetime.now().isoformat()}
    _messages = [{"role": "system", "content": SYSTEM_PROMPT}]


# ─────────────────────────────────────────────
#  Tool Çalıştırma Motoru
# ─────────────────────────────────────────────

def _run_tool(tool: str, payload: str, cfg: AgentConfig, allow_web: bool) -> str:
    """Bir tool'u çalıştır ve sonucu döndür."""
    if tool == "PYTHON":
        return run_python(payload, cfg.workspace, timeout_s=cfg.timeout)
    elif tool == "BASH":
        return run_bash(payload, cfg.workspace, timeout_s=cfg.timeout)
    elif tool == "WEB_SEARCH":
        if not allow_web:
            return "[BLOCKED] WEB_SEARCH devre dışı. Etkinleştirmek için mesajınıza ALLOW_WEB_SEARCH ekleyin."
        return web_search(payload)
    elif tool == "WEB_OPEN":
        return web_open(payload)
    elif tool == "READ_FILE":
        return read_file(payload, cfg.workspace)
    elif tool == "WRITE_FILE":
        return write_file(payload, cfg.workspace)
    elif tool == "TODO":
        return append_todo(payload, cfg.workspace)
    else:
        return f"[ERROR] Bilinmeyen tool: {tool}"


def _format_tool_output(tool: str, output: str) -> str:
    """Tool çıktısını Markdown formatına çevir."""
    icon_map = {
        "PYTHON": "🐍",
        "BASH": "💻",
        "WEB_SEARCH": "🌐",
        "WEB_OPEN": "📖",
        "READ_FILE": "📄",
        "WRITE_FILE": "✍️",
        "TODO": "📝",
    }
    icon = icon_map.get(tool, "🛠️")

    # Tool çıktısını code block olarak formatla
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
        except (json.JSONDecodeError, TypeError):
            return f"**{icon} Web Arama:**\n```\n{output}\n```"
    else:
        return f"**{icon} {tool}:**\n```\n{output}\n```"


# ─────────────────────────────────────────────
#  Chat İşleyicisi (Gradio)
# ─────────────────────────────────────────────

def process_message(
    user_msg: str,
    chat_history: List[Dict[str, str]],
    model: str,
    timeout: int,
    max_steps: int,
    files: List[str] = None,
):
    """Kullanıcı mesajını işle ve yanıt döndür (streaming destekli).

    Yields:
        (güncellenmiş_chat_history, durum_metni)
    """
    global _cfg, _messages

    # Config güncelle
    if _cfg is None or _cfg.model != model:
        _cfg = _init_config(model=model, timeout=timeout, max_steps=max_steps)
        _cfg.workspace.mkdir(parents=True, exist_ok=True)

    if not _messages:
        _reset_session()

    # Web araması izni kontrolü
    allow_web = "ALLOW_WEB_SEARCH" in user_msg.upper()

    try:
        from memory_manager import memory
        mem_context = memory.get_context_string(user_msg, n_results=2)
        base_text = f"{mem_context}\n\n[Mevcut Görev/Soru]:\n{user_msg}" if mem_context else user_msg
        
        if files:
            content = [{"type": "text", "text": base_text}]
            for f in files:
                content.append({"type": "file", "path": f})
            _messages.append({"role": "user", "content": content})
        else:
            _messages.append({"role": "user", "content": base_text})
    except Exception as e:
        if files:
            content = [{"type": "text", "text": user_msg}]
            for f in files:
                content.append({"type": "file", "path": f})
            _messages.append({"role": "user", "content": content})
        else:
            _messages.append({"role": "user", "content": user_msg})

    status_parts = []

    # Multi-step tool loop
    for step in range(max_steps):
        # LLM'den yanıt al
        try:
            from llm_backend import auto_create_backend, summarize_memory
            backend = auto_create_backend(model)
            
            # Bağlam boyutu aşılmışsa özetle (threshold=15 mesaj)
            _messages = summarize_memory(_messages, backend, threshold=15)
            
            assistant = ""
            chat_history.append({"role": "assistant", "content": ""})
            for chunk in backend.chat_stream(_messages):
                assistant += chunk
                chat_history[-1]["content"] = assistant
                yield chat_history, f"Düşünüyor... (adım {step + 1})"

        except Exception as e:
            error_msg = f"❌ LLM Hatası: {e}"
            if chat_history and getattr(chat_history[-1], "get", lambda x: None)("role") == "assistant":
                chat_history[-1]["content"] = error_msg
            else:
                chat_history.append({"role": "assistant", "content": error_msg})
            yield chat_history, f"Hata: {e}"
            return

        # Tool tespit et
        tool, payload, outside = extract_tool(assistant)

        if tool is None:
            # Fenced code block kontrolü
            py_m = FENCED_PY_RE.search(assistant)
            bash_m = FENCED_BASH_RE.search(assistant)
            if py_m and (not bash_m or len(py_m.group(1)) >= len(bash_m.group(1))):
                tool, payload = "PYTHON", py_m.group(1)
                outside = FENCED_PY_RE.sub("", assistant).strip()
            elif bash_m:
                tool, payload = "BASH", bash_m.group(1)
                outside = FENCED_BASH_RE.sub("", assistant).strip()
            else:
                # Düz metin yanıtı
                _messages.append({"role": "assistant", "content": assistant})
                
                try:
                    from memory_manager import memory
                    memory.store_interaction(_session_id, user_msg, assistant)
                except Exception as e:
                    pass
                
                # chat_history, stream aşamasında son halini aldı zaten
                save_conversation(_cfg.history_dir, _session_id, _messages, _session_metadata)
                yield chat_history, f"✅ Tamamlandı (adım {step + 1})"
                return

        # Tool dışı metin varsa göster
        if outside:
            chat_history.append({"role": "assistant", "content": outside})

        # Tool'u çalıştır
        try:
            out = _run_tool(tool, payload, _cfg, allow_web)
            tool_display = _format_tool_output(tool, out)
        except AgentError as e:
            out = e.tool_output()
            tool_display = f"⚠️ **Hata ({type(e).__name__}):**\n```\n{e.user_message()}\n```"
        except Exception as e:
            out = f"[UNEXPECTED_ERROR] {type(e).__name__}: {e}"
            tool_display = f"❌ **Beklenmeyen Hata:**\n```\n{e}\n```"

        # Chat history'e tool çıktısını ekle
        chat_history.append({"role": "assistant", "content": tool_display})
        yield chat_history, f"Adım {step + 1}: {tool} tamamlandı"

        # İç mesaj listesini güncelle
        _messages.append({"role": "assistant", "content": assistant})
        _messages.append({
            "role": "user",
            "content": f"TOOL_OUTPUT ({tool}):\n{out}\n\nContinue. If done, answer normally (no tool).",
        })

        save_conversation(_cfg.history_dir, _session_id, _messages, _session_metadata)
        status_parts.append(f"Adım {step + 1}: {tool}")

    yield chat_history, f"⚠️ Maksimum adım ({max_steps}) aşıldı"


# ─────────────────────────────────────────────
#  Gradio Arayüzü
# ─────────────────────────────────────────────

def create_ui():
    """Gradio arayüzünü oluştur."""
    import gradio as gr

    app_config = load_config()

    # Koyu tema
    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.gray,
        font=gr.themes.GoogleFont("Inter"),
    )

    custom_css = """
    .gradio-container { max-width: 1200px !important; }
    .tool-output { background: #1e1e2e; border-radius: 8px; padding: 12px; }
    footer { display: none !important; }
    """

    with gr.Blocks(
        title="🧠 Bio-ML Agent",
    ) as demo:
        # Başlık
        gr.Markdown(
            "# 🧠 Bio-ML Agent\n"
            "**Yerel LLM destekli biyomühendislik ML proje asistanı**\n\n"
            "Merhaba! Bir ML projesi oluşturmak, veri analizi yapmak veya "
            "biyomühendislik soruları sormak için mesaj yazın."
        )

        with gr.Tabs():
            with gr.Tab("💬 Konuşma"):
                with gr.Row():
                    # Sol panel: Chat
                    with gr.Column(scale=4):
                        chatbot = gr.Chatbot(
                            label="💬 Konuşma",
                            height=550,
                            type="messages",
                        )
                        with gr.Row():
                            msg_input = gr.MultimodalTextbox(
                                label="Mesajınız (Görüntü/Tıbbi Veri eklenebilir)",
                                placeholder="Örn: Breast cancer analizini yap...",
                                file_types=["image", "audio"],
                                lines=2,
                                scale=4,
                            )
                            audio_input = gr.Audio(
                                sources=["microphone"], 
                                type="filepath", 
                                label="🎤 Sesli Komut", 
                                scale=1
                            )
                            send_btn = gr.Button("Gönder 🚀", variant="primary", scale=1)
                        status_box = gr.Textbox(
                            label="📊 Durum",
                            interactive=False,
                            value="Hazır",
                        )

                    # Sağ panel: Ayarlar
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ Ayarlar")
                        model_input = gr.Textbox(
                            label="Model",
                            value=app_config.agent.model,
                            info="Ollama model adı",
                        )
                        timeout_input = gr.Slider(
                            label="Timeout (s)",
                            minimum=30,
                            maximum=600,
                            value=app_config.agent.timeout,
                            step=30,
                        )
                        max_steps_input = gr.Slider(
                            label="Maks. Adım",
                            minimum=1,
                            maximum=30,
                            value=app_config.agent.max_steps,
                            step=1,
                        )

                        gr.Markdown("---")
                        gr.Markdown("### 📋 Bilgi")
                        session_info = gr.Markdown(
                            f"**Oturum:** `{generate_session_id()[:12]}...`\n\n"
                            f"**Workspace:** `{app_config.workspace.base_dir}`"
                        )

                        new_session_btn = gr.Button("🔄 Yeni Oturum", variant="secondary")

            with gr.Tab("📂 Data Explorer"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Workspace Dosyaları")
                        gr.Markdown("Desteklenen türler: CSV, JSON, TXT, LOG, HTML")
                        file_dropdown = gr.Dropdown(label="Dosya Seç", choices=[], interactive=True)
                        refresh_files_btn = gr.Button("🔄 Yenile")
                    with gr.Column(scale=3):
                        data_preview = gr.Dataframe(label="Veri Önizleme", interactive=False, visible=False)
                        text_preview = gr.Textbox(label="Metin Önizleme", lines=20, max_lines=40, interactive=False, visible=True)
                        html_preview = gr.HTML(label="HTML Önizleme", visible=False)

                def update_file_list():
                    work_dir = _cfg.workspace if _cfg else Path("workspace")
                    if not work_dir.exists():
                        return gr.update(choices=[])
                    files = [str(p.relative_to(work_dir)) for p in work_dir.rglob("*") 
                             if p.is_file() and p.suffix.lower() in ['.csv', '.json', '.txt', '.log', '.html']]
                    return gr.update(choices=sorted(files))

                def preview_file(filepath):
                    if not filepath:
                        return gr.update(visible=False), gr.update(value="", visible=True), gr.update(visible=False)
                    
                    work_dir = _cfg.workspace if _cfg else Path("workspace")
                    full_path = work_dir / filepath
                    if not full_path.exists():
                        return gr.update(visible=False), gr.update(value="Dosya bulunamadı.", visible=True), gr.update(visible=False)
                    
                    try:
                        ext = full_path.suffix.lower()
                        if ext == '.csv':
                            import pandas as pd
                            df = pd.read_csv(full_path, nrows=100)
                            return gr.update(value=df, visible=True), gr.update(visible=False), gr.update(visible=False)
                        elif ext == '.json':
                            import pandas as pd
                            try:
                                df = pd.read_json(full_path)
                                return gr.update(value=df.head(100), visible=True), gr.update(visible=False), gr.update(visible=False)
                            except ValueError:
                                with open(full_path, 'r', encoding='utf-8') as f:
                                    text = f.read(10000)
                                return gr.update(visible=False), gr.update(value=text, visible=True), gr.update(visible=False)
                        elif ext == '.html':
                            with open(full_path, 'r', encoding='utf-8') as f:
                                html_text = f.read()
                            return gr.update(visible=False), gr.update(visible=False), gr.update(value=html_text, visible=True)
                        else: # txt, log
                            with open(full_path, 'r', encoding='utf-8') as f:
                                text = f.read(10000)
                            return gr.update(visible=False), gr.update(value=text, visible=True), gr.update(visible=False)
                    except Exception as e:
                        return gr.update(visible=False), gr.update(value=f"Hata: {str(e)}", visible=True), gr.update(visible=False)

                refresh_files_btn.click(fn=update_file_list, outputs=file_dropdown)
                file_dropdown.change(fn=preview_file, inputs=file_dropdown, outputs=[data_preview, text_preview, html_preview])
                demo.load(fn=update_file_list, outputs=file_dropdown)

        # Event handlers
        def on_send(user_data, audio_path, history, model, timeout, max_steps):
            if isinstance(user_data, dict):
                user_msg = user_data.get("text", "")
                files = user_data.get("files", [])
            else:
                user_msg = user_data if user_data else ""
                files = []

            if audio_path:
                files.append(audio_path)

            if not user_msg.strip() and not files:
                yield history, gr.update(), gr.update(), "Boş mesaj gönderilemez."
                return
            
            history = history or []
            
            # Formulate chat history message text based on files vs pure text
            display_text = user_msg if user_msg else "(Multimodal Dosya İletildi)"
            history.append({"role": "user", "content": display_text})
            
            yield history, gr.update(value=None), gr.update(value=None), "Başlatılıyor..."
            
            for updated_history, status in process_message(
                user_msg, history, model, int(timeout), int(max_steps), files=files
            ):
                yield updated_history, gr.update(), gr.update(), status

        def on_new_session():
            _reset_session()
            sid = _session_id[:12]
            return [], "Hazır — Yeni oturum", f"**Oturum:** `{sid}...`"

        # Gönder butonu
        send_btn.click(
            fn=on_send,
            inputs=[msg_input, audio_input, chatbot, model_input, timeout_input, max_steps_input],
            outputs=[chatbot, msg_input, audio_input, status_box],
        )

        # Enter tuşu
        msg_input.submit(
            fn=on_send,
            inputs=[msg_input, audio_input, chatbot, model_input, timeout_input, max_steps_input],
            outputs=[chatbot, msg_input, audio_input, status_box],
        )

        # Yeni oturum
        new_session_btn.click(
            fn=on_new_session,
            outputs=[chatbot, status_box, session_info],
        )

    demo._bio_theme = theme
    demo._bio_css = custom_css
    return demo


# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────

def main():
    """Web arayüzünü başlat."""
    # Logger kur
    log_dir = Path("logs").resolve()
    log_dir.mkdir(exist_ok=True)
    global log
    log = setup_logger(log_dir, "INFO")

    # Config başlat
    global _cfg
    _cfg = _init_config()
    _cfg.workspace.mkdir(parents=True, exist_ok=True)

    # Oturum başlat
    _reset_session()

    print("🧠 Bio-ML Agent Web Arayüzü başlatılıyor...")
    print(f"   Model: {_cfg.model}")
    print(f"   Workspace: {_cfg.workspace}")
    print(f"   Oturum: {_session_id}")
    print()

    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=demo._bio_theme,
        css=demo._bio_css,
    )


if __name__ == "__main__":
    main()
