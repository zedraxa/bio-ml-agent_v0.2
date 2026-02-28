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
from services.agent_service import AgentService
from agent import setup_logger, generate_session_id

log = logging.getLogger("bio_ml_agent")

# ─────────────────────────────────────────────
#  Core Service Entegrasyonu
# ─────────────────────────────────────────────
_agent_service: Optional[AgentService] = None

def get_agent_service(model: str, timeout: int, max_steps: int) -> AgentService:
    global _agent_service
    if _agent_service is None or _agent_service.config.model != model:
        _agent_service = AgentService(model=model, timeout=timeout, max_steps=max_steps)
    return _agent_service


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
    """Kullanıcı mesajını işle ve AgentService'den gelen stream eventlerini Gradio'ya aktar."""
    service = get_agent_service(model, timeout, max_steps)

    if not chat_history:
        service.reset_session()

    for event in service.process_message(user_msg, files):
        ev_type = event.get("type")
        
        if ev_type == "status":
            yield chat_history, event.get("content", "")
            
        elif ev_type == "assistant_start":
            chat_history.append({"role": "assistant", "content": ""})
            
        elif ev_type == "chunk":
            if not chat_history or chat_history[-1]["role"] != "assistant":
                chat_history.append({"role": "assistant", "content": ""})
            chat_history[-1]["content"] += event.get("content", "")
            yield chat_history, "Düşünüyor..."
            
        elif ev_type == "tool_start":
            tool_name = event.get("tool")
            yield chat_history, f"Araç çalıştırılıyor: {tool_name}"
            
        elif ev_type == "tool_output":
            formatted = event.get("formatted", "")
            chat_history.append({"role": "assistant", "content": formatted})
            yield chat_history, "Araç tamamlandı."
            
        elif ev_type == "error":
            error_msg = event.get("content", "")
            if chat_history and chat_history[-1]["role"] == "assistant":
                chat_history[-1]["content"] += f"\n\n{error_msg}"
            else:
                chat_history.append({"role": "assistant", "content": error_msg})
            yield chat_history, "Hata oluştu."
            
        elif ev_type == "done":
            break


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
                    work_dir = Path(app_config.workspace.base_dir).expanduser().resolve()
                    if not work_dir.exists():
                        return gr.update(choices=[])
                    files = [str(p.relative_to(work_dir)) for p in work_dir.rglob("*") 
                             if p.is_file() and p.suffix.lower() in ['.csv', '.json', '.txt', '.log', '.html']]
                    return gr.update(choices=sorted(files))

                def preview_file(filepath):
                    if not filepath:
                        return gr.update(visible=False), gr.update(value="", visible=True), gr.update(visible=False)
                    
                    work_dir = Path(app_config.workspace.base_dir).expanduser().resolve()
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
            
            # Gradio 4.40+ (type="messages") standardı: Dosyalar tuple olarak ayrı mesaja konur
            for f_path in files:
                history.append({"role": "user", "content": (f_path,)})
            
            if user_msg.strip():
                history.append({"role": "user", "content": user_msg})
            
            yield history, gr.update(value=None), gr.update(value=None), "Başlatılıyor..."
            
            for updated_history, status in process_message(
                user_msg, history, model, int(timeout), int(max_steps), files=files
            ):
                yield updated_history, gr.update(), gr.update(), status

        def on_new_session(model, timeout, max_steps):
            service = get_agent_service(model, int(timeout), int(max_steps))
            service.reset_session()
            sid = service.session_id[:12]
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
            inputs=[model_input, timeout_input, max_steps_input],
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

    # Config al
    app_config = load_config()
    work_dir = Path(app_config.workspace.base_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    print("🧠 Bio-ML Agent Web Arayüzü başlatılıyor...")
    print(f"   Model: {app_config.agent.model}")
    print(f"   Workspace: {work_dir}")
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
