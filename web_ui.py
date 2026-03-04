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
from agent import setup_logger, generate_session_id, list_conversations, load_conversation

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

        elif ev_type == "approval_required":
            reason = event.get("reason", "Onay gerekiyor")
            step = event.get("step", "?")
            approval_msg = f"⏸️ **Durakladı — Adım {step}**\n\n{reason}\n\n*Devam etmek için 'Devam Et' butonuna basın.*"
            chat_history.append({"role": "assistant", "content": approval_msg})
            yield chat_history, f"⏸️ Onay bekleniyor (adım {step})"
            
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
                        )
                        with gr.Row():
                            msg_input = gr.MultimodalTextbox(
                                label="Mesajınız (Görüntü/Ses/TXT/CSV/JSON eklenebilir)",
                                placeholder="Örn: Breast cancer analizini yap... veya .txt dosyası ekleyin",
                                file_types=["image", "audio", ".txt", ".csv", ".json", ".md", ".py", ".log", ".yml", ".yaml"],
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
                        model_input = gr.Dropdown(
                            label="Model",
                            choices=[
                                "gemini-2.5-flash",
                                "gemini-2.5-pro",
                                "gemini-2.0-flash",
                                "gpt-4o",
                                "gpt-4o-mini",
                                "claude-sonnet-4-20250514",
                                "claude-3-5-haiku-20241022",
                                "qwen2.5:7b-instruct",
                                "qwen2.5:14b-instruct",
                                "qwen2.5:32b-instruct",
                                "llama3.1:8b-instruct-q4_0",
                                "deepseek-r1:7b",
                                "codestral:latest",
                            ],
                            value=app_config.agent.model,
                            allow_custom_value=True,
                            info="Listeden seç veya özel model adı yaz",
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
                            maximum=9999,
                            value=app_config.agent.max_steps,
                            step=1,
                        )

                        gr.Markdown("---")
                        gr.Markdown("### 🎮 Çalışma Modu")
                        mode_radio = gr.Radio(
                            choices=[
                                ("🚀 Tam Otomatik", 1),
                                ("🔢 Adım Onaylı", 2),
                                ("🧠 Akıllı Onay", 3),
                                ("📋 Plan + Checkpoint", 4),
                            ],
                            value=1,
                            label="Mod",
                            info="Agent'ın ne zaman duraklatılacağını belirler",
                        )
                        with gr.Row():
                            approval_interval_input = gr.Number(
                                label="Her N. adımda dur",
                                value=5,
                                minimum=1,
                                maximum=500,
                                visible=False,
                                precision=0,
                            )
                            checkpoint_step_input = gr.Number(
                                label="Checkpoint adımı",
                                value=150,
                                minimum=1,
                                maximum=9999,
                                visible=False,
                                precision=0,
                            )

                        gr.Markdown("---")
                        swarm_toggle = gr.Checkbox(
                            label="🐝 Swarm Modu (Alt Ajanlar)",
                            value=False,
                            info="DataEngineer → MLExpert → BioinfoExpert pipeline",
                        )

                        continue_btn = gr.Button(
                            "▶️ Devam Et",
                            variant="primary",
                            visible=False,
                        )

                        def on_mode_change(mode):
                            return (
                                gr.update(visible=(mode == 2)),   # interval input
                                gr.update(visible=(mode == 4)),   # checkpoint input
                            )
                        mode_radio.change(
                            fn=on_mode_change,
                            inputs=mode_radio,
                            outputs=[approval_interval_input, checkpoint_step_input],
                        )

                        gr.Markdown("---")
                        gr.Markdown("### 📋 Bilgi")
                        session_info = gr.Markdown(
                            f"**Oturum:** `{generate_session_id()[:12]}...`\n\n"
                            f"**Workspace:** `{app_config.workspace.base_dir}`"
                        )

                        new_session_btn = gr.Button("🔄 Yeni Oturum", variant="secondary")

                        gr.Markdown("---")
                        gr.Markdown("### 📜 Geçmiş Oturumlar")
                        session_dropdown = gr.Dropdown(
                            label="Oturum Seç",
                            choices=[],
                            interactive=True,
                        )
                        with gr.Row():
                            refresh_sessions_btn = gr.Button("🔄", variant="secondary", scale=1)
                            load_session_btn = gr.Button("📂 Yükle", variant="primary", scale=3)

            with gr.Tab("🔍 Açıklanabilirlik (XAI)"):
                gr.Markdown("### Makine Öğrenimi Model Karar Açıklamaları (SHAP/LIME)")
                gr.Markdown("Agent tarafından üretilen SHAP, LIME ve diğer analiz grafikleri burada görüntülenir.")
                
                with gr.Row():
                    xai_project_dropdown = gr.Dropdown(
                        label="Proje Seç",
                        choices=[],
                        interactive=True,
                    )
                    xai_refresh_btn = gr.Button("🔄 Yenile", variant="primary")
                
                with gr.Row():
                    xai_gallery = gr.Gallery(label="Analiz Grafikleri", show_label=True, elem_id="xai_gallery", columns=[2], rows=[2], object_fit="contain", height="auto")
                
                def list_xai_projects():
                    work_dir = Path(app_config.workspace.base_dir).expanduser().resolve()
                    if not work_dir.exists():
                        return gr.update(choices=[])
                    projects = sorted([d.name for d in work_dir.iterdir() if d.is_dir()], reverse=True)
                    # Aktif projeyi ön seçili yap
                    current = None
                    if _agent_service and _agent_service.project_name:
                        current = _agent_service.project_name
                    return gr.update(choices=projects, value=current)

                def load_xai_plots(project_name):
                    work_dir = Path(app_config.workspace.base_dir).expanduser().resolve()
                    if not project_name:
                        # Proje seçilmemişse aktif projeyi dene
                        if _agent_service and _agent_service.project_name:
                            project_name = _agent_service.project_name
                        else:
                            return []
                    project_dir = work_dir / project_name
                    if not project_dir.exists():
                        return []
                    plots = []
                    for p in project_dir.rglob("*.png"):
                        plots.append(str(p))
                    return sorted(plots)
                    
                xai_refresh_btn.click(fn=list_xai_projects, outputs=xai_project_dropdown)
                xai_project_dropdown.change(fn=load_xai_plots, inputs=xai_project_dropdown, outputs=xai_gallery)
                demo.load(fn=list_xai_projects, outputs=xai_project_dropdown)

            with gr.Tab("📂 Data Explorer"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Workspace Dosyaları")
                        gr.Markdown("Desteklenen türler: CSV, JSON, TXT, LOG, HTML, PNG, JPG")
                        file_dropdown = gr.Dropdown(label="Dosya Seç", choices=[], interactive=True)
                        refresh_files_btn = gr.Button("🔄 Yenile")
                    with gr.Column(scale=3):
                        data_preview = gr.Dataframe(label="Veri Önizleme", interactive=False, visible=False)
                        text_preview = gr.Textbox(label="Metin Önizleme", lines=20, max_lines=40, interactive=False, visible=True)
                        html_preview = gr.HTML(label="HTML Önizleme", visible=False)
                        image_preview = gr.Image(label="Görüntü Önizleme", visible=False)

                def update_file_list():
                    work_dir = Path(app_config.workspace.base_dir).expanduser().resolve()
                    if not work_dir.exists():
                        return gr.update(choices=[])
                    allowed_suffixes = [
                        '.csv', '.json', '.txt', '.log', '.html', '.png', '.jpg', '.jpeg',
                        '.py', '.md', '.yml', '.yaml', '.pkl', '.ipynb', '.parquet',
                    ]
                    files = [str(p.relative_to(work_dir)) for p in work_dir.rglob("*") 
                             if p.is_file() and p.suffix.lower() in allowed_suffixes]
                    return gr.update(choices=sorted(files))

                def preview_file(filepath):
                    if not filepath:
                        return gr.update(visible=False), gr.update(value="", visible=True), gr.update(visible=False), gr.update(visible=False)
                    
                    work_dir = Path(app_config.workspace.base_dir).expanduser().resolve()
                    full_path = work_dir / filepath
                    if not full_path.exists():
                        return gr.update(visible=False), gr.update(value="Dosya bulunamadı.", visible=True), gr.update(visible=False), gr.update(visible=False)
                    
                    try:
                        ext = full_path.suffix.lower()
                        if ext == '.csv':
                            import pandas as pd
                            df = pd.read_csv(full_path, nrows=100)
                            return gr.update(value=df, visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
                        elif ext == '.json':
                            import pandas as pd
                            try:
                                df = pd.read_json(full_path)
                                return gr.update(value=df.head(100), visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
                            except ValueError:
                                with open(full_path, 'r', encoding='utf-8') as f:
                                    text = f.read(10000)
                                return gr.update(visible=False), gr.update(value=text, visible=True), gr.update(visible=False), gr.update(visible=False)
                        elif ext == '.html':
                            with open(full_path, 'r', encoding='utf-8') as f:
                                html_text = f.read()
                            return gr.update(visible=False), gr.update(visible=False), gr.update(value=html_text, visible=True), gr.update(visible=False)
                        elif ext in ['.png', '.jpg', '.jpeg']:
                            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value=str(full_path), visible=True)
                        else: # txt, log
                            with open(full_path, 'r', encoding='utf-8') as f:
                                text = f.read(10000)
                            return gr.update(visible=False), gr.update(value=text, visible=True), gr.update(visible=False), gr.update(visible=False)
                    except Exception as e:
                        return gr.update(visible=False), gr.update(value=f"Hata: {str(e)}", visible=True), gr.update(visible=False), gr.update(visible=False)

                refresh_files_btn.click(fn=update_file_list, outputs=file_dropdown)
                file_dropdown.change(fn=preview_file, inputs=file_dropdown, outputs=[data_preview, text_preview, html_preview, image_preview])
                demo.load(fn=update_file_list, outputs=file_dropdown)

        # Event handlers
        # Metin tabanlı dosya uzantıları
        TEXT_FILE_EXTENSIONS = {'.txt', '.csv', '.json', '.md', '.py', '.log', '.yml', '.yaml'}

        def _read_text_file(filepath: str) -> Optional[str]:
            """Metin dosyasını okur, None döner eğer binary ise."""
            try:
                p = Path(filepath)
                if p.suffix.lower() in TEXT_FILE_EXTENSIONS:
                    content = p.read_text(encoding='utf-8', errors='replace')
                    # Çok büyük dosyaları kırp (max 50K karakter)
                    if len(content) > 50_000:
                        content = content[:50_000] + f"\n\n... (dosya çok büyük, {len(content)} karakterden ilk 50.000'i alındı)"
                    return content
            except Exception as e:
                log.warning(f"Dosya okuma hatası: {filepath} — {e}")
            return None

        def on_send(user_data, audio_path, history, model, timeout, max_steps, mode, interval, checkpoint, swarm):
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

            # Modu AgentService'e uygula
            service = get_agent_service(model, int(timeout), int(max_steps))
            service.approval_mode = int(mode)
            service.approval_interval = int(interval)
            service.checkpoint_step = int(checkpoint)
            service.swarm_enabled = bool(swarm)

            # Dosyaları işle: metin dosyalarını oku, diğerlerini tuple olarak ekle
            text_file_contents = []
            non_text_files = []
            for f_path in files:
                fp = f_path if isinstance(f_path, str) else str(f_path)
                txt = _read_text_file(fp)
                if txt is not None:
                    fname = Path(fp).name
                    text_file_contents.append(f"📄 **{fname}** içeriği:\n```\n{txt}\n```")
                else:
                    non_text_files.append(f_path)

            # Binary dosyaları (görüntü, ses) tuple olarak ekle
            for f_path in non_text_files:
                history.append({"role": "user", "content": (f_path,)})

            # Metin dosya içeriklerini kullanıcı mesajına ekle
            combined_msg = user_msg.strip()
            if text_file_contents:
                file_block = "\n\n".join(text_file_contents)
                if combined_msg:
                    combined_msg = f"{combined_msg}\n\n{file_block}"
                else:
                    combined_msg = file_block

            if combined_msg:
                history.append({"role": "user", "content": combined_msg})
                user_msg = combined_msg  # Agent'a gönderilecek mesajı da güncelle
            
            yield history, gr.update(value=None), gr.update(value=None), "Başlatılıyor...", gr.update(visible=False)
            
            show_continue = False
            for updated_history, status in process_message(
                user_msg, history, model, int(timeout), int(max_steps), files=files
            ):
                if "⏸️" in status and "Onay bekleniyor" in status:
                    show_continue = True
                yield updated_history, gr.update(), gr.update(), status, gr.update(visible=show_continue)

        def on_continue(history, model, timeout, max_steps, mode, interval, checkpoint, swarm):
            """Duraklatılmış agent'ı devam ettir."""
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
                show_continue = "⏸️" in status and "Onay bekleniyor" in status
                yield updated_history, status, gr.update(visible=show_continue)

        def on_new_session(model, timeout, max_steps):
            service = get_agent_service(model, int(timeout), int(max_steps))
            service.reset_session()
            sid = service.session_id[:12]
            return [], "Hazır — Yeni oturum", f"**Oturum:** `{sid}...`"

        def on_refresh_sessions():
            history_dir = Path(app_config.history.directory).expanduser().resolve()
            sessions = list_conversations(history_dir, limit=50)
            if not sessions:
                return gr.update(choices=[], value=None)
            choices = []
            for s in sessions:
                label = f"{s['session_id'][:20]}  |  💬{s['message_count']}  |  {s['summary'][:40]}"
                choices.append((label, s['session_id']))
            return gr.update(choices=choices, value=None)

        def on_load_session(session_id, model, timeout, max_steps):
            if not session_id:
                return gr.update(), "Oturum seçilmedi.", gr.update()
            history_dir = Path(app_config.history.directory).expanduser().resolve()
            try:
                messages, metadata = load_conversation(history_dir, session_id)
                service = get_agent_service(model, int(timeout), int(max_steps))
                service.set_session(session_id, messages, metadata)
                # Chat geçmişini Gradio formatına dönüştür
                chat_history = []
                for m in messages:
                    if m["role"] == "system":
                        continue
                    if m["role"] == "user" and m["content"].startswith("TOOL_OUTPUT"):
                        continue
                    chat_history.append({"role": m["role"], "content": m["content"][:2000]})
                sid = session_id[:20]
                msg_count = len([m for m in messages if m['role'] != 'system'])
                proj = metadata.get("project_name", "—")
                return (
                    chat_history,
                    f"✅ Oturum yüklendi — {msg_count} mesaj | Proje: {proj}",
                    f"**Oturum:** `{sid}...`\n\n**Proje:** `{proj}`\n\n**Workspace:** `{app_config.workspace.base_dir}`",
                )
            except FileNotFoundError:
                return gr.update(), f"❌ Oturum bulunamadı: {session_id}", gr.update()
            except Exception as e:
                return gr.update(), f"❌ Hata: {str(e)}", gr.update()

        # Gönder butonu
        send_btn.click(
            fn=on_send,
            inputs=[msg_input, audio_input, chatbot, model_input, timeout_input, max_steps_input,
                    mode_radio, approval_interval_input, checkpoint_step_input, swarm_toggle],
            outputs=[chatbot, msg_input, audio_input, status_box, continue_btn],
        )

        # Enter tuşu
        msg_input.submit(
            fn=on_send,
            inputs=[msg_input, audio_input, chatbot, model_input, timeout_input, max_steps_input,
                    mode_radio, approval_interval_input, checkpoint_step_input, swarm_toggle],
            outputs=[chatbot, msg_input, audio_input, status_box, continue_btn],
        )

        # Devam Et butonu
        continue_btn.click(
            fn=on_continue,
            inputs=[chatbot, model_input, timeout_input, max_steps_input,
                    mode_radio, approval_interval_input, checkpoint_step_input, swarm_toggle],
            outputs=[chatbot, status_box, continue_btn],
        )

        # Yeni oturum
        new_session_btn.click(
            fn=on_new_session,
            inputs=[model_input, timeout_input, max_steps_input],
            outputs=[chatbot, status_box, session_info],
        )

        # Geçmiş oturumlar
        refresh_sessions_btn.click(fn=on_refresh_sessions, outputs=session_dropdown)
        load_session_btn.click(
            fn=on_load_session,
            inputs=[session_dropdown, model_input, timeout_input, max_steps_input],
            outputs=[chatbot, status_box, session_info],
        )
        demo.load(fn=on_refresh_sessions, outputs=session_dropdown)

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
