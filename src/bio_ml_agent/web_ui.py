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
import shutil
import tempfile
import time
import signal
import subprocess
import requests
import qrcode
import io
import base64
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import gradio as gr
import pandas as pd

# Proje kökünü path'e ekle
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

from bio_ml_agent.utils.config import load_config
from bio_ml_agent.utils.logger import setup_logger
from bio_ml_agent.services.agent_service import AgentService
from bio_ml_agent.services.dashboard_service import (
    seed_tasks, list_tasks, create_task, update_task, delete_task,
    approve_task, reject_task, list_projects, get_project_results,
    compare_all_models, list_models, get_modules, get_stats,
    get_report, load_config as dash_load_config, update_config as dash_update_config,
    get_api_keys_status, get_audit_log,
)
from bio_ml_agent.core.conversation import generate_session_id, list_conversations, load_conversation
from bio_ml_agent.ultra_agent.observability.audit_trail import AuditTrailLogger

log = logging.getLogger("bio_ml_agent")
config = load_config()

# ─────────────────────────────────────────────
#  UI Modülleri Entegrasyonu
# ─────────────────────────────────────────────
from bio_ml_agent.ui.session_handlers import on_new_session, on_load_session, on_refresh_sessions
from bio_ml_agent.ui.whatsapp import start_whatsapp_services, stop_whatsapp_services, get_whatsapp_status
from bio_ml_agent.ui.chat_handlers import process_message, try_read_as_text, on_send, on_continue
from bio_ml_agent.ui.data_explorer import update_file_list, preview_file, list_xai_projects, load_xai_plots

# ─────────────────────────────────────────────
#  Gradio Arayüzü
# ─────────────────────────────────────────────



def get_modern_theme():
    """Özel Bio-ML Modern Teması (Emerald & Indigo)"""
    return gr.themes.Soft(
        primary_hue="emerald",
        secondary_hue="indigo",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    ).set(
        body_background_fill="*neutral_50",
        block_background_fill="white",
        block_border_width="1px",
        block_title_text_weight="600",
        section_header_text_weight="600",
        button_primary_background_fill="linear-gradient(90deg, *primary_500, *secondary_500)",
        button_primary_background_fill_hover="linear-gradient(90deg, *primary_600, *secondary_600)",
        button_primary_text_color="white",
    )

CUSTOM_CSS = """
.gradio-container { max-width: 1300px !important; }
.main-header { 
    background: linear-gradient(135deg, #064e3b 0%, #1e1b4b 100%);
    padding: 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    color: white;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.1);
}
.main-header h1 { margin: 0; font-size: 2.5rem; font-weight: 800; letter-spacing: -0.025em; }
.main-header p { margin: 0.5rem 0 0; opacity: 0.8; font-size: 1.1rem; }

.tool-output { 
    background: #0f172a !important; 
    border-radius: 12px !important; 
    padding: 16px !important; 
    border-left: 4px solid #10b981 !important;
    font-family: 'Fira Code', 'Courier New', monospace !important;
}

.message-bubble { border-radius: 16px !important; }
.user-message { background: #f1f5f9 !important; }
.bot-message { background: white !important; border: 1px solid #e2e8f0 !important; }

footer { display: none !important; }
.tabs { border-bottom: 1px solid #e2e8f0 !important; }
.tab-nav button.selected { border-bottom-color: #10b981 !important; color: #10b981 !important; }
"""

def create_ui():
    """Ana Gradio arayüzünü oluşturur ve döndürür."""
    with gr.Blocks(
        title="🧠 Bio-ML Agent",
        theme=get_modern_theme(),
        css=CUSTOM_CSS,
    ) as demo:
        # Premium Header (Glassmorphism + Gradient)
        gr.HTML(
            """
            <div class="main-header">
                <h1>🧠 Bio-ML Agent <span style='font-weight:300; opacity:0.6'>v7.0</span></h1>
                <p>Otonom Biyomühendislik Laboratuvar Asistanı & Veri Bilimi Platformu</p>
            </div>
            """
        )

        with gr.Tabs(elem_id="main-tabs"):
            with gr.Tab("💬 Konuşma"):
                with gr.Row():
                    # Sol panel: Chat
                    with gr.Column(scale=4):
                        chatbot = gr.Chatbot(
                            label="Zekâ Kanalı",
                            height=700,
                            show_label=True,
                            elem_id="main-chatbot",
                            render_markdown=True,
                            avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=BioML"),
                        )
                        with gr.Row():
                            msg_input = gr.MultimodalTextbox(
                                label="Mesajınız (Dosya eklenebilir: TXT, CSV, JSON, görüntü, ses vb.)",
                                placeholder="Örn: Breast cancer analizini yap... veya herhangi bir dosya ekleyin",
                                file_types=None,  # Tüm dosya türlerini kabul et
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
                        model_input = gr.Dropdown(
                            label="LLM Modeli",
                            choices=[
                                "gemini-2.5-flash",
                                "gemini-2.5-pro",
                                "gemini-2.0-flash",
                                "gpt-4o",
                                "gpt-4o-mini",
                                "claude-3-5-sonnet-20241022",
                                "claude-3-5-haiku-20241022",
                                "qwen2.5:7b-instruct",
                                "qwen2.5:14b-instruct",
                                "qwen2.5:32b-instruct",
                                "llama3.1:8b-instruct-q4_0",
                                "deepseek-r1:7b",
                                "codestral:latest",
                            ],
                            value=config.agent.model,
                            allow_custom_value=True,
                            info="Listeden seç veya özel model adı yaz",
                        )
                        timeout_input = gr.Slider(
                            label="Timeout (s)",
                            minimum=30,
                            maximum=600,
                            value=config.agent.timeout,
                            step=30,
                        )
                        max_steps_input = gr.Slider(
                            label="Maks. Adım",
                            minimum=1,
                            maximum=9999,
                            value=config.agent.max_steps,
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
                            visible=False,
                        )

                        local_mode_toggle = gr.Checkbox(
                            label="💻 Offline / Local Execute Mode",
                            value=True,
                            info="Kapatılırsa API Gateway'e (Bulut) Remote Run atar.",
                            visible=True,
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
                            f"**Workspace:** `{config.workspace.base_dir}`"
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
                
                # XAI Logic using submodules handled below


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

                # Data Explorer Logic handled below (event bindings at end of create_ui)


            # ═══════════════════════════════════════
            #  Dashboard Tabları (dashboard.py'den taşındı)
            # ═══════════════════════════════════════

            with gr.Tab("📋 Görevler"):
                gr.Markdown("### Proje Görev Yönetimi")
                with gr.Row():
                    task_status_filter = gr.Dropdown(label="Durum", choices=["", "pending", "in_progress", "completed"], value="")
                    task_category_filter = gr.Dropdown(label="Kategori", choices=["", "core", "ml", "ui", "testing", "devops", "bioeng"], value="")
                    task_refresh_btn = gr.Button("🔄 Yenile", variant="primary")
                task_table = gr.Dataframe(
                    label="Görevler",
                    headers=["ID", "Başlık", "Durum", "Kategori", "Öncelik", "Güncelleme"],
                    interactive=False,
                    wrap=True,
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("#### ➕ Yeni Görev")
                        new_task_title = gr.Textbox(label="Başlık", placeholder="Görev başlığı...")
                        new_task_desc = gr.Textbox(label="Açıklama", placeholder="Görev detayı...", lines=2)
                        with gr.Row():
                            new_task_cat = gr.Dropdown(label="Kategori", choices=["core", "ml", "ui", "testing", "devops", "bioeng", "general"], value="general")
                            new_task_prio = gr.Dropdown(label="Öncelik", choices=["high", "medium", "low"], value="medium")
                        create_task_btn = gr.Button("✅ Oluştur", variant="primary")
                        task_action_status = gr.Textbox(label="Sonuç", interactive=False)

                def refresh_tasks(status, category):
                    tasks = list_tasks(status=status or None, category=category or None)
                    rows = [[t.get("id", "")[:8], t.get("title", ""), t.get("status", ""), t.get("category", ""), t.get("priority", ""), t.get("updated_at", "")] for t in tasks]
                    return rows

                def on_create_task(title, desc, cat, prio):
                    if not title.strip():
                        return "❌ Başlık gerekli."
                    t = create_task(title=title, description=desc, category=cat, priority=prio)
                    return f"✅ Görev oluşturuldu: {t['id'][:8]} — {t['title']}"

                task_refresh_btn.click(fn=refresh_tasks, inputs=[task_status_filter, task_category_filter], outputs=task_table)
                create_task_btn.click(fn=on_create_task, inputs=[new_task_title, new_task_desc, new_task_cat, new_task_prio], outputs=task_action_status)
                demo.load(fn=lambda: refresh_tasks("", ""), outputs=task_table)

            with gr.Tab("📁 Projeler"):
                gr.Markdown("### ML Proje Geçmişi")
                proj_refresh_btn = gr.Button("🔄 Yenile", variant="primary")
                proj_table = gr.Dataframe(
                    label="Projeler",
                    headers=["ID", "Ad", "Değiştirilme", "Dosya Sayısı", "Model Var?", "En İyi Model"],
                    interactive=False,
                    wrap=True,
                )
                proj_detail = gr.JSON(label="Proje Detayı", visible=False)

                def refresh_projects():
                    projects = list_projects()
                    rows = [[
                        p.get("id", ""), p.get("name", ""), p.get("modified", ""),
                        p.get("file_count", 0), "✅" if p.get("has_model") else "❌",
                        p.get("best_model", "—")
                    ] for p in projects]
                    return rows

                proj_refresh_btn.click(fn=refresh_projects, outputs=proj_table)
                demo.load(fn=refresh_projects, outputs=proj_table)

            with gr.Tab("📊 Modeller"):
                gr.Markdown("### Kaydedilmiş ML Modelleri")
                model_refresh_btn = gr.Button("🔄 Yenile", variant="primary")
                model_table = gr.Dataframe(
                    label="Modeller",
                    headers=["Ad", "Tür", "Boyut (KB)", "Oluşturma", "Yol"],
                    interactive=False,
                    wrap=True,
                )
                gr.Markdown("---")
                gr.Markdown("### 🔀 Model Karşılaştırma")
                compare_json = gr.JSON(label="Tüm Proje Karşılaştırmaları")
                compare_btn = gr.Button("📈 Karşılaştır", variant="primary")

                def refresh_models_table():
                    models = list_models()
                    return [[m.get("name", ""), m.get("task_type", ""), m.get("size_kb", 0), m.get("created", ""), m.get("path", "")] for m in models]

                def on_compare():
                    return compare_all_models()

                model_refresh_btn.click(fn=refresh_models_table, outputs=model_table)
                compare_btn.click(fn=on_compare, outputs=compare_json)
                demo.load(fn=refresh_models_table, outputs=model_table)

            with gr.Tab("📈 İstatistik"):
                gr.Markdown("### Proje İstatistikleri")
                stats_refresh_btn = gr.Button("🔄 Yenile", variant="primary")
                with gr.Row():
                    stat_total = gr.Number(label="Toplam Görev", interactive=False)
                    stat_completed = gr.Number(label="Tamamlanan", interactive=False)
                    stat_pending = gr.Number(label="Bekleyen", interactive=False)
                    stat_pct = gr.Number(label="Tamamlanma %", interactive=False)
                with gr.Row():
                    stat_lines = gr.Number(label="Toplam Satır", interactive=False)
                    stat_modules = gr.Number(label="Modül Sayısı", interactive=False)
                modules_table = gr.Dataframe(
                    label="Modül Detayları",
                    headers=["Dosya", "Açıklama", "Kategori", "Satır", "KB"],
                    interactive=False,
                    wrap=True,
                )
                report_md = gr.Markdown(label="📄 Proje Raporu")

                def refresh_stats():
                    s = get_stats()
                    mods = get_modules()
                    mod_rows = [[m["filename"], m["description"], m["category"], m["lines"], m["size_kb"]] for m in mods]
                    report = get_report() or "_Rapor dosyası bulunamadı._"
                    return s["total"], s["completed"], s["pending"], s["completion_pct"], s["total_lines"], s["total_modules"], mod_rows, report

                stats_refresh_btn.click(fn=refresh_stats, outputs=[stat_total, stat_completed, stat_pending, stat_pct, stat_lines, stat_modules, modules_table, report_md])
                demo.load(fn=refresh_stats, outputs=[stat_total, stat_completed, stat_pending, stat_pct, stat_lines, stat_modules, modules_table, report_md])

            with gr.Tab("⚙️ Ayarlar"):
                gr.Markdown("### Yapılandırma (config.yaml)")
                config_json = gr.JSON(label="Mevcut Ayarlar")
                config_refresh_btn = gr.Button("🔄 Yenile")
                gr.Markdown("---")
                gr.Markdown("### 🔑 API Key Durumu")
                api_keys_json = gr.JSON(label="API Key'ler (değerler gizli)")

                def refresh_config():
                    return dash_load_config(), get_api_keys_status()

                config_refresh_btn.click(fn=refresh_config, outputs=[config_json, api_keys_json])
                demo.load(fn=refresh_config, outputs=[config_json, api_keys_json])

            with gr.Tab("🔍 Denetim (Audit)"):
                gr.Markdown("### Ajan Denetim İzleri")
                with gr.Row():
                    audit_limit = gr.Slider(label="Satır Sayısı", minimum=10, maximum=200, value=50, step=10)
                    audit_filter = gr.Dropdown(label="Filtre", choices=["", "HITL", "PYTHON", "BASH", "BROWSER", "ERROR", "APPROVAL"], value="")
                    audit_refresh_btn = gr.Button("🔄 Yenile", variant="primary")
                audit_output = gr.Dataframe(
                    label="Log Kayıtları",
                    headers=["Zaman", "Seviye", "Mesaj"],
                    interactive=False,
                    wrap=True,
                )
                def refresh_audit(limit, filt):
                    result = get_audit_log(limit=int(limit), log_filter=filt or "")
                    rows = []
                    for e in result.get("entries", []):
                        rows.append([e.get("timestamp", ""), e.get("level", ""), e.get("raw", "")[:200]])
                    return rows

                audit_refresh_btn.click(fn=refresh_audit, inputs=[audit_limit, audit_filter], outputs=audit_output)

            with gr.Tab("📱 WhatsApp"):
                gr.Markdown("### 📱 WhatsApp Bağlantı Yönetimi")
                gr.Markdown("WhatsApp Web üzerinden Bio-ML Agent ile konuşmak için bu sekmeyi kullanın.")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        wa_status_md = gr.Markdown("Durum: **Çevrimdışı**", label="Bağlantı Durumu")
                    with gr.Column(scale=1):
                        wa_qr_img = gr.Image(show_label=False, interactive=False, height=450, elem_id="whatsapp-qr")
                        with gr.Row():
                            wa_start_btn = gr.Button("🚀 Başlat", variant="primary", scale=2)
                            wa_refresh_btn = gr.Button("🔄 Yenile", variant="secondary", scale=1)
                            wa_stop_btn = gr.Button("🛑 Durdur", variant="stop", scale=1)
                    
                    with gr.Column(scale=3):
                        gr.Markdown("#### 📖 Talimatlar")
                        gr.Markdown(
                            "1. **Servisi Başlat:** Yukarıdaki butona basarak WhatsApp köprüsünü çalıştırın.\n"
                            "2. **QR Kod:** Birkaç saniye içinde sol tarafta bir QR kod belirecektir.\n"
                            "3. **Taratın:** Telefonunuzdan WhatsApp > Bağlı Cihazlar > Cihaz Bağla yolunu izleyerek kodu okutun.\n"
                            "4. **Başlatın:** Bağlantı kurulduğunda durum 'CONNECTED' olacaktır. Web UI başlatma zaten çekirdek hattı açar; isterseniz telefondan **'STR'** de yazabilirsiniz.\n"
                            "5. **Komut Verin:** Artık **'AGT [mesajınız]'** yazarak her yerden ajana erişebilirsiniz."
                        )
                
                wa_start_btn.click(fn=start_whatsapp_services, outputs=[wa_status_md, wa_qr_img])
                wa_refresh_btn.click(fn=get_whatsapp_status, outputs=[wa_status_md, wa_qr_img])
                wa_stop_btn.click(fn=stop_whatsapp_services, outputs=wa_status_md)
                
                # Periyodik Yenileme (Açıkken her 2 saniyede bir durumu kontrol et)
                # WhatsApp periyodik güncelleme (Gradio 6+ için gr.Timer)
                wa_timer = gr.Timer(2)
                wa_timer.tick(fn=get_whatsapp_status, outputs=[wa_status_md, wa_qr_img])
                
                # İlk yükleme
                demo.load(fn=get_whatsapp_status, outputs=[wa_status_md, wa_qr_img])


        # Event handlers using submodules
        from bio_ml_agent.ui.chat_handlers import on_send, on_continue, get_agent_service, get_active_project_name
        from bio_ml_agent.ui.session_handlers import on_new_session, on_refresh_sessions, on_load_session

        # XAI Bindings
        xai_refresh_btn.click(
            fn=lambda: list_xai_projects(get_active_project_name()), 
            outputs=xai_project_dropdown
        )
        xai_project_dropdown.change(fn=load_xai_plots, inputs=xai_project_dropdown, outputs=xai_gallery)
        demo.load(fn=lambda: list_xai_projects(get_active_project_name()), outputs=xai_project_dropdown)

        # Data Explorer Bindings
        refresh_files_btn.click(fn=update_file_list, outputs=file_dropdown)
        file_dropdown.change(fn=preview_file, inputs=file_dropdown, outputs=[data_preview, text_preview, html_preview, image_preview])
        demo.load(fn=update_file_list, outputs=file_dropdown)

        # Chat & Session Bindings
        send_btn.click(
            fn=on_send,
            inputs=[msg_input, audio_input, chatbot, model_input, timeout_input, max_steps_input,
                    mode_radio, approval_interval_input, checkpoint_step_input, swarm_toggle, local_mode_toggle],
            outputs=[chatbot, msg_input, audio_input, status_box, continue_btn],
        )

        # Enter tuşu
        msg_input.submit(
            fn=on_send,
            inputs=[msg_input, audio_input, chatbot, model_input, timeout_input, max_steps_input,
                    mode_radio, approval_interval_input, checkpoint_step_input, swarm_toggle, local_mode_toggle],
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
            inputs=[gr.State(get_agent_service), model_input, timeout_input, max_steps_input],
            outputs=[chatbot, status_box, session_info],
        )

        # Geçmiş oturumlar
        refresh_sessions_btn.click(fn=on_refresh_sessions, outputs=session_dropdown)
        load_session_btn.click(
            fn=on_load_session,
            inputs=[session_dropdown, gr.State(get_agent_service), model_input, timeout_input, max_steps_input],
            outputs=[chatbot, status_box, session_info],
        )
        demo.load(fn=on_refresh_sessions, outputs=session_dropdown)

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

    # Workspace hazırla
    work_dir = Path(config.workspace.base_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    print("🧠 Bio-ML Agent Web Arayüzü başlatılıyor...")
    print(f"   Model: {config.agent.model}")
    print(f"   Workspace: {work_dir}")
    print()

    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
