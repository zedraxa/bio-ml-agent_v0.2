import os
import sys
import logging
import requests
from typing import Optional
from pathlib import Path
from flask import Flask, request, jsonify, abort
from twilio.twiml.messaging_response import MessagingResponse
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Proje kökünü path'e ekle
# sys.path.insert(0, str(Path(__file__).resolve().parent))

from bio_ml_agent.utils.logger import setup_logger
from bio_ml_agent.utils.config import get_config
from bio_ml_agent.ultra_agent.observability.audit_trail import AuditTrailLogger
from bio_ml_agent.brain.models import WhatsAppMissionCard, WhatsAppCardType, IntentType, MissionPriority, StepStatus
from bio_ml_agent.brain.comment_manager import CommentManager, Comment
from bio_ml_agent.brain.mission_brain import MissionBrain
from bio_ml_agent.brain.persistence import PROJECT_STORE, MISSION_STORE
from typing import Dict, List

# Flask app oluştur
app = Flask(__name__)

# Logger
log_dir = Path("logs").resolve()
log_dir.mkdir(exist_ok=True)
log = setup_logger(log_dir, "INFO")
config = get_config()

# Rate Limiter (Saniyede max/ip)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["100 per day", "20 per minute"]
)

# API Güvenlik Kontrolü
def require_api_key():
    config = get_config()
    expected_key = config.security.api_key
    gateway_key = config.gateway.secret_key
    
    if not expected_key and not config.gateway.enabled:
        return # Güvenlik anahtarı ayarlanmamışsa serbest geçiş

    # Twilio webhook'ları çoğu durumda custom header taşıyamaz; Body/From geldiğinde geçiş izni ver.
    if request.path == "/whatsapp" and request.values.get("From"):
        return
    # Local Node köprüsü loopback üzerinden çağırıyorsa API key olmadan da geçişe izin ver.
    if request.path == "/whatsapp-local" and request.remote_addr in {"127.0.0.1", "::1"}:
        return
    
    # Twilio ve Whatsapp-Web tarafı Authorization Header, X-API-Key veya URL parametresi kullanabilir.
    api_key = request.headers.get("X-API-Key") or request.args.get("api_key")
    if api_key != expected_key and api_key != gateway_key:
        log.warning(f"Yetkisiz Webhook Erişimi (IP: {get_remote_address()})")
        abort(403, description="Geçersiz veya eksik API Key")

# Bellek (AgentService Qdrant altyapısı üzerinden yönetilecek, burada durum tutulmuyor)
# session_histories = {}

# Node.js Push API adresi
PUSH_API_URL = "http://127.0.0.1:3001/push-message"


def _push_status(sender_id: str, text: str, media_path: Optional[str] = None):
    """Node.js üzerinden WhatsApp'a ara durum veya medya mesajı gönder."""
    try:
        payload = {"to": sender_id, "text": text}
        if media_path:
            payload["media_path"] = str(media_path)
        requests.post(PUSH_API_URL, json=payload, timeout=10)
    except Exception as e:
        log.warning(f"[Push] Mesaj gönderilemedi: {e}")


# Meşguliyet takibi (Hangi kullanıcı için ajan şu an çalışıyor?)
busy_sessions: Dict[str, bool] = {}
# E8: Kullanıcı bazlı aktif proje bağlamı (Context Switch)
user_active_projects: Dict[str, str] = {}
# F3: Toplu Yorum Geri Dönüş Teyidi Buffer
user_revision_buffers: Dict[str, List[str]] = {}

@app.route("/whatsapp-local", methods=["POST"])
@limiter.limit("20 per minute")
def whatsapp_local():
    """Node.js (whatsapp-web.js) üzerinden gelen mesajı Ajan'a ilet."""
    require_api_key()
    # Handle both JSON (Node.js) and Form (Twilio)
    data = request.json if request.is_json else request.form.to_dict()
    if not data:
        data = {}
        
    # Extract message and sender (supports both Twilio "Body"/"From" and Node "text"/"from")
    incoming_msg = data.get("text") or data.get("Body") or ""
    incoming_msg = incoming_msg.strip()
    sender_id = data.get("from") or data.get("From") or ""

    log.info(f"[Whatsapp-Local] Mesaj alındı ({sender_id}): {incoming_msg}")

    # Media Check
    has_media = data.get("hasMedia", False) or int(data.get("NumMedia", 0)) > 0
    if not incoming_msg and not has_media:
        return jsonify({"reply": "Lütfen geçerli bir mesaj veya dosya gönderin."})

    # 1. Meşguliyet Kontrolü
    if sender_id in busy_sessions:
        msg_upper = incoming_msg.upper()
        if any(kw in msg_upper for kw in ["DURUM", "NE YAPIYOR", "NABER", "GİDİŞAT"]):
            return jsonify({"reply": "🔄 Şu an hala önceki görevin üzerinde çalışıyorum. Birazdan raporun tamamlanacak, lütfen bekle! 😊"})
        return jsonify({"reply": "⚠️ Şu an bir görevi işliyorum. Lütfen o bitene kadar bekle ya da bitmesini bekle!"})

    app_config = get_config()
    
    # 2. Önce Gateway'i dene (Bulut/Platform Modu)
    gateway_url = f"http://127.0.0.1:8001/api/v1/platform/chat/async"
    try:
        headers = {"X-API-Key": app_config.security.api_key}
        payload = {
            "session_id": sender_id,
            "message": incoming_msg,
            "channel": "whatsapp_local",
            "callback_url": PUSH_API_URL,
            "callback_payload": {"to": sender_id},
        }
        response = requests.post(gateway_url, json=payload, headers=headers, timeout=2)
        if response.status_code in [200, 202]:
            return jsonify({"reply": "⏳ İsteğiniz platforma iletildi, sonuçlar birazdan gelecek."})
    except Exception:
        pass

    # 3. Yerel Mod (MissionBrain Entegrasyonu - Eksen E)
    import threading
    def background_process(sid, msg_data):
        busy_sessions[str(sid)] = True
        try:
            msg = str(msg_data.get("text") or msg_data.get("Body") or "").strip()
            
            log.info(f"[Background] {sid} için MissionBrain başlatıldı...")
            
            # Determine Active Project Context (E8)
            project_id = user_active_projects.get(str(sid), f"wa_project_{sid}")
            brain = MissionBrain(project_id=project_id)
            
            # E5 / E6: Check for Media attachments
            has_media = msg_data.get("hasMedia", False) or int(msg_data.get("NumMedia", 0)) > 0
            saved_file_path = None
            if has_media:
                media_type = msg_data.get("MediaContentType0") or msg_data.get("mimetype", "")
                
                # Dosyayı kaydet
                if msg_data.get("mediaData"):
                    try:
                        import base64
                        import uuid
                        
                        workspace_dir = Path("workspace").resolve()
                        project_dir = workspace_dir / project_id
                        project_dir.mkdir(parents=True, exist_ok=True)
                        
                        file_data = base64.b64decode(msg_data["mediaData"])
                        ext = msg_data.get("filename", "").split('.')[-1] if '.' in msg_data.get("filename", "") else "bin"
                        if not ext or len(ext) > 4:
                            if media_type.startswith("image/jpeg"): ext = "jpg"
                            elif media_type.startswith("image/png"): ext = "png"
                            elif media_type == "application/pdf": ext = "pdf"
                            else: ext = "dat"
                        
                        filename = f"wa_upload_{str(uuid.uuid4())[:6]}.{ext}"
                        saved_file_path = project_dir / filename
                        with open(saved_file_path, "wb") as f:
                            f.write(file_data)
                        log.info(f"Media saved to {saved_file_path}")
                        
                        # msg metnine dosya yolunu ekle
                        msg += f"\\n[Ekli Dosya Yolu: {saved_file_path}]"
                    except Exception as e:
                        log.error(f"Error saving media: {e}")

                # E6: Voice Notes -> Tasks
                if media_type.startswith("audio/"):
                    stt_text = f"SESLİ NOT (Simüle Edilen STT): Lütfen bu görevi analiz edin."
                    log.info(f"[E6] Voice note received. Simulated STT: {stt_text}")
                    msg = stt_text # Treat audio as the transcribed text as fallback
                    
                    voice_card = WhatsAppMissionCard(
                        card_type=WhatsAppCardType.VOICE_NOTE_PROCESSED,
                        title="Sesli Not Dinlendi",
                        body=f"🔊 Söylediklerinizi şöyle anladım: _{msg}_\\nİşleme alıyorum..."
                    )
                    _push_status(sid, voice_card.render_to_text())
                    
                # E5: File & Image Intake
                elif media_type.startswith("image/") or media_type.startswith("application/"):
                    log.info(f"[E5] File received: {media_type}")
                    
                    file_card = WhatsAppMissionCard(
                        card_type=WhatsAppCardType.MEDIA_RECEIVED,
                        title="Dosya Alındı",
                        body=f"📎 Dosya işlenmek üzere Swarm'a aktarılıyor..."
                    )
                    _push_status(sid, file_card.render_to_text())
            
            # E1: Check for explicit commands (Status, Reject, Approve, Comment)
            msg_upper = msg.upper()
            
            # 8. Project Context Switch (E8)
            if msg_upper.startswith("PROJE DEĞIŞTIR") or msg_upper.startswith("PROJE DEGISTIR"):
                new_project = msg[15:].strip()
                if new_project:
                    user_active_projects[str(sid)] = new_project
                    switch_card = WhatsAppMissionCard(
                        card_type=WhatsAppCardType.CONTEXT_SWITCH,
                        title="Proje Odası Değişti",
                        body=f"Şu an *{new_project}* projesi içindesiniz. Göndereceğiniz dosyalar/mesajlar bu projeye eklenecektir."
                    )
                    _push_status(sid, switch_card.render_to_text())
                    return
            
            # G1: Conversational Query Check
            if any(q in msg_upper for q in ["SON DURUM", "PROJE NE ALEMDE", "HANGI ARTIFACT", "NE DURUMDA"]):
                response_text = brain.handle_conversational_query(msg)
                info_card = WhatsAppMissionCard(
                    card_type=WhatsAppCardType.INFO_RESPONSE,
                    title="Proje Bilgisi",
                    body=response_text
                )
                _push_status(sid, info_card.render_to_text())
                return
                
            # G4: Conversational Summary Check
            if any(q in msg_upper for q in ["BANA Ozet VER", "BANA ÖZET VER", "KISACA ANLAT", "3 MADDELIK OZET"]):
                summary_text = brain.generate_whatsapp_summary()
                sum_card = WhatsAppMissionCard(
                    card_type=WhatsAppCardType.CONVERSATIONAL_SUMMARY,
                    title="Kısa Özet",
                    body=summary_text,
                    action_buttons=["Detay Göster"]
                )
                _push_status(sid, sum_card.render_to_text())
                return

            # 7. Session Briefing (E7)
            if msg_upper in ["ÖZET", "OZET", "DURUM RAPORU", "GÜNLÜK ÖZET", "GUNLUK OZET"]:
                project = PROJECT_STORE.load(project_id)
                if not project:
                    _push_status(sid, f"📝 *{project_id}* projesinde henüz bir veri yok.")
                    return
                    
                completed_tasks = 0
                pending_appr = 0
                if project.active_mission_id:
                    plan = MISSION_STORE.get_plan(project.active_mission_id)
                    if plan:
                        completed_tasks = len([s for s in plan.steps if s.status == StepStatus.COMPLETED])
                        pending_appr = len([s for s in plan.steps if s.status == StepStatus.WAITING_APPROVAL])
                
                artifacts_count = len(project.artifacts)
                
                brief_card = WhatsAppMissionCard(
                    card_type=WhatsAppCardType.PROJECT_BRIEFING,
                    title=f"Proje Özeti: {project_id}",
                    body=f"✅ Tamamlanan Adımlar: {completed_tasks}\n⏳ Bekleyen Onaylar: {pending_appr}\n📁 Üretilen Dosya: {artifacts_count}\n\nSon artifact: {project.artifacts[-1].type if project.artifacts else 'Yok'}"
                )
                _push_status(sid, brief_card.render_to_text())
                return
                
            # Quick Actions (F1/F2)
            if msg_upper in ["ÖZETLE", "OZETLE"]:
                project = PROJECT_STORE.load(project_id)
                if project and project.artifacts:
                    art_type = project.artifacts[-1].type
                    _push_status(sid, f"📄 *Artifact Özeti ({art_type})*: Aktif artifact yaklaşık {len(project.artifacts)} parçadan oluşuyor. Detaylar için 'SON SÜRÜM' diyebilirsiniz.")
                else:
                    _push_status(sid, "⚠️ Henüz oluşturulmuş bir artifact bulunmuyor.")
                return
                
            if msg_upper in ["SON SÜRÜM", "SON SURUM"]:
                project = PROJECT_STORE.load(project_id)
                if project and project.artifacts:
                    file_path = str(project.artifacts[-1].path)
                    _push_status(sid, f"📎 *İşte Son Sürüm*: {file_path} (Sistem lokalinde)")
                else:
                    _push_status(sid, "⚠️ Henüz oluşturulmuş bir artifact bulunmuyor.")
                return
                
            # F3: Comment Batching Check
            if msg_upper.startswith("YORUM:") or msg_upper.startswith("NOT:"):
                comment_text = msg.split(":", 1)[1].strip()
                if sid not in user_revision_buffers:
                    user_revision_buffers[sid] = []
                user_revision_buffers[sid].append(comment_text)
                count = len(user_revision_buffers[sid])
                _push_status(sid, f"✍️ {count}. yorumunuz kaydedildi. Eklemeye devam edebilir veya işleme dökmek için *UYGULA* diyebilirsiniz.")
                return
                
            if msg_upper in ["UYGULA", "REVİZE ET", "REVIZE ET"]:
                if sid in user_revision_buffers and user_revision_buffers[sid]:
                    batched_notes = "\\n- ".join(user_revision_buffers[sid])
                    batched_text = f"Toplu Revizyon Talebi:\\n- {batched_notes}"
                    # Clear buffer
                    user_revision_buffers[sid] = []
                    _push_status(sid, f"🔄 {batched_text.count('-')} madde halinde ilettiğiniz notlar derleniyor...")
                    
                    # Convert to single comment to hit the processing pipeline below
                    msg = batched_text 
                else:
                    _push_status(sid, "⚠️ Uygulanacak kaydedilmiş bir yorumunuz bulunmuyor.")
                    return
            
            # 1. System Status Query
            if msg_upper == "DURUM" or msg_upper == "STATUS":
                status_card = WhatsAppMissionCard(
                    card_type=WhatsAppCardType.STATUS_UPDATE,
                    title="Sistem Durumu",
                    body=f"Şu an *{len(busy_sessions)}* aktif görev işleniyor.\nBoşta olan ajanlar hazır bekliyor.",
                )
                _push_status(sid, status_card.render_to_text())
                return
                
            # 2. Approval Workflow (E3)
            # Check if this is a response to an approval card
            if msg_upper in ["ONAYLA", "APPROVE", "DEVAM", "1"]:
                brain.resolve_pending_approval(project_id, is_approved=True, feedback="WhatsApp üzerinden onaylandı.")
                ack_card = WhatsAppMissionCard(
                    card_type=WhatsAppCardType.STATUS_UPDATE,
                    title="Göreve Devam Ediliyor",
                    body="Onayınız alındı. İşlem arka planda devam ediyor."
                )
                _push_status(sid, ack_card.render_to_text())
                return
            elif msg_upper in ["REDDET", "İPTAL", "REJECT", "CANCEL", "2"]:
                brain.resolve_pending_approval(project_id, is_approved=False, feedback="WhatsApp üzerinden reddedildi.")
                ack_card = WhatsAppMissionCard(
                    card_type=WhatsAppCardType.ERROR_ALERT,
                    title="Görev Durduruldu",
                    body="Red işlemi alındı. İlgili adım iptal edildi."
                )
                _push_status(sid, ack_card.render_to_text())
                return
            
            # 3. Comment Threading (E4)
            # If there's an active project with artifacts, assume brief messages might be comments
            # (In a real system, LLM classification would be used here. We use a simple length/keyword heuristic)
            project = PROJECT_STORE.load(project_id)
            if project and project.active_mission_id and len(msg) < 150 and not msg_upper.startswith("YENİ GÖREV"):
                # Fast path: Treat as a comment on the latest artifact
                target_artifact_id = project.artifacts[-1].artifact_id if project.artifacts else project_id
                
                cm = CommentManager()
                new_comment = Comment(
                    content=msg,
                    author="WhatsApp User",
                    target_id=target_artifact_id,
                    target_type="artifact" if project.artifacts else "project",
                )
                cm.add_comment(new_comment)
                
                # Trigger Refinement Loop
                brain.process_feedback(project.active_mission_id, new_comment)
                
                comment_card = WhatsAppMissionCard(
                    card_type=WhatsAppCardType.REVIEW_BUNDLE,
                    title="Yorum İşleme Alındı",
                    body=f"Şu notunuz sisteme eklendi ve revizyon süreci başlatıldı:\n_{msg}_"
                )
                _push_status(sid, comment_card.render_to_text())
                return

            # 4. Standard Flow: Start new full Swarm execution
            start_card = WhatsAppMissionCard(
                card_type=WhatsAppCardType.MISSION_STARTED,
                title="Görev Analizi Başlatıldı",
                body=f"İsteğiniz Bio-ML Swarm Topluluğuna iletildi: _{msg[:50]}..._",
            )
            _push_status(sid, start_card.render_to_text())
            
            from bio_ml_agent.swarm.orchestrator import SwarmOrchestrator
            
            swarm = SwarmOrchestrator(app_config)
            messages = [{"role": "user", "content": msg}]
            
            final_report = ""
            for update in swarm.process(messages):
                if update["type"] == "status":
                    _push_status(sid, f"🔄 {update['content']}")
                elif update["type"] == "assistant":
                    final_report += update["content"]
                elif update["type"] == "error":
                    final_report += f"\\n❌ Hata: {update['content']}"

            # Check for generated artifacts in the workspace to send back (PDF or PNG)
            media_to_send = None
            import os
            import time
            
            swarm_workspace = Path(swarm.context.workspace_dir)
            recent_files = []
            for ext in ["*.pdf", "*.png", "*.jpg"]:
                recent_files.extend(swarm_workspace.rglob(ext))
                
            current_time = time.time()
            if recent_files:
                recent_files.sort(key=lambda p: os.path.getmtime(str(p)), reverse=True)
                newest = recent_files[0]
                # Modifiye tarihi son 5 dakika içindeyse
                if current_time - os.path.getmtime(str(newest)) < 300: 
                    media_to_send = str(newest)
                    log.info(f"Yollanacak taze artifact bulundu: {media_to_send}")
            
            end_card = WhatsAppMissionCard(
                card_type=WhatsAppCardType.MISSION_COMPLETED,
                title="Görev Tamamlandı",
                body=final_report[:1000] + ("..." if len(final_report) > 1000 else ""),
            )
                
            _push_status(sid, end_card.render_to_text(), media_path=media_to_send)
            
        except Exception as ex:
            log.error(f"[Background Error] {ex}")
            err_card = WhatsAppMissionCard(
                card_type=WhatsAppCardType.ERROR_ALERT,
                title="Kritik Hata",
                body=f"İşlem sırasında beklenmedik bir hata oluştu:\n_{str(ex)[:100]}_",
                action_buttons=["Yeniden Dene", "Logları Gör"]
            )
            _push_status(sid, err_card.render_to_text())
        finally:
            busy_sessions.pop(str(sid), None)

    # Arka planda çalıştır
    thread = threading.Thread(target=background_process, args=(sender_id, data))
    thread.start()
    return jsonify({"reply": "🚀 Görev alındı! Arka planda çalışmaya başlıyorum. Durum güncellemelerini buradan ileteceğim..."})


@app.route("/whatsapp", methods=["POST"])
@limiter.limit("20 per minute")
def whatsapp_webhook():
    """Twilio üzerinden gelen eski/yedek WhatsApp mesaj adaptörü."""
    require_api_key()
    incoming_msg = request.values.get("Body", "").strip()
    sender_id = request.values.get("From", "")

    log.info(f"Twilio WhatsApp mesajı alındı ({sender_id}): {incoming_msg}")

    resp = MessagingResponse()
    msg = resp.message()

    if not incoming_msg:
        msg.body("Lütfen geçerli bir mesaj gönderin.")
        return str(resp)

    if not incoming_msg.upper().startswith("AGT"):
        return str(resp)
        
    if incoming_msg.upper().startswith("AGT "):
        incoming_msg = incoming_msg[4:].strip()
    elif incoming_msg.upper().startswith("AGT"):
        incoming_msg = incoming_msg[3:].strip()

    app_config = get_config()
    
    audit_logger = AuditTrailLogger(app_config.agent.workspace)
    audit_logger.log_critical_action(
        agent_id=sender_id,
        action="WHATSAPP_MESSAGE",
        details={"message_length": len(incoming_msg), "source": "twilio-webhook"},
        approval_status="RECEIVED"
    )
    
    try:
        gateway_url = "http://127.0.0.1:8001/api/v1/platform/chat/async"
        headers = {"X-API-Key": app_config.security.api_key}
        payload = {
            "session_id": sender_id,
            "message": incoming_msg,
            "channel": "whatsapp_twilio",
            "callback_url": PUSH_API_URL,
            "callback_payload": {"to": sender_id},
        }
        
        response = requests.post(gateway_url, json=payload, headers=headers, timeout=5)
        
        if response.status_code in [200, 202]:
            msg.body("İsteğiniz kuyruğa alındı. Sonuçlar işlemler bitince bu numaraya iletilecektir.")
        else:
            msg.body("Sisteme erişilemiyor.")
            
    except Exception as e:
        msg.body(f"Sistemsel hata: {str(e)}")

    return str(resp)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("📱 Bio-ML WhatsApp Çekirdek Sunucusu Başlatılıyor...")
    app.run(host="0.0.0.0", port=5000, debug=False)
