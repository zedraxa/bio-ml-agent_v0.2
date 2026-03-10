import os
import sys
import logging
import requests
from pathlib import Path
from flask import Flask, request, jsonify, abort
from twilio.twiml.messaging_response import MessagingResponse
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Proje kökünü path'e ekle
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.logger import setup_logger
from utils.config import load_config
# AgentService importu kaldırıldı -> Control Plane Gateway'e HTTP istek atılacak
from ultra_agent.observability.audit_trail import AuditTrailLogger

# Flask app oluştur
app = Flask(__name__)

# Logger
log_dir = Path("logs").resolve()
log_dir.mkdir(exist_ok=True)
log = setup_logger(log_dir, "INFO")

# Rate Limiter (Saniyede max/ip)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["100 per day", "20 per minute"]
)

# API Güvenlik Kontrolü
def require_api_key():
    config = load_config()
    expected_key = config.security.api_key
    if not expected_key:
        return # Güvenlik anahtarı ayarlanmamışsa serbest geçiş
    
    # Twilio ve Whatsapp-Web tarafı Authorization Header, X-API-Key veya URL parametresi kullanabilir.
    api_key = request.headers.get("X-API-Key") or request.args.get("api_key")
    if api_key != expected_key:
        log.warning(f"Yetkisiz Webhook Erişimi (IP: {get_remote_address()})")
        abort(403, description="Geçersiz veya eksik API Key")

# Bellek (AgentService Qdrant altyapısı üzerinden yönetilecek, burada durum tutulmuyor)
# session_histories = {}

# Node.js Push API adresi
PUSH_API_URL = "http://127.0.0.1:3001/push-message"


def _push_status(sender_id: str, text: str):
    """Node.js üzerinden WhatsApp'a ara durum mesajı gönder."""
    try:
        requests.post(PUSH_API_URL, json={"to": sender_id, "text": text}, timeout=5)
    except Exception as e:
        log.warning(f"[Push] Mesaj gönderilemedi: {e}")


@app.route("/whatsapp-local", methods=["POST"])
@limiter.limit("10 per minute")
def whatsapp_local():
    """Node.js (whatsapp-web.js) üzerinden gelen mesajı Ajan'a ilet."""
    require_api_key()
    data = request.json or {}
    incoming_msg = data.get("text", "").strip()
    sender_id = data.get("from", "")

    log.info(f"[Whatsapp-Local] Mesaj alındı ({sender_id}): {incoming_msg}")

    if not incoming_msg:
        return jsonify({"reply": "Lütfen geçerli bir mesaj gönderin."})

    app_config = load_config()
    timeout = app_config.agent.timeout
    max_steps = app_config.agent.max_steps
    
    audit_logger = AuditTrailLogger()
    audit_logger.log_critical_action(
        action_type="WHATSAPP_MESSAGE",
        user_id=sender_id,
        details={"message_length": len(incoming_msg), "source": "whatsapp-local"},
        status="RECEIVED"
    )

    try:
        # Ajan doğrudan burada çalıştırılmayacak. Mesajı Gateway'e iletiyoruz.
        gateway_url = "http://127.0.0.1:8000/api/v1/platform/chat/async"
        headers = {"X-API-Key": app_config.security.api_key}
        payload = {
            "session_id": sender_id,
            "message": incoming_msg,
            "channel": "whatsapp_local"
        }
        
        response = requests.post(gateway_url, json=payload, headers=headers, timeout=5)
        
        if response.status_code in [200, 202]:
            return jsonify({"reply": "Mesajınız bulut aracıma iletildi. İşlem tamamlanınca sonuçlar size gönderilecek."})
        else:
            return jsonify({"reply": "Sistem şu an meşgul. Lütfen daha sonra tekrar deneyin."})
            
    except Exception as e:
        error_text = f"Sistemsel bir hata oluştu: {str(e)}"
        log.error(error_text)
        _push_status(sender_id, f"💥 {error_text}")
        return jsonify({"reply": error_text})


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

    app_config = load_config()
    
    audit_logger = AuditTrailLogger()
    audit_logger.log_critical_action(
        action_type="WHATSAPP_MESSAGE",
        user_id=sender_id,
        details={"message_length": len(incoming_msg), "source": "twilio-webhook"},
        status="RECEIVED"
    )
    
    try:
        gateway_url = "http://127.0.0.1:8000/api/v1/platform/chat/async"
        headers = {"X-API-Key": app_config.security.api_key}
        payload = {
            "session_id": sender_id,
            "message": incoming_msg,
            "channel": "whatsapp_twilio"
        }
        
        response = requests.post(gateway_url, json=payload, headers=headers, timeout=5)
        
        if response.status_code in [200, 202]:
            msg.body("İsteğiniz kuyruğa alındı. Sonuçlar işlemler bitince bu numaraya iletilecektir.")
        else:
            msg.body("Sisteme erişilemiyor.")
            
    except Exception as e:
        msg.body(f"Sistemsel hata: {str(e)}")

    return str(resp)


if __name__ == "__main__":
    print("📱 Bio-ML WhatsApp Çekirdek Sunucusu Başlatılıyor...")
    app.run(host="0.0.0.0", port=5000, debug=False)
