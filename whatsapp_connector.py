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
from services.agent_service import AgentService
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
        # LLMRouter model kararını içerde kendi verecek, burada `model=""` veya null geçiyoruz.
        service = AgentService(model="", timeout=timeout, max_steps=max_steps)
        
        # Sadece sender_id'yi set ediyoruz. AgentCore arka planda Qdrant'a bakar.
        service.session_id = sender_id
        
        step_count = 0
        tool_count = 0
        
        for event in service.process_message(user_msg=incoming_msg):
            ev_type = event.get("type")
            
            if ev_type == "status":
                status_text = event.get("content", "")
                if status_text:
                    _push_status(sender_id, f"📊 {status_text}")
                    
            elif ev_type == "tool_start":
                tool_name = event.get("tool", "")
                tool_count += 1
                emoji_map = {
                    "PYTHON": "🐍",
                    "BASH": "🔧",
                    "WRITE_FILE": "📝",
                    "READ_FILE": "📖",
                    "WEB_SEARCH": "🔍",
                    "WEB_OPEN": "🌐",
                    "RAG_SEARCH": "🔎",
                }
                emoji = emoji_map.get(tool_name, "🛠️")
                _push_status(sender_id, f"{emoji} Araç çalışıyor: {tool_name}")
                    
            elif ev_type == "tool_output":
                tool_name = event.get("tool", "araç")
                output = event.get("output", "")
                # Kısa özet gönder (ilk 200 karakter)
                summary = output[:200].replace("\n", " ").strip()
                if len(output) > 200:
                    summary += "..."
                _push_status(sender_id, f"✅ {tool_name} tamamlandı\n{summary}")
                    
            elif ev_type == "chunk":
                step_count += 1
                # Her 3 chunk'ta bir düşünme durumu bildir
                if step_count % 3 == 0:
                    _push_status(sender_id, f"🧠 Düşünüyor... (adım {step_count})")
                    
            elif ev_type == "error":
                error_text = event.get("content", "Hata oluştu.")
                _push_status(sender_id, f"❌ {error_text}")
                
        final_history = service.messages
        
        if final_history and final_history[-1]["role"] == "assistant":
            agent_reply = final_history[-1]["content"]
            if len(agent_reply) > 1500:
                agent_reply = agent_reply[:1500] + "\n\n... (Mesaj sınırına ulaşıldı)"
            return jsonify({"reply": agent_reply})
        else:
            return jsonify({"reply": "Ajan bir yanıt üretemedi."})
            
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
        service = AgentService(model="", timeout=app_config.agent.timeout, max_steps=app_config.agent.max_steps)
        service.session_id = sender_id

        for event in service.process_message(incoming_msg):
            pass
            
        final_history = service.messages
        
        if final_history and final_history[-1]["role"] == "assistant":
            agent_reply = final_history[-1]["content"]
            if len(agent_reply) > 1500:
                agent_reply = agent_reply[:1500] + "\n\n... (Mesaj sınırına ulaşıldı)"
            msg.body(agent_reply)
        else:
            msg.body("Ajan bir yanıt üretemedi.")
    except Exception as e:
        msg.body(f"Sistemsel hata: {str(e)}")

    return str(resp)


if __name__ == "__main__":
    print("📱 Bio-ML WhatsApp Çekirdek Sunucusu Başlatılıyor...")
    app.run(host="0.0.0.0", port=5000, debug=False)
