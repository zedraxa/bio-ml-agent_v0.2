import os
import sys
import logging
from pathlib import Path
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from flask import jsonify

# Proje kökünü path'e ekle
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import setup_logger, load_config
from web_ui import process_message

# Flask app oluştur
app = Flask(__name__)

# Logger
log_dir = Path("logs").resolve()
log_dir.mkdir(exist_ok=True)
log = setup_logger(log_dir, "INFO")

# Bellek (Her telefon numarası için geçici mesaj geçmişi)
session_histories = {}


@app.route("/whatsapp-local", methods=["POST"])
def whatsapp_local():
    """Node.js (whatsapp-web.js) üzerinden gelen mesajı Ajan'a ilet."""
    data = request.json or {}
    incoming_msg = data.get("text", "").strip()
    sender_id = data.get("from", "")

    log.info(f"[Whatsapp-Local] Mesaj alındı ({sender_id}): {incoming_msg}")

    if not incoming_msg:
        return jsonify({"reply": "Lütfen geçerli bir mesaj gönderin."})

    if not incoming_msg:
        return jsonify({"reply": "Lütfen geçerli bir mesaj gönderin."})

    # Oturum geçmişini al veya oluştur
    if sender_id not in session_histories:
        session_histories[sender_id] = []
    
    history = session_histories[sender_id]
    
    # Konfigürasyonu yükle
    app_config = load_config()
    model = "gemini-2.5-flash"  # Kullanıcı isteği: STR çalıştırıldığında özel olarak bu model kullanılsın
    timeout = app_config.agent.timeout
    max_steps = app_config.agent.max_steps

    try:
        final_history = history
        final_status = ""
        
        for updated_history, status in process_message(
            user_msg=incoming_msg, 
            chat_history=history, 
            model=model, 
            timeout=timeout, 
            max_steps=max_steps
        ):
            final_history = updated_history
            final_status = status
            
        session_histories[sender_id] = final_history
        
        if final_history and final_history[-1]["role"] == "assistant":
            agent_reply = final_history[-1]["content"]
            return jsonify({"reply": agent_reply})
        else:
            return jsonify({"reply": "Ajan bir yanıt üretemedi. Durum: " + final_status})
            
    except Exception as e:
        error_text = f"Sistemsel bir hata oluştu: {str(e)}"
        log.error(error_text)
        return jsonify({"reply": error_text})


@app.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():
    """Twilio üzerinden gelen eski/yedek WhatsApp mesaj adaptörü."""
    incoming_msg = request.values.get("Body", "").strip()
    sender_id = request.values.get("From", "")

    log.info(f"Twilio WhatsApp mesajı alındı ({sender_id}): {incoming_msg}")

    resp = MessagingResponse()
    msg = resp.message()

    if not incoming_msg:
        msg.body("Lütfen geçerli bir mesaj gönderin.")
        return str(resp)

    if not incoming_msg.upper().startswith("AGT"):
        # AGT ile başlamıyorsa sessizce yoksay
        return str(resp)
        
    # Ajanın anlaması için "AGT " kısmını temizle
    if incoming_msg.upper().startswith("AGT "):
        incoming_msg = incoming_msg[4:].strip()
    elif incoming_msg.upper().startswith("AGT"):
        incoming_msg = incoming_msg[3:].strip()

    if sender_id not in session_histories:
        session_histories[sender_id] = []
    history = session_histories[sender_id]
    
    app_config = load_config()
    
    try:
        final_history = history
        for updated_history, status in process_message(
            user_msg=incoming_msg, chat_history=history, 
            model="gemini-2.5-flash", timeout=app_config.agent.timeout, max_steps=app_config.agent.max_steps
        ):
            final_history = updated_history
            
        session_histories[sender_id] = final_history
        
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
