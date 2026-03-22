import logging

logger = logging.getLogger(__name__)

def start_ngrok(port: int = 5000):
    """
    Starts an ngrok tunnel to expose the local Flask webhook port to the internet.
    Automatically logs the full Twilio URL.
    """
    try:
        from pyngrok import ngrok
        logger.info(f"Starting ngrok tunnel for port {port}...")
        public_url = ngrok.connect(port).public_url
        
        print("\n" + "="*65)
        print("🚀 NGROK TUNNEL ACTIVE")
        print(f"🔗 Base URL: {public_url}")
        print(f"🔗 TWILIO WEBHOOK URL: {public_url}/whatsapp-webhook")
        print("="*65 + "\n")
        
        return public_url
    except ImportError:
        logger.error("pyngrok is not installed. Run 'pip install pyngrok' to use automated tunneling.")
        print("\n[WARNING] pyngrok not installed. Automated WhatsApp tunneling will not start.")
        print("To install: pip install pyngrok")
        return None
    except Exception as e:
        logger.error(f"Failed to start ngrok: {str(e)}")
        print(f"\n[ERROR] Failed to start ngrok: {e}")
        return None
