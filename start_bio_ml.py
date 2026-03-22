import subprocess
import sys
import os
import time
from pathlib import Path

# Explicitly set PYTHONPATH so that imports from "bio_ml_agent" work automatically
src_path = Path(__file__).resolve().parent / "src"
os.environ["PYTHONPATH"] = str(src_path) + os.pathsep + os.environ.get("PYTHONPATH", "")

def main():
    print("🧬 Starting Bio-ML Agent OS [Real-World Deployment]")
    
    # Load Environment
    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        print("⚠️ Warning: .env file not found. Falling back to system environment variables.")
    else:
        print("✅ Found .env file.")

    # 1. Start automated WhatsApp Tunnel (ngrok)
    try:
        from bio_ml_agent.utils.ngrok_manager import start_ngrok
        ngrok_url = start_ngrok(port=5000)
    except Exception as e:
        print(f"⚠️ Could not load ngrok manager: {e}")

    # 2. Spawn local Web UI (Gradio)
    print("\n🖥️ Starting Agent Intelligence Web UI...")
    ui_process = subprocess.Popen(
        [sys.executable, "src/bio_ml_agent/web_ui.py"],
        env=os.environ
    )
    
    # 3. Spawn WhatsApp Connector Backend
    print("📱 Starting WhatsApp Operational Router (Port 5000)...")
    wa_process = subprocess.Popen(
        [sys.executable, "src/bio_ml_agent/whatsapp_connector.py"],
        env=os.environ
    )
    
    try:
        while True:
            time.sleep(1)
            # Check if any child process crashed
            if ui_process.poll() is not None:
                print("❌ Web UI process exited unexpectedly.")
                break
            if wa_process.poll() is not None:
                print("❌ WhatsApp Connector process exited unexpectedly.")
                break
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Bio-ML Agent OS gracefully...")
    finally:
        ui_process.terminate()
        wa_process.terminate()
        print("✅ Shutdown complete.")
        sys.exit(0)

if __name__ == "__main__":
    main()
