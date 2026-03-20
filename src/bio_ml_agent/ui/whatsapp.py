import os
import sys
import shutil
import signal
import subprocess
import time
import logging
import requests
import qrcode
from pathlib import Path
from typing import Optional, Tuple
from bio_ml_agent.utils.config import get_config

log = logging.getLogger("bio_ml_agent")
config = get_config()

_whatsapp_node_proc = None
_whatsapp_flask_proc = None

def start_whatsapp_services() -> Tuple[str, Optional[Any]]:
    global _whatsapp_node_proc, _whatsapp_flask_proc
    
    # Resolve project root relative to this file
    # src/bio_ml_agent/ui/whatsapp.py -> project_root
    root = Path(__file__).resolve().parent.parent.parent.parent
    node_client_dir = root / "whatsapp-client"
    connector_script = root / "src" / "bio_ml_agent" / "whatsapp_connector.py"
    (root / "logs").mkdir(parents=True, exist_ok=True)
    
    if _whatsapp_node_proc is None or _whatsapp_node_proc.poll() is not None:
        node_bin = config.whatsapp.node_executable
        
        bash_cmd_cleanup = (
            "pkill -9 -f 'node index.js' || true && "
            "fuser -k 3001/tcp || true && "
            "rm -rf whatsapp-client/.wwebjs_auth/session/SingletonLock || true"
        )
        subprocess.run(["/bin/bash", "-c", bash_cmd_cleanup], check=False)
        time.sleep(1)

        qr_file_path = node_client_dir / "qr.png"
        if qr_file_path.exists():
            try: qr_file_path.unlink()
            except Exception: pass

        if not shutil.which(node_bin):
             # Fallback check
             node_bin = "node"
             if not shutil.which(node_bin):
                return "❌ **Node.js bulunamadı.**", None
        
        log_path = root / "logs" / "whatsapp_node.log"
        bash_cmd = f"'{node_bin}' index.js --accept-tos > '{log_path}' 2>&1"
        _whatsapp_node_proc = subprocess.Popen(
            ["/bin/bash", "-c", bash_cmd], 
            cwd=node_client_dir, 
            start_new_session=True
        )
        log.info(f"WhatsApp Node.js client started.")

    if _whatsapp_flask_proc is None or _whatsapp_flask_proc.poll() is not None:
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{root}/src:{env.get('PYTHONPATH', '')}"
        log_path = root / "logs" / "whatsapp_flask.log"
        f_flask = open(log_path, "w", encoding="utf-8")
        _whatsapp_flask_proc = subprocess.Popen(
            [sys.executable, str(connector_script)], 
            env=env, stdout=f_flask, stderr=f_flask, start_new_session=True
        )
        log.info(f"WhatsApp Flask connector started.")

    empty_img = None
    try:
        from PIL import Image
        empty_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
    except Exception: pass

    return "⌛ **Servis Başlatılıyor...**", empty_img

def stop_whatsapp_services():
    global _whatsapp_node_proc, _whatsapp_flask_proc
    if _whatsapp_node_proc:
        try: os.killpg(_whatsapp_node_proc.pid, signal.SIGTERM)
        except Exception: _whatsapp_node_proc.terminate()
        _whatsapp_node_proc = None
    if _whatsapp_flask_proc:
        try: os.killpg(_whatsapp_flask_proc.pid, signal.SIGTERM)
        except Exception: _whatsapp_flask_proc.terminate()
        _whatsapp_flask_proc = None
    return "Servisler durduruldu."

def get_whatsapp_status():
    global _whatsapp_node_proc
    empty_img = None
    try:
        from PIL import Image
        empty_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
    except Exception: pass

    try:
        resp = requests.get(f"http://localhost:{config.whatsapp.port}/status", timeout=1)
        if resp.status_code == 200:
            status = resp.json().get("status", "Bilinmiyor")
            root = Path(__file__).resolve().parent.parent.parent.parent
            node_client_dir = root / "whatsapp-client"

            qr_file_path = node_client_dir / "qr.png"
            if qr_file_path.exists() and status != "CONNECTED":
                try:
                    from PIL import Image
                    img = Image.open(str(qr_file_path)).convert('RGB')
                    return "📱 **QR Kod Hazır.**", img
                except Exception: pass

            if status == "CONNECTED":
                try:
                    from PIL import Image, ImageDraw
                    img = Image.new('RGB', (400, 400), color=(37, 211, 102))
                    draw = ImageDraw.Draw(img)
                    draw.line([(100, 200), (180, 280), (300, 120)], fill="white", width=30)
                    return "✅ **Bağlantı Kuruldu!**", img
                except Exception: return "✅ **Bağlantı Kuruldu!**", empty_img

            if status == "QR_READY":
                qr_resp = requests.get(f"http://localhost:{config.whatsapp.port}/qr", timeout=1)
                qr_str = qr_resp.json().get("qr")
                if qr_str:
                    qr_gen = qrcode.QRCode(version=1, box_size=12, border=10)
                    qr_gen.add_data(qr_str)
                    qr_gen.make(fit=True)
                    img = qr_gen.make_image(fill_color="black", back_color="white")
                    return "📱 **QR Kod Hazır.**", img.convert('RGB')
            
            return f"ℹ️ **Durum:** {status}", empty_img
            
    except Exception:
        if _whatsapp_node_proc is not None and _whatsapp_node_proc.poll() is None:
            return "⌛ **Servis Hazırlanıyor...**", empty_img
        return "❌ **Servis Çevrimdışı.**", empty_img
