import time
import json
import requests
import threading
from flask import Flask, request, jsonify
from pathlib import Path
import sys
import subprocess

# Add src to PYTHONPATH
src_path = str(Path(__file__).resolve().parent.parent / "src")
sys.path.append(src_path)

# 1. Mock Node.js Push API
mock_app = Flask(__name__)
received_messages = []

@mock_app.route("/push-message", methods=["POST"])
def push_message():
    data = request.json
    print(f"✅ Mock Node.js received: {data}")
    received_messages.append(data)
    return jsonify({"ok": True})

def run_mock_node():
    mock_app.run(port=3001)

# Start Mock Node in thread
threading.Thread(target=run_mock_node, daemon=True).start()
time.sleep(2)

# 2. Start WhatsApp Connector (Flask) in background
# We'll use subprocess to run it with PYTHONPATH properly
connector_env = {**subprocess.os.environ, "PYTHONPATH": src_path}
connector_proc = subprocess.Popen(
    [sys.executable, "src/bio_ml_agent/whatsapp_connector.py"],
    env=connector_env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

print("⏳ Waiting for Connector to start...")
time.sleep(3)

# 3. Trigger a simulated message
# Note: In real life, Node.js calls /whatsapp-local
print("🚀 Sending simulated 'AGT' message to connector...")
try:
    resp = requests.post(
        "http://127.0.0.1:5000/whatsapp-local",
        json={"text": "Hello Agent", "from": "123456789"},
        timeout=10
    )
    print(f"Connector Response: {resp.json()}")
except Exception as e:
    print(f"Error calling connector: {e}")

# Wait for background process to finish and call back our mock
print("⏳ Waiting for Agent response (max 20s)...")
for _ in range(20):
    if received_messages:
        break
    time.sleep(1)

# 4. Final Verification
if received_messages:
    print("✨ SUCCESS: WhatsApp flow verified!")
    for msg in received_messages:
        print(f"Final Message: {msg['text']}")
else:
    print("❌ FAILED: No message received back from Agent.")

# Cleanup
connector_proc.terminate()
print("Cleanup done.")

if not received_messages:
    sys.exit(1)
