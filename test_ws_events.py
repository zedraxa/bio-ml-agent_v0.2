import asyncio
import websockets
import json
import requests
import threading
import time

BASE_URL = "http://127.0.0.1:8001/api/v1/platform"
WS_URL = "ws://127.0.0.1:8001/api/v1/platform/ws/events/test-session-1"

async def ws_client():
    try:
        async with websockets.connect(WS_URL) as ws:
            print("WS Connected!")
            while True:
                msg = await ws.recv()
                print(f"WS Received: {msg}")
    except Exception as e:
        print(f"WS Exception: {e}")

def trigger_event():
    time.sleep(2) # ws baglantisini bekle
    payload = {
        "event_id": "evt-123",
        "session_id": "test-session-1",
        "event_type": "tool_start",
        "payload": {"tool_name": "Calculator", "args": "2+2"},
        "timestamp": "2026-03-10T12:00:00Z"
    }
    headers = {"X-API-Key": "YOUR_API_KEY_HERE"}
    try:
        print("Sending HTTP POST to trigger event...")
        res = requests.post(f"{BASE_URL}/runs/run-mock/events/emit", json=payload, headers=headers)
        print("HTTP Res:", res.status_code, res.json())
        time.sleep(1) # ws ye gelmesini bekle
    except Exception as e:
         print(f"HTTP Exception: {e}")

if __name__ == "__main__":
    t = threading.Thread(target=trigger_event)
    t.start()
    asyncio.run(ws_client())
