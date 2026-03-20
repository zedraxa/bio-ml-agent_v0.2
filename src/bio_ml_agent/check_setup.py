import os
import sys
import socket
import importlib
import requests
from pathlib import Path

# ─────────────────────────────────────────────
#  Configuration & Definitions
# ─────────────────────────────────────────────

CRITICAL_PACKAGES = [
    "pydantic", "requests", "yaml", "fastapi", "uvicorn", "gradio",
    "google.genai", "openai", "redis", "qdrant_client"
]

DATA_PACKAGES = [
    "numpy", "pandas", "sklearn", "shap", "lime", "Bio"
]

SERVICE_PORTS = {
    "Web UI (Gradio)": 5050,
    "API Server / Control Plane": 8001,
    "WhatsApp Node Client": 3001,
    "WhatsApp Flask Bridge": 5000,
    "Redis": 6379,
    "Qdrant": 6333
}

# ─────────────────────────────────────────────
#  Helper Functions
# ─────────────────────────────────────────────

def check_package(package_name):
    try:
        importlib.import_module(package_name.split('.')[0])
        return True, "Installed"
    except ImportError:
        return False, "MISSING"

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def check_redis():
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, socket_timeout=1)
        if r.ping():
            return True, "Connected"
    except Exception as e:
        return False, str(e)
    return False, "Not Reachable"

def check_qdrant():
    try:
        resp = requests.get("http://localhost:6333/healthz", timeout=1)
        if resp.status_code == 200:
            return True, "Healthy"
    except Exception:
        pass
    return False, "Not Reachable"

def check_env():
    env_path = Path(".env")
    if not env_path.exists():
        return False, "Missing .env"
    
    with open(env_path, 'r') as f:
        content = f.read()
    
    keys = ["GEMINI_API_KEY", "OPENAI_API_KEY"]
    found = [k for k in keys if k in content and not content.strip().startswith(f"#{k}")]
    
    if found:
        return True, f"Found {len(found)} keys ({', '.join(found)})"
    return False, "No active API keys found"

# ─────────────────────────────────────────────
#  Main Execution
# ─────────────────────────────────────────────

def main():
    print("\033[1;36m" + "="*50)
    print("🧠 BIO-ML AGENT | DEEP HEALTH CHECK")
    print("="*50 + "\033[0m")
    
    print(f"[*] OS:     {sys.platform}")
    print(f"[*] Python: {sys.version.split(' ')[0]}")
    print(f"[*] Cwd:    {os.getcwd()}")
    print("-" * 50)

    # 1. Dependency Check
    print("\n\033[1;33m[1] Dependency Check\033[0m")
    all_pkg = CRITICAL_PACKAGES + DATA_PACKAGES
    missing = []
    for pkg in all_pkg:
        ok, status = check_package(pkg)
        mark = "✅" if ok else "❌"
        print(f" {mark} {pkg:25} -> {status}")
        if not ok:
            missing.append(pkg)

    # 2. Port & Service Check
    print("\n\033[1;33m[2] Port & Service Monitoring\033[0m")
    for name, port in SERVICE_PORTS.items():
        in_use = is_port_in_use(port)
        mark = "🔹" if in_use else "⚪"
        status = "IN USE" if in_use else "FREE"
        print(f" {mark} {name:28} [Port {port}]: {status}")

    # 3. Connectivity Check (Optional/Live)
    print("\n\033[1;33m[3] Service Connectivity (Live Tests)\033[0m")
    
    r_ok, r_msg = check_redis()
    print(f" {'✅' if r_ok else '⚠️'} Redis Connection:       {r_msg}")
    
    q_ok, q_msg = check_qdrant()
    print(f" {'✅' if q_ok else '⚠️'} Qdrant Health:         {q_msg}")

    # 4. Environment Check
    print("\n\033[1;33m[4] Environment & Configuration\033[0m")
    e_ok, e_msg = check_env()
    print(f" {'✅' if e_ok else '❌'} .env File:              {e_msg}")

    # Summary & Mode Detection
    print("\n" + "="*50)
    if not missing and e_ok:
        print("\033[1;32m🚀 STATUS: READY FOR LAUNCH\033[0m")
        if r_ok and q_ok:
            print("[Mode] Recommendation: FULL DOCKER MODE (All services detected)")
        else:
            print("[Mode] Recommendation: MINIMAL LOCAL MODE (Partial services detected)")
    else:
        print("\033[1;31m⚠️ STATUS: ATTENTION REQUIRED\033[0m")
        if missing:
            print(f"[*] Missing dependencies: {', '.join(missing)}")
        if not e_ok:
            print("[*] Fix your .env file to enable LLM capabilities.")
    print("="*50)

if __name__ == "__main__":
    main()
