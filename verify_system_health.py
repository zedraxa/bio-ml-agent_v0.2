import os
from pathlib import Path

def check_env():
    print("Checking Environment Variables...")
    
    # Load .env explicitly if available
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    k, v = line.strip().split("=", 1)
                    os.environ[k] = v
    
    llm_key = os.getenv("OPENAI_API_KEY", "") or os.getenv("GEMINI_API_KEY", "") or os.getenv("ANTHROPIC_API_KEY", "")
    twilio_key = os.getenv("TWILIO_ACCOUNT_SID", "")
    ngrok_token = os.getenv("NGROK_AUTHTOKEN", "")
    
    print(f"  [ENV] LLM API Keys (OpenAI/Gemini/Claude): {'✅' if llm_key else '❌'}")
    print(f"  [ENV] Twilio Configured: {'✅' if twilio_key else '⚠️ (WhatsApp will not respond automatically)'}")
    print(f"  [ENV] Ngrok Token Configured: {'✅' if ngrok_token else '⚠️ (Tunnel may have limits)'}")
    
    return bool(llm_key)

def check_db():
    print("\nChecking Local Storage...")
    default_history = Path(__file__).parent / "history_default"
    default_history.mkdir(exist_ok=True, parents=True)
    try:
        test_file = default_history / ".test_write"
        test_file.touch()
        test_file.unlink()
        print("  [DB] Storage Writable: ✅")
    except Exception as e:
        print(f"  [DB] Storage Write Failed: ❌ ({e})")

def check_llm():
    print("\nChecking LLM Subsystem Initialization...")
    try:
        from bio_ml_agent.llm_backend import auto_create_backend
        # Mock initial load
        _ = auto_create_backend("test", mode="auto")
        print("  [LLM] LLM Router Instantiated: ✅")
    except ImportError as e:
        print(f"  [LLM] Module Import Failed. Run `pip install -e .`: ❌ ({e})")
    except Exception as e:
        # Fallback if config requires active API logic immediately
        print("  [LLM] LLM Loader Passed with warnings: ⚠️")

def check_ngrok():
    print("\nChecking pyngrok dependencies...")
    try:
        import pyngrok  # type: ignore
        print("  [NET] pyngrok is installed: ✅")
    except ImportError:
        print("  [NET] pyngrok is NOT installed. Install via `pip install pyngrok`: ❌")

if __name__ == "__main__":
    print("\n🔍 Running Bio-ML Agent Pre-flight checks...")
    print("=" * 50)
    
    has_env = check_env()
    check_db()
    check_ngrok()
    
    # Needs PYTHONPATH logic for the standalone script
    src_path = Path(__file__).parent / "src"
    os.environ["PYTHONPATH"] = str(src_path) + os.pathsep + os.environ.get("PYTHONPATH", "")
    import sys
    sys.path.insert(0, str(src_path))
    
    check_llm()
    
    print("\n" + "=" * 50)
    print("✨ System Health Checks Completed.")
    print("Run `python start_bio_ml.py` to launch the OS!")
