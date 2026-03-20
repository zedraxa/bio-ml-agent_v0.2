import os
import sys
import importlib
from pathlib import Path

def check_package(package_name):
    try:
        importlib.import_module(package_name.split('.')[0])
        print(f"✅ {package_name:25} is installed.")
        return True
    except ImportError:
        print(f"❌ {package_name:25} is NOT installed.")
        return False

def check_env_file():
    env_path = Path(".env")
    if env_path.exists():
        print("✅ .env file found.")
        # Check for critical keys
        with open(env_path, 'r') as f:
            content = f.read()
            keys = ["GEMINI_API_KEY", "OPENAI_API_KEY"]
            found = [k for k in keys if k in content and not content.strip().startswith(f"#{k}")]
            if found:
                print(f"   Found keys: {', '.join(found)}")
            else:
                print("   ⚠️ No active API keys found in .env")
        return True
    else:
        print("❌ .env file NOT found. Please copy .env.example to .env")
        return False

def main():
    print(f"🧠 Bio-ML Agent | Setup Verification")
    print("=" * 40)
    print(f"Python: {sys.version.split(' ')[0]}")
    print(f"Path:   {os.getcwd()}")
    print("-" * 40)

    packages_to_check = [
        "pydantic", "requests", "yaml", "fastapi", "uvicorn", "gradio",
        "google.genai", "openai", "redis", "qdrant_client",
        "numpy", "pandas", "sklearn", "shap", "lime", "Bio"
    ]

    success_count = 0
    for pkg in packages_to_check:
        if check_package(pkg):
            success_count += 1

    print("-" * 40)
    env_ok = check_env_file()
    
    print("-" * 40)
    if success_count == len(packages_to_check) and env_ok:
        print("🚀 System is READY for Bio-ML Agent.")
    else:
        print("⚠️ Some issues were detected. Check the log above.")

if __name__ == "__main__":
    main()
