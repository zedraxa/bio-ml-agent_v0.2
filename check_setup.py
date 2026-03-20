import sys
import importlib

def check_package(package_name):
    try:
        importlib.import_module(package_name)
        print(f"✅ {package_name} is installed.")
        return True
    except ImportError:
        print(f"❌ {package_name} is NOT installed.")
        return False

packages_to_check = [
    "pydantic", "requests", "yaml", "colorama", "rich", "tqdm", 
    "aiohttp", "qdrant_client", "redis", "ollama", 
    "google.genai", "google.generativeai", "huggingface_hub", "transformers",
    "sklearn", "scipy", "matplotlib", "seaborn", "Bio", "cv2", "PIL", "skimage",
    "shap", "lime", "imblearn", "torch", "torchvision", "pypdf", "docx", "rank_bm25",
    "sentence_transformers", "fastapi", "uvicorn", "flask", "gradio"
]

print(f"Python version: {sys.version}")
print("-" * 30)

success_count = 0
for pkg in packages_to_check:
    if check_package(pkg):
        success_count += 1

print("-" * 30)
print(f"Result: {success_count}/{len(packages_to_check)} packages verified.")

if success_count == len(packages_to_check):
    print("🚀 All core packages are correctly installed.")
else:
    print("⚠️ Some packages are missing or could not be loaded.")
