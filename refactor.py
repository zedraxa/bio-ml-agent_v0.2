import os
import re
from pathlib import Path

MODULES = [
    "core", "ml", "models", "routers", "services", "ultra_agent", "utils", "plugins", "data_streams",
    "agent", "api_server", "job_worker", "llm_backend", "mlflow_tracker", "web_ui", 
    "dataset_catalog", "exceptions", "patch_agent", "plugin_manager", "progress", 
    "report_generator", "whatsapp_connector"
]

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    new_content = content
    for mod in MODULES:
        new_content = re.sub(rf'(?<!bio_ml_agent\.)\bfrom\s+{mod}\b', f'from bio_ml_agent.{mod}', new_content)
        new_content = re.sub(rf'(?<!bio_ml_agent\.)\bimport\s+{mod}\b', f'import bio_ml_agent.{mod}', new_content)

    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated {filepath}")

def main():
    root = Path(".")
    paths = []
    
    target_dirs = ["core", "ml", "models", "routers", "services", "ultra_agent", "utils", "plugins", "data_streams", "scripts", "tests", "api"]
    for d in target_dirs:
        if (root / d).exists():
            paths.extend((root / d).rglob("*.py"))
            
    loose_scripts = ["agent.py", "api_server.py", "job_worker.py", "llm_backend.py", "mlflow_tracker.py", "web_ui.py", "dataset_catalog.py", "exceptions.py", "patch_agent.py", "plugin_manager.py", "progress.py", "report_generator.py", "whatsapp_connector.py", "check_setup.py"]
    for s in loose_scripts:
        if (root / s).exists():
            paths.append(root / s)

    for p in set(paths):
        if p.name == "refactor.py":
            continue
        process_file(p)

    print("Refactoring complete.")

if __name__ == "__main__":
    main()
