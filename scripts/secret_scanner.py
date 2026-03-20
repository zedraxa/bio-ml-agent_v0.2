#!/usr/bin/env python3
import os
import re
import sys
from pathlib import Path

# Sık rastlanan Secret Regex paternleri
SECRET_PATTERNS = {
    "OpenAI API Key": r"sk-[a-zA-Z0-9]{48}",
    "AWS Access Key": r"AKIA[0-9A-Z]{16}",
    "Slack Token": r"xox[baprs]-[0-9a-zA-Z]{10,48}",
    "Github Token": r"gh[pousr]_[0-9a-zA-Z]{36}"
}

def scan_directory(dir_path: Path) -> bool:
    found_secrets = False
    for root, dirs, files in os.walk(dir_path):
        # Ignore common non-source directories
        if any(ignored in root for ignored in ['.git', '.venv', '__pycache__', 'logs', 'node_modules']):
            continue
            
        for file_name in files:
            # Check source code files and configs
            if not file_name.endswith(('.py', '.json', '.yaml', '.yml', '.md', '.env', '.sh', '.txt')):
                continue
                
            file_path = Path(root) / file_name
            try:
                content = file_path.read_text(encoding='utf-8')
                for idx, line in enumerate(content.splitlines(), 1):
                    for secret_name, pattern in SECRET_PATTERNS.items():
                        if re.search(pattern, line):
                            print(f"[!] {secret_name} SIZINTISI BULUNDU: {file_path}:{idx}")
                            found_secrets = True
            except Exception:
                pass
                
    return found_secrets

if __name__ == "__main__":
    target_dir = Path(__file__).parent.parent
    has_leaks = scan_directory(target_dir)
    
    if has_leaks:
        print("\n❌ GÜVENLİK İHLALİ: Hardcoded şifre veya API key sistem taramasında yakalandı!")
        sys.exit(1)
    else:
        print("✅ GÜVENLİK TARAMASI: Kod tabanı temiz, sızıntı bulunamadı.")
        sys.exit(0)
