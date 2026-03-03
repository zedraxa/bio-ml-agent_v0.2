import os
import json
from pathlib import Path
from typing import Optional, Dict, Any

VAULT_DIR = Path.home() / ".bio-ml-agent"
VAULT_FILE = VAULT_DIR / "vault.json"

def _ensure_vault_exists():
    VAULT_DIR.mkdir(parents=True, exist_ok=True)
    if not VAULT_FILE.exists():
        with open(VAULT_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f)

def get_vault() -> Dict[str, Any]:
    _ensure_vault_exists()
    try:
        with open(VAULT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_vault(data: Dict[str, Any]):
    _ensure_vault_exists()
    with open(VAULT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def save_credential(platform_name: str, username: str, password: str, email: Optional[str] = None, api_key: Optional[str] = None):
    """
    Belirli bir platform ('kaggle', 'huggingface' vb.) için giriş bilgilerini kaydeder.
    """
    vault = get_vault()
    
    vault[platform_name] = {
        "username": username,
        "password": password,
        "email": email,
        "api_key": api_key
    }
    
    # Remove None values
    vault[platform_name] = {k: v for k, v in vault[platform_name].items() if v is not None}
    
    save_vault(vault)
    return f"[{platform_name}] Kimlik bilgileri başarıyla kasaya (vault) kaydedildi."

def get_credential(platform_name: str) -> Optional[Dict[str, str]]:
    """
    Kasadaki belirli bir platformun bilgilerini döner. Bulunamazsa None döner.
    """
    vault = get_vault()
    return vault.get(platform_name)

def list_platforms() -> list:
    """
    Kayıtlı olan tüm platform isimlerini döner.
    """
    return list(get_vault().keys())
