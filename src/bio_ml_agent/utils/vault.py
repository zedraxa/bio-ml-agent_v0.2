import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
from cryptography.fernet import Fernet
import base64

VAULT_DIR = Path.home() / ".bio-ml-agent"
VAULT_FILE = VAULT_DIR / "vault.enc"
KEY_FILE = VAULT_DIR / "vault.key"

def _get_or_create_key() -> bytes:
    if not KEY_FILE.exists():
        key = Fernet.generate_key()
        with open(KEY_FILE, "wb") as f:
            f.write(key)
        # Sadece kullanıcının okuyabileceği şekilde izni ayarla
        os.chmod(KEY_FILE, 0o600)
        return key
    with open(KEY_FILE, "rb") as f:
        return f.read()

def _ensure_vault_exists():
    VAULT_DIR.mkdir(parents=True, exist_ok=True)
    if not VAULT_FILE.exists():
        save_vault({})

def get_vault() -> Dict[str, Any]:
    _ensure_vault_exists()
    try:
        with open(VAULT_FILE, "rb") as f:
            encrypted_data = f.read()
        if not encrypted_data:
            return {}

        fernet = Fernet(_get_or_create_key())
        decrypted_data = fernet.decrypt(encrypted_data)
        return json.loads(decrypted_data.decode("utf-8"))
    except Exception:
        return {}

def save_vault(data: Dict[str, Any]):
    VAULT_DIR.mkdir(parents=True, exist_ok=True)
    fernet = Fernet(_get_or_create_key())

    json_data = json.dumps(data, indent=4, ensure_ascii=False)
    encrypted_data = fernet.encrypt(json_data.encode("utf-8"))

    with open(VAULT_FILE, "wb") as f:
        f.write(encrypted_data)
    os.chmod(VAULT_FILE, 0o600)

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
