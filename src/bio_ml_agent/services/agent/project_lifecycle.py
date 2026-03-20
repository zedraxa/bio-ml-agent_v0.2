# services/agent/project_lifecycle.py
import os
import re as _re
import json
import unicodedata
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

log = logging.getLogger("bio_ml_agent")

def slugify_project_text(text: str, max_words: int = 6, max_len: int = 48) -> str:
    """Kullanıcı mesajından proje klasörü adı için ASCII slug üretir."""
    text = text or "untitled-project"
    text = _re.sub(r"ALLOW_WEB_SEARCH", "", text, flags=_re.IGNORECASE).strip()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    words = _re.findall(r"[a-z0-9]+", text)
    if not words:
        return "untitled-project"
    slug = "-".join(words[:max_words]).strip("-")
    return slug[:max_len] or "untitled-project"

def ensure_project_context(
    user_msg: str, 
    session_id: str, 
    workspace: Path, 
    current_project_name: Optional[str] = None
) -> Dict:
    """İlk kullanıcı mesajında otomatik proje klasörü oluşturur."""
    if current_project_name:
        return {} # Zaten oluşmuş

    date_prefix = datetime.now().strftime("%Y-%m-%d")
    short_sid = session_id.split("_")[-1][:8]
    slug = slugify_project_text(user_msg)
    project_name = f"{date_prefix}_{slug}_{short_sid}"
    project_root = workspace / project_name
    project_root.mkdir(parents=True, exist_ok=True)

    os.environ["AGENT_PROJECT"] = project_name
    log.info("📁 Proje oluşturuldu: %s", project_name)

    # project.json index dosyası
    project_meta = {
        "project_name": project_name,
        "session_id": session_id,
        "created_at": datetime.now().isoformat(),
        "first_prompt": user_msg[:300],
        "status": "active",
    }
    try:
        (project_root / "project.json").write_text(
            json.dumps(project_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass

    return {
        "project_name": project_name,
        "project_path": str(project_root),
        "session_id": session_id,
        "first_user_prompt": user_msg[:200],
    }
