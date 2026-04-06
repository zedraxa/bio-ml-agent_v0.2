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

    # Phase 0: Ensure Project is registered in SQLite DB (Truth Layer)
    try:
        from bio_ml_agent.db.session import SessionLocal
        from bio_ml_agent.db.models import ProjectDB
        import uuid
        import time

        with SessionLocal() as db:
            exist = db.query(ProjectDB).filter(ProjectDB.name == project_name).first()
            if not exist:
                pid = f"prj-{uuid.uuid4().hex[:6]}"
                new_proj = ProjectDB(
                    project_id=pid,
                    name=project_name,
                    description=user_msg[:200],
                    goals=["Mission initiated from prompt"],
                    state="initialized",
                    workspace_mode="research",
                    created_at=time.time(),
                    updated_at=time.time(),
                    last_accessed_at=time.time()
                )
                db.add(new_proj)
                db.commit()
                log.info("📊 Proje SQLite DB'ye kaydedildi (Truth): %s", pid)
    except Exception as e:
        log.warning("⚠️ Proje SQLite senkronizasyon hatası: %s", e)

    return {
        "project_name": project_name,
        "project_path": str(project_root),
        "session_id": session_id,
        "first_user_prompt": user_msg[:200],
    }
