# core/conversation.py
# ═══════════════════════════════════════════════════════════
#  Bio-ML Agent — Konuşma Geçmişi Yönetimi
#  agent.py monolitinden ayrıştırılmıştır.
# ═══════════════════════════════════════════════════════════

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple

log = logging.getLogger("bio_ml_agent")


# ─────────────────────────────────────────────
#  Geçmiş Yönetimi
# ─────────────────────────────────────────────

def _ensure_history_dir(history_dir: Path) -> None:
    """Geçmiş klasörünü oluştur."""
    history_dir.mkdir(parents=True, exist_ok=True)


def generate_session_id() -> str:
    """Benzersiz oturum kimliği üret."""
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]


def save_conversation(history_dir: Path, session_id: str, messages: List[Dict[str, str]],
                      metadata: Optional[Dict] = None) -> Path:
    """Konuşma geçmişini JSON dosyasına kaydet."""
    _ensure_history_dir(history_dir)
    filepath = history_dir / f"{session_id}.json"

    # İlk kullanıcı mesajından özet çıkar
    first_user_msg = ""
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") in ("user", "user_message"):
            c = msg.get("content")
            if not c and "parts" in msg:
                c = msg["parts"]

            if isinstance(c, list):
                # Örn: langchain formatı [{"type": "text", "text": "hey"}] veya gemini ["hey", image]
                c = " ".join(
                    item.get("text", "") if isinstance(item, dict) and item.get("type") == "text"
                    else str(item)
                    for item in c
                )
            if not isinstance(c, str):
                c = str(c or "")

            first_user_msg = c[:120].replace("\n", " ")
            break

    data = {
        "session_id": session_id,
        "created_at": (metadata or {}).get("created_at", datetime.now().isoformat()),
        "updated_at": datetime.now().isoformat(),
        "summary": first_user_msg,
        "message_count": len(messages),
        "messages": messages,
    }
    if metadata:
        data["metadata"] = metadata

    filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    log.debug("💾 Oturum kaydedildi | session=%s | mesaj_sayısı=%d", session_id, len(messages))
    return filepath


def load_conversation(history_dir: Path, session_id: str) -> Tuple[List[Dict[str, str]], Dict]:
    """Konuşma geçmişini dosyadan yükle. (messages, metadata) döndürür."""
    filepath = history_dir / f"{session_id}.json"
    if not filepath.exists():
        raise FileNotFoundError(f"Oturum bulunamadı: {session_id}")

    data = json.loads(filepath.read_text(encoding="utf-8"))
    messages = data.get("messages", [])
    metadata = {
        "created_at": data.get("created_at", ""),
        "session_id": data.get("session_id", session_id),
    }
    saved_meta = data.get("metadata", {})
    if saved_meta:
        metadata.update(saved_meta)
    return messages, metadata


def list_conversations(history_dir: Path, limit: int = 20) -> List[Dict]:
    """Kayıtlı konuşma oturumlarını listele (en yeniden en eskiye)."""
    _ensure_history_dir(history_dir)
    sessions = []
    for f in sorted(history_dir.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            sessions.append({
                "session_id": data.get("session_id", f.stem),
                "created_at": data.get("created_at", "?"),
                "updated_at": data.get("updated_at", "?"),
                "summary": data.get("summary", "")[:80],
                "message_count": data.get("message_count", 0),
            })
        except (json.JSONDecodeError, KeyError):
            continue
        if len(sessions) >= limit:
            break
    return sessions


def delete_conversation(history_dir: Path, session_id: str) -> bool:
    """Bir konuşma oturumunu sil."""
    filepath = history_dir / f"{session_id}.json"
    if filepath.exists():
        filepath.unlink()
        return True
    return False


def print_history_help():
    """Geçmiş yönetimi komutlarının yardımını göster."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║               📜 Konuşma Geçmişi Komutları                  ║
╠══════════════════════════════════════════════════════════════╣
║  /history           → Kayıtlı oturumları listele            ║
║  /load <session_id> → Eski bir oturumu yükle                ║
║  /delete <session_id> → Bir oturumu sil                     ║
║  /new               → Yeni oturum başlat (mevcut kaydedilir)║
║  /save              → Mevcut oturumu şimdi kaydet           ║
║  /info              → Mevcut oturum bilgilerini göster       ║
╚══════════════════════════════════════════════════════════════╝
""")
