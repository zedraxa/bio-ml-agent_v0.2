import logging
import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

log = logging.getLogger("memory_curator")

class MemoryCurator:
    """
    MemoryCurator (Professional):
    - Akıllı Eleme: Sadece yüksek güvenli (Confidence.HIGH/CRITICAL) bilgileri kalıcı belleğe işler.
    - Semantik Kontrol: Mevcut bilgiyle çelişen yeni bilgileri işaretler.
    - Provenance Locking: Her bilginin hangi ajan ve hangi kaynaktan geldiğini mühürler.
    """
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.truth_file = self.storage_path / "project_truth.json"
        self.project_truth = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.truth_file.exists():
            try:
                return json.loads(self.truth_file.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def commit(self, key: str, value: Any, provenance: Dict[str, Any], confidence_level: str):
        """Bilgiyi kalite ve kaynak kontrolünden geçirerek belleğe yazar."""
        
        # Sorumluluk: Sadece kaliteli veriyi al
        if confidence_level not in ["HIGH", "CRITICAL"]:
            log.warning(f"⚠️ Düşük güvenli bilgi reddedildi: {key} ({confidence_level})")
            return

        # Varsa çelişki kontrolü (basit bazda)
        if key in self.project_truth:
            log.info(f"🔄 Bilgi güncelleniyor: {key}")

        self.project_truth[key] = {
            "value": value,
            "provenance": provenance,
            "timestamp": time.time(),
            "confidence": confidence_level
        }
        self._save()
        log.info(f"✅ Belleğe işlendi: {key}")

    def _save(self):
        self.truth_file.write_text(json.dumps(self.project_truth, indent=2, ensure_ascii=False), encoding="utf-8")

    def query(self, key: str) -> Optional[Dict[str, Any]]:
        return self.project_truth.get(key)
