import time
from typing import List, Dict, Any, Optional
from pathlib import Path

log = logging.getLogger("memory_curator")

class MemoryCurator:
    """
    Memory Curator: Hangi bilginin kalıcı belleğe (Project Truth) 
    yazılacağına karar veren ve belleği yöneten katman.
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

    def commit(self, key: str, value: Any, provenance: Dict[str, Any]):
        """Bilgiyi kanıtı ile birlikte belleğe işler."""
        self.project_truth[key] = {
            "value": value,
            "provenance": provenance,
            "timestamp": time.time()
        }
        self._save()

    def _save(self):
        self.truth_file.write_text(json.dumps(self.project_truth, indent=2, ensure_ascii=False), encoding="utf-8")

    def query(self, key: str) -> Optional[Dict[str, Any]]:
        return self.project_truth.get(key)
