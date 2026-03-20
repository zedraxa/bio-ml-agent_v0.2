import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

log = logging.getLogger("site_memory")

class SiteMemory:
    """
    Site Memory: Web sitelerine özgü davranışları ve başarılı selector'ları hatırlar.
    """
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.memory_file = self.storage_path / "site_profiles.json"
        self.profiles: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.memory_file.exists():
            try:
                return json.loads(self.memory_file.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def save(self):
        self.memory_file.write_text(json.dumps(self.profiles, indent=2, ensure_ascii=False), encoding="utf-8")

    def record_success(self, domain: str, action_type: str, target: str):
        """Başarılı bir aksiyonu kaydeder."""
        if domain not in self.profiles:
            self.profiles[domain] = {"success_actions": [], "best_selectors": {}}
        
        # Basit istatistik tutma
        self.profiles[domain]["success_actions"].append({
            "type": action_type,
            "target": target
        })
        self.save()

    def get_best_selector(self, domain: str, goal_element: str) -> Optional[str]:
        """Bir site için daha önce çalışan en iyi selectorü döner."""
        return self.profiles.get(domain, {}).get("best_selectors", {}).get(goal_element)
