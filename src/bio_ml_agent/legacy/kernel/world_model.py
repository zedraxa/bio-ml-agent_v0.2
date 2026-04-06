import logging
from typing import Dict, Any, List, Optional, cast
from threading import Lock
import time

log = logging.getLogger("kernel.world_model")

class WorldModel:
    """
    WorldModel: Agent OS içindeki tüm ajanların ortak "Dünya Bilgisi"ni tutar.
    Thread-safe bir yapıdadır ve tüm durum değişikliklerini (URL, Files, session) takip eder.
    """

    def __init__(self):
        self._state: Dict[str, Any] = {
            "session": {
                "start_time": time.time(),
                "active_agents": [],
                "current_goal": None
            },
            "browser": {
                "active_url": None,
                "tabs": [],
                "captured_coordinates": {}
            },
            "files": {
                "loaded_pdfs": [],
                "active_dataset": None
            },
            "knowledge": {
                "entities": {},
                "findings": []
            }
        }
        self._lock = Lock()

    def update(self, path: str, value: Any):
        """State içinde belirli bir yolu (Path) günceller. Örn: 'browser.active_url'"""
        with self._lock:
            keys = path.split('.')
            curr: Any = self._state

            for i in range(len(keys) - 1):
                key = keys[i]
                if key not in curr or not isinstance(curr[key], dict):
                    curr[key] = {}
                curr = curr[key]

            last_key = keys[len(keys) - 1]
            if isinstance(curr, dict):
                curr[last_key] = value
                log.info(f"🌐 WorldModel Update: {path} -> {str(value)[:50]}...")
            else:
                log.error(f"❌ WorldModel Update Failed: Path {path} is not in a dict.")

    def get(self, path: str, default: Any = None) -> Any:
        """State içinden veri çeker."""
        with self._lock:
            keys = path.split('.')
            curr: Any = self._state
            try:
                for key in keys:
                    curr = curr[key]
                return curr
            except (KeyError, TypeError):
                return default

    def snapshot(self) -> Dict[str, Any]:
        """Tüm state'in bir kopyasını döner (Gözlem için)."""
        with self._lock:
            import copy
            return copy.deepcopy(self._state)
