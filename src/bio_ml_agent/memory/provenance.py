import logging
from typing import Dict, Any

log = logging.getLogger("provenance_tracker")

class ProvenanceTracker:
    """
    Provenance Tracker: Her bir sonucun veya bellek kaydının 
    kaynağını (URL, dosya, adım, model) takip eder.
    """

    def __init__(self):
        pass

    @staticmethod
    def create_record(source_type: str, source_uri: str, step_id: int, model: str) -> Dict[str, Any]:
        """Yeni bir kanıt kaydı oluşturur."""
        return {
            "source_type": source_type, # 'web', 'file', 'tool'
            "source_uri": source_uri,
            "step_id": step_id,
            "origin_model": model,
            "version": "1.0"
        }
