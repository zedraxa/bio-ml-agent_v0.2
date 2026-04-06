from typing import Optional
import logging
from .base import BaseMemoryStore
from .qdrant_store import QdrantMemoryStore

log = logging.getLogger("bio_ml_agent")

_memory_instance: Optional[BaseMemoryStore] = None

def get_memory_store() -> BaseMemoryStore:
    """
    Standardize edilmiş semantik hafıza erişim noktası.
    Konfigürasyona göre Qdrant veya legacy fallback döner.
    """
    global _memory_instance
    if _memory_instance is not None:
        return _memory_instance

    try:
        from bio_ml_agent.utils.config import get_config
        cfg = get_config()
        backend = cfg.memory.backend.lower()

        if backend == "qdrant":
            _memory_instance = QdrantMemoryStore(
                collection_name=cfg.memory.qdrant.collection,
                host=cfg.memory.qdrant.host,
                port=cfg.memory.qdrant.port
            )
            log.info(f"Semantik Hafıza Backend: Qdrant ({cfg.memory.qdrant.host}:{cfg.memory.qdrant.port})")
        else:
            log.warning(f"Bilinmeyen hafıza backend: {backend}. Qdrant varsayılıyor.")
            _memory_instance = QdrantMemoryStore()

    except Exception as e:
        log.warning(f"Hafıza başlatılamadı, Qdrant varsayılıyor: {e}")
        _memory_instance = QdrantMemoryStore()

    return _memory_instance
