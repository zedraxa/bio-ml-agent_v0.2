import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

log = logging.getLogger("bio_ml_agent")

# ── Embedding Engine (Lazy Singleton) ──────────────────────────
_embedding_model = None
_EMBEDDING_DIM = 384  # MiniLM-L12-v2 boyutu


def _get_embedding_model():
    """Sentence-Transformers modelini tembel olarak yükler."""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    try:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        log.info("Embedding modeli yüklendi: paraphrase-multilingual-MiniLM-L12-v2")
        return _embedding_model
    except ImportError:
        log.warning("sentence-transformers yüklü değil. Dummy embedding kullanılacak.")
        return None
    except Exception as e:
        log.warning(f"Embedding modeli yüklenemedi: {e}")
        return None


def _encode_text(text: str) -> List[float]:
    """Metni vektöre dönüştürür. Model yoksa rastgele vektör üretir."""
    model = _get_embedding_model()
    if model is not None:
        vec = model.encode(text, normalize_embeddings=True)
        return vec.tolist()
    # Fallback: dummy
    import random
    return [random.uniform(-1, 1) for _ in range(_EMBEDDING_DIM)]


class QdrantMemoryStore:
    """
    S6-1, S6-2, S6-3: Qdrant Bazlı Semantik Hafıza Yönetimi.
    Agent bellek katmanı için Şemalar, Schema/TTL desteği ve
    Provenance (kaynak takip) ile birleştirilmiş Vektör Arama içerir.

    v1.1: Gerçek sentence-transformers embedding desteği eklendi.
    """
    def __init__(self, collection_name: str = "agent_semantic_memory"):
        self.collection_name = collection_name
        self.host = "localhost"
        self.port = 6333

        try:
            from qdrant_client import QdrantClient
            self.client = QdrantClient(host=self.host, port=self.port)
            self._ensure_collection()
            self.enabled = True
        except ImportError:
            log.warning("qdrant_client yüklü değil, gelişmiş RAG özellikleri devre dışı kalacak.")
            self.enabled = False
        except Exception as e:
            log.warning(f"Qdrant DB'ye bağlanılamadı: {e}. Vektör hafıza devre dışı.")
            self.enabled = False

    def _ensure_collection(self):
        from qdrant_client.models import Distance, VectorParams
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=_EMBEDDING_DIM, distance=Distance.COSINE),
            )
            log.info(f"Qdrant koleksiyonu oluşturuldu: {self.collection_name} (dim={_EMBEDDING_DIM})")

    def store_memory(self, text: str, source_file: str, ttl_days: int = 30,
                     project: str = "", tags: Optional[List[str]] = None) -> str:
        """
        S6-3 (Provenance): Kaynak adı ile vektör metnini indexler.
        v1.1: Gerçek embedding ve proje/tag metadata desteği.
        """
        if not self.enabled:
            return "Qdrant Disabled"

        vector = _encode_text(text)

        mem_id = str(uuid.uuid4())
        payload = {
            "text": text,
            "source": source_file,
            "ttl": ttl_days,
            "project": project,
            "tags": tags or [],
            "created_at": datetime.now().isoformat()
        }

        from qdrant_client.models import PointStruct
        self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=mem_id, vector=vector, payload=payload)]
        )
        return mem_id

    def search_memory(self, query: str, limit: int = 5, min_score: float = 0.5,
                      project_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        S6-4 (Hybrid Arama): Gerçek semantik arama + metadata filtre.
        v1.1: Proje bazlı filtreleme (Cross-Project Pollination) desteği.
        """
        if not self.enabled:
            return []

        query_vector = _encode_text(query)

        # Opsiyonel proje filtresi
        query_filter = None
        if project_filter:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            query_filter = Filter(
                must=[FieldCondition(key="project", match=MatchValue(value=project_filter))]
            )

        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=min_score,
            query_filter=query_filter,
        )

        results = []
        for hit in search_result:
            result = dict(hit.payload) if hit.payload else {}
            result["score"] = round(hit.score, 4)
            results.append(result)
        return results

    def cross_project_recall(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Tüm projelerden semantik hatırlama (Cross-Project Pollination)."""
        return self.search_memory(query, limit=limit, min_score=0.4)
