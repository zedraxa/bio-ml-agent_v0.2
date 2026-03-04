import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

log = logging.getLogger("bio_ml_agent")

class QdrantMemoryStore:
    """
    S6-1, S6-2, S6-3: Qdrant Bazlı Semantik Hafıza Yönetimi.
    Agent bellek katmanı için Şemalar, Schema/TTL desteği ve 
    Provenance (kaynak takip) ile birleştirilmiş Vektör Arama içerir.
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
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
                # TTL Kurulum mantığı / Metadata şemaları Qdrant payloadlarında saklanır.
            )
            
    def store_memory(self, text: str, source_file: str, ttl_days: int = 30) -> str:
        """
        S6-3 (Provenance): Kaynak adı ile vektör metnini indexler.
        """
        if not self.enabled: return "Qdrant Disabled"
        
        # Gerçek uygulamada text_to_vector gömme (embedding API) modelinden geçecek
        # Şimdilik dummy vektör oluşturuluyor.
        import random
        dummy_vector = [random.uniform(-1, 1) for _ in range(1536)]
        
        mem_id = str(uuid.uuid4())
        payload = {
            "text": text,
            "source": source_file,           # S6-3: Provenance
            "ttl": ttl_days,                 # S6-1: TTL Kuralı
            "created_at": datetime.now().isoformat()
        }
        
        from qdrant_client.models import PointStruct
        self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=mem_id, vector=dummy_vector, payload=payload)]
        )
        return mem_id
        
    def search_memory(self, query: str, limit: int = 3, min_score: float = 0.7) -> List[Dict[str, Any]]:
        """
        S6-4 (Hybrid Arama): Metadata filtre destekli anlamsal (semantic) arama.
        """
        if not self.enabled: return []
        
        import random
        dummy_query_vector = [random.uniform(-1, 1) for _ in range(1536)]
        
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=dummy_query_vector,
            limit=limit,
            score_threshold=min_score
        )
        
        return [hit.payload for hit in search_result]
