import logging
from typing import List, Dict, Any, Optional

log = logging.getLogger("bio_ml_agent.rag.retrieval")

class HybridRetriever:
    """
    Qdrant (Vektörel) araması ve Qdrant Text Payload (Keyword) aramasını birleştirir,
    ardından Cross-Encoder kullanarak sonuçları (Reranking) sıralar.
    """
    def __init__(self,
                 vector_store,  # Qdrant bellek nesnesi (QdrantMemoryStore)
                 reranker_model: str = "cross-encoder/ms-marco-TinyBERT-L-2-v2",
                 top_k: int = 5):
        self.vector_store = vector_store
        self.top_k = top_k
        self.encoder = None
        self._reranker_model = reranker_model

    def _lazy_load_encoder(self):
        if self.encoder is None:
            log.info(f"Yükleniyor (Cross-Encoder): {self._reranker_model}")
            try:
                from sentence_transformers import CrossEncoder
                self.encoder = CrossEncoder(self._reranker_model, max_length=512)
            except ImportError:
                log.error("sentence_transformers paketi eksik. Reranker devre dışı.")
                self.encoder = "fallback"
            except Exception as e:
                log.error(f"Cross-Encoder yükleme hatası: {e}")
                self.encoder = "fallback" # Error state

    def search(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not query.strip():
            return []

        search_limit = self.top_k * 3

        # 1. Semantik Arama
        semantic_results = []
        if hasattr(self.vector_store, "search_memory"):
            semantic_results = self.vector_store.search_memory(
                query=query,
                limit=search_limit,
                project_filter=filters.get("project") if filters else None,
                session_filter=filters.get("session_id") if filters else None,
                type_filter=filters.get("memory_type") if filters else None,
                tag_filter=filters.get("tags") if filters else None
            )
        elif hasattr(self.vector_store, "search"):
            # Fallback legacy support
            semantic_results = self.vector_store.search(query, limit=search_limit, filters=filters)

        # 2. Keyword Arama (BM25 - Qdrant MatchText Payload Index aracılığıyla)
        keyword_results = []
        if hasattr(self.vector_store, "search_memory"):
            keyword_results = self.vector_store.search_memory(
                query=query, # Vector required by qdrant for distance fallback, but keyword is strict filter
                keyword_query=query, # Actual keyword MatchText payload
                limit=search_limit,
                project_filter=filters.get("project") if filters else None,
                session_filter=filters.get("session_id") if filters else None,
                type_filter=filters.get("memory_type") if filters else None,
                tag_filter=filters.get("tags") if filters else None
            )

        # 3. Sonuçları RRF (Reciprocal Rank Fusion) ile birleştir
        pool = {}
        rrf_k = 60

        for rank, res in enumerate(semantic_results):
            doc_id = res.get("id", res.get("content", ""))
            if doc_id not in pool:
                pool[doc_id] = {"doc": res, "rrf_score": 0.0}
            pool[doc_id]["rrf_score"] += 1.0 / (rrf_k + rank + 1)

        for rank, res in enumerate(keyword_results):
            doc_id = res.get("id", res.get("content", ""))
            if doc_id not in pool:
                pool[doc_id] = {"doc": res, "rrf_score": 0.0}
            pool[doc_id]["rrf_score"] += 1.0 / (rrf_k + rank + 1)

        fused_pool = sorted(pool.values(), key=lambda x: x["rrf_score"], reverse=True)
        top_candidates = [item["doc"] for item in fused_pool[:self.top_k * 2]]

        if not top_candidates:
            return []

        # 4. Re-Ranking (Cross-Encoder)
        self._lazy_load_encoder()

        if self.encoder and self.encoder != "fallback":
            texts = [doc.get("content") or doc.get("text", "") for doc in top_candidates]
            pairs = [[query, txt] for txt in texts]
            try:
                scores = self.encoder.predict(pairs)
                for doc, score in zip(top_candidates, scores):
                    doc["rerank_score"] = float(score)

                final_results = sorted(top_candidates, key=lambda x: x["rerank_score"], reverse=True)
                return final_results[:self.top_k]

            except Exception as e:
                log.error(f"Reranking hatası: {e}, RRF sıralaması ile devam ediliyor.")

        return top_candidates[:self.top_k]
