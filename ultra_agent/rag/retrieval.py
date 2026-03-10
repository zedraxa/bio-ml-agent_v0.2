import logging
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

log = logging.getLogger("bio_ml_agent.rag.retrieval")

class HybridRetriever:
    """
    Qdrant (Vektörel) araması ve BM25 (Keyword) aramasını birleştirir,
    ardından Cross-Encoder kullanarak sonuçları (Reranking) sıralar.
    """
    def __init__(self, 
                 vector_store,  # Qdrant bellek nesnesi
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
                self.encoder = CrossEncoder(self._reranker_model, max_length=512)
            except Exception as e:
                log.error(f"Cross-Encoder yükleme hatası: {e}")
                self.encoder = "fallback" # Error state

    def search(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not query.strip():
            return []

        # 1. Semantik Arama (Qdrant'tan)
        # Örnekleme (Gerçek implementasyonda Qdrant API çağrısı yapılmalı)
        # semantic_results = self.vector_store.search(query, top_k=self.top_k*2, filters=filters)
        semantic_results = []
        if hasattr(self.vector_store, "search"):
             semantic_results = self.vector_store.search(query, limit=self.top_k*2)

        # 2. BM25 / Keyword (Hybrid Aşaması - Basitleştirilmiş Yer tutucu)
        # (Eğer veritabanı Qdrant v1.10+ Payload Index ile full-text aramayı destekliyorsa 
        # burası direkt Qdrant API üzerinden full_text index ile sorgulanır.
        # Lokal in-memory liste için Rank-BM25 kullanılabilir)
        # ...
        
        # Sonuçları Havuzda Topla
        pool = {res.get("text") or res.get("content"): res for res in semantic_results if (res.get("text") or res.get("content"))}
        pool_docs = list(pool.keys())
        
        if not pool_docs:
            return []

        # 3. Re-Ranking (Cross-Encoder)
        self._lazy_load_encoder()
        if self.encoder and self.encoder != "fallback":
            pairs = [[query, doc] for doc in pool_docs]
            try:
                scores = self.encoder.predict(pairs)
                # Dokümanları ve skorları eşleştir, skora göre azalan sırala
                scored_results = sorted(zip(pool_docs, scores), key=lambda x: x[1], reverse=True)
                
                final_results = []
                for doc, score in scored_results[:self.top_k]:
                    original_res = pool[doc]
                    original_res["rerank_score"] = float(score)
                    final_results.append(original_res)
                    
                return final_results
            except Exception as e:
                log.error(f"Reranking hatası: {e}, normal sıralama ile devam ediliyor.")

        # Fallback (Encoder yüklenemezse veya hata verirse mevcut skora göre kes)
        return list(pool.values())[:self.top_k]
