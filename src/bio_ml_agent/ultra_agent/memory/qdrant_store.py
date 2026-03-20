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


from .base import BaseMemoryStore
from .schema import MemoryEntry

class QdrantMemoryStore(BaseMemoryStore):
    """
    S6-1, S6-2, S6-3: Qdrant Bazlı Semantik Hafıza Yönetimi.
    Agent bellek katmanı için Şemalar, Schema/TTL desteği ve
    Provenance (kaynak takip) ile birleştirilmiş Vektör Arama içerir.
    """
    def __init__(self, collection_name: str = "agent_semantic_memory", 
                 host: str = "localhost", port: int = 6333):
        self.collection_name = collection_name
        self.host = host
        self.port = port

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
        from qdrant_client.models import Distance, VectorParams, TextIndexParams, TokenizerType
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=_EMBEDDING_DIM, distance=Distance.COSINE),
            )
            log.info(f"Qdrant koleksiyonu oluşturuldu: {self.collection_name} (dim={_EMBEDDING_DIM})")
        
        # Tam metin arama (Full-Text Search) için index oluştur
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name, 
                field_name="content", 
                field_schema=TextIndexParams(
                    type="text",
                    tokenizer=TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=20,
                    lowercase=True,
                )
            )
        except Exception as e:
            # Index varsa zaten hata atabilir, kritik değil
            pass

    def store_memory(self, entry: MemoryEntry) -> str:
        """Sözleşmeye uygun, şema bazlı bellek kaydı."""
        if not self.enabled:
            return "Qdrant Disabled"

        vector = _encode_text(entry.content)
        mem_id = str(uuid.uuid4())
        
        # Pydantic modelini dict'e çevir ve Qdrant payload'ı hazırla
        payload = entry.dict()
        payload["created_at"] = payload["created_at"].isoformat()
        payload["last_accessed_at"] = payload["last_accessed_at"].isoformat()

        from qdrant_client.models import PointStruct
        self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=mem_id, vector=vector, payload=payload)]
        )
        return mem_id

    def upsert_memory(self, entry: MemoryEntry, min_similarity: float = 0.85) -> str:
        """
        Deduplication (mükerrerlik) kontrolü yaparak anıyı kaydeder veya günceller.
        """
        if not self.enabled:
            return "Qdrant Disabled"

        # Önce benzer bir anı var mı kontrol et
        similar = self.search_memory(entry.content, limit=1, min_score=min_similarity, 
                                     project_filter=entry.project)
        
        if similar:
            existing = similar[0]
            existing_id = existing["id"]
            log.info(f"🔄 Benzer anı bulundu ({existing['score']:.2f}), güncelleniyor: {existing_id}")
            
            # Mevcut anıyı güncelle
            # Not: Qdrant 'upsert' ile aynı ID kullanıldığında üzerine yazar.
            # Bazı alanları birleştirmek isteyebiliriz (örn: tags)
            new_tags = list(set(existing.get("tags", []) + entry.tags))
            entry.tags = new_tags
            
            # Önem skorunu en yükseğiyle güncelle
            entry.importance = max(existing.get("importance", 0.0), entry.importance)
            
            # last_accessed_at'i şimdiye ayarla (varsayılan zaten şimdi)
            
            # Vector'ü de yeni içeriğe göre güncelle
            vector = _encode_text(entry.content)
            payload = entry.dict()
            payload["created_at"] = payload["created_at"].isoformat() if hasattr(payload["created_at"], "isoformat") else payload["created_at"]
            payload["last_accessed_at"] = payload["last_accessed_at"].isoformat() if hasattr(payload["last_accessed_at"], "isoformat") else payload["last_accessed_at"]

            from qdrant_client.models import PointStruct
            self.client.upsert(
                collection_name=self.collection_name,
                points=[PointStruct(id=existing_id, vector=vector, payload=payload)]
            )
            return existing_id
        
        # Benzeri yoksa yeni olarak kaydet
        return self.store_memory(entry)

    def search_memory(self, query: str, limit: int = 5, min_score: float = 0.5,
                      keyword_query: Optional[str] = None,
                      project_filter: Optional[str] = None,
                      session_filter: Optional[str] = None,
                      type_filter: Optional[List[str]] = None,
                      tag_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Sözleşmeye uygun semantik arama ve ağırlıklı re-ranking."""
        if not self.enabled:
            return []

        query_vector = _encode_text(query)

        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, MatchText
        must_conditions = []
        
        if keyword_query:
            must_conditions.append(FieldCondition(key="content", match=MatchText(text=keyword_query)))
        
        if project_filter:
            must_conditions.append(FieldCondition(key="project", match=MatchValue(value=project_filter)))
        
        if session_filter:
            must_conditions.append(FieldCondition(key="session_id", match=MatchValue(value=session_filter)))
        
        if type_filter:
            must_conditions.append(FieldCondition(key="memory_type", match=MatchAny(any=type_filter)))

        if tag_filter:
            must_conditions.append(FieldCondition(key="tags", match=MatchAny(any=tag_filter)))

        query_filter = Filter(must=must_conditions) if must_conditions else None

        # Re-ranking için limit'in 2 katını çekiyoruz
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit * 2,
            score_threshold=min_score * 0.5, # Re-ranking öncesi esnek eşik
            query_filter=query_filter,
        )

        final_results = []
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)

        for hit in search_result:
            if not hit.payload:
                continue
            
            payload: Dict[str, Any] = dict(hit.payload)
            semantic_score: float = float(hit.score)
            
            # 1. Importance (payload'dan)
            importance = float(payload.get("importance", 0.5))
            
            # 2. Recency (Zaman Bazlı)
            created_at_str = payload.get("created_at")
            recency_score = 1.0
            if created_at_str:
                try:
                    created_at = datetime.fromisoformat(created_at_str)
                    if created_at.tzinfo is None:
                        created_at = created_at.replace(tzinfo=timezone.utc)
                    
                    diff = (now - created_at).total_seconds()
                    age_days = diff / 86400
                    # Logaritmik veya hiperbolik azalma
                    recency_score = 1.0 / (1.0 + age_days)
                except Exception:
                    pass
            
            # 3. Reliability (Kaynak Güvenilirliği)
            reliability = 1.0 if payload.get("source_kind") == "manual" else 0.8
            
            # 4. Formül: 0.55 similarity + 0.20 importance + 0.15 recency + 0.10 reliability
            weighted_score = (
                (0.55 * semantic_score) + 
                (0.20 * importance) + 
                (0.15 * recency_score) + 
                (0.10 * reliability)
            )

            # S6-5: Provenance (Gerekçe) oluştur
            prov_details = [
                f"Semantik: {semantic_score:.2f} (w=0.55)",
                f"Önem: {importance:.2f} (w=0.20)",
                f"Güncellik: {recency_score:.2f} (w=0.15)",
                f"Güvenilirlik: {reliability:.2f} (w=0.10)"
            ]
            provenance = " + ".join(prov_details) + f" = {weighted_score:.2f}"
            
            # Kaynak bazlı ek açıklama
            if project_filter and payload.get("project") == project_filter:
                provenance += " | [Aynı Proje]"
            if session_filter and payload.get("session_id") == session_filter:
                provenance += " | [Aynı Oturum]"

            if weighted_score >= min_score:
                payload["score"] = float(round(float(weighted_score), 4))
                payload["id"] = str(hit.id)
                payload["semantic_score"] = float(round(float(semantic_score), 4))
                payload["provenance"] = str(provenance)
                final_results.append(payload)

        # Skora göre yeniden sırala ve limit uygula
        final_results = sorted(final_results, key=lambda x: x.get("score", 0.0), reverse=True)[:limit]

        # S6-5: Erişim zamanını güncelle (Ideal Mimari)
        if final_results:
            self._update_access_times([res["id"] for res in final_results])

        return final_results

    def _update_access_times(self, memory_ids: List[str]):
        """Hafıza erişildiğinde zaman damgasını günceller."""
        if not self.enabled or not memory_ids:
            return
            
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            for m_id in memory_ids:
                self.client.set_payload(
                    collection_name=self.collection_name,
                    payload={"last_accessed_at": now_iso},
                    points=[m_id]
                )
        except Exception as e:
            log.warning(f"Hafıza erişim zamanı güncellenemedi: {e}")

    def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """ID ile anı getirir."""
        if not self.enabled:
            return None
        try:
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[memory_id]
            )
            if points:
                res = dict(points[0].payload)
                res["id"] = str(points[0].id)
                return res
        except Exception:
            pass
        return None

    def delete_memory(self, memory_id: str) -> bool:
        """ID ile anı siler."""
        if not self.enabled:
            return False
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[memory_id]
            )
            return True
        except Exception:
            return False

    def expire_memories(self):
        """TTL süresi dolan anıları arar ve siler."""
        if not self.enabled:
            return
        
        from qdrant_client.models import Filter, FieldCondition, Range
        from datetime import datetime, timezone, timedelta
        
        now = datetime.now(timezone.utc)
        log.info(f"Qdrant TTL temizliği başlatıldı (Zaman: {now.isoformat()})")
        
        # Basit yaklaşım: listele ve kontrol et (koleksiyon çok büyük değilse)
        # Daha verimli: Qdrant Range filtreleri.
        # Payload'da created_at string olarak tutuluyor.
        
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=1000,
            with_payload=True,
            with_vectors=False
        )
        
        deleted_count = 0
        for p in points:
            payload = p.payload
            if not payload: continue
            
            created_at_str = payload.get("created_at")
            ttl_days = payload.get("ttl_days", 30)
            
            if created_at_str:
                try:
                    created_at = datetime.fromisoformat(created_at_str)
                    if created_at.tzinfo is None:
                        created_at = created_at.replace(tzinfo=timezone.utc)
                    
                    expiry_date = created_at + timedelta(days=ttl_days)
                    if now > expiry_date:
                        self.delete_memory(str(p.id))
                        deleted_count += 1
                except Exception as e:
                    log.error(f"TTL kontrol hatası ({p.id}): {e}")
        
        if deleted_count > 0:
            log.info(f"🗑️ {deleted_count} adet süresi dolan anı silindi.")

    def maintenance(self):
        """Hafıza bakımı yapar."""
        if not self.enabled:
            return
        
        log.info("Hafıza bakım görevi (maintenance) başladı.")
        self.expire_memories()
        
        # Düşük öneme sahip ve uzun süredir erişilmeyenleri 'archived' olarak işaretle
        # (Şimdilik metadata üzerinden)
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        archive_threshold = now - timedelta(days=60)
        
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=500,
            with_payload=True
        )
        
        archived_count = 0
        for p in points:
            payload = p.payload
            if not payload: continue
            
            importance = payload.get("importance", 0.5)
            last_access_str = payload.get("last_accessed_at")
            
            if last_access_str:
                try:
                    last_access = datetime.fromisoformat(last_access_str)
                    if last_access.tzinfo is None:
                        last_access = last_access.replace(tzinfo=timezone.utc)
                    
                    # Önem < 0.3 ve 60 gündür erişilmemişse arşivle
                    if importance < 0.3 and last_access < archive_threshold:
                        if "archived" not in payload.get("tags", []):
                            tags = payload.get("tags", [])
                            tags.append("archived")
                            payload["tags"] = tags
                            # Vector dahil point struct oluşturup upsert
                            self.client.set_payload(
                                collection_name=self.collection_name,
                                payload={"tags": tags},
                                points=[p.id]
                            )
                            archived_count += 1
                except Exception:
                    pass
        
        if archived_count > 0:
            log.info(f"📦 {archived_count} adet anı arşivlendi.")

    def merge_similar_memories(self, project: str) -> int:
        """
        Faz 9: Belirtilen proje için `MemoryMerger` kullanarak birleştirme (Synthesis) yapar.
        """
        if not self.enabled:
            return 0
        from bio_ml_agent.ultra_agent.memory.maintenance import MemoryMerger
        merger = MemoryMerger(self)
        try:
            return merger.merge_project_memories(project)
        except Exception as e:
            log.error(f"Merge işlemi başarısız ({project}): {e}")
            return 0

    def report_feedback(self, memory_id: str, helpful: bool):
        """Geri bildirim kaydeder."""
        if not self.enabled:
            return
        
        # Metadata içinde bir feedback listesi veya counter tutabiliriz
        try:
            points = self.client.retrieve(self.collection_name, ids=[memory_id])
            if not points: return
            
            payload = dict(points[0].payload)
            metadata = payload.get("metadata", {})
            feedbacks = metadata.get("feedbacks", [])
            feedbacks.append({
                "time": datetime.now().isoformat(),
                "helpful": helpful
            })
            metadata["feedbacks"] = feedbacks
            
            # Importance'ı biraz artır/azalt
            if helpful:
                payload["importance"] = min(1.0, payload.get("importance", 0.5) + 0.05)
            else:
                payload["importance"] = max(0.0, payload.get("importance", 0.5) - 0.05)
            
            self.client.set_payload(
                collection_name=self.collection_name,
                payload={"metadata": metadata, "importance": payload["importance"]},
                points=[memory_id]
            )
            log.info(f"Feedback kaydedildi ({memory_id}): helper={helpful}")
        except Exception as e:
            log.error(f"Feedback hatası: {e}")

    def list_memories_for_project(self, project: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Proje filtreli liste."""
        if not self.enabled:
            return []
        
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        scroll_result = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="project", match=MatchValue(value=project))]
            ),
            limit=limit
        )
        
        points, _ = scroll_result
        return [dict(p.payload) | {"id": str(p.id)} for p in points]

    def explain_recall(self, memory_id: str, query: str) -> str:
        """Anı geri çağırmasını açıklar."""
        mem = self.get_memory_by_id(memory_id)
        if not mem:
            return "Anı bulunamadı."
        
        source = mem.get("source_kind", "bilinmiyor")
        created = mem.get("created_at", "bilinmiyor")
        provenance = mem.get("provenance", "Detaylı skor bilgisi yok.")
        
        return f"Bu anı '{source}' kaynağından ({created}) geliyor.\nNeden Hatırlandı: {provenance}"

    def summarize_memory_scope(self, project: str) -> str:
        """Kapsam özeti."""
        if not self.enabled:
            return "Hafıza devre dışı."
        
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        count_res = self.client.count(
            collection_name=self.collection_name,
            count_filter=Filter(
                must=[FieldCondition(key="project", match=MatchValue(value=project))]
            )
        )
        return f"Proje '{project}' için {count_res.count} adet kayıtlı anı bulunuyor."

    def delete_by_project(self, project: str):
        """Projeye ait tüm anıları siler."""
        if not self.enabled:
            return
        
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="project", match=MatchValue(value=project))]
            )
        )
        log.info(f"Proje anıları silindi: {project}")

    def cross_project_recall(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Tüm projelerden semantik hatırlama."""
        return self.search_memory(query, limit=limit, min_score=0.4)
