from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from .schema import MemoryEntry

class BaseMemoryStore(ABC):
    """
    Semantik Hafıza Katmanı için Soyut Base Sınıf (Sözleşme).
    """

    @abstractmethod
    def store_memory(self, entry: MemoryEntry) -> str:
        """Anıyı semantik veritabanına kaydeder ve ID döner."""
        pass

    @abstractmethod
    def upsert_memory(self, entry: MemoryEntry, min_similarity: float = 0.85) -> str:
        """
        Deduplication (mükerrerlik) kontrolü yaparak anıyı kaydeder veya günceller.
        Benzer bir anı varsa, mevcut anıyı günceller ve ID'sini döner.
        """
        pass

    @abstractmethod
    def search_memory(self, query: str, limit: int = 5, min_score: float = 0.5,
                      project_filter: Optional[str] = None,
                      session_filter: Optional[str] = None,
                      type_filter: Optional[List[str]] = None,
                      tag_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Benzer anıları semantik ve ağırlıklı olarak arar."""
        pass

    @abstractmethod
    def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """ID ile tek bir anı getirir."""
        pass

    @abstractmethod
    def delete_memory(self, memory_id: str) -> bool:
        """Tek bir anıyı siler."""
        pass

    @abstractmethod
    def expire_memories(self):
        """TTL süresi dolan anıları temizler."""
        pass

    @abstractmethod
    def list_memories_for_project(self, project: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Belirli bir projeye ait anıları listeler."""
        pass

    @abstractmethod
    def explain_recall(self, memory_id: str, query: str) -> str:
        """Bir anının neden geri çağrıldığını (provenance/score) açıklar."""
        pass

    @abstractmethod
    def summarize_memory_scope(self, project: str) -> str:
        """Bir projenin hafıza kapsamını özetler (kaç anı, hangi tagler vb)."""
        pass

    @abstractmethod
    def delete_by_project(self, project: str):
        """Belirli bir projeye ait tüm anıları siler."""
        pass

    @abstractmethod
    def maintenance(self):
        """
        S6-5: Hafıza bakımı yapar. 
        TTL temizliği, benzer anıları birleştirme ve arşivleme işlemlerini yürütür.
        """
        pass

    @abstractmethod
    def report_feedback(self, memory_id: str, helpful: bool):
        """Evaluation için bir anının ne kadar yardımcı olduğunu raporlar."""
        pass

    def get_context_string(self, query: str, limit: int = 5, min_score: float = 0.5,
                           project_filter: Optional[str] = None,
                           session_id: Optional[str] = None,
                           memory_types: Optional[List[str]] = None) -> str:
        """
        S6-4: Çok Katmanlı Geri Çağırma (Multi-layered Recall).
        Sırasıyla: Proje -> Oturum -> Cross-project araması yapar.
        """
        all_memories = []
        seen_ids = set()

        # 1. Katman: Proje Bazlı (Aynı proje içi)
        if project_filter:
            p_mems = self.search_memory(query, limit=limit, min_score=min_score,
                                        project_filter=project_filter,
                                        type_filter=memory_types)
            for m in p_mems:
                if m["id"] not in seen_ids:
                    all_memories.append(m)
                    seen_ids.add(m["id"])

        # 2. Katman: Oturum Bazlı (Yakın geçmiş)
        if session_id and len(all_memories) < limit:
            s_mems = self.search_memory(query, limit=limit, min_score=min_score,
                                        session_filter=session_id,
                                        type_filter=memory_types)
            for m in s_mems:
                if m["id"] not in seen_ids:
                    all_memories.append(m)
                    seen_ids.add(m["id"])

        # 3. Katman: Cross-Project (Tüm sistem, yüksek eşik ile)
        if len(all_memories) < limit:
            cp_mems = self.search_memory(query, limit=limit, min_score=min_score + 0.1,
                                        type_filter=memory_types)
            for m in cp_mems:
                if m["id"] not in seen_ids:
                    all_memories.append(m)
                    seen_ids.add(m["id"])

        if not all_memories:
            return ""

        # Limit uygula
        memories = all_memories[:limit]

        context = "Geçmiş konuşmalardan hatırladıkların (Semantik Bellek):\n"
        context += "-" * 50 + "\n"
        for i, mem in enumerate(memories):
            score = mem.get("score", 0.0)
            importance = mem.get("importance", 0.5)
            text = mem.get("content", mem.get("text", ""))
            m_type = mem.get("memory_type", "fact")
            created = mem.get("created_at", "unknown")

            context += f"Anı {i+1} [Tip: {m_type}, Önem: {importance}, Skor: {score:.2f}, Tarih: {created}]:\n{text}\n"
            context += "-" * 50 + "\n"

        return context
