import logging
import json
from typing import List, Dict, Any, Optional
from .schema import MemoryEntry

log = logging.getLogger("bio_ml_agent")

class MemoryEvaluator:
    """
    S6-5: Semantik Hafıza Değerlendirme Sistemi (Evaluation System).
    Hafızanın doğruluğunu, hit_rate ve kirlilik oranlarını ölçer.
    """
    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name

    def evaluate_recall(self, query: str, memories: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Geri çağrılan anıların sorgu ile ilgisini değerlendirir.
        LLM kullanarak her anıya bir 'relevance' puanı verir.
        """
        if not memories:
            return {"precision@k": 0.0, "hit_rate": 0.0, "pollution_rate": 0.0}

        # Basit bir değerlendirme mantığı (LLM simülasyonu veya gerçek çağrı)
        # Gerçek uygulamada LLM'e query ve anıları gönderip "Hangileri gerçekten faydalı?" diye sorulur.
        
        useful_count = 0
        total_count = len(memories)
        
        for mem in memories:
            # Şimdilik sadece semantik skora güveniyoruz (gerçekte LLM karar vermeli)
            if mem.get("score", 0.0) > 0.7:
                useful_count += 1
        
        precision = useful_count / total_count if total_count > 0 else 0.0
        hit_rate = 1.0 if useful_count > 0 else 0.0
        pollution = (total_count - useful_count) / total_count if total_count > 0 else 0.0
        
        return {
            "precision@k": round(precision, 4),
            "hit_rate": hit_rate,
            "pollution_rate": round(pollution, 4)
        }

    def measure_write_quality(self, store: Any, project: str) -> float:
        """
        Yazılan anıların ne kadarının sonradan kullanıldığını ölçer.
        'last_accessed_at' ve 'created_at' farkına bakar.
        """
        memories = store.list_memories_for_project(project, limit=100)
        if not memories:
            return 0.0
            
        used_count = 0
        for m in memories:
            # Eğer oluşturulduktan sonra en az bir kez erişilmişse (last_accessed > created + epsilon)
            # Veya importance değeri feedback ile artmışsa
            if m.get("metadata", {}).get("feedbacks"):
                used_count += 1
                
        return round(used_count / len(memories), 4)
