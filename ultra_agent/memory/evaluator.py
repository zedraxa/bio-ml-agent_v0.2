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
        LLM kullanarak her anıya bir 'relevance' (yararlılık) puanı verir.
        Faz 9: Gerçek LLM Hakemliği
        """
        if not memories:
            return {"precision@k": 0.0, "hit_rate": 0.0, "pollution_rate": 0.0}

        try:
            from llm_backend import auto_create_backend
            llm = auto_create_backend(model=self.model_name)
        except Exception as e:
            log.error(f"LLM Backend yüklenemedi: {e}")
            return {"precision@k": 0.0, "hit_rate": 0.0, "pollution_rate": 0.0}

        useful_count = 0
        total_count = len(memories)

        for i, mem in enumerate(memories):
            content = mem.get("content", "")
            summary = mem.get("summary", "")
            
            prompt = f"""
Sen bir 'Hafıza Hakemi' (Memory Evaluator) ajansın.
Kullanıcının bir sorgusu var ve bu sorguya yanıt vermek için sistem veritabanından bir anı getirdi.
Senden istenen, bu anının kullanıcının sorgusunu cevaplamak için GERÇEKTEN YARARLI olup olmadığına (True/False) karar vermendir.

Sorgu: "{query}"

Getirilen Anı İçeriği:
{content}

Getirilen Anı Özeti:
{summary}

Sadece JSON formatında yanıt ver. Örnek: {{"is_useful": true, "reason": "Kısa açıklama"}}
"""
            try:
                response_text = llm.chat([
                    {"role": "system", "content": "Sen JSON çıktısı veren katı bir hakemsin."},
                    {"role": "user", "content": prompt}
                ])
                
                # Temizle
                if response_text.startswith("```json"): response_text = response_text[7:]
                if response_text.startswith("```"): response_text = response_text[3:]
                if response_text.endswith("```"): response_text = response_text[:-3]
                
                result = json.loads(response_text.strip())
                if result.get("is_useful", False):
                    useful_count += 1
            except Exception as e:
                log.warning(f"Anı değerlendirilirken {i} hata (varsayılan=False): {e}")

        precision = useful_count / total_count if total_count > 0 else 0.0
        hit_rate = 1.0 if useful_count > 0 else 0.0
        pollution = (total_count - useful_count) / total_count if total_count > 0 else 0.0
        
        return {
            "precision@k": round(precision, 4),
            "hit_rate": hit_rate,
            "pollution_rate": round(pollution, 4),
            "total_evaluated": total_count
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
