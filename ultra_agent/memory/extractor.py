import json
import logging
from typing import List, Optional
from .schema import MemoryEntry
from llm_backend import auto_create_backend

log = logging.getLogger("bio_ml_agent")

EXTRACTION_PROMPT = """
Sana bir kullanıcı ve asistan diyaloğu vereceğim. Bu diyalogdan çıkarılabilecek, 'gelecekteki görevler için kritik öneme sahip' 0 ile 3 adet arasında kalıcı anı üretmeni istiyorum.

Kriterler:
1. **interaction**: Diyalogdan çıkan çok önemli bir bağlam (örn: "Kullanıcı veri kümesini temizlememi istedi").
2. **decision**: Teknik bir karar veya mimari seçim (örn: "Bundan sonra pandas yerine polars kullanılmasına karar verildi").
3. **artifact**: Önemli bir dosya veya model üretimi (örn: "diabetes_model_v1.joblib eğitildi ve kaydedildi").
4. **fact**: Görev sırasında keşfedilen somut ve değerli bilgi (örn: "Veri kümesinde %15 eksik değer olduğu keşfedildi").
5. **preference**: Kullanıcının çalışma şekli veya format tercihi (örn: "Kullanıcı her zaman grafiklerin karanlık modda olmasını istiyor").

Önemli: Eğer diyalog sıradan bir "merhaba", "tamam" veya "dosyayı oku" gibi düşük değerli bir adım ise '[]' (boş liste) döndür. Sadece 'YÜKSEK DEĞERLİ' bilgileri anılaştır.

Çıktı Formatı (Strict JSON List):
[
  {
    "content": "Anı metni",
    "memory_type": "decision|fact|preference|artifact|interaction",
    "importance": 0.0-1.0,
    "confidence": 0.0-1.0,
    "tags": ["tag1", "tag2"],
    "summary": "Tek cümlelik özet"
  }
]

DİYALOG:
User: {user_msg}
Assistant: {assistant_msg}
"""

class MemoryExtractor:
    """
    S6-3: Diyaloglardan akıllı anı çıkarımı yapan servis (Memory Extraction).
    """

    def __init__(self, model_name: str):
        self.backend = auto_create_backend(model_name)

    def extract_memories(self, user_msg: str, assistant_msg: str, 
                         project: str = "", session_id: str = "") -> List[MemoryEntry]:
        """
        LLM kullanarak diyalogdan 0-3 arası anı çıkarır.
        """
        prompt = EXTRACTION_PROMPT.format(user_msg=user_msg, assistant_msg=assistant_msg)
        
        try:
            # Token tasarrufu için user_msg ve assistant_msg çok uzunsa kırp
            if len(user_msg) > 2000: user_msg = user_msg[:2000] + "..."
            if len(assistant_msg) > 2000: assistant_msg = assistant_msg[:2000] + "..."
            
            response = self.backend.chat([{"role": "user", "content": prompt}])
            
            # JSON temizleme (Markdown fence'leri varsa kaldır)
            clean_res = response.strip()
            if "```json" in clean_res:
                clean_res = clean_res.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_res:
                clean_res = clean_res.split("```")[1].strip()
            
            raw_memories = json.loads(clean_res)
            if not isinstance(raw_memories, list):
                return []
                
            entries = []
            for m in raw_memories[:5]: # En fazla 5 anı
                entries.append(MemoryEntry(
                    content=m.get("content", ""),
                    memory_type=m.get("memory_type", "fact"),
                    importance=m.get("importance", 0.5),
                    confidence=m.get("confidence", 1.0),
                    summary=m.get("summary"),
                    tags=m.get("tags", []),
                    project=project,
                    session_id=session_id,
                    source_kind="chat"
                ))
            return entries
            
        except Exception as e:
            log.warning("Memory Extraction hatası: %s", e)
            return []
