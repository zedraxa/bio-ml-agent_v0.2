import logging
from typing import List, Dict, Any, Optional
import os

log = logging.getLogger(__name__)

COMPRESSION_PROMPT = """
Aşağıdaki semantik hafıza kayıtlarını (anıları), kullanıcının mevcut sorusuyla en alakalı olacak şekilde 3 maddelik kısa bir briefing'e (özet) çevir.
Gereksiz detayları at, sadece eyleme geçirilebilir veya bağlam kurucu bilgileri tut. 
Eğer anılar soruyla alakasızsa "Önemli bir anı bulunamadı" yaz.

Kullanıcı Sorusu: {query}

Anılar:
{context}

Briefing (Maks 3 madde, Türkçe, Markdown formatında):
"""

class MemoryCompressor:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.environ.get("AGENT_MODEL", "gpt-4o")

    def compress(self, context_string: str, query: str) -> str:
        """
        Ham anı listesini LLM ile kısa bir briefing'e dönüştürür.
        """
        if not context_string or len(context_string.strip()) < 20:
            return ""

        try:
            from llm_backend import llm_chat
            
            prompt = COMPRESSION_PROMPT.format(query=query, context=context_string)
            messages = [{"role": "user", "content": prompt}]
            
            briefing = llm_chat(self.model_name, messages)
            
            if briefing and "Önemli bir anı bulunamadı" not in briefing:
                return f"🧠 **Bellek Özeti (Recall Briefing):**\n{briefing.strip()}\n"
            
            return "" # Alakasızsa boş döneriz ki prompt şişmesin
        except Exception as e:
            log.warning("MemoryCompressor Hatası: %s", e)
            # Hata durumunda ham veriyi dönmek yerine sessizce başarısız olabiliriz
            # veya çok kısa bir versiyonunu dönebiliriz. Şimdilik boş dönelim.
            return ""
