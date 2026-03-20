# services/agent/memory_context.py
import logging
from typing import Optional
from bio_ml_agent.ultra_agent.memory import get_memory_store
from bio_ml_agent.ultra_agent.memory.compressor import MemoryCompressor

log = logging.getLogger("bio_ml_agent")

def get_compressed_context(user_msg: str, model_name: str, project_name: Optional[str] = None, session_id: Optional[str] = None) -> Optional[str]:
    """Hafıza store'undan bağlamı alır ve LLM için sıkıştırır."""
    try:
        mem_store = get_memory_store()
        raw_context = mem_store.get_context_string(
            user_msg, 
            limit=10, 
            project_filter=project_name, 
            session_id=session_id,
            memory_types=["decision", "artifact", "fact"]
        )
        
        if not raw_context:
            return None

        compressor = MemoryCompressor(model_name=model_name)
        mem_context = compressor.compress(raw_context, query=user_msg)
        
        if mem_context:
            log.info("🧠 Hafıza Briefing'i oluşturuldu (MemoryContext).")
            return mem_context
            
    except Exception as e:
        log.warning(f"Bağlam sıkıştırılırken hata oluştu: {e}")
    
    return None
