import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class SwarmContext:
    """Paylaşılan Swarm Belleği ve Context'i"""
    def __init__(self, workspace_path: str, model: str):
        self.workspace = workspace_path
        self.model = model
        self.shared_memory: Dict[str, Any] = {}
        self.history: List[Dict[str, str]] = []

class SwarmOrchestrator:
    """Ana yönetici ajan. Gelen isteği analiz edip doğru alt ajana yönlendirir."""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.context = SwarmContext(str(cfg.workspace), cfg.model)
        
        # Alt Ajanları Başlat (Lazy load engellemek için doğrudan initialize edebiliriz)
        from .data_engineer import DataEngineerAgent
        from .ml_expert import MLExpertAgent
        from .bioinfo_expert import BioinfoExpertAgent
        
        self.agents = {
            "DATA_ENGINEER": DataEngineerAgent(self.context),
            "ML_EXPERT": MLExpertAgent(self.context),
            "BIOINFORMATICIAN": BioinfoExpertAgent(self.context)
        }
        
    def _route_intent(self, user_msg: str) -> str:
        """Kullanıcının mesajına göre hangi alt ajanın devreye girmesi gerektiğini seçer."""
        msg_lower = user_msg.lower()
        
        # Basit kural tabanlı intent routing (Geliştirilip LLM'ye de yaptırılabilir)
        if any(kw in msg_lower for kw in ["pdb", "protein", "dna", "rna", "smiles", "lipinski", "genom", "dizi"]):
            return "BIOINFORMATICIAN"
            
        elif any(kw in msg_lower for kw in ["eğit", "model", "scikit", "kıyasla", "grafik", "roc", "accuracy", "knn", "svm"]):
            return "ML_EXPERT"
            
        elif any(kw in msg_lower for kw in ["veri", "yükle", "csv", "dataset", "temizle", "eksik", "korelasyon"]):
            return "DATA_ENGINEER"
            
        # Varsayılan olarak Orchestrator'ın karar veremediği durumda Data Engine'den başlatıp pipeline kurabiliriz
        # veya ML Expert'e atabiliriz. Şimdilik NLP ile analiz için varsayılan bir 'Genel' akış tutuyoruz.
        return "DATA_ENGINEER" 
        
    def process(self, messages: List[Dict[str, str]]) -> str:
        """LLM ile sohbet döngüsüne girmeden önce mesajı yakalayıp Swarm'a dağıtır."""
        if not messages:
            return "Boş mesaj."
            
        last_user_msg = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                last_user_msg = str(msg["content"])
                break
                
        # 1. Intent belirle
        target_agent_id = self._route_intent(last_user_msg)
        target_agent = self.agents.get(target_agent_id)
        
        logger.info(f"[Swarm Orchestrator] Görev '{target_agent_id}' ajanına yönlendirildi.")
        
        # 2. Görevi ilgili ajana ilet
        if target_agent:
            # Context'i history ile güncelle
            self.context.history = messages
            
            # Sub-agent process'i çağır
            # Sub-agentlar kendi sistem promptlarıyla LLM'e gidecek
            return target_agent.execute()
        
        return "Uygun bir alt ajan bulunamadı."
