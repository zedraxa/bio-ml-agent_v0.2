import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

from .base import SwarmContext

class SwarmOrchestrator:
    """Ana yönetici ajan. Gelen isteği analiz edip doğru alt ajana yönlendirir."""
    
    def __init__(self, cfg):
        self.cfg = cfg
        # Hem AppConfig (utils.config) hem de AgentConfig (agent.py) objelerini destekle
        if hasattr(cfg, "agent") and hasattr(cfg.agent, "model"):
            model_name = cfg.agent.model
            workspace_dir = str(cfg.workspace.base_dir)
        else:
            model_name = cfg.model
            workspace_dir = str(cfg.workspace)
            
        self.context = SwarmContext(workspace_dir, model_name)
        
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
        
        # Ardışık boru hattı (Pipeline) gerektiren genel analiz komutları
        if any(kw in msg_lower for kw in ["kanser", "analiz", "uçtan uca", "pipeline", "hepsini"]):
            return "PIPELINE"
            
        if any(kw in msg_lower for kw in ["pdb", "protein", "dna", "rna", "smiles", "lipinski", "genom", "dizi"]):
            return "BIOINFORMATICIAN"
            
        elif any(kw in msg_lower for kw in ["eğit", "model", "scikit", "kıyasla", "grafik", "roc", "accuracy", "knn", "svm"]):
            return "ML_EXPERT"
            
        elif any(kw in msg_lower for kw in ["veri", "yükle", "csv", "dataset", "temizle", "eksik", "korelasyon"]):
            return "DATA_ENGINEER"
            
        # Varsayılan olarak Pipeline çalıştıralım ki otonom çalışabilsin
        return "PIPELINE" 
        
    def _run_pipeline(self, user_msg: str) -> str:
        """Data Engineer -> ML Expert -> Bioinformatician zincirini çalıştırır ve hata döngüsü içerir."""
        logger.info("[Swarm Orchestrator] Uçtan uca Pipeline başlatılıyor...")
        
        de_agent = self.agents.get("DATA_ENGINEER")
        ml_agent = self.agents.get("ML_EXPERT")
        bio_agent = self.agents.get("BIOINFORMATICIAN")
        
        # 1. Aşama: Veri Mühendisliği (Agentic Error Loop içerir)
        max_retries = 2
        de_task_prompt = f"Kullanıcı İsteği: {user_msg}\nLütfen bu isteğe uygun veriyi bul, indir ve temizleyerek '.csv' olarak kaydet."
        
        for attempt in range(max_retries):
            logger.info(f"[Pipeline] Data Engineer calistiriliyor (Deneme {attempt+1}/{max_retries})...")
            
            error_msg = self.context.shared_memory.get("pipeline_error", "")
            de_result = de_agent.execute(task_prompt=de_task_prompt, error_history=error_msg)
            
            # 2. Aşama: ML Uzmanı
            logger.info("[Pipeline] ML Expert calistiriliyor...")
            ml_task_prompt = f"Kullanıcı İsteği: {user_msg}\nData Engineer şu veriyi hazırladı: {de_result}\nLütfen bu veriyi kullanarak makine öğrenmesi modelleri eğit, değerlendir ve XAI grafiklerini oluştur."
            
            ml_result = ml_agent.execute(task_prompt=ml_task_prompt)
            
            # Hata kontrolü (Basit bir heuristic: Eğer ML Expert veride sorun var derse veya empty dataset hatası alırsa)
            ml_lower = ml_result.lower()
            if "hata" in ml_lower and ("veride" in ml_lower or "eksik" in ml_lower or "bulunamadı" in ml_lower or "boş" in ml_lower):
                logger.warning(f"[Pipeline] ML Expert veride sorun tespit etti! Hata geri beslemesi yapılıyor. Hata: {ml_result}")
                self.context.shared_memory["pipeline_error"] = f"ML Uzmanı veride şu hatayı buldu: {ml_result}"
                continue # Tekrar Data Engineer'a dön (Error Loop)
            else:
                self.context.shared_memory.pop("pipeline_error", None) # Hata yoksa temizle
                break # Döngüden çık
                
        # 3. Aşama: Biyoinformatik / Nihai Raporlama
        logger.info("[Pipeline] Bioinfo Expert (Raporlayıcı) calistiriliyor...")
        bio_task_prompt = f"Kullanıcı İsteği: {user_msg}\nVeri Mühendisliği Çıktısı: {de_result}\nML Analiz Çıktısı ve XAI (SHAP) Önemli Özellikleri: {ml_result}\nLütfen tüm bu bilgileri harmanlayarak kullanıcıya markdown formatında detaylı ve biyolojik/klinik açıdan yorumlanmış bir sonuç raporu sun. Özellikle ML çıktısındaki XAI (SHAP/LIME) özelliklerini '## Klinik Karar Özeti' başlığı altında biyolojik etkileriyle analiz et."
        
        final_report = bio_agent.execute(task_prompt=bio_task_prompt)
        
        return f"### 🐝 Bio-ML Swarm Topluluğu Raporu\n\n{final_report}"
        
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
        
        self.context.history = messages
        
        if target_agent_id == "PIPELINE":
            return self._run_pipeline(last_user_msg)
            
        target_agent = self.agents.get(target_agent_id)
        
        logger.info(f"[Swarm Orchestrator] Görev '{target_agent_id}' ajanına yönlendirildi.")
        
        # 2. Görevi ilgili ajana ilet
        if target_agent:
            # Sub-agent process'i çağır
            return target_agent.execute(task_prompt=last_user_msg)
        
        return "Uygun bir alt ajan bulunamadı."
