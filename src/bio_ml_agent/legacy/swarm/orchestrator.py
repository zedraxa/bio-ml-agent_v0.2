import json
import logging
from typing import List, Dict, Any, Optional, Generator

logger = logging.getLogger(__name__)

from bio_ml_agent.swarm.base import SwarmContext
from bio_ml_agent.services.agent_registry import agent_registry
from bio_ml_agent.models.workspace_ux import AgentRole
from bio_ml_agent.models.lifecycle import AgentTier

class SwarmOrchestrator:
    """Ana yönetici ajan. Gelen isteği analiz edip doğru alt ajana yönlendirir."""

    def __init__(self, cfg):
        self.cfg = cfg
        # Hem AppConfig (utils.config), AgentConfig (core.config) hem de eski AgentConfig objelerini destekle
        if hasattr(cfg, "agent") and hasattr(cfg.agent, "model"):
            model_name = cfg.agent.model
            workspace_dir = str(cfg.workspace.base_dir) if hasattr(cfg.workspace, "base_dir") else str(cfg.workspace)
        elif hasattr(cfg, "model"):
            model_name = cfg.model
            workspace_dir = str(cfg.workspace)
        else:
            # Fallback for unexpected structures
            model_name = "qwen2.5:7b-instruct"
            workspace_dir = "workspace"

        self.context = SwarmContext(workspace_dir, model_name)

        # Registry üzerinden ajanları yükle (Sadece STABLE ve ACTIVE olanlar varsayılan akışa dahil)
        from bio_ml_agent.swarm.data_engineer import DataEngineerAgent
        from bio_ml_agent.swarm.ml_expert import MLExpertAgent
        from bio_ml_agent.swarm.bioinfo_expert import BioinfoExpertAgent
        from bio_ml_agent.swarm.researcher import ResearchAgent
        from bio_ml_agent.swarm.in_silico_expert import InSilicoExpertAgent
        from bio_ml_agent.swarm.academic_publishing_expert import AcademicPublishingExpertAgent

        # Registry'den STABLE/ACTIVE olanları eşle
        self.agents = {}

        # Haritalama Sözlüğü (Role -> Class)
        role_class_map = {
            AgentRole.DATA_ENGINEER: DataEngineerAgent,
            AgentRole.ML_EXPERT: MLExpertAgent,
            AgentRole.BIOINFORMATICIAN: BioinfoExpertAgent,
            AgentRole.RESEARCHER: ResearchAgent,
            AgentRole.IN_SILICO_EXPERT: InSilicoExpertAgent,
            AgentRole.ACADEMIC_EXPERT: AcademicPublishingExpertAgent
        }

        # Sadece Registry'de tanımlı ve Tier uygun olan ajanları ayağa kaldır
        for role, agent_class in role_class_map.items():
            # En az ACTIVE düzeyindeki ajanları sisteme dahil et
            registry_matches = agent_registry.find_by_role(role, min_tier=AgentTier.BETA)
            if registry_matches:
                self.agents[role.name] = agent_class(self.context)
                logger.info(f"[Swarm] Loaded agent class for role: {role.name} (Registry confirmed)")

    def _route_intent(self, user_msg: str) -> str:
        """Kullanıcının mesajına göre hangi alt ajanın devreye girmesi gerektiğini seçer."""
        msg_lower = user_msg.lower()

        # Ardışık boru hattı (Pipeline) gerektiren genel analiz komutları
        if any(kw in msg_lower for kw in ["kanser", "analiz", "uçtan uca", "pipeline", "hepsini"]):
            return "PIPELINE"

        elif any(kw in msg_lower for kw in ["docking", "alphafold", "screening", "omics", "plddt", "variant", "target", "yapı", "pocket"]):
            return AgentRole.IN_SILICO_EXPERT.name

        elif any(kw in msg_lower for kw in ["makale", "rapor", "lab raporu", "tez", "poster", "sunum", "hakem", "yazdır", "yaz", "draft", "paper", "review", "citation", "academic"]):
            return AgentRole.ACADEMIC_EXPERT.name

        elif any(kw in msg_lower for kw in ["pdb", "protein", "dna", "rna", "smiles", "lipinski", "genom", "dizi"]):
            return AgentRole.BIOINFORMATICIAN.name

        elif any(kw in msg_lower for kw in ["eğit", "model", "scikit", "kıyasla", "grafik", "roc", "accuracy", "knn", "svm"]):
            return AgentRole.ML_EXPERT.name

        elif any(kw in msg_lower for kw in ["veri", "yükle", "csv", "dataset", "temizle", "eksik", "korelasyon"]):
            return AgentRole.DATA_ENGINEER.name

        # Varsayılan olarak Pipeline çalıştıralım ki otonom çalışabilsin
        return "PIPELINE"

    def _run_pipeline(self, user_msg: str) -> str:
        """Data Engineer -> ML Expert -> Bioinformatician zincirini çalıştırır ve hata döngüsü içerir."""
        logger.info("[Swarm Orchestrator] Uçtan uca Pipeline başlatılıyor...")

        de_agent = self.agents.get(AgentRole.DATA_ENGINEER.name)
        ml_agent = self.agents.get(AgentRole.ML_EXPERT.name)
        bio_agent = self.agents.get(AgentRole.BIOINFORMATICIAN.name)

        # 1. Aşama: Veri Mühendisliği (Agentic Error Loop içerir)
        max_retries = 2
        de_task_prompt = f"Kullanıcı İsteği: {user_msg}\nLütfen bu isteğe uygun veriyi bul, indir ve temizleyerek '.csv' olarak kaydet."

        for attempt in range(max_retries):
            if not de_agent: break
            logger.info(f"[Pipeline] Data Engineer calistiriliyor (Deneme {attempt+1}/{max_retries})...")

            error_msg = self.context.shared_memory.get("pipeline_error", "")
            de_result = de_agent.execute(task_prompt=de_task_prompt, error_history=error_msg)

            # 2. Aşama: ML Uzmanı
            if not ml_agent: break
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
        if not bio_agent: return "Pipeline failed: Required agents not available."
        logger.info("[Pipeline] Bioinfo Expert (Raporlayıcı) calistiriliyor...")
        bio_task_prompt = f"Kullanıcı İsteği: {user_msg}\nVeri Mühendisliği Çıktısı: {de_result}\nML Analiz Çıktısı ve XAI (SHAP) Önemli Özellikleri: {ml_result}\nLütfen tüm bu bilgileri harmanlayarak kullanıcıya markdown formatında detaylı ve biyolojik/klinik açıdan yorumlanmış bir sonuç raporu sun. Özellikle ML çıktısındaki XAI (SHAP/LIME) özelliklerini '## Klinik Karar Özeti' başlığı altında biyolojik etkileriyle analiz et."

        final_report = bio_agent.execute(task_prompt=bio_task_prompt)

        return f"### 🐝 Bio-ML Swarm Topluluğu Raporu\n\n{final_report}"

    def process(self, messages: List[Dict[str, str]]) -> Generator[Dict[str, Any], None, None]:
        """LLM ile sohbet döngüsüne girmeden önce mesajı yakalayıp Swarm'a dağıtır (Generator versiyon)."""
        if not messages:
            yield {"type": "assistant", "content": "Boş mesaj."}
            return

        last_user_msg = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                last_user_msg = str(msg["content"])
                break

        # 1. Intent belirle
        target_agent_id = self._route_intent(last_user_msg)
        yield {"type": "status", "content": f"🎯 Swarm Modu: {target_agent_id} seçildi."}

        self.context.history = messages

        if target_agent_id == "PIPELINE":
            # Pipeline generator olarak çalışacak
            yield from self._run_pipeline_gen(last_user_msg)
            return

        target_agent = self.agents.get(target_agent_id)

        if target_agent:
            yield {"type": "status", "content": f"🕵️ {target_agent.name} görevlendirildi..."}
            result = target_agent.execute(task_prompt=last_user_msg)
            yield {"type": "assistant", "content": result}
        else:
            yield {"type": "assistant", "content": "Uygun bir alt ajan bulunamadı."}

    def _run_pipeline_gen(self, user_msg: str) -> Generator[Dict[str, Any], None, None]:
        """Uçtan uca Pipeline (Generator versiyon)."""
        logger.info("[Swarm Orchestrator] Generator Pipeline başlatılıyor...")

        res_agent = self.agents.get(AgentRole.RESEARCHER.name)
        de_agent = self.agents.get(AgentRole.DATA_ENGINEER.name)
        ml_expert = self.agents.get(AgentRole.ML_EXPERT.name)
        bio_expert = self.agents.get(AgentRole.BIOINFORMATICIAN.name)

        # Güvenlik Kontrolü
        if not all([res_agent, de_agent, ml_expert, bio_expert]):
            missing = [k for k, v in self.agents.items() if v is None]
            yield {"type": "error", "content": f"❌ Swarm ajanları başlatılamadı veya Registry izni yok: {missing}"}
            return

        # 0. Aşama: Literatür ve Klinik Araştırma
        yield {"type": "status", "content": f"🔍 {res_agent.name} literatür ve klinik veri tarıyor..."}
        res_result = res_agent.execute(task_prompt=f"Kullanıcı İsteği: {user_msg}\nLütfen bu konuyla ilgili en güncel klinik bulguları araştır.")

        # 1. Aşama: Veri Mühendisliği
        max_retries = 2
        de_task_prompt = f"Kullanıcı İsteği: {user_msg}\nLütfen bu isteğe uygun veriyi bul, indir ve temizleyerek '.csv' olarak kaydet."

        de_result = ""
        for attempt in range(max_retries):
            yield {"type": "status", "content": f"🧹 {de_agent.name} veri temizliyor (Deneme {attempt+1})..."}
            error_msg = self.context.shared_memory.get("pipeline_error", "")
            de_result = de_agent.execute(task_prompt=de_task_prompt, error_history=error_msg)

            # 2. Aşama: ML Uzmanı
            yield {"type": "status", "content": f"🤖 {ml_expert.name} model eğitiyor ve XAI analizi yapıyor..."}
            ml_task_prompt = f"Kullanıcı İsteği: {user_msg}\nData Engineer şu veriyi hazırladı: {de_result}\nLütfen bu veriyi kullanarak modeller eğit ve XAI grafiklerini oluştur."

            ml_result = ml_expert.execute(task_prompt=ml_task_prompt)

            ml_lower = ml_result.lower()
            if "hata" in ml_lower and any(kw in ml_lower for kw in ["veride", "eksik", "boş", "bulunamadı"]):
                yield {"type": "status", "content": "⚠️ ML Uzmanı veride hata buldu, Data Engineer'a geri dönülüyor..."}
                self.context.shared_memory["pipeline_error"] = f"ML Uzmanı veride şu hatayı buldu: {ml_result}"
                continue
            else:
                self.context.shared_memory.pop("pipeline_error", None)
                break

        # 3. Aşama: Biyoinformatik / Nihai Raporlama
        yield {"type": "status", "content": f"🧬 {bio_expert.name} tıbbi raporu harmanlıyor..."}
        bio_task_prompt = (
            f"Kullanıcı İsteği: {user_msg}\n"
            f"Araştırma Bulguları: {res_result}\n"
            f"Veri Mühendisliği Çıktısı: {de_result}\n"
            f"ML Analiz Çıktısı: {ml_result}\n"
            "Lütfen tüm bu bilgileri biyolojik/klinik açıdan yorumlayarak markdown formatında detaylı bir sonuç raporu sun. "
            "Araştırma bulgularını ve ML sonuçlarını birleştirerek 'Gelecek Çalışmalar ve Klinik Öneriler' bölümü ekle."
        )

        final_report = bio_expert.execute(task_prompt=bio_task_prompt)

        yield {"type": "assistant", "content": f"### 🐝 Bio-ML Swarm Topluluğu Raporu\n\n{final_report}"}
