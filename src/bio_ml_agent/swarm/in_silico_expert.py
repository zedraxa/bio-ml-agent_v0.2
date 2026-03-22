import logging
import json
from bio_ml_agent.swarm.base import BaseAgent, SwarmContext

# Import the new Eksen F missions
from bio_ml_agent.missions.discovery.sequence_to_structure import SequenceToStructureBrief
from bio_ml_agent.missions.discovery.target_evaluation import ProteinTargetEvaluationMission
from bio_ml_agent.missions.discovery.structure_to_screening import StructureToScreeningPrep
from bio_ml_agent.missions.discovery.omics_target_prioritization import OmicsToTargetPrioritization
from bio_ml_agent.missions.discovery.variant_structural_hypothesis import VariantToStructuralHypothesis

logger = logging.getLogger(__name__)

class InSilicoExpertAgent(BaseAgent):
    """
    Bu ajan Part III'teki Struktur, Doking ve Omik (Eksen A-F) altyapısını Swarm'a bağlar.
    Kullanıcı isteğini alıp ilgili Discovery Mission'a (F1-F5) yönlendirir.
    """
    def __init__(self, context: SwarmContext):
        super().__init__(
            name="IN_SILICO_EXPERT",
            role="Sen Bio-ML Swarm Topluluğunun In-Silico Keşif ve Yapısal Biyoloji Uzmanısın. Görevin, gelen metne göre uygun yapısal biyoloji/omics/docking Mission'unu tetikleyip eldeki verilerle en iyi biyo-mühendislik/farmakoloji raporunu üretmek.",
            context=context
        )

    def execute(self, task_prompt: str, error_history: str = "") -> str:
        logger.info(f"[{self.name}] Görev alindi. Hangi Discovery Mission tetiklenecegi seciliyor...")
        
        # Basit bir Intent Analyzer (LLM ile de yapilabilir, heuristic de yapilabilir)
        # Guvenli olmasi acisindan LLM router kullanacagiz.
        analysis_prompt = f"""
        Aşağıdaki isteği analiz et ve hangi Research Mission'ın uygun olduğuna karar ver:
        1: Sequence to Structure (AlphaFold, pLDDT, Domain, Disorder)
        2: Protein Target Evaluation (Sequence + Structure + Pathway risk)
        3: Structure to Screening Prep (Pocket finding, Docking Prep)
        4: Omics to Target Prioritization (Expression data, RNA-seq, Target Feasibility)
        5: Variant to Structural Hypothesis (Mutasyon, Stabilite, Yüzey/Çekirdek Etkisi)

        Kullanıcı İsteği: {task_prompt}

        Sadece 1, 2, 3, 4, veya 5 rakamından birini ve json olarak dondur.
        Ornek: {{"mission": 3, "extracted_payload": "hedef protein veya parametreler"}}
        """
        
        try:
            decision_raw = self.llm.chat([{"role": "user", "content": analysis_prompt}])
            decision_text = decision_raw.replace("```json", "").replace("```", "").strip()
            decision = json.loads(decision_text)
            mission_id = decision.get("mission", 1)
            payload_data = decision.get("extracted_payload", task_prompt)
        except Exception as e:
            logger.warning(f"InSilicoExpert intent ayıklanırken hata: {e}. Defaulting to M1.")
            mission_id = 1
            payload_data = task_prompt

        # Mission Tetikleme Modulu
        result_payload = {}
        try:
            if mission_id == 1:
                mission = SequenceToStructureBrief(mission_id="AF_BRIEF_01")
                result_payload = mission.execute({"sequence": payload_data})
            elif mission_id == 2:
                mission = ProteinTargetEvaluationMission(mission_id="TARG_EVAL_01")
                result_payload = mission.execute({"target_name": payload_data})
            elif mission_id == 3:
                mission = StructureToScreeningPrep(mission_id="SCREEN_PREP_01")
                result_payload = mission.execute({"structure_mesh": payload_data})
            elif mission_id == 4:
                mission = OmicsToTargetPrioritization(mission_id="OMICS_TAR_01")
                result_payload = mission.execute({"rnaseq_matrix": payload_data})
            elif mission_id == 5:
                mission = VariantToStructuralHypothesis(mission_id="VAR_STRUCT_01")
                result_payload = mission.execute({"variant_list": [payload_data]})
            else:
                return "❌ Uygun In-Silico Mission bulunamadi."
        except Exception as e:
            return f"❌ Mission {mission_id} calistirilirken hata olustu: {str(e)}"

        # Misyondan donen karmasik dictionary yapisini Markdown raporuna cevir
        report_prompt = f"""
        Sen Swarm Biyoinformatik Uzmanısın. Alt otonom laboratuvarından (Mission {mission_id}) şu sonuçlar dondu:
        {json.dumps(result_payload, indent=2)}

        Kullanıcının asil istedigi: {task_prompt}

        Bu JSON verisini kullanarak mükemmel, raporlamaya hazir, anlasilir ve bilimsel bir Markdown (.md) 
        raporu olustur. Tablolar ve madde isaretleri kullan. Halusinasyon yapmadan SADECE JSON'daki bulgulari turkce / ingilizce aktar.
        """
        
        final_report = self.llm.chat([{"role": "user", "content": report_prompt}])
        
        return final_report
