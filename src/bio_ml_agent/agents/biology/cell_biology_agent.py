import logging
from typing import Dict, Any, List
import json

from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from bio_ml_agent.llm_backend import auto_create_backend

log = logging.getLogger("biology.cell")

class CellBiologyAgent(BaseSubAgent):
    """
    C1: Cell Biology Agent
    Orkestrasyonun "Biyolog" katmanı. Yazılım / Mimar ajanlarından ziyade,
    Microscopy veya Omics verilerinden elde edilen sayısal özelliklerin (features)
    hücre biyolojisi açısından 'ne anlama geldiğini' yorumlar.
    
    Yetenekleri:
    - Organel Yorumlama (Fluorescence/Localization)
    - Hücre Döngüsü (Cell cycle phases)
    - Mitoz/Mayoz Evre Tespiti
    - Sinyal, Hasar ve Ölüm Morfolojisi (Apoptosis vs Necrosis)
    - Hücre Kültürü Gözlemleri (Proliferasyon / Confluency)
    """

    def __init__(self, model_name: str = "gemini-2.5-pro"):
        # We use a highly capable model for deep biological reasoning
        super().__init__("CellBiologyAgent", model_name)
        self.llm = auto_create_backend(model_name)
        self.microscopy_features: Dict[str, Any] = {}
        self.biological_evaluation: Dict[str, Any] = {}

    def perceive(self, context: Dict[str, Any]) -> None:
        """
        Girdi olarak MicroscopyAgent'tan veya dış metriklerden alınan:
        - intensity_stats
        - morphological_measurements (area, circularity vs)
        - texture_features
        - time_lapse data
        gibi verileri algılar.
        """
        self.microscopy_features = context.get("features", {})
        if not self.microscopy_features:
            log.warning("No physical microscopy/cell features provided to CellBiologyAgent.")
        else:
            log.info(f"🧬 CellBiolgist received measurements array: {len(self.microscopy_features)} features.")

    def plan(self, goal: str) -> List[str]:
        return [
            "Perform comprehensive cell biology evaluation (Organelles, Cycle, Morphology, Health)."
        ]

    def act(self, step: str) -> Any:
        # The agent acts as an interpretative biological consultant

        prompt = f"""
        You are a Senior Cell Biologist analyzing raw computational measurements from a microscopy mission.
        Your goal is to perform '{step}'.
        
        Input Features/Measurements:
        {json.dumps(self.microscopy_features, indent=2)}
        
        Evaluate these metrics through the lens of Cell Biology.
        Respond STRICTLY with a JSON dictionary containing your expert assessment:
        {{
            "organelle_status": "detailed interpretation...",
            "cell_cycle_phase_estimate": "interpretation...",
            "mitosis_meiosis_tracking": "interpretation...",
            "morphology_and_damage": "apoptosis/necrosis/healthy interpretation...",
            "culture_confluency_health": "interpretation...",
            "biological_conclusion": "overall summary..."
        }}
        Do NOT write navigational text. Only return the JSON (optionally wrapped in ```json markdown).
        """

        messages = [{"role": "user", "content": prompt}]

        try:
            response_text = self.llm.chat(messages)
            clean_text = response_text.replace("```json", "").replace("```", "").strip()
            self.biological_evaluation = json.loads(clean_text)
        except Exception as e:
            log.warning(f"CellBio parsing fell back to strings. Error: {e}")
            self.biological_evaluation[step.replace(' ', '_')] = response_text

        return f"Completed Biological Evaluation phase: {step}"

    def verify(self, action_result: Any) -> bool:
        """JSON'ın biyolojik alanları kapsayıp kapsamadığına bakar."""
        keys = ["organelle_status", "biological_conclusion"]
        valid = any(k in self.biological_evaluation for k in keys)
        if not valid:
            log.warning("Biological Evaluation may have missed key interpretive keys.")
        return valid

    def summarize(self) -> AgentResult:
        is_valid = self.verify("")
        conf = Confidence.HIGH if is_valid else Confidence.LOW

        conclusion = self.biological_evaluation.get("biological_conclusion", "Analysis complete but unformatted.")

        return AgentResult(
            success=is_valid,
            data={"biological_evaluation": self.biological_evaluation},
            confidence=conf,
            evidence=[Evidence(source="biology_engine", content_snippet=f"{conclusion[:100]}...")],
            message=f"Cell Biology analysis complete: {conclusion}"
        )
