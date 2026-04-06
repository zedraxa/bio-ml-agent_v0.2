import logging
from typing import Dict, Any, List
import json

from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from bio_ml_agent.llm_backend import auto_create_backend

log = logging.getLogger("biology.histology")

class HistologyAgent(BaseSubAgent):
    """
    C2: Histology Agent
    Doku (tissue) seviyesindeki yapıları ve mikroskobik patolojileri yorumlayan
    uzman histolog ajanıdır.
    
    Yetenekleri:
    - Doku Tipi Ayrımı ve Histolojik Yapı Tanıma
    - Katman (Layer) ayrımı (örneğin epitel, stroma)
    - Boyama (H&E, IHC vb.) Kaynaklı Yorumlar
    - Anomali / Tümör benzeri alanların işaretlenmesi ve patolojik çıkarımlar
    """

    def __init__(self, model_name: str = "gemini-2.5-pro"):
        super().__init__("HistologyAgent", model_name)
        self.llm = auto_create_backend(model_name)
        self.wsi_features: Dict[str, Any] = {}
        self.histology_evaluation: Dict[str, Any] = {}

    def perceive(self, context: Dict[str, Any]) -> None:
        """Whole Slide Image (WSI) segmentasyon veya özellik çıkarım metriklerini alır."""
        self.wsi_features = context.get("wsi_features", {})
        if not self.wsi_features:
            log.warning("No histology/WSI features provided.")
        else:
            log.info(f"🔬 HistologyAgent received tissue metrics.")

    def plan(self, goal: str) -> List[str]:
        return [
            "Perform comprehensive histological evaluation (Tissue structure, Layers, Staining, Anomalies/Pathology)."
        ]

    def act(self, step: str) -> Any:
        prompt = f"""
        You are a Senior Pathologist / Histologist.
        Analyze the following computational metric extractions from a tissue slide.
        
        Input Features/Measurements:
        {json.dumps(self.wsi_features, indent=2)}
        
        Provide a deep histological interpretation.
        Respond STRICTLY with a JSON dictionary:
        {{
            "tissue_type_recognition": "interpretation...",
            "layer_separation_analysis": "interpretation...",
            "staining_based_insights": "interpretation...",
            "anomaly_and_pathology_localization": "interpretation...",
            "histological_conclusion": "overall pathological / structural summary..."
        }}
        Do NOT write navigational text. Only return the JSON (optionally wrapped in ```json markdown).
        """

        messages = [{"role": "user", "content": prompt}]

        try:
            response_text = self.llm.chat(messages)
            clean_text = response_text.replace("```json", "").replace("```", "").strip()
            self.histology_evaluation = json.loads(clean_text)
        except Exception as e:
            log.warning(f"Histology parsing fell back to strings. Error: {e}")
            self.histology_evaluation["unformatted_evaluation"] = response_text

        return f"Completed Histological Evaluation"

    def verify(self, action_result: Any) -> bool:
        keys = ["tissue_type_recognition", "histological_conclusion"]
        return any(k in self.histology_evaluation for k in keys)

    def summarize(self) -> AgentResult:
        is_valid = self.verify("")
        conf = Confidence.HIGH if is_valid else Confidence.LOW
        conclusion = self.histology_evaluation.get("histological_conclusion", "Unformatted pathology report.")

        return AgentResult(
            success=is_valid,
            data={"histology_evaluation": self.histology_evaluation},
            confidence=conf,
            evidence=[Evidence(source="histology_engine", content_snippet=f"{conclusion[:100]}...")],
            message=f"Histology analysis complete: {conclusion}"
        )
