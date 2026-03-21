import logging
from typing import Dict, Any
from bio_ml_agent.core.agent_base import BaseSubAgent

logger = logging.getLogger("academic.microscopy_narration")

class MicroscopyResultNarrationAgent(BaseSubAgent):
    """
    D4: Microscopy Result Narration Agent
    Transforms object detection / segmentation outputs into biological microscopy texts.
    Linked closely with Part II.
    """
    def __init__(self):
        super().__init__(
            role_name="MicroscopyNarrator",
            system_prompt=(
                "You are an expert Histopathologist/Cell Biologist. "
                "Translate machine learning bounding box/segmentation JSONs "
                "into a formal microscopic observation narrative. "
                "E.g., turn '50 cells, area 20px' into 'Microscopic evaluation revealed a dense accumulation of cells...' Output JSON."
            )
        )
        self.cv_output = ""

    def perceive(self, context: Dict[str, Any]):
        self.cv_output = context.get("computer_vision_metrics", "")

    def plan(self) -> str:
        return "Translate bounding boxes and mask morphologies into biological narrative."

    def act(self, instructions: str) -> None:
        prompt = f"""
        Computer Vision / Segmentation Output:
        {self.cv_output}
        
        Generate JSON:
        {{
            "microscopy_narrative_md": "Formal paragraph detailing the visual field findings.",
            "identified_structures": ["List of distinct cellular/tissue features mentioned"]
        }}
        """
        self.current_result = self.llm.chat([{"role": "user", "content": prompt}])

    def verify(self) -> bool:
        return "microscopy_narrative_md" in self.current_result

    def summarize(self) -> Any:
        try:
            import json
            data = json.loads(self.current_result.replace("```json", "").replace("```", "").strip())
        except Exception:
            data = {"raw": self.current_result}
        return self.create_checkpoint("Narrate Microscopy Findings", data, "HIGH")
