import logging
from typing import Dict, Any
from bio_ml_agent.core.agent_base import BaseSubAgent

logger = logging.getLogger("academic.figure_caption_agent")

class FigureCaptionAgent(BaseSubAgent):
    """
    D1: Figure Caption Agent
    Generates publication-style captions for various types of scientific figures.
    """
    def __init__(self):
        super().__init__(
            role_name="FigureCaptionGenerator",
            system_prompt=(
                "You are an expert Scientific Editor. Your task is to write a detailed, "
                "publication-ready figure caption based on the provided figure context. "
                "Start with a short title sentence, then describe the axes/colors, "
                "and conclude with the key observation (without overly interpreting). Output JSON."
            )
        )
        self.figure_context = ""

    def perceive(self, context: Dict[str, Any]):
        self.figure_context = context.get("figure_description", "")
        self.figure_type = context.get("figure_type", "Plot")

    def plan(self) -> str:
        return f"Generate professional caption for {self.figure_type}."

    def act(self, instructions: str) -> None:
        prompt = f"""
        Draft a publication-style caption for this {self.figure_type}.
        Data/Visual details: {self.figure_context}
        
        Output JSON:
        {{
            "short_caption": "Brief 1-sentence description",
            "detailed_caption_md": "Full paragraph suitable for a journal article"
        }}
        """
        self.current_result = self.llm.chat([{"role": "user", "content": prompt}])

    def verify(self) -> bool:
        return "detailed_caption_md" in self.current_result

    def summarize(self) -> Any:
        try:
            import json
            data = json.loads(self.current_result.replace("```json", "").replace("```", "").strip())
        except Exception:
            data = {"raw": self.current_result}
        return self.create_checkpoint("Write Figure Caption", data, "HIGH")
