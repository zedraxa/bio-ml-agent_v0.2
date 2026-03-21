import logging
from typing import Dict, Any
from bio_ml_agent.core.agent_base import BaseSubAgent

logger = logging.getLogger("academic.poster_content_agent")

class PosterContentAgent(BaseSubAgent):
    """
    F2: Poster Content Agent
    Condenses the paper into high-impact bullet points for A0 academic posters.
    """
    def __init__(self):
        super().__init__(
            role_name="PosterContentGenerator",
            system_prompt=(
                "You are an Academic Poster Designer. Break down a complex manuscript "
                "into ultra-concise, high-impact bullet points suitable for an A0 poster. "
                "Less text, more focus on key figures and takeaways. Output JSON."
            )
        )
        self.manuscript = ""

    def perceive(self, context: Dict[str, Any]):
        self.manuscript = context.get("manuscript_text", "")

    def plan(self) -> str:
        return "Distill manuscript into concise poster format."

    def act(self, instructions: str) -> None:
        prompt = f"""
        Convert this manuscript into an A0 scientific poster layout:
        {self.manuscript}
        
        Produce a JSON layout:
        {{
            "poster_title": "Catchy short title",
            "background_bullets": ["Max 3 points"],
            "methods_bullets": ["Max 4 flowchart steps"],
            "results_highlights": ["Max 4 bullet points referring to figures"],
            "takeaway_message": "The single massive conclusion"
        }}
        """
        self.current_result = self.llm.chat([{"role": "user", "content": prompt}])

    def verify(self) -> bool:
        return "poster_title" in self.current_result

    def summarize(self) -> Any:
        try:
            import json
            data = json.loads(self.current_result.replace("```json", "").replace("```", "").strip())
        except Exception:
            data = {"raw": self.current_result}
        return self.create_checkpoint("Generate Poster Content", data, "HIGH")
