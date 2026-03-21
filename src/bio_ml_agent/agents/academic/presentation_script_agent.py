import logging
from typing import Dict, Any
from bio_ml_agent.core.agent_base import BaseSubAgent

logger = logging.getLogger("academic.presentation_script_agent")

class PresentationScriptAgent(BaseSubAgent):
    """
    F1: Presentation Script Agent
    Converts a drafted manuscript into a slide-by-slide presentation script.
    """
    def __init__(self):
        super().__init__(
            role_name="PresentationScriptWriter",
            system_prompt=(
                "You are an Academic Presentation Coach. Take a complex manuscript and "
                "convert it into a compelling slide-by-slide 15-minute conference presentation. "
                "Outline what goes on the slide vs what the speaker actually says. Output JSON."
            )
        )
        self.manuscript = ""

    def perceive(self, context: Dict[str, Any]):
        self.manuscript = context.get("manuscript_text", "")
        self.duration_mins = context.get("target_duration_mins", 15)

    def plan(self) -> str:
        return f"Draft a {self.duration_mins}-minute presentation script based on manuscript."

    def act(self, instructions: str) -> None:
        prompt = f"""
        Draft a slide-by-slide presentation script for this manuscript:
        {self.manuscript}
        
        Output JSON format:
        {{
            "presentation_title": "Engaging title",
            "slides": [
                {{
                    "slide_number": 1,
                    "slide_visual_content": "Bullet points and graphics",
                    "speaker_notes": "What the presenter should actually say"
                }}
            ],
            "estimated_time": "15 mins"
        }}
        """
        self.current_result = self.llm.chat([{"role": "user", "content": prompt}])

    def verify(self) -> bool:
        return "slides" in self.current_result

    def summarize(self) -> Any:
        try:
            import json
            data = json.loads(self.current_result.replace("```json", "").replace("```", "").strip())
        except Exception:
            data = {"raw": self.current_result}
        return self.create_checkpoint("Generate Presentation Script", data, "HIGH")
