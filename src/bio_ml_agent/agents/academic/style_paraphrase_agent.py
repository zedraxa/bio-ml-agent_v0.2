import logging
from typing import Dict, Any
from bio_ml_agent.core.agent_base import BaseSubAgent

logger = logging.getLogger("academic.style_paraphrase_agent")

class AcademicStyleAgent(BaseSubAgent):
    """
    E1: Academic Style & Paraphrase Agent
    Adjusts the tone of the draft to match specific journal guidelines (e.g. Nature vs PLOS).
    """
    def __init__(self):
        super().__init__(
            role_name="AcademicStyleEditor",
            system_prompt=(
                "You are an Academic Copy Editor. Your job is to read a scientific draft "
                "and rewrite it to perfectly match the requested target journal's tone. "
                "Enhance flow, remove colloquialisms, and ensure dense, precise phrasing. Output JSON."
            )
        )
        self.draft = ""

    def perceive(self, context: Dict[str, Any]):
        self.draft = context.get("draft_text", "")
        self.target_journal = context.get("target_journal", "Standard Scientific")

    def plan(self) -> str:
        return f"Adapt draft to {self.target_journal} style."

    def act(self, instructions: str) -> None:
        prompt = f"""
        Target Style: {self.target_journal}
        Draft Text:
        {self.draft}
        
        Rewrite the text to match the target style securely. Output JSON:
        {{
            "styled_text_md": "The polished markdown text",
            "style_changes_made": ["List of major tone adjustments"]
        }}
        """
        self.current_result = self.llm.chat([{"role": "user", "content": prompt}])

    def verify(self) -> bool:
        return "styled_text_md" in self.current_result

    def summarize(self) -> Any:
        try:
            import json
            data = json.loads(self.current_result.replace("```json", "").replace("```", "").strip())
        except Exception:
            data = {"raw": self.current_result}
        return self.create_checkpoint("Apply Academic Style", data, "HIGH")
