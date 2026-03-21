import logging
from typing import Dict, Any
from bio_ml_agent.core.agent_base import BaseSubAgent

logger = logging.getLogger("academic.abstract_agent")

class ScientificAbstractAgent(BaseSubAgent):
    """
    B2: Scientific Abstract Agent
    Writes structured or unstructured abstracts given the full paper content.
    """
    def __init__(self):
        super().__init__(
            role_name="ScientificAbstractGenerator",
            system_prompt=(
                "You are an expert at distilling complex scientific papers into powerful, concise Abstracts. "
                "Write clearly, emphasizing the gap, the approach, key results, and broader impact. "
                "Output JSON."
            )
        )
        self.full_text = ""

    def perceive(self, context: Dict[str, Any]):
        self.full_text = context.get("paper_draft", "")
        self.format_type = context.get("abstract_type", "unstructured")

    def plan(self) -> str:
        return "Distill paper draft into an abstract."

    def act(self, instructions: str) -> None:
        prompt = f"""
        Write a {self.format_type} abstract (max 250 words) based on the following paper draft:
        -----------------
        {self.full_text}
        -----------------
        
        Output JSON:
        {{
            "abstract_text": "The final abstract text",
            "keywords": ["5-6 relevant keywords"],
            "graphical_abstract_prompt": "Prompt for an AI image generator to make a graphical abstract"
        }}
        """
        self.current_result = self.llm.chat([{"role": "user", "content": prompt}])

    def verify(self) -> bool:
        return "abstract_text" in self.current_result

    def summarize(self) -> Any:
        try:
            import json
            data = json.loads(self.current_result.replace("```json", "").replace("```", "").strip())
        except Exception:
            data = {"raw": self.current_result}
        return self.create_checkpoint("Write Abstract", data, "HIGH")
