import logging
from typing import Dict, Any
from bio_ml_agent.core.agent_base import BaseSubAgent

logger = logging.getLogger("academic.gap_finder")

class GapFinderAgent(BaseSubAgent):
    """
    C4: Gap Finder Agent
    Identifies what is missing in the literature to justify the current study.
    """
    def __init__(self):
        super().__init__(
            role_name="GapFinder",
            system_prompt=(
                "You are a Scientific Strategist. Your goal is to review the current literature matrix "
                "and clearly define the 'Knowledge Gap' that a new study will fill. "
                "Output JSON."
            )
        )
        self.literature_matrix = ""

    def perceive(self, context: Dict[str, Any]):
        self.literature_matrix = context.get("literature_matrix", "")

    def plan(self) -> str:
        return "Evaluate literature boundary and declare the undiscovered zone."

    def act(self, instructions: str) -> None:
        prompt = f"""
        Based on this literature landscape:
        {self.literature_matrix}
        
        Write JSON output:
        {{
            "identified_gap": "The specific scientific question left unanswered",
            "gap_justification_paragraph": "A formal academic paragraph stating 'While previous studies have shown X, the mechanism of Y remains unclear...'"
        }}
        """
        self.current_result = self.llm.chat([{"role": "user", "content": prompt}])

    def verify(self) -> bool:
        return "identified_gap" in self.current_result

    def summarize(self) -> Any:
        try:
            import json
            data = json.loads(self.current_result.replace("```json", "").replace("```", "").strip())
        except Exception:
            data = {"raw": self.current_result}
        return self.create_checkpoint("Identify Literature Gap", data, "HIGH")
