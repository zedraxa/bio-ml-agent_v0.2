import logging
from typing import Dict, Any
from bio_ml_agent.core.agent_base import BaseSubAgent

logger = logging.getLogger("academic.structure_compliance_agent")

class StructureComplianceAgent(BaseSubAgent):
    """
    E2: Structure Compliance Agent
    Enforces word counts, references limits, and section ordering based on journal guidelines.
    """
    def __init__(self):
        super().__init__(
            role_name="StructureComplianceChecker",
            system_prompt=(
                "You are a Journal Submission Formatting Checker. "
                "You inspect a manuscript draft against strict journal guidelines (word limits, section orders). "
                "You flag violations and suggest trims. Output JSON."
            )
        )
        self.draft = ""

    def perceive(self, context: Dict[str, Any]):
        self.draft = context.get("manuscript_draft", "")
        self.guidelines = context.get("journal_guidelines", {})

    def plan(self) -> str:
        return "Check manuscript against formatting limits."

    def act(self, instructions: str) -> None:
        prompt = f"""
        Journal Guidelines: {self.guidelines}
        Manuscript Outline/Draft:
        {self.draft}
        
        Output JSON:
        {{
            "compliance_status": "Pass/Fail",
            "violations": ["E.g. Abstract exceeds 250 words (currently 280)"],
            "suggested_fixes": ["Trim Background section in Abstract"]
        }}
        """
        self.current_result = self.llm.chat([{"role": "user", "content": prompt}])

    def verify(self) -> bool:
        return "compliance_status" in self.current_result

    def summarize(self) -> Any:
        try:
            import json
            data = json.loads(self.current_result.replace("```json", "").replace("```", "").strip())
        except Exception:
            data = {"raw": self.current_result}
        return self.create_checkpoint("Check Structure Compliance", data, "HIGH")
