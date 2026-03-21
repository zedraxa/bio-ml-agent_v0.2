import logging
from typing import Dict, Any
from bio_ml_agent.core.agent_base import BaseSubAgent

logger = logging.getLogger("academic.lab_report_critic")

class LabReportCriticAgent(BaseSubAgent):
    """
    A5: Lab Report Critic Agent
    Flags missing sections, logic gaps, checks tense usage, and grades against rubric.
    """
    def __init__(self):
        super().__init__(
            role_name="LabReportCritic",
            system_prompt=(
                "You are a strict, detail-oriented Academic Reviewer for laboratory reports. "
                "Evaluate the drafted lab report for logical consistency, required sections, "
                "academic tone (passive voice check), and alignment between Methods and Results."
                "Output JSON."
            )
        )
        self.draft = ""

    def perceive(self, context: Dict[str, Any]):
        self.draft = context.get("report_draft", "")

    def plan(self) -> str:
        return "Critique sections, flag inconsistencies, and generate revision list."

    def act(self, instructions: str) -> None:
        prompt = f"""
        Review the following laboratory report draft:
        -----------------
        {self.draft}
        -----------------
        
        Generate a JSON critique:
        {{
            "missing_sections": ["list"],
            "logic_gaps": ["e.g. Conclusion mentions variable not in Results"],
            "tone_warnings": ["Check paragraph 2 for informal language"],
            "revision_notes_md": "Markdown summary of required changes"
        }}
        """
        self.current_result = self.llm.chat([{"role": "user", "content": prompt}])

    def verify(self) -> bool:
        return "revision_notes_md" in self.current_result

    def summarize(self) -> Any:
        try:
            import json
            data = json.loads(self.current_result.replace("```json", "").replace("```", "").strip())
        except Exception:
            data = {"raw": self.current_result}
        return self.create_checkpoint("Critique Lab Report", data, "HIGH")
