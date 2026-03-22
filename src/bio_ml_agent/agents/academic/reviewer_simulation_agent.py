import logging
from typing import Dict, Any, List
from bio_ml_agent.core.agent_base import BaseSubAgent

logger = logging.getLogger("academic.reviewer_simulation_agent")

class ReviewerSimulationAgent(BaseSubAgent):
    """
    E4: Reviewer Simulation Agent (Reviewer #2)
    Critiques the full manuscript exactly like a harsh peer-reviewer.
    """
    def __init__(self):
        super().__init__(
            role_name="ReviewerSimulation",
            system_prompt=(
                "You are 'Reviewer 2' for a high-impact scientific journal. "
                "You are highly critical, detail-oriented, and skeptical. "
                "Read the manuscript and write a formal peer-review report highlighting methodological flaws, "
                "overstated conclusions, and missing controls. Output JSON."
            )
        )
        self.manuscript = ""

    def perceive(self, context: Dict[str, Any]):
        self.manuscript = context.get("final_manuscript", "")

    def plan(self, goal: str) -> List[str]:
        return ["Read manuscript thoroughly", "Identify methodological flaws", "Draft formal peer-review report"]

    def act(self, instructions: str) -> None:
        prompt = f"""
        Manuscript for Review:
        {self.manuscript}
        
        Write your peer review report. Output JSON:
        {{
            "decision": "Accept / Minor Revision / Major Revision / Reject",
            "major_criticisms": ["Points that invalidate the main claim"],
            "minor_criticisms": ["Typos, figure adjustments, citation missing"],
            "reviewer_report_md": "The formal letter to the editor/authors"
        }}
        """
        self.current_result = self.llm.chat([{"role": "user", "content": prompt}])

    def verify(self, action_result: Any) -> bool:
        return self.current_result and "reviewer_report_md" in self.current_result

    def summarize(self) -> Any:
        try:
            import json
            data = json.loads(self.current_result.replace("```json", "").replace("```", "").strip())
        except Exception:
            data = {"raw": self.current_result}
        return self.create_checkpoint("Simulate Peer Review", data, "HIGH")
