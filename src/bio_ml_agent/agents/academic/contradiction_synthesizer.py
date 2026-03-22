import logging
from typing import Dict, Any, List
from bio_ml_agent.core.agent_base import BaseSubAgent

logger = logging.getLogger("academic.contradiction_synthesizer")

class ContradictionSynthesizerAgent(BaseSubAgent):
    """
    C3: Contradiction Synthesizer Agent
    Reads papers that disagree and writes a balanced debate paragraph.
    """
    def __init__(self):
        super().__init__(
            role_name="ContradictionSynthesizer",
            system_prompt=(
                "You are a Scientific Reviewer specializing in resolving contradictions. "
                "When literature contains conflicting evidence, you write a mature, balanced paragraph "
                "exploring potential reasons for the discrepancy (e.g. sample size, assay differences). Output JSON."
            )
        )
        self.conflicting_papers = []

    def perceive(self, context: Dict[str, Any]):
        self.conflicting_papers = context.get("conflicting_papers", [])

    def plan(self, goal: str) -> List[str]:
        return ["Analyze methodology differences", "Synthesize findings", "Explain divergent results"]

    def act(self, instructions: str) -> None:
        prompt = f"""
        Analyze these conflicting studies:
        {self.conflicting_papers}
        
        Generate JSON:
        {{
            "synthesis_paragraph_md": "Academic text highlighting the debate and suggesting reasons for the discrepancy.",
            "suspected_cause_of_difference": "e.g. 'Study B used in-vivo vs Study A in-vitro'"
        }}
        """
        self.current_result = self.llm.chat([{"role": "user", "content": prompt}])

    def verify(self, action_result: Any) -> bool:
        return self.current_result and "synthesis_paragraph_md" in self.current_result

    def summarize(self) -> Any:
        try:
            import json
            data = json.loads(self.current_result.replace("```json", "").replace("```", "").strip())
        except Exception:
            data = {"raw": self.current_result}
        return self.create_checkpoint("Synthesize Contradictions", data, "HIGH")
