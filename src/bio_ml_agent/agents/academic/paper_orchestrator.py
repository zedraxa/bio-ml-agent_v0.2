import logging
from typing import Dict, Any, List
from bio_ml_agent.core.agent_base import BaseSubAgent

logger = logging.getLogger("academic.paper_orchestrator")

class PaperDraftOrchestrator(BaseSubAgent):
    """
    B1: Paper Draft Orchestrator Agent
    Scaffolds a paper's structure depending on its type (e.g. review, primary research, letter).
    """
    def __init__(self):
        super().__init__(
            role_name="PaperDraftOrchestrator",
            system_prompt=(
                "You are the Lead Editor of an Academic Publishing Engine. "
                "Determine the structure of a paper based on the raw experimental and literature inputs, "
                "and decide the flow of Abstract, Introduction, Methods, Results, Discussion, and Conclusion. "
                "Output as JSON."
            )
        )
        self.payload = {}

    def perceive(self, context: Dict[str, Any]):
        self.payload = context

    def plan(self, goal: str) -> List[str]:
        return ["Analyze journal requirements", "Establish structural flow", "Determine required visual materials (Figures/Tables)"]

    def act(self, instructions: str) -> None:
        prompt = f"""
        Draft the formal paper scaffold using the following raw inputs and requested format:
        {self.payload.get('paper_type', 'Original Research Article')}
        Inputs: {self.payload.get('materials', 'No data')}
        
        Output a JSON blueprint:
        {{
            "paper_type": "string",
            "section_outline": ["List of headers"],
            "target_journal_tone": "e.g. Nature style (concise) or PLOS (detailed)",
            "required_figures_tables": ["List of expected visual material based on data"]
        }}
        """
        self.current_result = self.llm.chat([{"role": "user", "content": prompt}])

    def verify(self, action_result: Any) -> bool:
        return self.current_result and "section_outline" in self.current_result

    def summarize(self) -> Any:
        try:
            import json
            data = json.loads(self.current_result.replace("```json", "").replace("```", "").strip())
        except Exception:
            data = {"raw": self.current_result}
        return self.create_checkpoint("Scaffold Paper Blueprint", data, "HIGH")
