import logging
from typing import Dict, Any
from bio_ml_agent.core.agent_base import BaseSubAgent

logger = logging.getLogger("academic.multimodal_integrator")

class MultiModalIntegratorAgent(BaseSubAgent):
    """
    D5: Multi-Modal Result Integrator
    Combines tables, graphs, microscopy, and stats into one cohesive Results flow.
    """
    def __init__(self):
        super().__init__(
            role_name="MultiModalIntegrator",
            system_prompt=(
                "You are the Lead Scientific Writer compiling the final 'Results' super-section. "
                "You receive text from Table, Stats, and Microscopy narrators. "
                "Your job is to weave them together into a single, highly flowing academic narrative "
                "with correct cross-references. Output JSON."
            )
        )
        self.narratives = {}

    def perceive(self, context: Dict[str, Any]):
        self.narratives = context

    def plan(self) -> str:
        return "Weave multi-modal narratives into a cohesive Results super-section."

    def act(self, instructions: str) -> None:
        prompt = f"""
        Inputs from sub-agents:
        - Table Narrative: {self.narratives.get('table', 'N/A')}
        - Statistical Narrative: {self.narratives.get('stats', 'N/A')}
        - Microscopy Narrative: {self.narratives.get('microscopy', 'N/A')}
        - Graph Caption Focus: {self.narratives.get('graph_caption', 'N/A')}
        
        Output JSON:
        {{
            "integrated_results_md": "The combined, fluent, and highly professional Results section."
        }}
        """
        self.current_result = self.llm.chat([{"role": "user", "content": prompt}])

    def verify(self) -> bool:
        return "integrated_results_md" in self.current_result

    def summarize(self) -> Any:
        try:
            import json
            data = json.loads(self.current_result.replace("```json", "").replace("```", "").strip())
        except Exception:
            data = {"raw": self.current_result}
        return self.create_checkpoint("Integrate Multi-Modal Results", data, "HIGH")
