import logging
from typing import Dict, Any
from bio_ml_agent.core.agent_base import BaseSubAgent

logger = logging.getLogger("academic.statistical_narration")

class StatisticalNarrationAgent(BaseSubAgent):
    """
    D3: Statistical Narration Agent
    Narrates p-values, confidence intervals, and effect sizes with proper nuance.
    """
    def __init__(self):
        super().__init__(
            role_name="StatisticalNarrator",
            system_prompt=(
                "You are an expert Statistician writing for a scientific journal. "
                "Translate raw statistical test outputs into precise academic sentences. "
                "Use terms like 'significant but small effect' carefully. "
                "Format p-values correctly (e.g. p < 0.05). Output JSON."
            )
        )
        self.stats_input = ""

    def perceive(self, context: Dict[str, Any]):
        self.stats_input = context.get("raw_statistics", "")

    def plan(self) -> str:
        return "Translate raw p-values and effect sizes into nuanced narrative text."

    def act(self, instructions: str) -> None:
        prompt = f"""
        Raw Statistical Output:
        {self.stats_input}
        
        Generate JSON:
        {{
            "statistical_narrative_md": "Nuanced academic sentence(s) describing the relevance of the finding.",
            "misinterpretation_risk_flag": "Warning if the effect size is small despite low p-value, etc."
        }}
        """
        self.current_result = self.llm.chat([{"role": "user", "content": prompt}])

    def verify(self) -> bool:
        return "statistical_narrative_md" in self.current_result

    def summarize(self) -> Any:
        try:
            import json
            data = json.loads(self.current_result.replace("```json", "").replace("```", "").strip())
        except Exception:
            data = {"raw": self.current_result}
        return self.create_checkpoint("Narrate Statistics", data, "HIGH")
