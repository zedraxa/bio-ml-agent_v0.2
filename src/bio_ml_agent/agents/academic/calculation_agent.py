import logging
from typing import Dict, Any
from bio_ml_agent.core.agent_base import BaseSubAgent

logger = logging.getLogger("academic.calculation_agent")

class CalculationIntegrationAgent(BaseSubAgent):
    """
    A4: Calculation & Formula Integration Agent
    Embeds formulas, executes unit conversions, and scaffolds 'sample calculation' sections.
    """
    def __init__(self):
        super().__init__(
            role_name="CalculationIntegration",
            system_prompt=(
                "You are a Scientific Calculation and Formatting Assistant. "
                "Your task is to take experimental numbers, format them into LaTeX-style equations, "
                "verify unit conversions, and write a 'Sample Calculation' block suitable for an engineering/biology report."
                "Output JSON."
            )
        )
        self.raw_data = ""

    def perceive(self, context: Dict[str, Any]):
        self.raw_data = context.get("math_data", "")

    def plan(self) -> str:
        return "Parse values, identify necessary formulas, and output a structured calculation narrative."

    def act(self, instructions: str) -> None:
        prompt = f"""
        Given the following raw measurements and expected formulas:
        {self.raw_data}
        
        Generate a JSON:
        {{
            "latex_formulas": ["LaTeX string 1", "LaTeX string 2"],
            "step_by_step_narrative": "Markdown explanation of how the calculation was performed",
            "unit_alerts": "Any issues with units (e.g., 'Ensure mL is converted to L')"
        }}
        """
        self.current_result = self.llm.chat([{"role": "user", "content": prompt}])

    def verify(self) -> bool:
        return "latex_formulas" in self.current_result

    def summarize(self) -> Any:
        try:
            import json
            data = json.loads(self.current_result.replace("```json", "").replace("```", "").strip())
        except Exception:
            data = {"raw": self.current_result}
        return self.create_checkpoint("Embed Calculations", data, "HIGH")
