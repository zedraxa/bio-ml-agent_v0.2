import logging
from typing import Dict, Any
from bio_ml_agent.core.agent_base import BaseSubAgent

logger = logging.getLogger("academic.table_narration_agent")

class TableNarrationAgent(BaseSubAgent):
    """
    D2: Table Narration Agent
    Extracts the main message from a table to narrate in text, avoiding listing every number.
    """
    def __init__(self):
        super().__init__(
            role_name="TableNarrator",
            system_prompt=(
                "You are a Scientific Results Narrator. Given a raw data table, "
                "extract the most significant differences or trends and write a short, "
                "elegant paragraph that narrates the table's main insight. "
                "Do NOT repeat every number. Output JSON."
            )
        )
        self.table_data = ""

    def perceive(self, context: Dict[str, Any]):
        self.table_data = context.get("table_csv", "")

    def plan(self) -> str:
        return "Extract main trends from table and narrate them."

    def act(self, instructions: str) -> None:
        prompt = f"""
        Extract the core message from this data table:
        {self.table_data}
        
        Produce JSON:
        {{
            "table_narrative_md": "A fluid academic paragraph summarizing to be placed in 'Results'",
            "key_highlights": ["List of top 3 statistical take-aways"]
        }}
        """
        self.current_result = self.llm.chat([{"role": "user", "content": prompt}])

    def verify(self) -> bool:
        return "table_narrative_md" in self.current_result

    def summarize(self) -> Any:
        try:
            import json
            data = json.loads(self.current_result.replace("```json", "").replace("```", "").strip())
        except Exception:
            data = {"raw": self.current_result}
        return self.create_checkpoint("Narrate Table", data, "HIGH")
