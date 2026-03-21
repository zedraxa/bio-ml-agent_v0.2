import logging
from typing import Dict, Any
from bio_ml_agent.core.agent_base import BaseSubAgent

logger = logging.getLogger("academic.results_discussion_agents")

class ResultsNarrationAgent(BaseSubAgent):
    """
    B4.1: Results Narration Agent
    Narrates tables, statistics, and trends scientifically.
    """
    def __init__(self):
        super().__init__(
            role_name="ResultsNarrator",
            system_prompt="You write precise scientific Results sections. You state exact numbers, distributions, and trends. You refer to figures (e.g. As seen in Figure 1). You do NOT interpret or discuss mechanism here. Output JSON."
        )
        self.inputs = {}

    def perceive(self, context: Dict[str, Any]): self.inputs = context
    def plan(self) -> str: return "Narrate results."

    def act(self, instructions: str) -> None:
        prompt = f"Data tables/Stats: {self.inputs.get('stats')}\nFormat as JSON: {{\"results_md\": \"...\", \"figure_references\": [\"\"]}}"
        self.current_result = self.llm.chat([{"role": "user", "content": prompt}])

    def verify(self) -> bool: return "results_md" in self.current_result
    def summarize(self) -> Any:
        import json
        try: return self.create_checkpoint("Write Results", json.loads(self.current_result.strip('` \njson')), "HIGH")
        except: return {"raw": self.current_result}

class DiscussionInterpretationAgent(BaseSubAgent):
    """
    B4.2: Discussion & Interpretation Agent
    Interprets findings, highlights limitations, relates to literature.
    """
    def __init__(self):
        super().__init__(
            role_name="DiscussionInterpreter",
            system_prompt="You write the Discussion section. Synthesize what results mean, correlate with literature, clarify limitations, and state future directions. Output JSON."
        )
        self.inputs = {}

    def perceive(self, context: Dict[str, Any]): self.inputs = context
    def plan(self) -> str: return "Interpret findings in discussion."

    def act(self, instructions: str) -> None:
        prompt = f"Results to interpret: {self.inputs.get('results')}\nLiterature support: {self.inputs.get('literature')}\nFormat JSON: {{\"discussion_md\": \"...\", \"limitations_identified\": [\"\"]}}"
        self.current_result = self.llm.chat([{"role": "user", "content": prompt}])

    def verify(self) -> bool: return "discussion_md" in self.current_result
    def summarize(self) -> Any:
        import json
        try: return self.create_checkpoint("Write Discussion", json.loads(self.current_result.strip('` \njson')), "HIGH")
        except: return {"raw": self.current_result}
