import logging
from typing import Dict, Any
from bio_ml_agent.core.agent_base import BaseSubAgent

logger = logging.getLogger("academic.intro_methods_agents")

class IntroductionBuilderAgent(BaseSubAgent):
    """
    B3.1: Introduction Builder Agent
    Frames problem, highlights literature gap, states objective.
    """
    def __init__(self):
        super().__init__(
            role_name="IntroductionBuilder",
            system_prompt="You write compelling scientific Introductions. Establish background, emphasize knowledge gaps, and clearly state objectives. Output JSON."
        )
        self.inputs = {}

    def perceive(self, context: Dict[str, Any]):
        self.inputs = context

    def plan(self) -> str: return "Draft introduction."

    def act(self, instructions: str) -> None:
        prompt = f"""
        Topic/Keywords: {self.inputs.get('topic')}
        Literature Core: {self.inputs.get('literature_summary')}
        Goal/Hypothesis: {self.inputs.get('objective')}
        
        Write the Introduction (Markdown text) and return as JSON:
        {{"introduction_md": "...", "identified_gaps": ["..."]}}
        """
        self.current_result = self.llm.chat([{"role": "user", "content": prompt}])

    def verify(self) -> bool: return "introduction_md" in self.current_result

    def summarize(self) -> Any:
        import json
        try: return self.create_checkpoint("Write Intro", json.loads(self.current_result.strip('` \njson')), "HIGH")
        except: return self.create_checkpoint("Write Intro", {"raw": self.current_result}, "LOW")


class MethodsFormalizerAgent(BaseSubAgent):
    """
    B3.2: Methods Formalizer Agent
    Converts messy protocols into rigorous passive-voice "Materials and Methods".
    """
    def __init__(self):
        super().__init__(
            role_name="MethodsFormalizer",
            system_prompt="You rewrite messy lab protocols into formal, strict, reproducible academic 'Materials and Methods' sections in passive voice. Output JSON."
        )
        self.inputs = {}

    def perceive(self, context: Dict[str, Any]):
        self.inputs = context

    def plan(self) -> str: return "Formalize methods."

    def act(self, instructions: str) -> None:
        prompt = f"""
        Raw Protocol/Notes: {self.inputs.get('protocol')}
        
        Write the Methods section (Markdown) and return JSON:
        {{"methods_md": "...", "missing_reproducibility_details": ["volumes missing", "time missing"]}}
        """
        self.current_result = self.llm.chat([{"role": "user", "content": prompt}])

    def verify(self) -> bool: return "methods_md" in self.current_result

    def summarize(self) -> Any:
        import json
        try: return self.create_checkpoint("Write Methods", json.loads(self.current_result.strip('` \njson')), "HIGH")
        except: return self.create_checkpoint("Write Methods", {"raw": self.current_result}, "LOW")
