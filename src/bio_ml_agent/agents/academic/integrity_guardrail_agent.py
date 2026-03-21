import logging
from typing import Dict, Any
from bio_ml_agent.core.agent_base import BaseSubAgent

logger = logging.getLogger("academic.integrity_guardrail_agent")

class IntegrityGuardrailAgent(BaseSubAgent):
    """
    E3: Integrity Guardrail Agent
    Prevents AI hallucinations. Cross-references the generated text against real data/paper inputs.
    """
    def __init__(self):
        super().__init__(
            role_name="IntegrityGuardrail",
            system_prompt=(
                "You are an AI Hallucination Detector for Scientific Papers. "
                "Your single goal is to compare the generated draft text against the RAW input data. "
                "If the text claims a value, finding, or citation that does not exist in the raw data, you MUST flag it as a hallucination. "
                "Output JSON."
            )
        )
        self.draft = ""
        self.raw_data = ""

    def perceive(self, context: Dict[str, Any]):
        self.draft = context.get("draft_text", "")
        self.raw_data = context.get("raw_data_anchor", "")

    def plan(self) -> str:
        return "Scan draft for unsupported claims against the raw data anchor."

    def act(self, instructions: str) -> None:
        prompt = f"""
        Raw Ground Truth Data:
        {self.raw_data}
        
        Generated Draft Text:
        {self.draft}
        
        Identify any claims in the draft that are NOT supported by the Ground Truth. Output JSON:
        {{
            "integrity_score": "High/Medium/Low",
            "detected_hallucinations": ["Quote the false claim -> Explain why it's unsupported"],
            "safe_draft_md": "Return the draft with hallucinations removed or marked [UNVERIFIED]"
        }}
        """
        self.current_result = self.llm.chat([{"role": "user", "content": prompt}])

    def verify(self) -> bool:
        return "integrity_score" in self.current_result

    def summarize(self) -> Any:
        try:
            import json
            data = json.loads(self.current_result.replace("```json", "").replace("```", "").strip())
        except Exception:
            data = {"raw": self.current_result}
        return self.create_checkpoint("Check Scientific Integrity", data, "HIGH")
