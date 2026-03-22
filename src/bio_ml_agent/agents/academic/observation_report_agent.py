import logging
from typing import Dict, Any
from bio_ml_agent.core.agent_base import BaseSubAgent

logger = logging.getLogger("academic.observation_report")

class ObservationToReportAgent(BaseSubAgent):
    """
    A3: Observation-to-Report Agent
    Translates informal lab notes (e.g. "it turned blue") to academic phrasing (e.g. "a colorimetric shift to blue was observed").
    """
    def __init__(self):
        super().__init__(
            role_name="ObservationToReport",
            system_prompt=(
                "You are an Academic Paraphrasing Expert specializing in translating informal, "
                "student-level laboratory observations into rigorous, passive-voice scientific prose."
                "Output JSON only."
            )
        )
        self.informal_notes = ""

    def perceive(self, context: Dict[str, Any]):
        self.informal_notes = context.get("informal_notes", "")

    def plan(self) -> str:
        return "Convert informal text to academic passive-voice results narrative."

    def act(self, instructions: str) -> None:
        if "refine" in instructions.lower() or "revision" in instructions.lower():
            # A5: Writing Review Refinement
            self.refine_narrative(self.context.get("refinement_task", {}))
            return

        prompt = f"""
        Convert the following informal lab observations into a high-quality academic 'Results' narrative format:
        Informal Notes: "{self.informal_notes}"
        
        Output JSON format:
        {{
            "academic_narrative": "Transformed high-quality scientific text",
            "detected_observations": ["list of distinct biological/chemical phenomena detected"]
        }}
        """
        self.current_result = self.llm.chat([{"role": "user", "content": prompt}])

    def refine_narrative(self, refinement_task: Dict[str, Any]):
        """
        A5: Writing Review Refinement logic.
        Applies editorial feedback to the scientific narrative.
        """
        instruction = refinement_task.get("instruction", "")
        current_text = getattr(self, "current_result", "")
        if isinstance(current_text, dict):
            current_text = current_text.get("academic_narrative", "")
            
        log_msg = f"✍️ A5: Refining narrative based on editorial feedback: '{instruction}'"
        logger.info(log_msg)
        
        prompt = f"""
        You are an Academic Writing Expert. 
        A researcher has provided editorial feedback on your scientific narrative.
        
        Feedback: {instruction}
        
        Current Narrative:
        {current_text}
        
        Tasks:
        1. Apply the feedback (e.g., shorten, simplify, increase academic rigor).
        2. Maintain scientific accuracy and passive voice where appropriate.
        
        Respond ONLY with the updated JSON (academic_narrative).
        """
        self.current_result = self.llm.chat([{"role": "user", "content": prompt}])

    def verify(self) -> bool:
        return "academic_narrative" in self.current_result

    def summarize(self) -> Any:
        try:
            import json
            data = json.loads(self.current_result.replace("```json", "").replace("```", "").strip())
        except Exception:
            data = {"error": "Failed to parse transformation", "raw": self.current_result}
            
        return self.create_checkpoint("Translate Observations", data, "HIGH")
