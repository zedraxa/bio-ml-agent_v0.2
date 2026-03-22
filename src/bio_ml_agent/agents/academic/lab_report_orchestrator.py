import logging
from typing import Dict, Any, List
from bio_ml_agent.core.agent_base import BaseSubAgent

logger = logging.getLogger("academic.lab_report_orchestrator")

class LabReportOrchestratorAgent(BaseSubAgent):
    """
    A1: Lab Report Orchestrator Agent
    Interprets lab notes, images, data, and rubric. Plans the report structure.
    """
    def __init__(self):
        super().__init__(
            role_name="LabReportOrchestrator",
            system_prompt=(
                "You are an Academic Lab Report Orchestrator. "
                "Your objective is to ingest raw lab notes, data matrices, protocols, and grading rubrics, "
                "and generate a highly structured plan for writing a formal scientific lab report. "
                "You must outline the exact sections required (e.g., Abstract, Introduction, Procedure, Results, Discussion, Conclusion) "
                "and list missing information risks. "
                "JSON format required."
            )
        )
        self.materials: Dict[str, Any] = {}

    def perceive(self, context: Dict[str, Any]):
        self.materials = context

    def plan(self, goal: str) -> List[str]:
        return ["Analyze inputs and lab notes", "Classify report type (formal/informal)", "Generate detailed section blueprint"]

    def act(self, instructions: str) -> None:
        prompt = f"""
        Design a Lab Report Blueprint based on these materials:
        - Experiment Type: {self.materials.get('experiment_type', 'Unknown')}
        - Raw Notes / Data: {self.materials.get('raw_notes', 'No notes provided')}
        - Rubric / Format requested: {self.materials.get('rubric', 'Standard Scientific Format')}
        
        Output a JSON object:
        {{
            "report_type": "string (e.g. Formal Lab Report)",
            "required_sections": ["list of exact section headers"],
            "writing_strategy": "string (e.g. focus heavily on discussion due to observation nature)",
            "missing_information_risks": ["list of missing data bits"]
        }}
        """
        self.current_result = self.llm.chat([{"role": "user", "content": prompt}])

    def verify(self, action_result: Any) -> bool:
        return self.current_result and "report_type" in self.current_result and "required_sections" in self.current_result

    def summarize(self) -> Any:
        try:
            import json
            data = json.loads(self.current_result.replace("```json", "").replace("```", "").strip())
        except Exception:
            data = {"error": "Failed to parse orchestrator output", "raw": self.current_result}
        
        return self.create_checkpoint(
            action="Orchestrate Lab Report",
            data=data,
            confidence="HIGH"
        )
