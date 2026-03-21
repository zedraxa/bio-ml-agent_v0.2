import logging
import json
from typing import Dict, Any, List

from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from bio_ml_agent.llm_backend import auto_create_backend

log = logging.getLogger("structural.docking_workflow")

class DockingWorkflowAgent(BaseSubAgent):
    """
    C1: Docking Workflow Agent
    Prepares protein structures (receptors) and ligand libraries for docking experiments.
    Generates preparation instructions and docking plans.
    """

    def __init__(self, model_name: str = "gemini-2.5-pro"):
        super().__init__("DockingWorkflowAgent", model_name)
        self.llm = auto_create_backend(model_name)
        self.workflow_context: Dict[str, Any] = {}
        self.docking_plan: Dict[str, Any] = {}

    def perceive(self, context: Dict[str, Any]) -> None:
        """Receives target PDB and ligand set formats."""
        self.workflow_context = context.get("docking_inputs", {})
        if not self.workflow_context:
            log.warning("No docking input specs provided to DockingWorkflowAgent.")
        else:
            log.info("🧪 DockingWorkflowAgent received structural/ligand setup formats.")

    def plan(self, goal: str) -> List[str]:
        return [
            "Parse receptor properties (needs protonation, metal ions handling).",
            "Organize ligand library input formats (SMILES/SDF to 3D).",
            "Determine bounding box (grid) logic based on active site.",
            "Generate raw executable docking plan (YAML equivalent)."
        ]

    def act(self, step: str) -> Any:
        prompt = f"""
        You are a Computational Chemist orchestrating a docking workflow.
        Review the following receptor and ligand inputs:
        {json.dumps(self.workflow_context, indent=2)}

        Tasks:
        1. Formulate processing steps for the receptor (e.g. adding hydrogens, cleaning water).
        2. Formulate conformer generation steps for the ligands.
        3. Define the virtual docking grid box parameters.

        Respond STRICTLY with a JSON dictionary matching:
        {{
            "receptor_preparation_steps": ["Remove H2O", "Add polar hydrogens"],
            "ligand_preparation_steps": ["Convert SMILES to SDF", "Protonate at pH 7.4"],
            "grid_box_logic": {{"center": "Pocket center", "dimensions": [20, 20, 20]}},
            "docking_plan_yaml_content": "yaml string representation of the workflow"
        }}
        """
        try:
            response = self.llm.chat([{"role": "user", "content": prompt}])
            clean_text = response.replace("```json", "").replace("```", "").strip()
            self.docking_plan = json.loads(clean_text)
            log.info("Docking plan workflow generated.")
        except Exception as e:
            log.error(f"Docking plan parsing failed: {e}")
            self.docking_plan["raw_error_text"] = response

        return "Docking preparation layout computed."

    def verify(self, action_result: Any) -> bool:
        return "docking_plan_yaml_content" in self.docking_plan

    def summarize(self) -> AgentResult:
        is_valid = self.verify("")
        conf = Confidence.HIGH if is_valid else Confidence.LOW
        return AgentResult(
            success=is_valid,
            data={"docking_blueprint": self.docking_plan},
            confidence=conf,
            evidence=[Evidence("DockingWorkflowAgent", "Generated prep & grid logic")],
            message="Docking workflow configuration is ready."
        )
