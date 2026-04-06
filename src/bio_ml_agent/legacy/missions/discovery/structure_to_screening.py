import logging
from typing import Dict, Any

from bio_ml_agent.missions.base_mission import BaseMission
from bio_ml_agent.agents.structural.binding_site_suggestion import BindingSiteSuggestionAgent
from bio_ml_agent.agents.structural.docking_workflow_agent import DockingWorkflowAgent

log = logging.getLogger("mission.structure_screening")

class StructureToScreeningPrep(BaseMission):
    """
    Mission 3: Structure to Screening Prep
    Flow: Predicted Structure -> Confidence Evaluation -> Pocket Candidate Sourcing ->
    Docking Prep Generation -> Screening Execution Plan.
    """

    def __init__(self, mission_id: str):
        super().__init__(mission_id)
        self.pocket_agent = BindingSiteSuggestionAgent()
        self.docking_agent = DockingWorkflowAgent()

    def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        log.info(f"🏗️ Starting [Structure to Screening Prep] Mission ID: {self.mission_id}")

        # Step 1: Pocket Finding
        self.pocket_agent.perceive({"surface_metrics": payload.get("structure_mesh")})
        self.pocket_agent.act("")
        pocket_res = self.pocket_agent.summarize()

        # Step 2: Docking Plan based on identified pocket
        self.docking_agent.perceive({
            "docking_inputs": {
                "receptor": "prepared_model.pdb",
                "pockets": pocket_res.data
            }
        })
        self.docking_agent.act("")
        docking_res = self.docking_agent.summarize()

        return {
            "mission_id": self.mission_id,
            "status": "COMPLETED",
            "pocket_candidates": pocket_res.data,
            "docking_plan": docking_res.data
        }
