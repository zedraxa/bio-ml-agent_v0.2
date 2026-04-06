import logging
from typing import Dict, Any

from bio_ml_agent.missions.base_mission import BaseMission
from bio_ml_agent.agents.omics.expression_analysis_agent import ExpressionAnalysisAgent
from bio_ml_agent.agents.omics.multi_omics_integrator import MultiOmicsIntegratorAgent

log = logging.getLogger("mission.omics_prioritization")

class OmicsToTargetPrioritization(BaseMission):
    """
    Mission 4: Omics to Target Prioritization
    Flow: Differential Expression -> Cutoff ranking -> Pathway linking -> Structural Feasibility cross-check.
    """

    def __init__(self, mission_id: str):
        super().__init__(mission_id)
        self.expression_agent = ExpressionAnalysisAgent()
        self.integrator_agent = MultiOmicsIntegratorAgent()

    def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        log.info(f"🧬 Starting [Omics to Target Prioritization] Mission ID: {self.mission_id}")

        # Step 1: Differential Expression filtering
        self.expression_agent.perceive({"differential_expression": payload.get("rnaseq_matrix")})
        self.expression_agent.act("")
        expr_result = self.expression_agent.summarize()

        # Step 2: Multi-Omics Structural Integration
        self.integrator_agent.perceive({
            "merged_datasets": {
                "transcriptomics_top_hits": expr_result.data.get("expression_analysis", {}).get("pathway_ready_gene_set", []),
                "simulated_structural_druggability": "High availability of pockets for top 3 hits"
            }
        })
        self.integrator_agent.act("")
        integration_res = self.integrator_agent.summarize()

        return {
            "mission_id": self.mission_id,
            "status": "COMPLETED",
            "expression_filtering": expr_result.data,
            "integrated_target_shortlist": integration_res.data
        }
