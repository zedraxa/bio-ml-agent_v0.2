import logging
import json
from typing import Dict, Any, List

from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from bio_ml_agent.llm_backend import auto_create_backend

log = logging.getLogger("omics.multi_omics_integrator")

class MultiOmicsIntegratorAgent(BaseSubAgent):
    """
    D3: Multi-Omics Integrator Agent
    The supreme contextualizer. Merges Transcriptomics (RNA), Proteomics (Protein),
    and Structure (AlphaFold) data to find targets that are both structurally viable
    and significantly dysregulated in disease.
    """

    def __init__(self, model_name: str = "gemini-2.5-pro"):
        super().__init__("MultiOmicsIntegratorAgent", model_name)
        self.llm = auto_create_backend(model_name)
        self.omics_context: Dict[str, Any] = {}
        self.integration_report: Dict[str, Any] = {}

    def perceive(self, context: Dict[str, Any]) -> None:
        """Receives mixed modalities: transcriptomic counts, mutational landscapes, structural scores."""
        self.omics_context = context.get("merged_datasets", {})
        if not self.omics_context:
            log.warning("No multi-omics matrices provided.")
        else:
            log.info("🧬 MultiOmicsIntegrator received multi-modal dataset.")

    def plan(self, goal: str) -> List[str]:
        return [
            "Cross-validate transcriptomic up-regulation with proteomic abundance.",
            "Map high-frequency mutations to structural disruption hot-spots.",
            "Synthesize an 'Integrated Target Viability' argument.",
            "Output multi-modal candidate intersection lists."
        ]

    def act(self, step: str) -> Any:
        prompt = f"""
        You are the Lead Systems & Structural Biologist.
        Integrate the following multi-modal omics signals alongside structural hints:
        {json.dumps(self.omics_context, indent=2)}

        Tasks:
        1. Find overlaps: e.g. "Gene X is up-regulated in RNA-Seq AND has a highly druggable structural pocket."
        2. Identify contradictions (e.g. high RNA expression but no protein abundance detected).
        3. Formulate the ultimate holistic argument for why a specific target matters in both disease expression and structural pharmacology.

        Respond STRICTLY with a JSON dictionary matching:
        {{
            "omics_intersections": ["Gene X: High RNA + Druggable Structure"],
            "holistic_arguments": ["Target Y is not just structurally interesting, but strongly over-expressed in mutant phenotypes."],
            "contradiction_flags": ["Gene Z has high RNA but low protein, suggesting translational repression."],
            "integrated_target_score": "High/Medium/Low with rationale"
        }}
        """
        try:
            response = self.llm.chat([{"role": "user", "content": prompt}])
            clean_text = response.replace("```json", "").replace("```", "").strip()
            self.integration_report = json.loads(clean_text)
            log.info("Multi-omics integration successful.")
        except Exception as e:
            log.error(f"Multi-omics mapping failed: {e}")
            self.integration_report["raw_error_text"] = response

        return "Multi-omics synthesis compiled."

    def verify(self, action_result: Any) -> bool:
        return "omics_intersections" in self.integration_report

    def summarize(self) -> AgentResult:
        is_valid = self.verify("")
        conf = Confidence.HIGH if is_valid else Confidence.LOW
        return AgentResult(
            success=is_valid,
            data={"integrated_omics_synthesis": self.integration_report},
            confidence=conf,
            evidence=[Evidence("MultiOmicsIntegrator", "Merged structure, mutation, and expression data")],
            message="Cross-modality biological insights generated."
        )
