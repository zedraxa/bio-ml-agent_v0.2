import logging
import json
from typing import Dict, Any, List

from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from bio_ml_agent.llm_backend import auto_create_backend

log = logging.getLogger("omics.biomarker")

class BiomarkerHypothesisAgent(BaseSubAgent):
    """
    D4: Biomarker Hypothesis Agent
    Generates actionable clinical biomarker hypotheses combining 
    structural vulnerabilities, functional impact, and OMICS expression support.
    Keeps predictions grounded in research/iterative contexts.
    """

    def __init__(self, model_name: str = "gemini-2.5-pro"):
        super().__init__("BiomarkerHypothesisAgent", model_name)
        self.llm = auto_create_backend(model_name)
        self.biomarker_context: Dict[str, Any] = {}
        self.diagnostic_hypothesis: Dict[str, Any] = {}

    def perceive(self, context: Dict[str, Any]) -> None:
        """Receives multi-omics candidate lists, pathways, and literature."""
        self.biomarker_context = context.get("candidate_markers", {})
        if not self.biomarker_context:
            log.warning("No candidate lists provided for biomarker generation.")
        else:
            log.info("💉 BiomarkerHypothesisAgent received candidate lists.")

    def plan(self, goal: str) -> List[str]:
        return [
            "Review candidate expression stability across cohorts.",
            "Assess structural/functional relevance as a diagnostic vs prognostic marker.",
            "Compile literature or pathway logical support.",
            "Assign a confidence rating and explicitly state it is a 'Research Hypothesis'."
        ]

    def act(self, step: str) -> Any:
        prompt = f"""
        You are a Clinical Translational Bioinformatics Scientist.
        Generate biomarker hypotheses from the following candidate data:
        {json.dumps(self.biomarker_context, indent=2)}

        Tasks:
        1. Nominate specific genes/proteins as diagnostic or prognostic biomarkers.
        2. Provide dual justification: Structural/Functional basis AND Expression data support.
        3. Rate your hypothesis confidence and add robust literature logical support.
        CRITICAL: Frame this explicitly as an in-silico research hypothesis, not a confirmed clinical fact.

        Respond STRICTLY with a JSON dictionary matching:
        {{
            "candidate_biomarkers": [
                {{
                    "marker": "Protein_ABC",
                    "type": "diagnostic",
                    "structural_basis": "Unique exposed surface motif suitable for ELISA",
                    "expression_support": "Persistently up-regulated in tumor RNA-Seq",
                    "confidence_rating": "Moderate",
                    "clinical_hypothesis_notes": "Could serve as an early tracking molecule. Requires wet-lab validation."
                }}
            ],
            "overall_literature_consensus": "markdown summary of known pathway connections"
        }}
        """
        try:
            response = self.llm.chat([{"role": "user", "content": prompt}])
            clean_text = response.replace("```json", "").replace("```", "").strip()
            self.diagnostic_hypothesis = json.loads(clean_text)
            log.info("Biomarker hypotheses formulated.")
        except Exception as e:
            log.error(f"Biomarker hypothesis failed: {e}")
            self.diagnostic_hypothesis["raw_error_text"] = response

        return "Clinical translational hypotheses drafted."

    def verify(self, action_result: Any) -> bool:
        return "candidate_biomarkers" in self.diagnostic_hypothesis

    def summarize(self) -> AgentResult:
        is_valid = self.verify("")
        conf = Confidence.HIGH if is_valid else Confidence.LOW
        return AgentResult(
            success=is_valid,
            data={"biomarker_hypotheses": self.diagnostic_hypothesis},
            confidence=conf,
            evidence=[Evidence("BiomarkerAgent", "Synthesized diagnostic hypotheses from integrated data")],
            message="Translational biomarker research completed."
        )
