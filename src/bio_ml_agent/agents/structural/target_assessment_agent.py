import logging
import json
from typing import Dict, Any, List

from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from bio_ml_agent.llm_backend import auto_create_backend

log = logging.getLogger("structural.target_assessment")

class TargetAssessmentAgent(BaseSubAgent):
    """
    B2: Target Assessment Agent
    Debates if a protein is a suitable target for research/development.
    Combines structure, conservation, domain risks, and literature context 
    into a holistic targetability score.
    """

    def __init__(self, model_name: str = "gemini-2.5-pro"):
        super().__init__("TargetAssessmentAgent", model_name)
        self.llm = auto_create_backend(model_name)
        self.target_context: Dict[str, Any] = {}
        self.assessment_report: Dict[str, Any] = {}

    def perceive(self, context: Dict[str, Any]) -> None:
        """Receives structural topology, pathway data, and conservation stats."""
        self.target_context = context.get("target_package", {})
        if not self.target_context:
            log.warning("No comprehensive target package provided.")
        else:
            log.info("🎯 TargetAssessment received structural/pathway context.")

    def plan(self, goal: str) -> List[str]:
        return [
            "Assess core targetability metrics (pockets vs disordered risk).",
            "Evaluate conserved versus highly variable functional blocks.",
            "Cross-validate structural viability against literature/pathway importance.",
            "Generate actionable druggability output files."
        ]

    def act(self, step: str) -> Any:
        prompt = f"""
        You are a Lead Pharmaceutical Target Discovery Scientist. 
        Evaluate this protein's viability as an in-silico or therapeutic target:
        {json.dumps(self.target_context, indent=2)}

        Tasks:
        1. Discuss targetability (druggability) using the intersection of structural domains and surface cavities.
        2. Identify risk areas: If a critical domain has high variability or low pLDDT (uncertainty).
        3. Weigh the target's pathway/literature importance vs its structural accessibility.

        Respond STRICTLY with a JSON dictionary matching:
        {{
            "targetability_score": "High/Medium/Low",
            "overall_assessment_md": "Markdown detailing whether this is a robust target or risky",
            "risk_notes": ["Low confidence in active site", "Highly disordered loop near cavity"],
            "conservation_summary": "Highly conserved signaling domain making it a generic but reliable target..."
        }}
        """
        try:
            response = self.llm.chat([{"role": "user", "content": prompt}])
            clean_text = response.replace("```json", "").replace("```", "").strip()
            self.assessment_report = json.loads(clean_text)
            log.info("Targetability evaluation finalized.")
        except Exception as e:
            log.error(f"Target assessment parsing failed: {e}")
            self.assessment_report["raw_error_text"] = response

        return "Target assessment drafted."

    def verify(self, action_result: Any) -> bool:
        return "targetability_score" in self.assessment_report

    def summarize(self) -> AgentResult:
        is_valid = self.verify("")
        conf = Confidence.HIGH if is_valid else Confidence.LOW
        return AgentResult(
            success=is_valid,
            data={"target_assessment": self.assessment_report},
            confidence=conf,
            evidence=[Evidence("TargetAssessment", "Cross-evaluated pathway against struct access")],
            message="Target feasibility and therapeutic potential evaluated."
        )
