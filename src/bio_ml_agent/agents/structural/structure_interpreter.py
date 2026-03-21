import logging
import json
from typing import Dict, Any, List

from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from bio_ml_agent.llm_backend import auto_create_backend

log = logging.getLogger("structural.interpreter")

class StructureInterpreterAgent(BaseSubAgent):
    """
    A2: Structure Interpreter Agent
    Analyzes AlphaFold/PDB structure arrays (pLDDT, PAE).
    Separates noise from biology by finding domains, disordered regions,
    and evaluating structural compactness.
    """

    def __init__(self, model_name: str = "gemini-2.5-pro"):
        super().__init__("StructureInterpreterAgent", model_name)
        self.llm = auto_create_backend(model_name)
        self.structural_metrics: Dict[str, Any] = {}
        self.analysis_report: Dict[str, Any] = {}

    def perceive(self, context: Dict[str, Any]) -> None:
        """Receives PAE, pLDDT arrays, and PDB metadata."""
        self.structural_metrics = context.get("alphafold_metrics", {})
        if not self.structural_metrics:
            log.warning("No AlphaFold metrics (pLDDT, PAE) provided.")
        else:
            log.info("📊 StructureInterpreter received confidence metrics.")

    def plan(self, goal: str) -> List[str]:
        return [
            "Parse pLDDT and PAE matrices.",
            "Identify high/low confidence regions.",
            "Flag intrinsically disordered regions (IDRs).",
            "Propose stable domain-like segments and assess structural compactness."
        ]

    def act(self, step: str) -> Any:
        prompt = f"""
        You are the ultimate Protein Structure Interpreter. Analyze the following summary of AlphaFold metrics:
        {json.dumps(self.structural_metrics, indent=2)}

        Tasks:
        1. Evaluate pLDDT bands (low, medium, high confidence).
        2. Evaluate PAE (Predicted Aligned Error) for inter-domain relationships.
        3. Delineate intrinsically disordered regions versus stabilized domains.
        4. Provide an overall structural compactness assessment and rough preliminary binding pocket assessment.

        Respond STRICTLY with a JSON object containing:
        {{
            "structure_summary": "Detailed markdown string summarizing the topology",
            "confidence_regions": {{"high_trust": "ranges...", "low_trust": "ranges..."}},
            "domain_candidates": ["domain1_range", "domain2_range"],
            "disorder_flags": ["disordered_loop1", "n_term_tail"],
            "compactness_score": "e.g. Globular, Flexible, Multi-domain"
        }}
        """
        try:
            response = self.llm.chat([{"role": "user", "content": prompt}])
            clean_text = response.replace("```json", "").replace("```", "").strip()
            self.analysis_report = json.loads(clean_text)
            log.info("Interpreted structural parameters successfully.")
        except Exception as e:
            log.error(f"Interpreter output parsing failed: {e}")
            self.analysis_report["raw_error_text"] = response

        return "AlphaFold Interpretation completed."

    def verify(self, action_result: Any) -> bool:
        return "confidence_regions" in self.analysis_report

    def summarize(self) -> AgentResult:
        is_valid = self.verify("")
        conf = Confidence.HIGH if is_valid else Confidence.LOW
        return AgentResult(
            success=is_valid,
            data=self.analysis_report,
            confidence=conf,
            evidence=[Evidence("StructureInterpreter", "Detected domains and disordered regions")],
            message="Structural interpretation parsed."
        )
