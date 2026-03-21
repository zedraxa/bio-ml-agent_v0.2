import logging
import json
from typing import Dict, Any, List

from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from bio_ml_agent.llm_backend import auto_create_backend

log = logging.getLogger("structural.comparison")

class StructureComparisonAgent(BaseSubAgent):
    """
    A3: Structure Comparison Agent
    Compares two structures (WT vs Mutant, Apo vs Bound, Species A vs B).
    Evaluates RMSD shifts in a biological and confidence-aware manner.
    """

    def __init__(self, model_name: str = "gemini-2.5-pro"):
        super().__init__("StructureComparisonAgent", model_name)
        self.llm = auto_create_backend(model_name)
        self.comparison_data: Dict[str, Any] = {}
        self.comparison_report: Dict[str, Any] = {}

    def perceive(self, context: Dict[str, Any]) -> None:
        """Receives structural alignment metrics (RMSD, delta distances)."""
        self.comparison_data = context.get("alignment_metrics", {})
        if not self.comparison_data:
            log.warning("No alignment alignment_metrics provided.")
        else:
            log.info("⚖️ StructureComparison received alignment pairs.")

    def plan(self, goal: str) -> List[str]:
        return [
            "Assess global and local RMSD values.",
            "Account for pLDDT confidence (ignoring noise in disordered loops).",
            "Highlight areas of functional divergence or mutational impact."
        ]

    def act(self, step: str) -> Any:
        prompt = f"""
        You are a Structural Bioinformatics Validator. Compare these two protein states (e.g. Mutant vs WT) based on spatial alignment metrics:
        {json.dumps(self.comparison_data, indent=2)}

        Tasks:
        1. Evaluate significant RMSD shifts. Ignore shifts occurring in regions where pLDDT is historically low (noise).
        2. Provide area-based structural difference summaries.
        3. Generate hypotheses on how these shifts impact the protein's function.

        Respond STRICTLY with a JSON dictionary matching:
        {{
            "comparison_summary": "High level structural shift explanation",
            "region_differences": [
                {{"region": "residues 10-20", "rmsd_deviation": 2.5, "is_confident_shift": true}}
            ],
            "impact_hypotheses": "markdown string on how function is altered"
        }}
        """
        try:
            response = self.llm.chat([{"role": "user", "content": prompt}])
            clean_text = response.replace("```json", "").replace("```", "").strip()
            self.comparison_report = json.loads(clean_text)
            log.info("Completed structural comparison.")
        except Exception as e:
            log.error(f"Comparison parsing failed: {e}")
            self.comparison_report["raw_error_text"] = response

        return "Comparison evaluation generated."

    def verify(self, action_result: Any) -> bool:
        return "region_differences" in self.comparison_report

    def summarize(self) -> AgentResult:
        is_valid = self.verify("")
        conf = Confidence.HIGH if is_valid else Confidence.LOW
        return AgentResult(
            success=is_valid,
            data=self.comparison_report,
            confidence=conf,
            evidence=[Evidence("StructureComparison", "Identified confident topological divergence")],
            message="Structural differences isolated and interpreted."
        )
