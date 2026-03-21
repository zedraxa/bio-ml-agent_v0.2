import logging
import json
from typing import Dict, Any, List

from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from bio_ml_agent.llm_backend import auto_create_backend

log = logging.getLogger("omics.expression_analysis")

class ExpressionAnalysisAgent(BaseSubAgent):
    """
    D1: Expression Analysis Agent
    Analyzes differential expression (DE) starting workflows.
    Evaluates volcano/heatmap data, prioritizes candidate gene flags,
    and prepares raw lists for pathway reasoning.
    """

    def __init__(self, model_name: str = "gemini-2.5-pro"):
        super().__init__("ExpressionAnalysisAgent", model_name)
        self.llm = auto_create_backend(model_name)
        self.expression_context: Dict[str, Any] = {}
        self.prioritized_candidates: Dict[str, Any] = {}

    def perceive(self, context: Dict[str, Any]) -> None:
        """Receives fold-change arrays, p-values, or raw sequencing DE matrices."""
        self.expression_context = context.get("differential_expression", {})
        if not self.expression_context:
            log.warning("No differential expression payload provided.")
        else:
            log.info("📊 ExpressionAnalysisAgent received DE matrices/volcano coordinates.")

    def plan(self, goal: str) -> List[str]:
        return [
            "Parse log2FoldChange and adjusted p-values from dataset.",
            "Flag extremely significant up/down-regulated genes.",
            "Translate top-ranked genes into a prioritized pathway input list.",
            "Generate statistical reasoning notes for the selected cutoff."
        ]

    def act(self, step: str) -> Any:
        prompt = f"""
        You are a Computational Biologist specializing in Transcriptomics RNA-Seq Data.
        Evaluate this differential expression dataset snippet:
        {json.dumps(self.expression_context, indent=2)}

        Tasks:
        1. Identify the most robust up-regulated and down-regulated candidate genes based on statistical cutoffs (e.g. log2FC > 2, FDR < 0.05).
        2. Rank these targets for functional relevance prioritization.
        3. Formulate the clean top-hit list to feed into exactly downstream pathway arrays.

        Respond STRICTLY with a JSON dictionary matching:
        {{
            "up_regulated_candidates": [
                {{"gene": "TP53", "l2fc": 2.4, "significance": "High"}}
            ],
            "down_regulated_candidates": [],
            "cutoffs_applied": "Log2FC > 2.0, adj-p < 0.01",
            "pathway_ready_gene_set": ["TP53", "BRCA1", "CDK4"]
        }}
        """
        try:
            response = self.llm.chat([{"role": "user", "content": prompt}])
            clean_text = response.replace("```json", "").replace("```", "").strip()
            self.prioritized_candidates = json.loads(clean_text)
            log.info("Expression DE analysis and prioritization completed.")
        except Exception as e:
            log.error(f"Expression data parsing failed: {e}")
            self.prioritized_candidates["raw_error_text"] = response

        return "Differential expression workflow executed."

    def verify(self, action_result: Any) -> bool:
        return "pathway_ready_gene_set" in self.prioritized_candidates

    def summarize(self) -> AgentResult:
        is_valid = self.verify("")
        conf = Confidence.HIGH if is_valid else Confidence.LOW
        return AgentResult(
            success=is_valid,
            data={"expression_analysis": self.prioritized_candidates},
            confidence=conf,
            evidence=[Evidence("ExpressionAnalysisAgent", "Prioritized DE targets for pathway arrays")],
            message="Transcriptomics fold-change arrays parsed and prioritized."
        )
