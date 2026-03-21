import logging
import json
from typing import Dict, Any, List

from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from bio_ml_agent.llm_backend import auto_create_backend

log = logging.getLogger("structural.sequence_function")

class SequenceToFunctionAgent(BaseSubAgent):
    """
    B1: Sequence-to-Function Agent
    Extracts biological meaning from sequence motifs, domain families,
    and localization hints. Predicts catalytic and binding residue candidates.
    Works heavily in tandem with structure context.
    """

    def __init__(self, model_name: str = "gemini-2.5-pro"):
        super().__init__("SequenceToFunctionAgent", model_name)
        self.llm = auto_create_backend(model_name)
        self.sequence_context: Dict[str, Any] = {}
        self.function_hints: Dict[str, Any] = {}

    def perceive(self, context: Dict[str, Any]) -> None:
        """Receives raw sequence, functional annotations (e.g. InterPro/PFAM)."""
        self.sequence_context = context.get("sequence_data", {})
        if not self.sequence_context:
            log.warning("No sequence properties provided to SequenceToFunctionAgent.")
        else:
            log.info("🧬 SequenceToFunction received functional annotations.")

    def plan(self, goal: str) -> List[str]:
        return [
            "Parse sequence motif signals and domain family hints.",
            "Formulate possible localization and generic function hints.",
            "Highlight primary catalytic or binding residue candidates based on conservation."
        ]

    def act(self, step: str) -> Any:
        prompt = f"""
        You are a Computational Biologist focusing on sequence-to-function translation.
        Analyze the following protein sequence metadata and motif hints:
        {json.dumps(self.sequence_context, indent=2)}

        Tasks:
        1. Identify the overarching domain family and likely cellular localization.
        2. Propose broad function hints based on motifs.
        3. Nominate specific residue indices as catalytic or binding candidates.

        Respond STRICTLY with a JSON dictionary matching:
        {{
            "domain_family_hints": ["Kinase", "Transmembrane Receptor"],
            "localization_hints": ["Nucleus", "Cytoplasm"],
            "catalytic_residue_candidates": [
                {{"residue": "Asp124", "reason": "Highly conserved catalytic triad anchor"}}
            ]
        }}
        """
        try:
            response = self.llm.chat([{"role": "user", "content": prompt}])
            clean_text = response.replace("```json", "").replace("```", "").strip()
            self.function_hints = json.loads(clean_text)
            log.info("Successfully extracted sequence-to-function predictions.")
        except Exception as e:
            log.error(f"Function hint parsing failed: {e}")
            self.function_hints["raw_error_text"] = response

        return "Sequence function evaluation summarized."

    def verify(self, action_result: Any) -> bool:
        return "catalytic_residue_candidates" in self.function_hints

    def summarize(self) -> AgentResult:
        is_valid = self.verify("")
        conf = Confidence.HIGH if is_valid else Confidence.LOW
        return AgentResult(
            success=is_valid,
            data={"function_hints": self.function_hints},
            confidence=conf,
            evidence=[Evidence("SequenceFunction", "Mapped generic sequence motifs to function")],
            message="Sequence capabilities and catalytic hypotheses formulated."
        )
