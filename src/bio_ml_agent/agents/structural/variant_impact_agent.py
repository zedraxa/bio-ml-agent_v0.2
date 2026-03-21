import logging
import json
from typing import Dict, Any, List

from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from bio_ml_agent.llm_backend import auto_create_backend

log = logging.getLogger("structural.variant_impact")

class VariantImpactReasoningAgent(BaseSubAgent):
    """
    B3: Variant Impact Reasoning Agent
    Evaluates specific genetic variants (amino acid substitutions).
    Produces structural impact hypotheses (surface vs core, stability logic, sensitive regions)
    tempered by AlphaFold confidence metrics.
    """

    def __init__(self, model_name: str = "gemini-2.5-pro"):
        super().__init__("VariantImpactReasoningAgent", model_name)
        self.llm = auto_create_backend(model_name)
        self.variant_context: Dict[str, Any] = {}
        self.impact_hypothesis: Dict[str, Any] = {}

    def perceive(self, context: Dict[str, Any]) -> None:
        """Receives variant list (e.g. ['A124T', 'R300W']) and mapped structural metrics."""
        self.variant_context = context.get("variants", {})
        if not self.variant_context:
            log.warning("No structural variant data provided.")
        else:
            log.info("🧬 VariantImpactAgent received genetic mutations mapped to structure.")

    def plan(self, goal: str) -> List[str]:
        return [
            "Parse requested amino acid substitutions.",
            "Assess regional confidence (pLDDT) to issue cautionary structural notes.",
            "Determine if the mutation is buried (core) or exposed (surface).",
            "Generate hypotheses regarding protein stability or functional disruption."
        ]

    def act(self, step: str) -> Any:
        prompt = f"""
        You are a Computational Geneticist and Structural Biologist.
        Analyze the impact of these amino acid substitutions on the protein structure:
        {json.dumps(self.variant_context, indent=2)}

        Tasks:
        1. Warn if the variant is in a low-confidence region (disordered loop/tail) where 'impact' is ambiguous.
        2. Classify the variant's location (Buried Hydrophobic Core vs Exposed Polar Surface).
        3. Formulate a hypothesis on structural stability (e.g. placing a bulky Tryp in a tight core).
        4. Predict if the mutation falls in a functionally sensitive region.

        Respond STRICTLY with a JSON dictionary matching:
        {{
            "variant_analysis": [
                {{
                    "variant": "R300W",
                    "confidence_warning": "Low pLDDT region, structural inference uncertain.",
                    "location_type": "surface",
                    "stability_hypothesis": "Likely destabilizing, introduces bulky hydrophobic group to solvent.",
                    "functional_sensitivity_probability": "High"
                }}
            ]
        }}
        """
        try:
            response = self.llm.chat([{"role": "user", "content": prompt}])
            clean_text = response.replace("```json", "").replace("```", "").strip()
            self.impact_hypothesis = json.loads(clean_text)
            log.info("Variant impact evaluation complete.")
        except Exception as e:
            log.error(f"Variant impact parsing failed: {e}")
            self.impact_hypothesis["raw_error_text"] = response

        return "Variant assessment finalized."

    def verify(self, action_result: Any) -> bool:
        return "variant_analysis" in self.impact_hypothesis

    def summarize(self) -> AgentResult:
        is_valid = self.verify("")
        conf = Confidence.HIGH if is_valid else Confidence.LOW
        return AgentResult(
            success=is_valid,
            data={"variant_hypotheses": self.impact_hypothesis},
            confidence=conf,
            evidence=[Evidence("VariantImpactAgent", "Generated stability hypotheses for mutations")],
            message="Structural inferences for sequence variants generated."
        )
