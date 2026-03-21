import logging
import json
from typing import Dict, Any, List

from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from bio_ml_agent.llm_backend import auto_create_backend

log = logging.getLogger("structural.binding_site")

class BindingSiteSuggestionAgent(BaseSubAgent):
    """
    A4: Binding Site Suggestion Agent
    An early-stage in-silico discovery module. 
    Predicts pocket candidates, assesses surface cavities, cross-references with conserved regions, 
    and applies trust scoring using AlphaFold structure confidence.
    """

    def __init__(self, model_name: str = "gemini-2.5-pro"):
        super().__init__("BindingSiteSuggestionAgent", model_name)
        self.llm = auto_create_backend(model_name)
        self.pocket_context: Dict[str, Any] = {}
        self.pocket_predictions: Dict[str, Any] = {}

    def perceive(self, context: Dict[str, Any]) -> None:
        """Receives structural surface, cavity coordinates, and conservation scores."""
        self.pocket_context = context.get("surface_metrics", {})
        if not self.pocket_context:
            log.warning("No surface/cavity metrics provided to Binding Site Agent.")
        else:
            log.info("🧪 BindingSiteAgent received topography metrics.")

    def plan(self, goal: str) -> List[str]:
        return [
            "Analyze surface topography for viable pockets/cavities.",
            "Cross-reference pocket locations with evolutionary conservation data.",
            "Apply structural confidence (pLDDT) to rate pocket trust.",
            "Rank candidates for downstream docking simulations."
        ]

    def act(self, step: str) -> Any:
        prompt = f"""
        You are a Computational Biophysicist specializing in Ligand Discovery.
        Analyze the following protein structural topography and cavity data:
        {json.dumps(self.pocket_context, indent=2)}

        Tasks:
        1. Identify the most probable ligand-binding pockets or active cavities.
        2. Evaluate these pockets against evolutionary conservation (if provided).
        3. Rate each pocket's structural reliability (Trust Score) based on AlphaFold pLDDT. High pLDDT = Real Pocket; Low pLDDT = Artifactual Gap.

        Respond STRICTLY with a JSON dictionary matching:
        {{
            "pocket_candidates": [
                {{"id": "Pocket1", "residues": [15, 16, 17], "volume_estimate": "large", "trust_score": "High"}}
            ],
            "conserved_intersections": ["Pocket1 lies in highly conserved generic motif"],
            "docking_recommendation": "markdown text guiding virtual screening"
        }}
        """
        try:
            response = self.llm.chat([{"role": "user", "content": prompt}])
            clean_text = response.replace("```json", "").replace("```", "").strip()
            self.pocket_predictions = json.loads(clean_text)
            log.info("Successfully suggested binding sites.")
        except Exception as e:
            log.error(f"Binding site parsing failed: {e}")
            self.pocket_predictions["raw_error_text"] = response

        return "Binding site predictions tabulated."

    def verify(self, action_result: Any) -> bool:
        return "pocket_candidates" in self.pocket_predictions

    def summarize(self) -> AgentResult:
        is_valid = self.verify("")
        conf = Confidence.HIGH if is_valid else Confidence.LOW
        return AgentResult(
            success=is_valid,
            data={"binding_site_suggestions": self.pocket_predictions},
            confidence=conf,
            evidence=[Evidence("BindingSiteAgent", "Identified candidate pockets")],
            message="Binding site and cavity discovery completed."
        )
