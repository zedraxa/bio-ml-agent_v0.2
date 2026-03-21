import logging
import json
from typing import Dict, Any, List

from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from bio_ml_agent.llm_backend import auto_create_backend

log = logging.getLogger("structural.residue_annotation")

class ResidueLevelAnnotationAgent(BaseSubAgent):
    """
    B4: Residue-Level Annotation Agent
    Provides highly granular, residue-by-residue structural classifications.
    Pinpoints specific buried/exposed amino acids, interfaces, and discrete motifs.
    """

    def __init__(self, model_name: str = "gemini-2.5-pro"):
        super().__init__("ResidueLevelAnnotationAgent", model_name)
        self.llm = auto_create_backend(model_name)
        self.residue_context: Dict[str, Any] = {}
        self.annotations: Dict[str, Any] = {}

    def perceive(self, context: Dict[str, Any]) -> None:
        """Receives atomic properties array, SASA metrics (Solvent Accessibility), etc."""
        self.residue_context = context.get("residue_properties", {})
        if not self.residue_context:
            log.warning("No residue property arrays provided.")
        else:
            log.info("🔍 ResidueAnnotationAgent received atomic properties array.")

    def plan(self, goal: str) -> List[str]:
        return [
            "Parse Solvent Accessible Surface Area (SASA) for each residue.",
            "Classify residues as strictly buried or exposed.",
            "Flag motif regions locally.",
            "Nominate high-suspicion residues for catalysis or structural cross-bracing."
        ]

    def act(self, step: str) -> Any:
        prompt = f"""
        You are a Residue Annotation System specializing in granular atomic mapping.
        Read the provided biophysical arrays (e.g., SASA, local geometries):
        {json.dumps(self.residue_context, indent=2)}

        Tasks:
        1. Classify critical segments into Buried (Core), Intermediate, or Exposed (Surface).
        2. Propose 'suspicion' tags for specific active residues (e.g. 'Catalytic Suspicion', 'Interface Suspicion').
        3. Highlight contiguous spatial motifs.

        Respond STRICTLY with a JSON dictionary matching:
        {{
            "critical_residues": [
                {{"residue_id": 45, "aa": "His", "classification": "Buried", "suspicion_tags": ["Catalytic Suspicion"]}}
            ],
            "motif_regions": ["Helix-Turn-Helix spanning residues 50-70"],
            "buried_exposed_stats": {{"buried_ratio": 0.45, "exposed_ratio": 0.55}}
        }}
        """
        try:
            response = self.llm.chat([{"role": "user", "content": prompt}])
            clean_text = response.replace("```json", "").replace("```", "").strip()
            self.annotations = json.loads(clean_text)
            log.info("Residue-level annotation complete.")
        except Exception as e:
            log.error(f"Residue parsing failed: {e}")
            self.annotations["raw_error_text"] = response

        return "Residue biophysical annotations built."

    def verify(self, action_result: Any) -> bool:
        return "critical_residues" in self.annotations

    def summarize(self) -> AgentResult:
        is_valid = self.verify("")
        conf = Confidence.HIGH if is_valid else Confidence.LOW
        return AgentResult(
            success=is_valid,
            data={"residue_annotations": self.annotations},
            confidence=conf,
            evidence=[Evidence("ResidueAnnotationAgent", "Classified single-residue biophysics")],
            message="Granular structural properties formulated."
        )
