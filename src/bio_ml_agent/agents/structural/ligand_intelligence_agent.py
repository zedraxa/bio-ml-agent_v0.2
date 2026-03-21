import logging
import json
from typing import Dict, Any, List

from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from bio_ml_agent.llm_backend import auto_create_backend

log = logging.getLogger("structural.ligand_intelligence")

class LigandIntelligenceAgent(BaseSubAgent):
    """
    C4: Ligand Intelligence Agent
    Aggregates ligand metadata, notes chemical similarities/scaffolds, 
    flags physicochemical issues, and pulls literature evidence.
    Creates rich candidate profiles ('ligand cards') instead of just IDs.
    """

    def __init__(self, model_name: str = "gemini-2.5-pro"):
        super().__init__("LigandIntelligenceAgent", model_name)
        self.llm = auto_create_backend(model_name)
        self.ligand_context: Dict[str, Any] = {}
        self.ligand_profiles: Dict[str, Any] = {}

    def perceive(self, context: Dict[str, Any]) -> None:
        """Receives top candidate SMILES, PubChem CID hints, and literature snippets."""
        self.ligand_context = context.get("candidate_metadata", {})
        if not self.ligand_context:
            log.warning("No candidate metadata provided to LigandIntelligenceAgent.")
        else:
            log.info("🔍 LigandIntelligenceAgent received molecular metadata.")

    def plan(self, goal: str) -> List[str]:
        return [
            "Extract molecular scaffolds and core chemical structures.",
            "Flag critical physicochemical violations (e.g. high toxicity risk).",
            "Synthesize available literature or known targets for the compound/scaffold.",
            "Generate comprehensive JSON candidate profiles (Ligand Cards)."
        ]

    def act(self, step: str) -> Any:
        prompt = f"""
        You are a Chemoinformatics and Pharmacognosy Intelligence Entity.
        Profile the following list of candidate molecules provided via SMILES or IDs:
        {json.dumps(self.ligand_context, indent=2)}

        Tasks:
        1. Identify the primary chemical scaffold and note any similarity to known drugs.
        2. Raise flags for poor physicochemical traits (e.g., promiscuous binders, highly reactive).
        3. Provide brief literature-based evidence or target-class associations.

        Respond STRICTLY with a JSON dictionary matching:
        {{
            "ligand_cards": [
                {{
                    "id": "Candidate_1",
                    "scaffold_type": "Indole derivative",
                    "physicochemical_flags": ["Low solubility risk"],
                    "literature_association": "Scaffold common in kinase inhibitors"
                }}
            ]
        }}
        """
        try:
            response = self.llm.chat([{"role": "user", "content": prompt}])
            clean_text = response.replace("```json", "").replace("```", "").strip()
            self.ligand_profiles = json.loads(clean_text)
            log.info("Ligand candidate profiles generated successfully.")
        except Exception as e:
            log.error(f"Ligand profiling failed: {e}")
            self.ligand_profiles["raw_error_text"] = response

        return "Ligand intelligence profiles constructed."

    def verify(self, action_result: Any) -> bool:
        return "ligand_cards" in self.ligand_profiles

    def summarize(self) -> AgentResult:
        is_valid = self.verify("")
        conf = Confidence.HIGH if is_valid else Confidence.LOW
        return AgentResult(
            success=is_valid,
            data={"ligand_intelligence": self.ligand_profiles},
            confidence=conf,
            evidence=[Evidence("LigandIntelligenceAgent", "Created chemical profiles for candidates")],
            message="Ligand profiles (Cards) enriched."
        )
