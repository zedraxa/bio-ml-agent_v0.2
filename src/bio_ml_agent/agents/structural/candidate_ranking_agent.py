import logging
import json
from typing import Dict, Any, List

from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from bio_ml_agent.llm_backend import auto_create_backend

log = logging.getLogger("structural.candidate_ranking")

class CandidateRankingAgent(BaseSubAgent):
    """
    C3: Candidate Ranking Agent
    Post-processes docking execution scores. 
    Filters out chemically illogical "high scorers", enforces molecular diversity,
    and isolates risky/redundant compounds.
    """

    def __init__(self, model_name: str = "gemini-2.5-pro"):
        super().__init__("CandidateRankingAgent", model_name)
        self.llm = auto_create_backend(model_name)
        self.raw_scores: Dict[str, Any] = {}
        self.ranked_candidates: Dict[str, Any] = {}

    def perceive(self, context: Dict[str, Any]) -> None:
        """Receives CSV-like dictionaries of ligand docking scores."""
        self.raw_scores = context.get("docking_scores", {})
        if not self.raw_scores:
            log.warning("No docking scores provided for ranking.")
        else:
            log.info("📊 CandidateRankingAgent received score distributions.")

    def plan(self, goal: str) -> List[str]:
        return [
            "Sort ligands by best (lowest) binding free energy.",
            "Analyze chemical structures (if SMILES provided) for generic hits (PAINS).",
            "Adjust ranks based on scaffold diversity to prevent redundancy.",
            "Flag highly polar/reactive false positives."
        ]

    def act(self, step: str) -> Any:
        prompt = f"""
        You are a Medicinal Chemist acting as a post-docking filter. 
        Evaluate these top docking scores:
        {json.dumps(self.raw_scores, indent=2)}

        Tasks:
        1. Ignore the sheer score if the molecule is a known 'PAIN' (Pan-Assay Interference Compounds) or chemically absurd.
        2. Identify redundancy (e.g. 5 ligands with the exact same scaffold).
        3. Nominate the true 'Top Diverse Candidates'.

        Respond STRICTLY with a JSON dictionary matching:
        {{
            "chemical_logic_notes": "markdown text discussing structural viabilities",
            "risk_flags": ["LigandA looks like a reactive quinone", "LigandB violates diversity"],
            "final_ranked_selections": ["LigandC", "LigandF"]
        }}
        """
        try:
            response = self.llm.chat([{"role": "user", "content": prompt}])
            clean_text = response.replace("```json", "").replace("```", "").strip()
            self.ranked_candidates = json.loads(clean_text)
            log.info("Candidate ranking filtered successfully.")
        except Exception as e:
            log.error(f"Candidate ranking parsing failed: {e}")
            self.ranked_candidates["raw_error_text"] = response

        return "Refined candidate tracking applied."

    def verify(self, action_result: Any) -> bool:
        return "final_ranked_selections" in self.ranked_candidates

    def summarize(self) -> AgentResult:
        is_valid = self.verify("")
        conf = Confidence.HIGH if is_valid else Confidence.LOW
        return AgentResult(
            success=is_valid,
            data={"refined_candidates": self.ranked_candidates},
            confidence=conf,
            evidence=[Evidence("CandidateRankingAgent", "Applied chemical logic filtering to raw scores")],
            message="Top candidates rationally ranked and diversified."
        )
