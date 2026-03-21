import logging
import json
from typing import Dict, Any, List

from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from bio_ml_agent.llm_backend import auto_create_backend

log = logging.getLogger("structural.critic")

class StructuralCriticAgent(BaseSubAgent):
    """
    C5: Structural Critic for Screening
    Acts as the final safety buffer against AI hallucinations in docking.
    Evaluates whether a docking pose is structurally logical or purely speculatory 
    (e.g., binding to a low pLDDT unstructured tail).
    """

    def __init__(self, model_name: str = "gemini-2.5-pro"):
        super().__init__("StructuralCriticAgent", model_name)
        self.llm = auto_create_backend(model_name)
        self.pose_context: Dict[str, Any] = {}
        self.critic_report: Dict[str, Any] = {}

    def perceive(self, context: Dict[str, Any]) -> None:
        """Receives the final ligand pose environment (distance to residues, pLDDT scores)."""
        self.pose_context = context.get("pose_details", {})
        if not self.pose_context:
            log.warning("No docking pose context provided to StructuralCriticAgent.")
        else:
            log.info("⚖️ StructuralCriticAgent received pose topology for auditing.")

    def plan(self, goal: str) -> List[str]:
        return [
            "Cross-check binding site residue confidence (pLDDT).",
            "Evaluate physical logic of the ligand pose (e.g. clashing, floating).",
            "Downgrade or reject highly scored but physically speculative results.",
            "Emit final structural verdict."
        ]

    def act(self, step: str) -> Any:
        prompt = f"""
        You are the Supreme Structural Critic. It is your job to stop AI hallucinations in drug discovery.
        Review the following docking pose parameters (interactions, local pLDDT confidence, distances):
        {json.dumps(self.pose_context, indent=2)}

        Tasks:
        1. Check if the pocket residues have low confidence. If so, the pocket might not exist structurally!
        2. Check for physical impossibility (floating ligand, extreme clashes).
        3. Issue a severe downgrade if the result is speculative, regardless of a 'good docking score'.

        Respond STRICTLY with a JSON dictionary matching:
        {{
            "docking_pose_verdict": "Valid/Speculative/Invalid",
            "pocket_confidence_check": "Pocket formed by disordered loop (Low Trust) vs Rigid Core (High Trust)",
            "critic_notes": ["Rejecting high score due to binding to disordered N-terminus"],
            "anti_hallucination_adjusted_score": "float or classification"
        }}
        """
        try:
            response = self.llm.chat([{"role": "user", "content": prompt}])
            clean_text = response.replace("```json", "").replace("```", "").strip()
            self.critic_report = json.loads(clean_text)
            log.info("Structural critique applied to docking results.")
        except Exception as e:
            log.error(f"Structural critic parsing failed: {e}")
            self.critic_report["raw_error_text"] = response

        return "Screening poses critiqued."

    def verify(self, action_result: Any) -> bool:
        return "docking_pose_verdict" in self.critic_report

    def summarize(self) -> AgentResult:
        is_valid = self.verify("")
        conf = Confidence.HIGH if is_valid else Confidence.LOW
        return AgentResult(
            success=is_valid,
            data={"critic_report": self.critic_report},
            confidence=conf,
            evidence=[Evidence("StructuralCriticAgent", "Audited pose against pLDDT halluciation risks")],
            message="Docking poses subjected to structural reality check."
        )
