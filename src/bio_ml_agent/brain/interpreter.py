import logging
from typing import Optional, Dict, Any, List
from .models import Comment, IntentType, TargetType, MissionStep, TaskType, AgentRole

logger = logging.getLogger("bio_ml_agent.brain.interpreter")

class CommentInterpreter:
    """
    R7-2: Feedback-to-Action Intelligence.
    Interprets human feedback to determine if/how the agent should refine its work.
    """
    def __init__(self, llm_bridge: Any = None):
        self.llm = llm_bridge

    def interpret_feedback(self, comment: Comment, target_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Axis B1/B2: Comment-to-Task Generator.
        Analyzes a comment to extract the refinement intent and step sequence.
        """
        if not comment.is_actionable and comment.intent not in [IntentType.APPROVE, IntentType.REJECT]:
            return None
            
        # B1 Logic: Classify intent if it's default/general
        if comment.intent == IntentType.GENERAL:
            comment.intent = self._classify_intent(comment.content)
            
        # B2 Logic: Generate a sequence of concrete steps
        steps = self.generate_task_sequence(comment)
            
        refinement_task = {
            "comment_id": comment.comment_id,
            "target_id": comment.target_id,
            "intent": comment.intent,
            "instruction": comment.requested_action or comment.content,
            "suggested_role": self._map_target_to_role(comment.target_type),
            "priority": comment.severity,
            "sub_steps": steps
        }
        
        logger.info(f"[Interpreter:B1/B2] Comment {comment.comment_id} decomposed into {len(steps)} steps")
        return refinement_task

    def generate_task_sequence(self, comment: Comment) -> List[str]:
        """B2: Decomposes high-level feedback into concrete agent steps."""
        text = comment.content.lower()
        if comment.target_type == TargetType.IMAGE_REGION:
            return ["Reload microscopy artifact", "Verify ROI label", "Run critic validation", "Update report section"]
        if "akademik" in text or "rigor" in text:
            return ["Analyze current tone", "Apply passive voice transformation", "Cross-check with literature template"]
        if "function" in text or "modular" in text:
            return ["Analyze block dependencies", "Extract function logic", "Update call sites", "Run syntax check"]
            
        return [f"Address feedback: {comment.content[:30]}..."]

    def _classify_intent(self, text: str) -> IntentType:
        """Heuristic-based or LLM-based intent classification."""
        text = text.lower()
        if any(w in text for w in ["uzun", "kısalt", "fazla"]): return IntentType.SIMPLIFY
        if any(w in text for w in ["yanlış", "hata", "bozuk"]): return IntentType.VERIFY
        if any(w in text for w in ["destekle", "ekle", "literatür"]): return IntentType.EXPAND
        if any(w in text for w in ["profesyonel", "akademik", "düzelt"]): return IntentType.REWRITE
        if any(w in text for w in ["araştır", "neden", "niçin"]): return IntentType.INVESTIGATE
        if any(w in text for w in ["onay", "tamam", "okey"]): return IntentType.APPROVE
        
        return IntentType.REVISE

    def _map_target_to_role(self, target_type: TargetType) -> AgentRole:
        """Determines which agent role is best suited to handle this target type."""
        mapping = {
            TargetType.IMAGE_REGION: AgentRole.MICROSCOPY_AGENT,
            TargetType.CODE_LINE: AgentRole.CODING_AGENT,
            TargetType.PARAGRAPH: AgentRole.WRITING_AGENT,
            TargetType.SECTION: AgentRole.WRITING_AGENT,
            TargetType.TABLE_ROW: AgentRole.DATA_ENGINEER,
            TargetType.FIGURE: AgentRole.ACADEMIC_EXPERT,
            TargetType.STEP: AgentRole.PLANNER  # Feedback on plans goes to the planner
        }
        return mapping.get(target_type, AgentRole.RESEARCHER)
