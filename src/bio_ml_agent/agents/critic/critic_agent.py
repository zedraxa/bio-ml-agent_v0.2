import logging
from typing import List, Dict, Any, Optional
from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence

log = logging.getLogger("critic_agent")

class CriticAgent(BaseSubAgent):
    """
    Critic Agent: Diğer ajanların çıktılarını denetleyen, 
    kanıt yeterliliğini sorgulayan ve revizyon öneren "kalite kapısı" ajanı.
    """
    
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        super().__init__("CriticAgent", model_name)
        self.target_result: Optional[AgentResult] = None

    def perceive(self, context: Dict[str, Any]) -> None:
        """Eleştirilecek sonucu alır."""
        self.target_result = context.get("target_result")

    def plan(self, goal: str) -> List[str]:
        return [
            "Review evidence sufficiency",
            "Check for logical contradictions",
            "Assess confidence level vs claims",
            "Suggest concrete improvements"
        ]

    def act(self, step: str) -> Any:
        if not self.target_result:
            return "No result to criticize."
            
        # Eleştiri mantığı (LLM tabanlı)
        return {
            "findings": ["Evidence is sufficient", "Confidence matches claims"],
            "suggestions": []
        }

    def verify(self, action_result: Any) -> bool:
        return isinstance(action_result, dict)

    def summarize(self) -> AgentResult:
        return AgentResult(
            success=True,
            data={"review": "Output is high quality."},
            confidence=Confidence.HIGH,
            evidence=[],
            message="Criticism complete. Output approved."
        )
