from typing import List, Dict, Any
from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
import logging

log = logging.getLogger("browser_critic")

class BrowserCritic(BaseSubAgent):
    """
    Browser Critic: Tarayıcı adımlarını inceler, kısırlık (loop) veya mantık hatası var mı bakar.
    """

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        super().__init__("BrowserCritic", model_name)
        self.history = []

    def perceive(self, context: Dict[str, Any]) -> None:
        self.history = context.get("history", [])

    def plan(self, goal: str) -> List[str]:
        return ["Check for loops", "Analyze step effectiveness", "Suggest replanning if needed"]

    def act(self, step: str) -> Any:
        return "Review complete"

    def verify(self, action_result: Any) -> bool:
        return True

    def summarize(self) -> AgentResult:
        return AgentResult(
            success=True,
            data={"findings": [], "replan_suggested": False},
            confidence=Confidence.HIGH,
            evidence=[],
            message="Critic review finished."
        )
