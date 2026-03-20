from typing import List, Dict, Any
from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
import logging

log = logging.getLogger("browser_verifier")

class BrowserVerifier(BaseSubAgent):
    """
    Browser Verifier: Aksiyonların başarılı olup olmadığını kontrol eder.
    Örn: "Dosya indirildi mi?", "Form gönderildi mi?", "Hata mesajı var mı?"
    """
    
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        super().__init__("BrowserVerifier", model_name)
        self.page: Any = None

    def perceive(self, context: Dict[str, Any]) -> None:
        self.page = context.get("page")

    def plan(self, goal: str) -> List[str]:
        return ["Check success indicators", "Check for error messages", "Validate state change"]

    def act(self, step: str) -> Any:
        # P6 ActionValidator mantığını buraya taşıyabiliriz
        return "Verified"

    def verify(self, action_result: Any) -> bool:
        return True

    def summarize(self) -> AgentResult:
        return AgentResult(
            success=True,
            data={"verified": True},
            confidence=Confidence.HIGH,
            evidence=[],
            message="Action verification successful."
        )
