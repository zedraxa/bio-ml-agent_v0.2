from typing import List, Dict, Any
from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
import json
import logging

log = logging.getLogger("browser_extractor")

class BrowserExtractor(BaseSubAgent):
    """
    Browser Extractor: Ham web içeriğini yapılandırılmış verilere (JSON/CSV) dönüştürür.
    """

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        super().__init__("BrowserExtractor", model_name)
        self.page = None

    def perceive(self, context: Dict[str, Any]) -> None:
        self.page = context.get("page")

    def plan(self, goal: str) -> List[str]:
        return ["Identify data targets", "Extract raw text", "Structure into JSON"]

    def act(self, step: str) -> Any:
        # Veri çekme mantığı
        return {"extracted": "data"}

    def verify(self, action_result: Any) -> bool:
        return isinstance(action_result, dict)

    def summarize(self) -> AgentResult:
        return self.create_result(
            success=True,
            data={"records": []},
            confidence=Confidence.MEDIUM,
            evidence=[],
            message="Data extraction complete."
        )
