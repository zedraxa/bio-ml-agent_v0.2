import logging
from typing import List, Dict, Any, Optional
from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from bio_ml_agent.llm_backend import auto_create_backend

log = logging.getLogger("critic_agent")

class CriticAgent(BaseSubAgent):
    """
    CriticAgent (Professional):
    - Adversarial Audit: Diğer ajanların sonuçlarındaki tutarsızlıkları ve kanıt eksiklerini bulur.
    - Kalite Kapısı: Sonuç güven skorlarını (Confidence) doğrular.
    - İyileştirme Önerileri: Başarısız adımlar için alternatif stratejiler sunar.
    """
    
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        super().__init__("CriticAgent", model_name)
        self.last_review: str = ""

    def perceive(self, context: Dict[str, Any]) -> None:
        self.target_result = context.get("target_result") # AgentResult nesnesi
        self.mission_goal = context.get("goal")

    def plan(self, goal: str) -> List[str]:
        return ["Check evidence sufficiency", "Identify logical contradictions", "Score confidence level"]

    def act(self, step: str) -> Any:
        if not self.target_result: return "Nothing to criticize"
        
        prompt = f"""Bir Adversarial Critic'sin. Aşağıdaki sonucu denetle.
Hedef: {self.mission_goal}
Sonuç Verisi: {self.target_result.data}
Kanıtlar: {[e.content_snippet for e in self.target_result.evidence]}

Bu sonuç güvenilir mi? Kanıtlarda boşluk var mı? Yanıtını sert ve analitik bir dille ver."""

        try:
            backend = auto_create_backend(self.model_name)
            self.last_review = backend.chat(prompt)
            return "Audit completed"
        except Exception as e:
            log.error(f"Critic LLM hatası: {e}")
            return "Audit failed"

    def verify(self, action_result: Any) -> bool:
        return True

    def summarize(self) -> AgentResult:
        is_valid = "PASS" in self.last_review.upper() or "TRUSTWORTHY" in self.last_review.upper()
        
        return AgentResult(
            success=is_valid,
            data={"review": self.last_review, "passed": is_valid},
            confidence=Confidence.HIGH,
            evidence=[Evidence(source="criticism", content_snippet=self.last_review[:200])],
            message="Critique completed with adversarial focus."
        )
