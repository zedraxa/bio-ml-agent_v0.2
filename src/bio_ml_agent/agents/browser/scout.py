import logging
from typing import List, Dict, Any, Optional
from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from bio_ml_agent.agents.browser.dom_processor import DOMPruner
from bio_ml_agent.agents.browser.fingerprint import FingerprintDetector
from bio_ml_agent.agents.browser.autopilot import BrowserAutopilot
from pathlib import Path

log = logging.getLogger("browser.scout")

class BrowserScout(BaseSubAgent):
    """
    BrowserScout (Professional Grade):
    - Sayfayı derinlemesine analiz eder (Deep Perception).
    - Otonom temizlik (Autopilot) yapar.
    - Riskleri (Fingerprint) tespit eder ve raporlar.
    """
    
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        super().__init__("BrowserScout", model_name)
        self.page: Any = None
        self.last_pruned_dom: Dict[str, Any] = {}
        self.detected_risks: Dict[str, Any] = {}

    async def perceive(self, context: Dict[str, Any]) -> None:
        self.page = context.get("page")
        if not self.page:
            log.error("Page object is missing in context.")
            return

        # 1. Otonom Temizlik (Cookie banner vb.)
        autopilot = BrowserAutopilot(self.page)
        await autopilot.cleanup()

        # 2. Derin DOM Ekstraksiyonu
        try:
            script_path = Path(__file__).parent / "scripts" / "extract_dom.js"
            raw_dom = await self.page.evaluate(script_path.read_text())
            self.last_pruned_dom = DOMPruner.prune(raw_dom)
        except Exception as e:
            log.error(f"DOM extraction failed: {e}")

        # 3. Risk ve Fingerprint Analizi
        content = await self.page.content()
        title = await self.page.title()
        self.detected_risks = FingerprintDetector.detect_risks(content, title)
        
        risk_score = FingerprintDetector.get_risk_score(self.detected_risks)
        if risk_score > 0.5:
             log.warning(f"⚠️ High Risk Detected ({risk_score}): {self.detected_risks}")

    def plan(self, goal: str) -> List[str]:
        return ["Analyze page structure", "Identify potential interactive targets", "Sanitize environment"]

    async def act(self, step: str) -> Any:
        # Scout operasyonel eylem yapmaz, sadece gözlem sonucunu hazırlar.
        return "Observation ready"

    def summarize(self) -> AgentResult:
        success = self.detected_risks.get("access_denied") is False
        
        return AgentResult(
            success=success,
            data={
                "pruned_dom_size": len(str(self.last_pruned_dom)),
                "risks": self.detected_risks,
                "url": self.page.url if self.page else "unknown"
            },
            confidence=Confidence.HIGH if success else Confidence.LOW,
            evidence=[
                Evidence(source=self.page.url, content_snippet=f"Scanned page. Risks: {self.detected_risks}")
            ] if self.page else [],
            message="Deep page perception completed successfully."
        )
