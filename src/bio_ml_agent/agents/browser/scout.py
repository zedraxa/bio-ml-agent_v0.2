import logging
from typing import List, Dict, Any, Optional
from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from bio_ml_agent.agents.browser.dom_processor import DOMPruner
from bio_ml_agent.agents.browser.fingerprint import FingerprintDetector
from bio_ml_agent.agents.browser.visual_evidence import VisualEvidenceGenerator
from bio_ml_agent.agents.browser.autopilot import BrowserAutopilot
from pathlib import Path
import time

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
        
        # 4. Görsel Kanıt ve Denetim (Hardening)
        screenshot_path = f"artifacts/scout_shot_{int(time.time())}.png"
        await self.page.screenshot(path=screenshot_path)
        
        ev_gen = VisualEvidenceGenerator()
        # Sadece en önemli 5 elemanı işaretle (POI)
        poi = self.last_pruned_dom.get("children", [])[:5]
        ev_gen.draw_bboxes(screenshot_path, poi, screenshot_path.replace(".png", "_audit.png"))
        
        # Denetim Raporu
        ev_gen.generate_audit_report({
            "url": self.page.url,
            "risks": self.detected_risks,
            "dom_summary": str(self.last_pruned_dom)[:200]
        }, f"artifacts/audit_{int(time.time())}.json")

    def plan(self, goal: str) -> List[str]:
        return ["Analyze page structure", "Identify potential interactive targets", "Sanitize environment"]

    async def act(self, step: str) -> Any:
        # Scout operasyonel eylem yapmaz, sadece gözlem sonucunu hazırlar.
        return "Observation ready"

    def verify(self, action_result: Any) -> bool:
        return action_result == "Observation ready"

    def summarize(self) -> AgentResult:
        success = self.detected_risks.get("access_denied") is False
        
        return self.create_result(
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
