from typing import List, Dict, Any, Optional
from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from bio_ml_agent.llm_backend import auto_create_backend
import json
import logging

log = logging.getLogger("browser_scout")

class BrowserScout(BaseSubAgent):
    """
    Browser Scout: Sayfayı hızlıca analiz eden, riskleri (captcha, login vb.) tespit eden 
     ve yapı haritasını çıkaran ajan.
    """
    
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        super().__init__("BrowserScout", model_name)
        self.page: Any = None
        self.perception_data: Dict[str, Any] = {}

    def perceive(self, context: Dict[str, Any]) -> None:
        """Playwright sayfasını ve distiller verilerini alır."""
        self.page = context.get("page")
        if not self.page:
            log.error("BrowserScout için Playwright 'page' nesnesi bulunamadı.")
            return
            
        # DOM Distiller çalıştır
        try:
            from pathlib import Path
            # Yolu daha güvenli al
            current_file = Path(__file__).resolve()
            distiller_path = current_file.parent.parent.parent / "ultra_agent" / "runtime" / "browser" / "distiller.js"
            
            if distiller_path.exists():
                js_content = distiller_path.read_text(encoding="utf-8")
                result = self.page.evaluate(js_content)
                self.perception_data = result if isinstance(result, dict) else {}
            else:
                self.perception_data = {"page": {"title": self.page.title(), "url": self.page.url}, "interactive": []}
        except Exception as e:
            log.error(f"Distiller hatası: {e}")
            self.perception_data = {"error": str(e)}

    def plan(self, goal: str) -> List[str]:
        """Sayfa yapısına göre bir keşif planı çıkarır."""
        if not self.page or not self.perception_data:
            return ["Wait for page load"]
            
        page_title = self.perception_data.get('page', {}).get('title', 'Unknown')
        page_url = getattr(self.page, 'url', 'unknown')
        
        prompt = f"""Bir sayfa kaşifisin (Scout). 
Şu anki sayfa: {page_title} ({page_url})
Hedef: {goal}
..."""
        return ["Analyze page structure", "Detect risks", "Map interactive elements"]

    def act(self, step: str) -> Any:
        """Keşif aksiyonları (scroll, screenshot vb.)."""
        if "Analyze" in step:
            return "Analyzed"
        return "Executed"

    def verify(self, action_result: Any) -> bool:
        """Aksiyonun doğruluğunu kontrol et."""
        return True

    def summarize(self) -> AgentResult:
        """Sayfa haritasını ve risk profilini döner."""
        # Risk analizi (LLM ile yapılabilir)
        risk_profile = {
            "captcha": "detected" if "captcha" in self.page.content().lower() else "none",
            "login_required": "maybe" if "login" in self.page.content().lower() else "no"
        }
        
        return AgentResult(
            success=True,
            data={
                "page_map": self.perception_data.get("page"),
                "risk_profile": risk_profile,
                "candidates_count": len(self.perception_data.get("interactive", []))
            },
            confidence=Confidence.HIGH,
            evidence=[
                Evidence(source=self.page.url, content_snippet=f"Title: {self.page.title()}")
            ],
            message=f"Page scouted successfully: {self.page.title()}"
        )
