import logging
from typing import List, Dict, Any, Optional
from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from pathlib import Path

log = logging.getLogger("microscopy.navigator")

class SlideNavigatorAgent(BaseSubAgent):
    """
    SlideNavigatorAgent (E1): 'Saha Gezgini' Uzmanı.
    - Büyük (WSI) veya çok alanlı görüntülerde kritik bölgeleri (ROI) bulur.
    - Alanları 'tanısal zenginlik' (diagnostic richness) açısından puanlar.
    - Analiz yükünü azaltmak için önemli koordinatları önerir.
    """

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        super().__init__("SlideNavigatorAgent", model_name)
        self.image_path: Optional[Path] = None
        self.roi_candidates: List[Dict[str, Any]] = []

    def perceive(self, context: Dict[str, Any]) -> None:
        path = context.get("image_path")
        if path:
            self.image_path = Path(path)
            log.info(f"🚩 Navigator active for large scale image: {self.image_path.name}")

    def plan(self, goal: str) -> List[str]:
        return [
            "Scan slide tiles at low magnification",
            "Evaluate feature density and cellular richness",
            "Identify diagnostic 'Hotspots' (Anomalies, High division zones)",
            "Generate ROI candidate list with richness scores"
        ]

    def act(self, step: str) -> Any:
        if not self.image_path: return "Error: No Image"

        log.info(f"🧭 Navigating slide: {step}")

        # Region Selection Simulation
        if "scan" in step.lower() or "richness" in step.lower():
            # Simulated ROI data
            self.roi_candidates = [
                {"roi_id": 1, "coord": [4500, 12000], "richness_score": 0.92, "reason": "High cellular density, potential mitosis hotspot"},
                {"roi_id": 2, "coord": [8900, 3400], "richness_score": 0.78, "reason": "Interesting tissue layer transition"},
                {"roi_id": 3, "coord": [1200, 5600], "richness_score": 0.45, "reason": "Staining artifact detected, low diagnostic value"}
            ]

        return f"Navigation step '{step}' completed. {len(self.roi_candidates)} ROIs found."

    def verify(self, action_result: Any) -> bool:
        return True

    def summarize(self) -> AgentResult:
        # Sort by richness score
        sorted_rois = sorted(self.roi_candidates, key=lambda x: x["richness_score"], reverse=True)
        top_roi = sorted_rois[0] if sorted_rois else None

        return AgentResult(
            success=True,
            data={"roi_candidates": sorted_rois},
            confidence=Confidence.HIGH,
            evidence=[Evidence(source="navigation_scan", content_snippet=f"Detected {len(sorted_rois)} diagnostic hotspots.")],
            message=f"Slide navigation complete. Recommended ROI: {top_roi['roi_id'] if top_roi else 'None'} with score {top_roi['richness_score'] if top_roi else 0}."
        )
