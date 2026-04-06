import logging
from typing import List, Dict, Any, Optional
from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from pathlib import Path

log = logging.getLogger("microscopy.annotator")

class AnnotationSuggestionAgent(BaseSubAgent):
    """
    AnnotationSuggestionAgent (E4): 'Yarı-Otomatik Etiketleme' Uzmanı.
    - Görüntüdeki yapılar için potansiyel etiketler önerir (Örn: 'Şunlar metaphase olabilir').
    - Güven skoru düşük olan (uncertain) bölgeleri bilerek 'Human Review' (İnsan Onayı) için işaretler.
    - Patologlar ve Biyomühendisler için bir 'Copilot' gibi çalışır.
    """

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        super().__init__("AnnotationSuggestionAgent", model_name)
        self.image_path: Optional[Path] = None
        self.suggestions: List[Dict[str, Any]] = []

    def perceive(self, context: Dict[str, Any]) -> None:
        path = context.get("image_path")
        if path:
            self.image_path = Path(path)
            log.info(f"✍️ Annotation Copilot online for: {self.image_path.name}")

    def plan(self, goal: str) -> List[str]:
        return [
            "Perform rapid preliminary scan of structures",
            "Generate high-confidence label suggestions (e.g., 'Metaphase')",
            "Identify anomalous or ambiguous zones",
            "Flag uncertain regions specifically for 'Human Review'",
            "Export suggestions in a machine-readable format (JSON/GeoJSON)"
        ]

    def act(self, step: str) -> Any:
        if not self.image_path: return "Error: No Image"

        log.info(f"💡 Generating suggestion for step: {step}")

        # Copilot Logic Simulation
        if "high-confidence" in step.lower() or "preliminary" in step.lower():
            self.suggestions.append({
                "loc": [1200, 3400],
                "suggested_label": "Metaphase",
                "confidence": 0.89,
                "note": "Clear chromosomal alignment detected."
            })
            self.suggestions.append({
                "loc": [450, 890],
                "suggested_label": "Necrotic Region",
                "confidence": 0.75,
                "note": "Loss of membrane integrity and pyknotic nuclei."
            })

        elif "ambiguous" in step.lower() or "human review" in step.lower():
            self.suggestions.append({
                "loc": [2100, 1500],
                "suggested_label": "UNKNOWN_ARTIFACT_OR_CELL",
                "confidence": 0.35,
                "note": "REQUIRE HUMAN REVIEW: Ambiguous stain aggregation vs cluster."
            })

        return f"Annotation step '{step}' completed."

    def verify(self, action_result: Any) -> bool:
        return True

    def summarize(self) -> AgentResult:
        review_count = sum(1 for s in self.suggestions if "HUMAN REVIEW" in s.get("note", ""))

        return AgentResult(
            success=True,
            data={"suggestions": self.suggestions, "needs_human_review": review_count},
            confidence=Confidence.HIGH, # Confident in its assessment of uncertainty
            evidence=[Evidence(source="annotation_copilot", content_snippet=f"Generated {len(self.suggestions)} suggestions.")],
            message=f"Annotation hints generated. {review_count} regions explicitly flagged for Human Review."
        )
