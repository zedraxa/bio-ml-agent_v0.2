import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

log = logging.getLogger("document.citation_mapper")

@dataclass
class CitationPoint:
    """Belge içindeki bir referans noktasını temsil eder."""
    page_number: int
    coordinates: Dict[str, float] # x, y, width, height
    text_snippet: str
    target_claim: Optional[str] = None

class CitationMapper:
    """
    CitationMapper: Metin parçalarını PDF koordinatları ile eşleştirir.
    - Kanıt (Evidence) üretiminde "Sayfa X, Sağ Üst Köşe" gibi detaylı bilgi sağlar.
    - Çapraz referansları (Cross-references) takip eder.
    """

    def __init__(self):
        self.points: List[CitationPoint] = []

    def add_point(self, page: int, x: float, y: float, w: float, h: float, text: str):
        point = CitationPoint(
            page_number=page,
            coordinates={"x": x, "y": y, "width": w, "height": h},
            text_snippet=text
        )
        self.points.append(point)
        log.debug(f"📍 Citation added: Page {page} -> {text[:30] if len(text) >= 30 else text}...")

    def find_best_evidence(self, query: str) -> Optional[CitationPoint]:
        """Query ile en iyi eşleşen kanıt noktasını bulur (Basit keyword match)."""
        for point in self.points:
            if query.lower() in point.text_snippet.lower():
                return point
        return None

    def serialize_citation(self, point: CitationPoint) -> str:
        """Kullanıcı dostu atıf metni üretir."""
        return f"(PDF Page {point.page_number} at x={point.coordinates['x']}, y={point.coordinates['y']})"
