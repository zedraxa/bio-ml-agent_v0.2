import logging
import json
from typing import List, Dict, Any, Optional
from PIL import Image, ImageDraw

log = logging.getLogger("browser.visual_evidence")

class VisualEvidenceGenerator:
    """
    VisualEvidenceGenerator: Gözlemleri görsel raporlara dönüştürür.
    - Screenshot üzerine bounding box çizer.
    - Önemli elemanları (POI) işaretler.
    - Denetim (Audit) dosyası oluşturur.
    """

    @staticmethod
    def draw_bboxes(image_path: str, elements: List[Dict[str, Any]], output_path: str):
        """Screenshot üzerine verilen elemanların rect'lerini çizer."""
        try:
            with Image.open(image_path) as img:
                draw = ImageDraw.Draw(img)

                for el in elements:
                    rect = el.get("rect")
                    if not rect: continue

                    x, y, w, h = rect['x'], rect['y'], rect['width'], rect['height']
                    # Draw rectangle
                    draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
                    # Label
                    label = f"{el.get('tag', '')}#{el.get('id', '')}"
                    draw.text((x, y - 10), label, fill="red")

                img.save(output_path)
                log.info(f"🖼️ Visual evidence saved to: {output_path}")
        except Exception as e:
            log.error(f"❌ Failed to generate visual evidence: {e}")

    @staticmethod
    def generate_audit_report(data: Dict[str, Any], output_path: str):
        """JSON formatında detaylı bir denetim raporu kaydeder."""
        try:
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=4)
            log.info(f"📊 Audit report saved to: {output_path}")
        except Exception as e:
            log.error(f"❌ Failed to save audit report: {e}")
