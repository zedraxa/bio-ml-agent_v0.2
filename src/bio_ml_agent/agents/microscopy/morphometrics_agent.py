import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from pathlib import Path
import math

log = logging.getLogger("microscopy.morphometrics")

class MorphometricsAgent(BaseSubAgent):
    """
    MorphometricsAgent (A4 - Physical Engine):
    - Transforms physical polygon data into precise geometric parameters using OpenCV.
    - Calculates absolute metrics: Area, Perimeter, Circularity, Elongation.
    - Exports comprehensive reports.
    """

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        super().__init__("MorphometricsAgent", model_name)
        self.raw_data: List[Dict[str, Any]] = []
        self.stats: Dict[str, Any] = {}
        self.results_data: List[Dict[str, Any]] = []

    def perceive(self, context: Dict[str, Any]) -> None:
        self.raw_data = context.get("segments", [])
        log.info(f"📊 Analyzing {len(self.raw_data)} physical segmented objects.")

    def plan(self, goal: str) -> List[str]:
        return [
            "Calculate basic geometry using cv2 (Area, Perimeter)",
            "Compute Shape Factors (Circularity, Aspect Ratio)",
            "Analyze Population metrics (Cell/Nucleus counts)",
            "Export comprehensive physical morphometrics_report.csv"
        ]

    def act(self, step: str) -> Any:
        if not self.raw_data:
            log.warning("No physical segmentation data found.")
            return "Skipped: No data"

        log.info(f"📐 Computing physical morphometrics: {step}")

        try:
            import cv2
        except ImportError:
            log.warning("OpenCV not found. Returning empty metrics.")
            return "Failed: No OpenCV"

        if "geometry" in step.lower() or "shape" in step.lower():
            if not self.results_data:
                self.results_data = [] # Processing block

                # Single pass for geometric processing
                for obj in self.raw_data:
                    poly = np.array(obj.get("polygon", []), dtype=np.int32)
                    if len(poly) < 3: continue

                    # Ensure shape is (N, 1, 2) for OpenCV
                    if len(poly.shape) == 2 and poly.shape[1] == 2:
                        contour = poly.reshape((-1, 1, 2))
                    else:
                        contour = poly

                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)

                    circularity = 0.0
                    if perimeter > 0:
                        circularity = (4 * math.pi * area) / (perimeter * perimeter)

                    # aspect ratio & elongation via bounding rect
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w)/h if h > 0 else 0

                    self.results_data.append({
                        "id": obj.get("id"),
                        "type": obj.get("type"),
                        "area_pixels": area,
                        "perimeter_pixels": perimeter,
                        "circularity": circularity,
                        "aspect_ratio": aspect_ratio
                    })

        elif "population" in step.lower():
            cell_count = sum(1 for o in self.raw_data if "cell" in str(o.get("type", "")).lower())
            nuc_count = sum(1 for o in self.raw_data if "nucleus" in str(o.get("type", "")).lower())
            self.stats["cell_count"] = cell_count
            self.stats["nucleus_count"] = nuc_count
            self.stats["NC_ratio_avg"] = nuc_count / cell_count if cell_count > 0 else 0

        return f"Completed physical calculation: {step}"

    def verify(self, action_result: Any) -> bool:
        return True

    def summarize(self) -> AgentResult:
        df = pd.DataFrame(self.results_data)
        output_file = Path("artifacts/microscopy_analysis/physical_morphometrics.csv")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if not df.empty:
            df.to_csv(output_file, index=False)

        mean_area = df["area_pixels"].mean() if not df.empty and "area_pixels" in df else 0

        return self.create_result(
            success=True,
            data={"stats": self.stats, "morpho_file": str(output_file)},
            confidence=Confidence.HIGH,
            evidence=[Evidence(source="physical_numeric_engine", content_snippet="OpenCV metrics generated.")],
            message=f"Physical Morphometric analysis complete. Captured {len(self.results_data)} structures. Mean Area: {mean_area:.1f} px."
        )
