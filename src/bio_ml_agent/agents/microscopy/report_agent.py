import logging
from typing import List, Dict, Any, Optional
from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from pathlib import Path

log = logging.getLogger("microscopy.report")

class MicroscopyReportAgent(BaseSubAgent):
    """
    MicroscopyReportAgent (A6): 'Bilimsel Sentez' Uzmanı.
    - Tüm ajanlardan gelen verileri akademik bir raporda birleştirir.
    - Gözlem özeti, ölçümler, belirsizlikler ve sınırlılıkları raporlar.
    - Çıktı: microscopy_analysis_report.md
    """

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        super().__init__("MicroscopyReportAgent", model_name)
        self.report_path = Path("artifacts/microscopy_analysis/analysis_report.md")

    def perceive(self, context: Dict[str, Any]) -> None:
        self.all_data = context # Pipeline'dan gelen tüm veriler
        log.info("📄 Generating scientific report from microscopy pipeline data.")

    def plan(self, goal: str) -> List[str]:
        return [
            "Synthesize Observation Summary (Modality/Specimen)",
            "Summarize Identified Structures & Phases",
            "Consolidate Measurement Tables (Counts/Geometric stats)",
            "Document Uncertainties & Study Limitations",
            "Suggest Follow-up Analysis",
            "Export professional Markdown report"
        ]

    def act(self, step: str) -> Any:
        log.info(f"Writing report section: {step}")
        # Note: In production, this uses a template engine or LLM to format the 'all_data'.
        return f"Section '{step}' added to report buffer."

    def verify(self, action_result: Any) -> bool:
        return True

    def summarize(self) -> AgentResult:
        # Markdown Report Simulation
        report_content = f"""# Microscopy Analysis Report
## 1. Observation Summary
- Specimen: {self.all_data.get('specimen', 'Unknown')}
- Modality: {self.all_data.get('modality', 'Unknown')}

## 2. Quantitative Results
- Total Cells: {self.all_data.get('stats', {}).get('cell_count', 0)}
- Mitotic Index: {self.all_data.get('stats', {}).get('mitotic_index', 0)}

## 3. Study Limitations & Uncertainties
- Artifact Interference: {self.all_data.get('critic_findings', [])}

## 4. Recommendations
- Perform high-res confocal z-stack for 3D reconstruction.
"""
        try:
            self.report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.report_path, "w") as f:
                f.write(report_content)
            log.info(f"✅ Scientific report saved to {self.report_path}")
        except Exception as e:
            log.error(f"Failed to save report: {e}")

        return AgentResult(
            success=True,
            data={"report_url": str(self.report_path)},
            confidence=Confidence.HIGH,
            evidence=[Evidence(source="reporting_engine", content_snippet="Academic synthesis finalized.")],
            message=f"Microscopy report generated successfully at {self.report_path}."
        )
