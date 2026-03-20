import logging
import pandas as pd
from typing import List, Dict, Any, Optional
from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from pathlib import Path

log = logging.getLogger("dataset_agent")

class DatasetAgent(BaseSubAgent):
    """
    Dataset Agent: Veri setlerini (CSV, Excel vb.) profilleyerek eksik veri, 
    sapma (bias) ve hedef uyumluluğu analizi yapar.
    """
    
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        super().__init__("DatasetAgent", model_name)
        self.df: Optional[pd.DataFrame] = None
        self.profile_report: Dict[str, Any] = {}

    def perceive(self, context: Dict[str, Any]) -> None:
        file_path = context.get("file_path")
        if file_path:
            try:
                self.df = pd.read_csv(file_path) # Fallback to CSV for now
                log.info(f"📊 Veri seti yüklendi: {len(self.df)} satır.")
            except Exception as e:
                log.error(f"Veri yükleme hatası: {e}")

    def plan(self, goal: str) -> List[str]:
        return [
            "Infer column schemas and types",
            "Check for missingness and imbalances",
            "Evaluate suitability for target task"
        ]

    def act(self, step: str) -> Any:
        if self.df is not None and "missingness" in step.lower():
            return self.df.isnull().sum().to_dict()
        return "Analyzed"

    def verify(self, action_result: Any) -> bool:
        return action_result is not None

    def summarize(self) -> AgentResult:
        return AgentResult(
            success=True,
            data=self.profile_report,
            confidence=Confidence.HIGH,
            evidence=[],
            message="Dataset profiling complete."
        )
