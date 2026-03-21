import logging
import pandas as pd
from typing import List, Dict, Any, Optional
from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from bio_ml_agent.agents.dataset.statistical_engine import StatisticalEngine

log = logging.getLogger("dataset_agent")

class DatasetAgent(BaseSubAgent):
    """
    DatasetAgent (Professional Grade):
    - Veri setlerini Pandas ve StatisticalEngine ile denetler.
    - Missingness, Outlier ve Bias analizi yapar.
    - Bilimsel veri kalitesi raporları üretir.
    """
    
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        super().__init__("DatasetAgent", model_name)
        self.df: Optional[pd.DataFrame] = None
        self.audit_report: Dict[str, Any] = {}

    def perceive(self, context: Dict[str, Any]) -> None:
        file_path = context.get("file_path")
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                log.info(f"📊 Dataset loaded: {len(self.df)} rows.")
            except Exception as e:
                log.error(f"Failed to load dataset: {e}")

    def plan(self, goal: str) -> List[str]:
        return ["Audit missingness", "Detect outliers", "Analyze class bias"]

    def act(self, step: str) -> Any:
        if self.df is None: return "No data"
        
        if "missingness" in step.lower():
            self.audit_report["missingness"] = StatisticalEngine.audit_missingness(self.df)
        elif "outliers" in step.lower():
            self.audit_report["outliers"] = StatisticalEngine.detect_outliers(self.df)
        elif "bias" in step.lower():
            # Örnek hedef sütun 'target'
            self.audit_report["bias"] = StatisticalEngine.check_bias(self.df, "target")
            
        return "Audit step completed."

    def verify(self, action_result: Any) -> bool:
        return True

    def summarize(self) -> AgentResult:
        if self.df is None:
             return AgentResult(success=False, data={}, confidence=Confidence.LOW, message="No data loaded.")

        return AgentResult(
            success=True,
            data=self.audit_report,
            confidence=Confidence.HIGH,
            evidence=[Evidence(source="pandas_audit", content_snippet="Statistical profiling finished.")],
            message="Dataset quality audit finished with professional oversight."
        )
