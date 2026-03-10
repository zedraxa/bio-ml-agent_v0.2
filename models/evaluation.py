from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# --- GÖREV 1 & 4: Benchmarks & Scenarios ---
class EvalScenario(BaseModel):
    """Değerlendirme senaryosu (Örn: "Repo Editing - Easy")."""
    scenario_id: str
    name: str
    difficulty: str # "easy", "medium", "hard", "adversarial"
    domain: str # "coding", "browser", "bioinformatics"
    input_data: Dict[str, Any]
    expected_output: Optional[Any] = None
    constraints: List[str] = Field(default_factory=list)

class BenchmarkSuite(BaseModel):
    """Bir grup senaryodan oluşan benchmark seti."""
    suite_id: str
    name: str
    scenarios: List[EvalScenario]
    version: str

# --- GÖREV 2: Eval Metrics ---
class EvalMetricResult(BaseModel):
    """Bir görevin başarı metrikleri."""
    run_id: str
    scenario_id: str
    task_completion: float = Field(ge=0.0, le=1.0) # 0-1 arası
    groundedness: float = Field(ge=0.0, le=1.0)
    citation_faithfulness: float = Field(ge=0.0, le=1.0)
    code_correctness: Optional[float] = None
    cost_efficiency: float
    latency_ms: float
    additional_metrics: Dict[str, float] = Field(default_factory=dict)

# --- GÖREV 3: Regression Gates ---
class RegressionGateConfig(BaseModel):
    """Merge öncesi kalite kapısı kuralı."""
    gate_id: str
    metric_name: str
    threshold: float
    comparison_operator: str # "ge", "le", "gt", "lt"
    source_benchmark: str # BenchmarkSuite.suite_id
