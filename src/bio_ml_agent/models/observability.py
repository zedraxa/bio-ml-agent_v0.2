from enum import Enum
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field

# --- GÖREV 1 & 2: OTel Span Modeli & Trace Viewer ---
class SpanType(str, Enum):
    PLAN = "plan"
    TOOL = "tool"
    BROWSER = "browser"
    MODEL_CALL = "model_call"
    RETRIEVAL = "retrieval"
    APPROVAL = "approval"
    WORKFLOW = "workflow"

class ObservabilitySpan(BaseModel):
    """OpenTelemetry uyumlu izleme (trace) span'i."""
    span_id: str
    trace_id: str
    parent_span_id: Optional[str] = None
    name: str
    span_type: SpanType
    start_time: str # ISO format
    end_time: Optional[str] = None
    status: str = Field(default="unset") # "ok", "error", "unset"
    attributes: Dict[str, Any] = Field(default_factory=dict) # latency, cost, tokens, etc.
    events: List[Dict[str, Any]] = Field(default_factory=list) # logs, screenshots
    error_message: Optional[str] = None

class TraceTree(BaseModel):
    """Hiyerarşik trace görünümü."""
    root_span: ObservabilitySpan
    children: List['TraceTree'] = Field(default_factory=list)
    total_latency_ms: float
    total_cost: float = 0.0
    total_tokens: int = 0

# --- GÖREV 3: Run Diff ---
class RunComparison(BaseModel):
    """İki AgentRun arasındaki farkların analizi."""
    comparison_id: str
    run_id_a: str
    run_id_b: str
    diff_summary: str
    metric_deltas: Dict[str, float] # accuracy, cost, latency change
    changed_spans: List[str] # Değişen span_id listesi
    findings: List[str]

# --- GÖREV 4: Alerting ---
class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class SystemAlert(BaseModel):
    """Sistem genelinde tetiklenen uyarılar."""
    alert_id: str
    severity: AlertSeverity
    category: str # "cost", "crash", "latency", "plugin"
    message: str
    timestamp: str
    context: Dict[str, Any] = Field(default_factory=dict)
    is_resolved: bool = Field(default=False)
