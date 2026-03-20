from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum
from datetime import datetime

# --- 1. Run Timeline & Tool Spans ---
class SpanType(str, Enum):
    THINKING = "thinking"
    TOOL_EXECUTION = "tool_execution"
    BROWSER_ACTION = "browser_action"
    HTTP_REQUEST = "http_request"
    HUMAN_INTERACTION = "human_interaction"

class TraceSpan(BaseModel):
    """Bir problemin çözümündeki tekil adımları kaydeden izleme (O-Tel span)."""
    span_id: str
    run_id: str
    parent_span_id: Optional[str] = None
    span_type: SpanType
    name: str  # Örn: "Search PubMed", "Evaluate Code"
    status: str = "ok" # "ok", "error"
    started_at: datetime
    ended_at: Optional[datetime] = None
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None

class RunTimeline(BaseModel):
    """Bir Run (çalıştırma) boyunca üretilen tüm span'ların silsilesi."""
    timeline_id: str
    run_id: str
    spans: List[TraceSpan] = Field(default_factory=list)
    total_duration_ms: int = 0
    created_at: datetime

# --- 2. Browser Replay & Screenshots ---
class BrowserTimelineEvent(BaseModel):
    """Tarayıcıdaki her bir DOM Snapshot veya ekran görüntüsünün paket hali."""
    event_id: str
    run_id: str
    span_id: Optional[str] = None
    url: str
    action_type: str # "click", "type", "navigate", "error_snapshot"
    screenshot_ref: Optional[str] = None # Storage bucket artifact ID'si
    dom_diff: Optional[Dict[str, Any]] = None
    timestamp: datetime

class FailureReplaySession(BaseModel):
    """Sistem çöktüğünde veya ajan takıldığında izlenecek "Hata Oynatma" objesi."""
    replay_id: str
    run_id: str
    failure_reason: str
    events: List[BrowserTimelineEvent] = Field(default_factory=list)
    recorded_at: datetime

# --- 3. Cloud Cost Trace ---
class CostCategory(str, Enum):
    COMPUTE_CPU = "compute_cpu"
    COMPUTE_GPU = "compute_gpu"
    LLM_TOKEN = "llm_token"
    STORAGE = "storage"
    NETWORK = "network"

class CloudCostEntry(BaseModel):
    """Herhangi bir eylemin (Span) arkasında yatan tahmini/kesin bulut faturası girdisi."""
    cost_id: str
    run_id: str
    span_id: Optional[str] = None
    category: CostCategory
    resource_provider: str # "openai", "aws_ec2", "anthropic"
    amount_usd: float
    usage_metric: str # "1024 tokens", "2 GPU hours"
    timestamp: datetime

class ProjectBillingTrace(BaseModel):
    """Proje veya spesifik run bazında toplanmış tüketim fişi."""
    trace_id: str
    project_id: str
    total_cost_usd: float = 0.0
    entries: List[CloudCostEntry] = Field(default_factory=list)
    generated_at: datetime

# --- 4. Per-user Audit & Approvals ---
class SystemActionType(str, Enum):
    PROJECT_CREATE = "project_create"
    ROLE_ASSIGN = "role_assign"
    SETTINGS_CHANGE = "settings_change"
    TASK_APPROVE = "task_approve"
    TASK_REJECT = "task_reject"

class UserAuditLog(BaseModel):
    """Sistemde HANGİ insan HANGİ ayarlara ne kararı verdi? (Onay geçmişi)"""
    audit_id: str
    user_id: str
    action_type: SystemActionType
    target_resource_id: str # Hangi projeye / taska yapıldı
    context_snapshot: Dict[str, Any] = Field(default_factory=dict)
    ip_address: Optional[str] = None
    timestamp: datetime

# --- 5. Secret Access Receipts ---
class SecretAccessReceipt(BaseModel):
    """Kasadan yetki kullanılarak çekilmiş gizli bilgilerin erişim makbuzu."""
    receipt_id: str
    secret_name: str
    accessed_by_user: Optional[str] = None
    accessed_by_agent: Optional[str] = None  # Örn: role "coder" in run-123
    run_id: str
    justification: str # Neden çekildi? (Örn: "For AWS S3 upload task")
    accessed_at: datetime
    is_authorized: bool = True
