from enum import Enum
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field

# --- GÖREV 1 & 2: Durable Workflow Engine ---
class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    TIMED_OUT = "timed_out"

class DurableWorkflow(BaseModel):
    """Kalıcı ve dayanıklı iş akışı modeli."""
    workflow_id: str
    name: str
    status: WorkflowStatus = Field(default=WorkflowStatus.PENDING)
    input_data: Dict[str, Any] = Field(default_factory=dict)
    state_snapshot: Dict[str, Any] = Field(default_factory=dict) # Resumable state
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_context: Optional[str] = None

class ChildWorkflow(BaseModel):
    """Alt workflow yapısı (Örn: Browser, Training)."""
    child_id: str
    parent_workflow_id: str
    workflow_type: str # "browser", "training", "literature", "report"
    status: WorkflowStatus
    result_artifact_path: Optional[str] = None

# --- GÖREV 3: Retry Policies ---
class ErrorType(str, Enum):
    TRANSIENT = "transient" # Yeniden denenebilir (Network vb.)
    TERMINAL = "terminal" # Denenmemeli (Logic/Auth error)

class RetryPolicy(BaseModel):
    """Hata türüne göre yeniden deneme politikası."""
    policy_id: str
    max_retries: int
    backoff_factor: float = 2.0
    retryable_error_types: List[ErrorType]
    timeout_seconds: int

# --- GÖREV 4: Schedules ---
class WorkflowSchedule(BaseModel):
    """Zamanlanmış görevler (Cron vb.)."""
    schedule_id: str
    workflow_template_id: str
    cron_expression: str # "0 0 * * *"
    next_run_at: str
    is_active: bool = Field(default=True)

# --- GÖREV 5: Manual Intervention ---
class WorkflowCheckpoint(BaseModel):
    """İnsan müdahalesi (HITL) duraklama noktası."""
    checkpoint_id: str
    workflow_id: str
    reason: str
    required_action: str # "approve", "edit_params", "provide_secret"
    is_resolved: bool = Field(default=False)
    resolved_at: Optional[str] = None
