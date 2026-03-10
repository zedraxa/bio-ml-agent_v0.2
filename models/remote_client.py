from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# --- GÖREV 1: Remote Dashboard Summary ---
class DashboardModule(str, Enum):
    PROJECT_LIST = "project_list"
    ACTIVE_RUNS = "active_runs"
    LIVE_LOG = "live_log"
    ARTIFACT_BROWSER = "artifact_browser"
    TASK_QUEUE = "task_queue"
    COST_SUMMARY = "cost_summary"

class DashboardSummaryTemplate(BaseModel):
    """Ana dashboard üzerinde istemciye dönecek yüzeysel/hafif veri paketi."""
    dashboard_id: str
    user_id: str
    active_modules: List[DashboardModule] = Field(default_factory=list)
    total_cost_mtd: float # Ay başından bugüne bakiye
    pending_approvals_count: int = 0
    active_runs_count: int = 0

# --- GÖREV 2: Mobile/PWA Arayüz ---
class MobileActionType(str, Enum):
    QUICK_PROMPT = "quick_prompt"
    APPROVE_STEP = "approve_step"
    REJECT_STEP = "reject_step"
    DOWNLOAD_ARTIFACT = "download_artifact"
    DISMISS_NOTIFICATION = "dismiss_notification"

class MobileClientAction(BaseModel):
    """Telefondan gönderilen sınırlı yetkili komut. (PWA Command)"""
    action_id: str
    session_id: str
    action_type: MobileActionType
    target_run_id: Optional[str] = None
    target_artifact_id: Optional[str] = None
    input_text: Optional[str] = None
    device_fingerprint: str

# --- GÖREV 3: Read-only Hızlı Görünüm ---
class SharedReadonlyView(BaseModel):
    """Ekip arkadaşlarına atılan görüntüleme amaçlı sınırlı bağlantı sözleşmesi."""
    share_id: str
    target_project_id: str
    target_run_id: str
    created_by_user_id: str
    expires_at: Optional[str] = None
    is_live_tracking_enabled: bool = False # Misafir run'ı canlı görebilir mi?
    allowed_guest_emails: List[str] = Field(default_factory=list)
    secret_share_token: str

# --- GÖREV 4: Device Handoff ---
class HandoffState(str, Enum):
    INITIATED = "initiated"
    RECEIVED = "received"
    COMPLETED = "completed"
    EXPIRED = "expired"

class DeviceHandoffEvent(BaseModel):
    """Masaüstünde başlayan işin telefonda devam etmesini sağlayan etkinlik."""
    handoff_id: str
    source_device_id: str
    target_device_id: Optional[str] = None
    user_id: str
    current_run_id: str
    pending_approval_id: Optional[str] = None
    state: HandoffState = Field(default=HandoffState.INITIATED)
    created_at: str
    expires_at: str
