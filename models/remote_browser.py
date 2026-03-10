from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# --- GÖREV 1: Browser Live Stream ---
class LiveBrowserStreamFrame(BaseModel):
    """Tarayıcıdaki her step'in canlı izleme kanalına düşen JSON veri paketi."""
    frame_id: str
    session_id: str
    step_number: int
    screenshot_base64_or_url: str
    target_dom_element_xpath: Optional[str] = None
    target_bounding_box: Optional[Dict[str, int]] = None # {"x": 10, "y": 20, "width": 100, "height": 50}
    action_type: str # "click", "type", "scroll"
    timestamp: str

# --- GÖREV 2: Takeover Mode ---
class TakeoverStatus(str, Enum):
    AGENT_IN_CONTROL = "agent"
    HUMAN_IN_CONTROL = "human"
    TRANSITIONING = "transitioning"

class BrowserTakeoverEvent(BaseModel):
    """Ajanın web gezintisini kullanıcının devralmasını temsil eden olay."""
    event_id: str
    session_id: str
    requested_by_user_id: str
    status: TakeoverStatus
    reason_for_takeover: Optional[str] = None
    timestamp: str

# --- GÖREV 3: Approval Required Actions ---
class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical" # Ödeme formları vb.

class RiskActionApprovalState(BaseModel):
    """Bir butona tıklayacağı veya form dolduracağı zaman insan onayı isteyen durum tutucu."""
    state_id: str
    session_id: str
    detected_action: str # "form_submit", "payment_click"
    dom_target_context: Dict[str, Any] # "Form labels, input values"
    risk_level: RiskLevel
    is_approved: Optional[bool] = None # None: Bekliyor, True: Onaylandı, False: Reddedildi
    timeout_seconds: int = 120

# --- GÖREV 4: Session Recording (Timeline Replay) ---
class TimelineSnapshot(BaseModel):
    """Sonradan oynatabilmek için kaydedilen zaman çizelgesi anlık görüntüsü."""
    snapshot_id: str
    step_index: int
    url: str
    screenshot_ref: str
    dom_changes_diff: str # HTML delta
    agent_reasoning: str # "I clicked this because..."
    timestamp_offset_ms: int

class SessionReplayTimeline(BaseModel):
    """Kaydedilmiş tüm oturum verisinin yeniden oynatılabilir veri bloğu."""
    timeline_id: str
    session_id: str
    total_duration_ms: int
    snapshots: List[TimelineSnapshot] = Field(default_factory=list)
