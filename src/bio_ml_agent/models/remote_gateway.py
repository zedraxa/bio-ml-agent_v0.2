from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# --- GÖREV 1: API Gateway Katmanı ---
class GatewayRequestLog(BaseModel):
    """API gateway üzerinden geçen remotel requestlerin logu."""
    request_id: str
    ip_address: str
    device_id: Optional[str] = None
    endpoint: str
    rate_limit_hits: int = 0
    is_signed: bool = False
    reverse_proxy_target: Optional[str] = None
    timestamp: str

# --- GÖREV 2: Kimlik Doğrulama (Auth) ---
class AuthMethod(str, Enum):
    EMAIL_PASSWORD = "email_password"
    MAGIC_LINK = "magic_link"
    OAUTH = "oauth"

class DeviceSession(BaseModel):
    """Cihaz oturumu nesnesi (telefon, tarayıcı vs.)."""
    device_id: str
    device_name: str
    last_active: str
    ip_address: str
    is_current: bool = False

class AuthSession(BaseModel):
    """Kullanıcı token ve doğrulama oturumu."""
    session_id: str
    user_id: str
    auth_method: AuthMethod
    access_token: str
    refresh_token: str
    expires_at: str
    device_sessions: List[DeviceSession] = Field(default_factory=list)

# --- GÖREV 3: Remote Session Broker ---
class RemoteSessionRegistry(BaseModel):
    """Kullanıcının aktif paneli, runları ve devam edebilme (resume) durumu."""
    registry_id: str
    user_id: str
    active_run_ids: List[str] = Field(default_factory=list)
    pending_approval_ids: List[str] = Field(default_factory=list)
    latest_artifact_ids: List[str] = Field(default_factory=list)
    last_connected_at: str
    resume_context_point: Optional[str] = None # Kaldığı yerden bağlanmak için context tutucu

# --- GÖREV 4: WebSocket/SSE Event Akışı ---
class EventStreamType(str, Enum):
    THINKING = "thinking"
    TOOL_START = "tool_start"
    TOOL_OUTPUT = "tool_output"
    ARTIFACT_READY = "artifact_ready"
    HUMAN_APPROVAL = "human_approval"
    ERROR_SIGNAL = "error_signal"

class StreamEvent(BaseModel):
    """WebSocket/SSE üzerinden istemcilere push edilen canlı durum olayı."""
    event_id: str
    session_id: str
    event_type: EventStreamType
    payload: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str
