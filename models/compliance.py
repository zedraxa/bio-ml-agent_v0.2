from enum import Enum
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field

# --- GÖREV 1: RBAC (Role Based Access Control) ---
class AccessRole(str, Enum):
    OWNER = "owner"
    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"
    GUEST = "guest"

class PermissionSet(BaseModel):
    """Belirli bir rol için izin tanımları."""
    can_read: bool = Field(default=True)
    can_write: bool = Field(default=False)
    can_execute_shell: bool = Field(default=False)
    can_access_secrets: bool = Field(default=False)
    can_publish_artifacts: bool = Field(default=False)
    can_modify_policies: bool = Field(default=False)

# --- GÖREV 2: Policy Templates ---
class PolicyMode(str, Enum):
    STUDENT = "student"
    LAB = "lab"
    ENTERPRISE = "enterprise"
    REGULATED_DATA = "regulated_data"
    FREEDOM = "freedom" # Sınırsız mod

class PolicyTemplate(BaseModel):
    """Çalışma modu ve davranış kuralları."""
    mode: PolicyMode
    enforce_hitl: bool = Field(default=True)
    max_tools_per_turn: int = Field(default=5)
    allowed_file_extensions: List[str] = Field(default_factory=lambda: [".py", ".md", ".txt", ".json"])
    forbidden_commands: List[str] = Field(default_factory=lambda: ["rm -rf /", "mkfs"])
    pii_masking_enabled: bool = Field(default=True)

# --- GÖREV 3: Consent Log ---
class ConsentRecord(BaseModel):
    """Kullanıcı onay mührü."""
    consent_id: str
    user_id: str
    action_type: str
    granted_at: str
    expires_at: Optional[str] = None
    scope: Dict[str, Any] = Field(default_factory=dict)
    digital_signature: str

# --- GÖREV 4 & 5: Violation & Degradation ---
class ViolationSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    BLOCKING = "blocking"

class ComplianceViolation(BaseModel):
    """Politika ihlal kaydı."""
    violated_rule_id: str
    policy_mode: PolicyMode
    agent_id: str
    detected_at: str
    severity: ViolationSeverity
    description: str
    remediation_action: str # Örn: "Switch to Read-Only"

class DegradationState(BaseModel):
    """İzin eksikliğinde kısıtlı çalışma steyti."""
    is_degraded: bool = Field(default=False)
    original_intent: str
    degration_reason: str
    active_limitations: List[str]
    suggested_fix: str
