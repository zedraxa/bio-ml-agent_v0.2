from enum import Enum
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field

# --- GÖREV 1: Action Approval Matrix ---
class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ActionType(str, Enum):
    SHELL_EXECUTION = "shell_execution"
    FILE_OVERWRITE = "file_overwrite"
    EXTERNAL_PUBLISH = "external_publish"
    EMAIL_SEND = "email_send"
    CLOUD_SPEND = "cloud_spend"
    DESTRUCTIVE_DELETE = "destructive_delete"
    CODE_MODIFICATION = "code_modification"

class ActionApprovalPolicy(BaseModel):
    """Eylem bazlı onay politikası."""
    action_type: ActionType
    risk_level: RiskLevel
    requires_approval: bool = Field(default=True)
    auto_approve_threshold: float = Field(default=0.0, description="Güven puanı bu değerin üzerindeyse onay isteme.")
    notification_required: bool = Field(default=True)

# --- GÖREV 2: Interrupt/Resume Mekanizması ---
class HITLResponseType(str, Enum):
    APPROVE = "approve"
    DENY = "deny"
    EDIT = "edit"
    ALTERNATIVE = "alternative"
    ABORT_ALL = "abort_all"

class HITLResponse(BaseModel):
    """Kullanıcının müdahale yanıtı."""
    response_type: HITLResponseType
    modified_action: Optional[Any] = None
    alternative_command: Optional[str] = None
    reason: Optional[str] = None
    responder_id: str
    responded_at: str

# --- GÖREV 3: Review UI Sözleşmesi ---
class SecurityReviewRequest(BaseModel):
    """Kullanıcıya sunulacak güvenlik incelemesi detayı."""
    request_id: str
    action_type: ActionType
    proposed_action: Dict[str, Any]
    intent_explanation: str = Field(..., description="Ajanın bu eylemi neden yapmak istediğine dair açıklaması.")
    expected_impact: str
    risk_assessment: str
    rollback_supported: bool = Field(default=False)
    rollback_plan: Optional[str] = None
    timeout_seconds: int = Field(default=300)
