from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# --- GÖREV 1 & 2: Channel Adapters & New Channels ---
class ChannelType(str, Enum):
    WHATSAPP = "whatsapp"
    TELEGRAM = "telegram"
    DISCORD = "discord"
    EMAIL = "email"
    WEBHOOK = "webhook"
    WEB_DASHBOARD = "web_dashboard"

class ChannelAdapterConfig(BaseModel):
    """Herhangi bir iletişim kanalını sisteme bağlayan adaptör yapılandırması."""
    adapter_id: str
    channel_type: ChannelType
    is_active: bool = True
    credentials_secret_ref: str # Güvenli vault referansı (Token vb.)
    webhook_url: Optional[str] = None
    allowed_user_ids: List[str] = Field(default_factory=list)

# --- GÖREV 3: Kanal Politikaları ---
class ChannelPolicy(BaseModel):
    """Hangi kanalın ne kadar veriyi gösterebileceğini belirleyen kısıtlar."""
    policy_id: str
    channel_type: ChannelType
    allow_full_tool_output: bool = False
    allow_sensitive_artifacts: bool = False # Hasta verisi vb. mesajlaşma kanalına düşmesin
    max_message_length: int = 4096
    send_only_summaries: bool = True

# --- GÖREV 4: Notification Engine ---
class NotificationType(str, Enum):
    RUN_COMPLETED = "run_completed"
    APPROVAL_PENDING = "approval_pending"
    ERROR_OCCURRED = "error_occurred"
    BUDGET_EXCEEDED = "budget_exceeded"
    MODEL_TRAINED = "model_trained"

class NotificationEvent(BaseModel):
    """Notification motorunun dağıtımını yapacağı olay veri sözleşmesi."""
    notification_id: str
    target_user_id: str
    notification_type: NotificationType
    message_title: str
    message_body: str
    action_url: Optional[str] = None
    delivered_channels: List[ChannelType] = Field(default_factory=list)
    timestamp: str
