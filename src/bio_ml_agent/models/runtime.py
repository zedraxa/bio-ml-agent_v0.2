from enum import Enum
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field

class MessageType(str, Enum):
    PLAN = "plan"
    ACTION = "action"
    OBSERVATION = "observation"
    CRITIQUE = "critique"
    APPROVAL_REQUEST = "approval_request"
    ARTIFACT_READY = "artifact_ready"
    FAILURE_RECOVERY = "failure_recovery"

class MessageContract(BaseModel):
    """Tek tip mesajlaşma sözleşmesi."""
    type: MessageType
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AgentRun(BaseModel):
    """Tek tip AgentRun veri modeli."""
    run_id: str
    parent_run_id: Optional[str] = None
    workspace_id: str
    project_id: str
    task_id: Optional[str] = None
    step_id: Optional[str] = None

    tool_call: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    artifact: Optional[Dict[str, Any]] = None

    # Telemetri
    cost: float = Field(default=0.0)
    token: int = Field(default=0)
    latency: float = Field(default=0.0)

# --- GÖREV 4: Tool ABI Standardı ---

class SideEffectClass(str, Enum):
    """Aracın sistem üzerindeki yan etki sınıfları."""
    READ_ONLY = "read_only"       # Sadece okur, zararsız
    STATE_MUTATION = "mutation"   # Sistem durumunu/objelerini değiştirir
    EXTERNAL_CALL = "external"    # Dış servislere istek atar (API)
    DESTRUCTIVE = "destructive"   # Dosya silme, DB drop gibi yıkıcı

class ApprovalLevel(str, Enum):
    """Aracın otonom çalışma onay seviyesi."""
    AUTO = "auto"                 # Hiç sormadan çalışır
    USER_NOTIFY = "notify"        # Çalışır ama kullanıcıya bildirir
    REQUIRE_APPROVAL = "require"  # Çalışmak için HITL onayı bekler

class ToolContract(BaseModel):
    """Araçların sistemde nasıl çağrılacağını standartlaştıran ABI modeli."""
    name: str = Field(..., description="Aracın benzersiz sistem adı")
    description: str = Field(..., description="Aracın amacı")
    input_schema: Dict[str, Any] = Field(default_factory=dict, description="JSON Schema (Girdi)")
    output_schema: Dict[str, Any] = Field(default_factory=dict, description="JSON Schema (Çıktı)")
    side_effect: SideEffectClass = Field(default=SideEffectClass.READ_ONLY)
    approval_level: ApprovalLevel = Field(default=ApprovalLevel.AUTO)
    retryability: bool = Field(default=True, description="Hata alınırsa güvenle tekrar edilebilir mi? (Idempotency)")
    idempotent: bool = Field(default=True, description="Çoklu çalıştırmalarda sonuç/etki aynı kalır mı?")

# --- GÖREV 5: Artifact Lifecycle ---

class ArtifactStatus(str, Enum):
    """Artifact (Çıktı) eserlerin yaşam döngüsü."""
    TEMP = "temp"           # Geçici dosyalar, loglar
    DRAFT = "draft"         # Üretim aşamasında olan/eksik taslaklar
    REVIEW = "review"       # Kullanıcı/Ajan incelemesi bekleyenler
    APPROVED = "approved"   # İncelemeden geçmiş geçerli dosyalar
    PUBLISHED = "published" # Üretime veya son sunuma hazır (rapor/viz)
    ARCHIVED = "archived"   # Artık kullanılmayan ama saklanan eserler

class ArtifactContract(BaseModel):
    """Ajanın ürettiği birim dosyaların standart şeması."""
    artifact_id: str = Field(..., description="UUID veya benzersiz ID")
    run_id: str = Field(..., description="Bağlı olduğu AgentRun ID'si")
    name: str = Field(..., description="Dosya veya eserin kısa adı")
    type: str = Field(..., description="Rapor, kod, grafik, model_weight vb.")
    status: ArtifactStatus = Field(default=ArtifactStatus.DRAFT)
    storage_path: Optional[str] = Field(None, description="Workspace içindeki yolu (Uri)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Mime_type, token_size vb.")
