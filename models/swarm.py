from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# --- GÖREV 4: Sub-agent Registry ---

class AgentRole(str, Enum):
    ORCHESTRATOR = "orchestrator"   # İşleri dağıtan ana beyin
    DATA_SCIENTIST = "data_scientist" # ML ve Veri analizi uzmanı
    BIOINFORMATICIAN = "bioinformatician" # Genomik ve lab verisi uzmanı
    CODE_REVIEWER = "code_reviewer" # Kod kalitesini denetleyen ajan
    WEB_RESEARCHER = "web_researcher" # Tarayıcıyı kullanan bilgi toplayıcı

class AgentCapability(BaseModel):
    """Bir ajanın sahip olduğu yetenek seti."""
    role: AgentRole
    tools_available: List[str] = Field(default_factory=list)
    can_delegate: bool = Field(default=False)

# --- GÖREV 1: Swarm Message Contract ---

class SwarmMessageType(str, Enum):
    TASK_DELEGATION = "task_delegation" # "Şu analizi sen yap"
    REPORT_RESULT = "report_result"     # "Analiz bitti, sonuç budur"
    DEBUG_REQUEST = "debug_request"     # "Kodum hata verdi, bakar mısın?"
    CONSENSUS_VOTE = "consensus_vote"   # "Bu model iyi mi? Oyla"
    HEARTBEAT = "heartbeat"             # "Buradayım, aktifim"

class SwarmMessage(BaseModel):
    """Ajanlar arası asenkron RPC ve mesajlaşma şeması."""
    message_id: str
    sender_id: str
    receiver_id: str
    type: SwarmMessageType
    payload: Dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=1, ge=1, le=5, description="1: En düşük, 5: Kritik")
    timestamp: str = Field(..., description="ISO formattaki zaman damgası")

# --- GÖREV 2: Multi-Agent Workspace Sharing ---

class WorkspaceLock(BaseModel):
    """Aynı dosyaya birden fazla ajanın yazmasını engelleyen kilit mekanizması."""
    resource_path: str
    owner_agent_id: str
    locked_at: str
    reason: str = Field(default="editing")

# --- GÖREV 3: Shared Memory (Vektörel) ---

class MemoryTag(str, Enum):
    BIO_LAYER = "bio-layer"
    CODE_LAYER = "code-layer"
    ML_LAYER = "ml-layer"
    UI_LAYER = "ui-layer"

class SharedVectorState(BaseModel):
    """Vektör veritabanının tüm ajanlar tarafından ortak kullanım şeması."""
    document_id: str
    content_preview: str
    tags: List[MemoryTag] = Field(default_factory=list)
    created_by_agent: str
    relevance_score: float = Field(default=1.0)

# --- GÖREV 5: Task Delegation & Aggregation ---

class DelegationPolicy(BaseModel):
    """Görevin kime devredileceğine dair kurallar seti."""
    require_review: bool = Field(default=True)
    timeout_seconds: int = Field(default=3600)
    max_retries: int = Field(default=3)

class TaskDelegationContract(BaseModel):
    """İşin parçalara ayrılması ve devredilmesini ifade eden sözleşme."""
    task_id: str
    orchestrator_id: str
    worker_id: str
    instructions: str
    policy: DelegationPolicy = Field(default_factory=DelegationPolicy)
    status: str = Field(default="assigned") # assigned, in_progress, completed, failed
