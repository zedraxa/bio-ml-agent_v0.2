from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum
from datetime import datetime

# --- 1. Unified Run Graph ---
class ExecutionRole(str, Enum):
    PLANNER = "planner"
    BROWSER_OPERATOR = "browser_operator"
    CODER = "coder"
    BIO_ANALYST = "bio_analyst"
    REVIEWER = "reviewer"
    REPORT_WRITER = "report_writer"
    CLOUD_NODE = "cloud_node"
    NOTIFICATION_ADAPTER = "notification_adapter"

class GraphEdge(BaseModel):
    source_node_id: str
    target_node_id: str
    condition: Optional[str] = None # Edge geçiş kuralı (örn: IF success)

class ExecutionNode(BaseModel):
    """Run Graph (Çalışma zinciri) içerisindeki tek bir durak."""
    node_id: str
    role: ExecutionRole
    task_description: str
    dependencies: List[str] = Field(default_factory=list) # source node ID'leri
    status: str = "pending" # pending, running, completed, suspended, failed
    result: Optional[Dict[str, Any]] = None

class UnifiedRunGraph(BaseModel):
    """Bir veya birden fazla uzmanın/sistemin senkron-asenkron ortaklaşa çalıştığı graf."""
    graph_id: str
    project_id: str
    nodes: Dict[str, ExecutionNode] = Field(default_factory=dict)
    edges: List[GraphEdge] = Field(default_factory=list)
    created_at: datetime
    status: str = "initialized"

# --- 2. Hybrid Execution Nodes ---
class NodeLocation(str, Enum):
    LOCAL = "local"
    REMOTE_CPU = "remote_cpu"
    REMOTE_GPU = "remote_gpu"
    SWARM_AGENT = "swarm_agent"
    WEBHOOK = "webhook"

class HybridExecutionTask(BaseModel):
    """Görevin bulut-lokal düzleminde nerede icra edileceğini belirten sözleşme."""
    task_id: str
    assigned_location: NodeLocation
    payload: Dict[str, Any]
    target_endpoint: Optional[str] = None # Webhook/RPC uç noktası
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

# --- 3. Omnichannel HITL & Handoff ---
class HandoffChannel(str, Enum):
    WEB_UI = "web_ui"
    WHATSAPP = "whatsapp"
    TELEGRAM = "telegram"
    DISCORD = "discord"

class HandoffStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"
    TIMEOUT = "timeout"

class HandoffRequest(BaseModel):
    """Ajanın bir kararda insan onayı beklerken (Suspend/Resume) duraksama hali."""
    handoff_id: str
    related_task_id: str
    requested_channel: HandoffChannel
    reason: str # Neden duraksadı? "Code review required", "High budget task"
    context_data: Dict[str, Any] # Kullanıcıya verilecek özet ekran (snapshot)
    status: HandoffStatus = HandoffStatus.PENDING
    resolution_data: Optional[Dict[str, Any]] = None # Kullanıcının döndüğü veri
    requested_at: datetime
    resolved_at: Optional[datetime] = None

# --- 4. Otonom Execution Router ---
class ResourceRequirement(BaseModel):
    """İşe ayrılacak kaynak hesabı yapısı."""
    min_cpu_cores: int = 1
    min_ram_gb: int = 2
    requires_gpu: bool = False
    requires_browser: bool = False

class ExecutionRouterDecision(BaseModel):
    """Kullanıcının 'Bunu yap' dediği işin donanım & ortam yönlendiricisi."""
    decision_id: str
    task_id: str
    requirements: ResourceRequirement
    selected_environment: NodeLocation
    estimated_cost_usd: float = 0.0
    reasoning: str # (Örn: "Has browser intent -> LOCAL", "Has Pytorch -> REMOTE_GPU")
    timestamp: datetime
