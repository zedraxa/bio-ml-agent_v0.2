from enum import Enum
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field

# --- GÖREV 1: Execution Classification ---
class WorkloadType(str, Enum):
    CPU_BOUND = "cpu_bound"     # Ağır hesaplama
    GPU_BOUND = "gpu_bound"     # ML eğitimi / Model çıkarımı
    MEMORY_BOUND = "memory_bound" # Büyük veri yükleme
    IO_BOUND = "io_bound"       # Çoklu dosya okuma/yazma
    BROWSER_HEAVY = "browser_heavy" # Yoğun tarayıcı otonomisi
    LATENCY_SENSITIVE = "latency_sensitive" # Hızlı yanıt gerektiren

class ResourceRequest(BaseModel):
    """İş yükü gereksinimleri."""
    workload_type: WorkloadType
    min_cpu_cores: int = Field(default=1)
    min_ram_gb: float = Field(default=1.0)
    requires_gpu: bool = Field(default=False)
    gpu_type: Optional[str] = None # Örn: "T4", "A100"
    estimated_duration_seconds: Optional[int] = None
    priority: int = Field(default=3, ge=1, le=5)

# --- GÖREV 2 & 3: Decision & Optimization ---
class ExecutionLocation(str, Enum):
    LOCAL = "local"
    REMOTE = "remote"
    HYBRID = "hybrid"

class ComputeNodeStatus(str, Enum):
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"

class ComputeNode(BaseModel):
    """Hesaplama düğümü (Local veya Cloud)."""
    node_id: str
    location: ExecutionLocation
    address: str # IP or URL
    total_cpu_cores: int
    total_ram_gb: float
    has_gpu: bool
    status: ComputeNodeStatus = Field(default=ComputeNodeStatus.IDLE)
    cost_per_hour: float = Field(default=0.0)

class ExecutionDecision(BaseModel):
    """Nerede çalıştırılacağına dair karar."""
    task_id: str
    selected_node_id: str
    location: ExecutionLocation
    reason: str
    estimated_cost: float
    confidence_score: float = Field(default=1.0)

# --- GÖREV 4: Remote Task Envelopes ---
class TaskEnvelope(BaseModel):
    """Uzak sunucuya gönderilen görev paketi."""
    envelope_id: str
    task_id: str
    container_image: str = Field(default="python:3.10-slim")
    entrypoint: List[str]
    env_vars: Dict[str, str] = Field(default_factory=dict)
    mounted_artifact_ids: List[str] = Field(default_factory=list)
    secrets_scope: List[str] = Field(default_factory=list)
    callback_url: Optional[str] = None
    timeout_seconds: int = Field(default=3600)

# --- GÖREV 5: Checkpoint & Resume ---
class ComputeCheckpoint(BaseModel):
    """Görev durumunu dondurma ve taşıma kaydı."""
    checkpoint_id: str
    task_id: str
    node_id: str
    timestamp: str
    state_payload_url: str # S3/Local path to serialized state
    iteration_count: int
    is_restorable: bool = Field(default=True)
