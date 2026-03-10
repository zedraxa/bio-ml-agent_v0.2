from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# --- GÖREV 1: Ephemeral Remote Workspace & Snapshot ---
class WorkspaceSnapshot(BaseModel):
    """Çalışma alanı (Project) anlık görüntüsü."""
    snapshot_id: str
    project_id: str
    file_indices: List[str] = Field(default_factory=list, description="Dosya listesi ve hash'leri")
    env_snapshot: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    git_commit_hash: Optional[str] = None
    created_at: str
    size_mb: float

# --- GÖREV 2 & 3: Warm Pool & Sandbox Classes ---
class SandboxType(str, Enum):
    CODE = "code"
    BROWSER = "browser"
    ML_TRAINING = "ml_training"
    NOTEBOOK = "notebook"

class SandboxStatus(str, Enum):
    PROVISIONING = "provisioning"
    READY = "ready"
    STOPPING = "stopping"
    TERMINATED = "terminated"

class SandboxConfig(BaseModel):
    """Bulut Sandbox konfigürasyonu."""
    sandbox_id: str
    sandbox_type: SandboxType
    image_tag: str
    resource_profile: str # Örn: "8cpu-32ram-t4"
    status: SandboxStatus = Field(default=SandboxStatus.PROVISIONING)
    active_since: Optional[str] = None

# --- GÖREV 4: Sync Protocol ---
class SyncEventType(str, Enum):
    UPLOAD = "upload"
    DOWNLOAD = "download"
    DELETE = "delete"

class SyncTrackRecord(BaseModel):
    """Lokal ve bulut arası senkronizasyon olay kaydı."""
    sync_id: str
    workspace_id: str
    event_type: SyncEventType
    file_paths: List[str]
    checksum_map: Dict[str, str]
    timestamp: str
    latency_ms: int

# --- GÖREV 5: Spend Guardrails ---
class SpendGuardrails(BaseModel):
    """Bulut bütçe ve kullanım sınırları."""
    project_id: str
    max_hours: float = Field(default=24.0)
    max_gpu_budget_usd: float = Field(default=10.0)
    current_spend_usd: float = Field(default=0.0)
    auto_stop_enabled: bool = Field(default=True)
    alert_threshold_percentage: int = Field(default=80)
