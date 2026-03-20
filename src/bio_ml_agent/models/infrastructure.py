from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# --- GÖREV 1: Helm Charts & Deployment ---
class DeploymentStatus(str, Enum):
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"

class HelmRelease(BaseModel):
    """Kubernetes Helm release metadatası."""
    release_id: str
    name: str # "bio-api", "browser-pool"
    namespace: str
    version: str
    values_override: Dict[str, Any] = Field(default_factory=dict)
    status: DeploymentStatus
    last_updated: str

# --- GÖREV 2: KServe Inference Services ---
class InferenceService(BaseModel):
    """Model sunumu (serving) konfigürasyonu."""
    service_id: str
    model_name: str
    model_version: str
    endpoint_url: str
    min_replicas: int = 1
    max_replicas: int = 10
    canary_percentage: Optional[float] = None # A/B test / Canary
    is_active: bool = Field(default=True)

# --- GÖREV 3 & 4: Node Classes & Autoscaling ---
class NodeClass(str, Enum):
    CPU_GENERAL = "cpu_general"
    GPU_TRAINING = "gpu_training"
    GPU_INFERENCE = "gpu_inference"
    BROWSER_NODE = "browser_node"

class ScalingPolicy(BaseModel):
    """Donanım ve kuyruk ölçekleme kuralı."""
    policy_id: str
    node_class: NodeClass
    metric_source: str # "cpu", "memory", "queue_depth"
    threshold: float
    scale_out_count: int
    scale_in_count: int

# --- GÖREV 5: Tenant Isolation ---
class TenantIsolation(BaseModel):
    """Müşteri/Organizasyon izolasyon sınırları."""
    tenant_id: str
    namespace: str
    quota_id: str
    secret_vault_path: str
    storage_root: str
    network_policy_enabled: bool = Field(default=True)
