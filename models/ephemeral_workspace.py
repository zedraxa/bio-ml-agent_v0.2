from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# --- GÖREV 2: Sıcak İmaj Havuzu (Sandbox Image Flavors) ---
class SandboxImageFlavor(str, Enum):
    PYTHON_STDLIB = "python_stdlib"
    BROWSER_AUTOMATION = "browser_automation"
    MACHINE_LEARNING = "machine_learning_gpu"
    JUPYTER_NOTEBOOK = "jupyter_notebook_server"
    R_BIOINFORMATICS = "r_bioinformatics"

class NodeState(str, Enum):
    PROVISIONING = "provisioning"
    WARMED_UP = "warmed_up" # Havuzda bekleyen sıcak makine
    RUNNING_JOB = "running_job"
    TERMINATING = "terminating"
    CRASHED = "crashed"

# --- GÖREV 3: Bütçe ve Süre Sınırı (Limits & Quotas) ---
class WorkspaceBudgetAndQuota(BaseModel):
    """Bulut iş alanlarının masraf ve boşa çıkma (idle) sınırları."""
    quota_id: str
    max_duration_seconds: int = 3600 # 1 Saat varsayılan limit
    idle_timeout_seconds: int = 600 # 10 dakika boş kalırsa kendini imha eder (Auto-shutdown)
    max_ram_mb: int = 8192 # 8GB
    max_gpu_count: int = 0
    max_cost_usd_per_node: float = 1.0

# --- GÖREV 1: Tek Tık Bulut Çalışma Alanı (Workspace Session) ---
class EphemeralWorkspaceInfo(BaseModel):
    """Bulut üzerinde geçici (tek kullanımlık) yaratılmış node bilgisini ve bağlantılarını tutar."""
    workspace_id: str
    run_id: str
    user_id: str
    image_flavor: SandboxImageFlavor
    state: NodeState = Field(default=NodeState.PROVISIONING)
    quota_rules: WorkspaceBudgetAndQuota
    connection_ssh_uri: Optional[str] = None # Bulut ortamına reverse debug bağlantısı
    connection_jupyter_url: Optional[str] = None # İmaj tipine bağlı ekstra url
    pulled_local_artifact_refs: List[str] = Field(default_factory=list) # Repodan kopyalananlar
    created_at: str
    terminated_at: Optional[str] = None
