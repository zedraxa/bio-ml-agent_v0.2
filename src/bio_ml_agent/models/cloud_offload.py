from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# --- GÖREV 1: İş Sınıflandırma (Execution Classification) ---
class ExecutionTarget(str, Enum):
    LOCAL_CPU = "local_cpu"
    LOCAL_BROWSER = "local_browser"
    REMOTE_CPU = "remote_cpu"
    REMOTE_GPU = "remote_gpu"
    REMOTE_LARGE_MEMORY = "remote_large_memory"

class JobClassification(BaseModel):
    """Bir görevin tahmini donanım ayak izine göre hedef node'u seçen analiz belgesi."""
    classification_id: str
    target: ExecutionTarget
    estimated_memory_mb: int = 1024
    requires_gpu: bool = False
    requires_browser: bool = False
    reasoning: Optional[str] = None # "Büyük tensor analizi olduğu için GPU'ya yönlendirildi."

# --- GÖREV 2: Policy Engine ---
class OffloadPolicy(BaseModel):
    """Buluta veya dış düğümlere görev atanırken uyulacak limit ve izinler kümesi."""
    policy_id: str
    force_remote: bool = False # Eğer cihaz kapasitesi baştan zayıfsa hep remote
    require_human_approval_for_remote: bool = True # Buluta taşımanın onayı gerekir mi
    max_cost_limit_usd: float = 5.0 # Remote işin en fazla harcayabileceği tutar
    fallback_to_local_if_fails: bool = False

# --- GÖREV 3: Runtime Paketleme ---
class RuntimePackage(BaseModel):
    """Uzak node'a geçerken sistemin mevcut halinin/ortamının referanslandığı paket veri sözleşmesi."""
    package_id: str
    run_id: str
    workspace_snapshot_uri: str # Lokaldeki klasörün bulut önbelleğindeki adresi (snapshot)
    env_vars: Dict[str, str] = Field(default_factory=dict)
    secret_scopes_allowed: List[str] = Field(default_factory=list) # Ulaşabileceği izole API keyler
    mounted_artifact_ids: List[str] = Field(default_factory=list) # Analize gerekli ek dosyalar
    callback_webhook_url: str # Bulut ayağı işi bitirince Agent'ı dürtmesi gereken webhook_url

# --- GÖREV 4: Checkpoint / Resume ---
class CheckpointResumeStrategy(BaseModel):
    """Crashed olan remote run'ları hayatta tutan ve duplicatelerden (race condition) koruyan yapı."""
    strategy_id: str
    run_id: str
    last_known_checkpoint_uri: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    in_progress_lock_id: str # Redis veya DB tabanlı Mutex kilit ID'si (çift çalışmayı önler)
    is_resumable: bool = True
