from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum
from datetime import datetime

# 1. Local Workspace Cache
class CacheItemStatus(str, Enum):
    SYNCED = "synced"
    DIRTY = "dirty"
    PENDING_UPLOAD = "pending_upload"

class LocalWorkspaceCache(BaseModel):
    """Ağ bağlantısı yokken veya aktif çalışma anında hızlı I/O için geçici/lokal disk dosya bağlamı."""
    cache_id: str
    project_id: str
    local_path: str
    size_bytes: int
    last_accessed: datetime
    status: CacheItemStatus
    sync_hash: Optional[str] = None

# 2. Object Storage Artifact Root
class ArtifactTier(str, Enum):
    HOT = "hot"
    COLD = "cold"
    ARCHIVE = "archive"

class ObjectStorageArtifact(BaseModel):
    """Proje çıktıları, grafikler, test raporları ve logların kalıcı olarak barındırıldığı yapı (AWS S3 vb.)."""
    artifact_id: str
    project_id: str
    run_id: Optional[str] = None
    object_key: str
    bucket_name: str
    tier: ArtifactTier = ArtifactTier.HOT
    url: Optional[str] = None
    created_at: datetime
    metadata: Dict[str, str] = Field(default_factory=dict)

# 3. Experiment Registry
class ExperimentRegistryMeta(BaseModel):
    """ML/DL hyperparameter, loss, metrik ve epoch sonuçlarını listeleyen kayıt defteri."""
    experiment_id: str
    run_name: str
    model_architecture: str
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(default_factory=dict)
    artifacts_ref: List[str] = Field(default_factory=list) # ObjectStorageArtifact ID'leri
    created_at: datetime

# 4. Vector Memory Store
class VectorMemoryIndex(BaseModel):
    """RAG için PDF ve metin embeddingleri (Qdrant, Chroma, Pinecone entegrasyonları)."""
    index_id: str
    collection_name: str
    embedding_model: str
    dimension_size: int
    document_count: int = 0
    last_updated: datetime

# 5. Relational Metadata & State Store
class RelationalMetadataRecord(BaseModel):
    """Kullanıcı id, Rol yetkileri, "Run" durumları (Running/Failed) vb. yapısal metadatalar (PostgreSQL mantığı)."""
    record_id: str
    entity_type: str # "user", "project", "run"
    entity_id: str
    attributes: Dict[str, Any] = Field(default_factory=dict)
    schema_version: str = "1.0"
    updated_at: datetime

# 6. Observability & Audit Trail Store
class AuditLogLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    SECURITY = "security"

class AuditTrailEvent(BaseModel):
    """Sistem içi kimlik denetimi ve eylem denetim (audit) logları, trace'ler."""
    event_id: str
    timestamp: datetime
    actor_id: str # User ID, Agent ID or System
    action: str
    target_resource: str
    level: AuditLogLevel
    ip_address: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)

# 7. Secret Management (Vault)
class SecretScope(str, Enum):
    PROJECT = "project"
    USER = "user"
    GLOBAL = "global"

class EncryptedSecretVault(BaseModel):
    """API Key'ler, Cloud Credentials ve Tokenların güvenli saklandığı şifreli depo alanı."""
    secret_id: str
    scope: SecretScope
    owner_id: str
    key_name: str
    encrypted_value: str
    kms_key_id: Optional[str] = None
    expires_at: Optional[datetime] = None
    created_at: datetime
