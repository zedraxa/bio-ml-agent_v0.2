from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# --- GÖREV 1: Object Storage Layer ---
class StorageArtifact(BaseModel):
    """Versiyonlu bulut varlığı."""
    artifact_id: str
    project_id: str
    path: str
    version_id: str
    checksum: str
    size_bytes: int
    content_type: str
    created_at: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SignedAccessURL(BaseModel):
    """Kısa süreli imzalı erişim URL'si."""
    artifact_id: str
    url: str
    expires_at: str
    http_method: str = Field(default="GET")

# --- GÖREV 2: Project Snapshotting ---
class ProjectSnapshot(BaseModel):
    """Bütünsel proje dondurma paketi (Kod + Veri + Raporlar)."""
    snapshot_id: str
    project_id: str
    timestamp: str
    code_version: str
    data_manifest_id: str # StorageArtifact id points to a manifest file
    report_ids: List[str] = Field(default_factory=list)
    recording_ids: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

# --- GÖREV 3: Branchable Data/Workspace ---
class DataBranch(BaseModel):
    """Çalışma alanı dalı (Branch)."""
    branch_id: str
    project_id: str
    name: str # Örn: "main", "experimental-model-v2"
    base_snapshot_id: str
    head_snapshot_id: str
    created_by_agent: str
    is_merged: bool = Field(default=False)

# --- GÖREV 4 & 5: Offline Cache & Conflict Resolution ---
class OfflineCacheMeta(BaseModel):
    """Yerel önbellek takip metadatası."""
    file_path: str
    last_synced_checksum: str
    last_synced_at: str
    local_modification_at: str
    is_dirty: bool = Field(default=False)

class SyncConflict(BaseModel):
    """Lokal ve bulut arası çelişki kaydı."""
    conflict_id: str
    file_path: str
    local_checksum: str
    remote_checksum: str
    local_updated_at: str
    remote_updated_at: str
    detected_at: str
    resolution_strategy: Optional[str] = None # Örn: "manual", "use_local", "use_remote"
