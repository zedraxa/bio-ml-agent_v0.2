from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# --- GÖREV 1: Cloud Artifact Storage ---
class CloudArtifactCategory(str, Enum):
    PROJECT_FILE = "project_file"
    REPORT = "report"
    EXPERIMENT_OUTPUT = "experiment_output"
    BROWSER_RECORDING = "browser_recording"
    MODEL_WEIGHT = "model_weight"

class CloudArtifactItem(BaseModel):
    """Bulut üzerinde yedeklenen / tutulan bir artifact verisi."""
    artifact_id: str
    project_id: str
    session_id: Optional[str] = None
    category: CloudArtifactCategory
    file_name: str
    size_bytes: int
    storage_path_uri: str
    checksum_sha256: Optional[str] = None
    created_at: str

# --- GÖREV 2: Remote File Browser ---
class RemoteFileActionType(str, Enum):
    VIEW = "view"
    DOWNLOAD = "download"
    UPLOAD = "upload"
    DELETE = "delete"

class RemoteFileBrowserAction(BaseModel):
    """Hafif istemcilerden (Mobil/Bulut arayüz) dosya okuma/yazma eylemi."""
    action_id: str
    user_id: str
    artifact_id: Optional[str] = None
    target_path: Optional[str] = None # Yükleme yapılacak ya da okunacak dizin
    action_type: RemoteFileActionType
    device_fingerprint: str
    supports_drag_and_drop: bool = False
    timestamp: str

# --- GÖREV 3: Project Sync ---
class SyncDirection(str, Enum):
    LOCAL_TO_CLOUD = "local_to_cloud"
    CLOUD_TO_LOCAL = "cloud_to_local"

class ConflictResolutionStrategy(str, Enum):
    KEEP_LOCAL = "keep_local"
    KEEP_CLOUD = "keep_cloud"
    MANUAL_MERGE = "manual_merge"
    ABORT = "abort"

class ProjectSyncSnapshot(BaseModel):
    """Bulut ve yerel cihaz (workspace) arasındaki proje senkronizasyon verisi."""
    sync_id: str
    project_id: str
    user_id: str
    direction: SyncDirection
    local_workspace_hash: str
    cloud_workspace_hash: str
    has_conflicts: bool = False
    conflict_resolution: Optional[ConflictResolutionStrategy] = None
    synced_files_count: int = 0
    timestamp: str

# --- GÖREV 4: Paylaşımlı Proje Erişimi ---
class ProjectAccessRole(str, Enum):
    OWNER = "owner"
    EDITOR = "editor"
    VIEWER = "viewer"
    CODE_REVIEWER = "code_reviewer"
    REPORT_VIEWER = "report_viewer"

class SharedProjectAccess(BaseModel):
    """Takım üyeleri arasındaki dosya / proje erişim yetki sınırları."""
    access_id: str
    project_id: str
    granted_user_id: str
    granted_by_user_id: str
    role: ProjectAccessRole
    expires_at: Optional[str] = None
