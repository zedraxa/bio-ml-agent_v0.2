from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# --- GÖREV 1: Canlı Notebook Session ---
class CellEditorRole(str, Enum):
    AGENT = "agent"
    HUMAN = "human"

class CellExecutionState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    ERROR = "error"
    SUCCESS = "success"

class NotebookCellActivity(BaseModel):
    """Bulut Jupyter ortamında çalışan / müdahale edilen tekil bir hücre eylemi."""
    cell_id: str
    last_edited_by: CellEditorRole
    code_content: str
    execution_state: CellExecutionState
    output_preview: Optional[str] = None
    execution_time_sec: float = 0.0

class LiveNotebookSession(BaseModel):
    """Agent ve insan kullanıcının ortak etkileştiği aktif notebook oturumu."""
    session_id: str
    workspace_id: str
    notebook_path: str
    cells: List[NotebookCellActivity] = Field(default_factory=list)
    human_interventions_count: int = 0
    connected_clients_count: int = 1

# --- GÖREV 2: Remote Patch Review ---
class PatchApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified_by_human"

class RemotePatchReviewEvent(BaseModel):
    """Mobil cihazdan / Slack üzerinden kolayca incelenip onaylanabilecek diff bloğu."""
    patch_id: str
    run_id: str
    pull_request_or_commit_title: str
    diff_content_preview: str # Kısmi diff özeti
    files_changed: int
    status: PatchApprovalStatus = Field(default=PatchApprovalStatus.PENDING)
    human_comment: Optional[str] = None
    generated_at: str

# --- GÖREV 3: Dataset Staging ---
class StagedCloudDataset(BaseModel):
    """Sadece bulutta duran, lokale inmeyen büyük makine öğrenmesi veri iskeleti."""
    dataset_id: str
    workspace_id: str
    dataset_name: str
    storage_size_gb: float
    is_mounted_on_workspace: bool = True
    local_proxy_placeholder_uri: str # Lokal taraf bu URI'yi görerek buluttaki veriyi stream edebilir
    access_format: str # e.g., "huggingface_dataset", "parquet_folder", "s3_bucket"
