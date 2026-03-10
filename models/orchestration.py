from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# --- GÖREV 1: Epic -> Story -> Task -> Run Hiyerarşisi ---
class ProjectStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    COMPLETED = "completed"
    ON_HOLD = "on_hold"
    ARCHIVED = "archived"

class ProjectEpic(BaseModel):
    """Büyük proje hedefi (Örn: Yeni bir ilaç molekülü simülasyonu)."""
    epic_id: str
    title: str
    description: str
    status: ProjectStatus = Field(default=ProjectStatus.DRAFT)
    owner_agent: str
    created_at: str

class UserStory(BaseModel):
    """Kullanıcı ihtiyacı bazlı ara hedef (Örn: Veri setinin temizlenmesi)."""
    story_id: str
    epic_id: str
    title: str
    acceptance_criteria: List[str]
    status: ProjectStatus = Field(default=ProjectStatus.DRAFT)

class ProjectTaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ProjectTask(BaseModel):
    """Somut, atomik görev."""
    task_id: str
    story_id: str
    title: str
    assigned_to: Optional[str] = None
    status: ProjectTaskStatus = Field(default=ProjectTaskStatus.PENDING)
    priority: int = Field(default=3, ge=1, le=5)
    metadata: Dict[str, Any] = Field(default_factory=dict)

# --- GÖREV 2 & 3: Bağımlılık Grafı & Kritik Yol ---
class DependencyType(str, Enum):
    FINISH_TO_START = "finish_to_start"
    START_TO_START = "start_to_start"

class TaskDependency(BaseModel):
    """Görevler arası ilişki."""
    task_id: str
    depends_on_id: str
    type: DependencyType = Field(default=DependencyType.FINISH_TO_START)

class CriticalPathMap(BaseModel):
    """Kritik yol analizi sonuçları."""
    epic_id: str
    critical_task_ids: List[str]
    estimated_duration_hours: float
    bottlenecks: List[str] = Field(default_factory=list)

# --- GÖREV 4 & 5: Kuyruk & İnsan Onayı ---
class ProjectTaskQueue(BaseModel):
    """Bekleyen görevler kuyruğu."""
    project_id: str
    pending_tasks: List[str] # Task ID listesi
    blocked_tasks: List[str]

class ApprovalNeededTask(ProjectTask):
    """İnsan onayı bekleyen kritik görev sınıfı."""
    requires_human_approval: bool = Field(default=True)
    approval_reason: str
    approver_id: Optional[str] = None # Kullanıcı ID

# --- GÖREV 6: Parçalı Teslimat ---
class PartialDeliveryRecord(BaseModel):
    """Incremental artifact yayınlama kaydı."""
    delivery_id: str
    project_id: str
    artifact_ids: List[str]
    version_tag: str
    description: str
    released_at: str
