from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

class WorkspaceMode(str, Enum):
    RESEARCH = "research"
    LAB = "lab"
    MICROSCOPY = "microscopy"
    CODING = "coding"
    WRITING = "writing"
    STRUCTURE = "structure"

class ProjectState(str, Enum):
    INITIALIZED = "initialized"
    DATA_COLLECTED = "data_collected"
    ANALYSIS_RUNNING = "analysis_running"
    REVIEW_NEEDED = "review_needed"
    DRAFT_READY = "draft_ready"
    FINALIZED = "finalized"

class ArtifactReviewStatus(str, Enum):
    DRAFT = "DRAFT"
    REVIEW_NEEDED = "REVIEW_NEEDED"
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    SUPERCEDED = "SUPERCEDED"
    FINAL = "FINAL"
    EXPORTED = "EXPORTED"

class TimelineEventType(str, Enum):
    INFO = "info"
    MISSION_STARTED = "mission_started"
    MISSION_COMPLETED = "mission_completed"
    ARTIFACT_GENERATED = "artifact_generated"
    USER_APPROVAL = "user_approval"
    CRITIC_REVIEW = "critic_review"
    ERROR = "error"

class TimelineEvent(BaseModel):
    event_id: str
    project_id: str
    event_type: TimelineEventType = Field(default=TimelineEventType.INFO)
    message: str
    timestamp: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    agent_name: Optional[str] = None

class ProjectArtifact(BaseModel):
    artifact_id: str
    project_id: str
    mission_id: Optional[str] = None
    title: str
    description: Optional[str] = None
    category: Optional[str] = "Documents" 
    file_type: str 
    created_by: str 
    confidence: Optional[float] = None
    review_status: ArtifactReviewStatus = Field(default=ArtifactReviewStatus.DRAFT)
    lineage_parents: List[str] = Field(default_factory=list) 
    content_uri: str 
    created_at: float
    updated_at: float

class ProjectMission(BaseModel):
    mission_id: str
    project_id: str
    name: str 
    status: str 
    progress_narrative: str
    active_agent: Optional[str] = None
    created_at: float

class WorkspaceProject(BaseModel):
    project_id: str
    name: str
    description: str
    goals: List[str] = Field(default_factory=list)
    state: ProjectState = Field(default=ProjectState.INITIALIZED)
    workspace_mode: WorkspaceMode = Field(default=WorkspaceMode.RESEARCH)
    open_questions: List[str] = Field(default_factory=list)
    next_actions: List[str] = Field(default_factory=list)
    created_at: float
    updated_at: float

class ProjectMemoryItem(BaseModel):
    memory_id: str
    project_id: str
    category: str # Decision, Dataset, Model, Finding, Issue
    title: str
    content: str
    importance: int = Field(default=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: float

class ProjectDashboardSummary(BaseModel):
    project: WorkspaceProject
    recent_missions: List[ProjectMission]
    recent_artifacts: List[ProjectArtifact]
    recent_memory: List[ProjectMemoryItem]
    active_agent_count: int = 0

