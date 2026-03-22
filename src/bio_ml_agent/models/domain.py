from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

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
    """C4: Professional artifact lifecycle states."""
    DRAFT = "DRAFT"
    REVIEW_NEEDED = "REVIEW_NEEDED"
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    SUPERCEDED = "SUPERCEDED"
    FINAL = "FINAL"
    EXPORTED = "EXPORTED"
    OUTDATED = "OUTDATED"
    CONFLICT = "CONFLICT"

class ArtifactType(str, Enum):
    """Types of artifacts an agent can produce."""
    REPORT = "report"
    DATA = "data"
    CODE = "code"
    IMAGE = "image"
    MODEL = "model"
    VISUALIZATION = "visualization"
    ANNOTATION = "annotation"
    STRUCTURED_JSON = "structured_json"
    CRITIQUE = "critique"
    OTHER = "other"

class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AgentRole(str, Enum):
    """B1: Unified Agent Roles across the ecosystem."""
    DATA_ENGINEER = "data_engineer"
    ML_EXPERT = "ml_expert"
    BIOINFORMATICIAN = "bioinformatician"
    RESEARCHER = "researcher"
    IN_SILICO_EXPERT = "in_silico_expert"
    ACADEMIC_EXPERT = "academic_expert"
    BROWSER_AGENT = "browser_agent"
    MICROSCOPY_AGENT = "microscopy_agent"
    CODING_AGENT = "coding_agent"
    WRITING_AGENT = "writing_agent"
    STRUCTURE_AGENT = "structure_agent"
    CRITIC = "critic"
    PLANNER = "planner"
    ORCHESTRATOR = "orchestrator"

class TaskType(str, Enum):
    """Universal task taxonomy for the Mission Decomposer (A3)."""
    DISCOVER = "discover"
    ANALYZE = "analyze"
    VERIFY = "verify"
    SYNTHESIZE = "synthesize"
    WRITE = "write"
    CRITIQUE = "critique"
    EXPORT = "export"

class MissionPriority(str, Enum):
    """G4: Priority levels for mission scheduling."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

class ResourceIntensity(str, Enum):
    """G4: Resource usage categories."""
    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"
    CRITICAL = "critical"

class TimelineEventType(str, Enum):
    INFO = "info"
    MISSION_STARTED = "mission_started"
    MISSION_COMPLETED = "mission_completed"
    ARTIFACT_GENERATED = "artifact_generated"
    USER_APPROVAL = "user_approval"
    CRITIC_REVIEW = "critic_review"
    ERROR = "error"

# --- New Canonical Enums ---

class MissionStatus(str, Enum):
    """Canonical mission lifecycle states."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    WAITING = "waiting" # Added for approval gates
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class CommentStatus(str, Enum):
    """Lifecycle of a feedback/review item."""
    NEW = "new"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    DROPPED = "dropped"

class ReviewerRole(str, Enum):
    USER = "user"
    SUPERVISOR = "supervisor"
    PEER = "peer"
    CRITIC_AGENT = "critic_agent"
    DOMAIN_AGENT = "domain_agent"
    WRITING_AGENT = "writing_agent"
    COMPLIANCE_AGENT = "compliance_agent"

class ReviewMode(str, Enum):
    QUICK = "quick"
    TECHNICAL = "technical"
    SCIENTIFIC = "scientific"
    WRITING = "writing"
    CODE = "code"
    MICROSCOPY = "microscopy"
    APPROVAL = "approval"

class IntentType(str, Enum):
    REVISE = "revise"
    CLARIFY = "clarify"
    EXPLAIN = "explain"
    VERIFY = "verify"
    CRITIQUE = "critique"
    APPROVE = "approve"
    REJECT = "reject"
    COMPARE = "compare"
    EXPAND = "expand"
    SIMPLIFY = "simplify"
    REWRITE = "rewrite"
    INVESTIGATE = "investigate"
    GENERAL = "general"

class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TargetType(str, Enum):
    PROJECT = "project"
    MISSION = "mission"
    STEP = "step"
    ARTIFACT = "artifact"

class ArtifactStateTransition(BaseModel):
    from_status: ArtifactReviewStatus
    to_status: ArtifactReviewStatus
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    reason: Optional[str] = None

# --- Common Primitives ---

class Confidence(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    explanation: Optional[str] = None

# --- Canonical Models ---

class Project(BaseModel):
    """The central project entity, unifying UX and Kernel states."""
    project_id: str
    name: str
    description: str = ""
    goals: List[str] = Field(default_factory=list)
    state: ProjectState = Field(default=ProjectState.INITIALIZED)
    workspace_mode: WorkspaceMode = Field(default=WorkspaceMode.RESEARCH)
    open_questions: List[str] = Field(default_factory=list)
    next_actions: List[str] = Field(default_factory=list)
    created_at: float
    updated_at: float
    last_accessed_at: Optional[float] = None
    
    # Kernel/Execution Metadata
    active_mission_id: Optional[str] = None
    critical_findings: List[str] = Field(default_factory=list)
    version: int = 1
    
    class Config:
        from_attributes = True

class Mission(BaseModel):
    """The canonical mission model, unifying Planning and UX."""
    mission_id: str
    project_id: str
    title: str
    objective: str
    user_prompt: Optional[str] = None
    status: MissionStatus = Field(default=MissionStatus.PENDING)
    progress_percentage: int = 0
    readable_progress: Optional[str] = None
    assigned_agents: List[AgentRole] = Field(default_factory=list)
    priority: MissionPriority = Field(default=MissionPriority.NORMAL)
    created_at: float
    completed_at: Optional[float] = None
    
    # Planning & Resource metrics
    total_resource_score: int = 0
    risk_summary: Optional[str] = None
    
    class Config:
        from_attributes = True

class Artifact(BaseModel):
    """The canonical artifact, unifying storage, produce, and review."""
    artifact_id: str
    project_id: str
    mission_id: Optional[str] = None
    title: str
    description: Optional[str] = None
    category: str = "Documents" # Documents, Data, Code, etc.
    artifact_type: ArtifactType = Field(default=ArtifactType.OTHER)
    file_path: Optional[str] = None # Physical path
    content_uri: Optional[str] = None # Public URI
    created_by: str # AgentRole or UserID
    confidence: Optional[float] = None
    status: ArtifactReviewStatus = Field(default=ArtifactReviewStatus.DRAFT)
    status_history: List[ArtifactStateTransition] = Field(default_factory=list)
    version: str = "1.0.0"
    lineage_parents: List[str] = Field(default_factory=list)
    created_at: float
    updated_at: float
    
    class Config:
        from_attributes = True

class ProjectMemoryItem(BaseModel):
    memory_id: str
    project_id: str
    category: str # Decision, Dataset, Model, Finding, Issue
    title: str
    content: str
    importance: int = Field(default=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: float

    class Config:
        from_attributes = True

class ProjectDashboardSummary(BaseModel):
    project: Project
    recent_missions: List[Mission]
    recent_artifacts: List[Artifact]
    recent_memory: List[ProjectMemoryItem]
    active_agent_count: int = 0

class Comment(BaseModel):
    """High-fidelity feedback primitive for the Review Engine."""
    comment_id: str
    author: str
    content: str
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    status: CommentStatus = Field(default=CommentStatus.NEW)
    
    # Contextual linkage
    target_id: str
    target_type: TargetType = Field(default=TargetType.ARTIFACT)
    
    # Review Metadata
    role: ReviewerRole = Field(default=ReviewerRole.USER)
    review_mode: ReviewMode = Field(default=ReviewMode.QUICK)
    intent: IntentType = Field(default=IntentType.GENERAL)
    severity: Severity = Field(default=Severity.INFO)
    requested_action: Optional[str] = None
    
    # Threading
    is_resolved: bool = False
    parent_comment_id: Optional[str] = None
    replies: List[str] = Field(default_factory=list)
    
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        from_attributes = True

class ReviewThread(BaseModel):
    """A collection of comments associated with a specific artifact/entity."""
    thread_id: str
    artifact_id: str
    status: str = "open" # open, resolved
    comments: List[Comment] = Field(default_factory=list)
    
    class Config:
        from_attributes = True

class ProjectTruthSnapshot(BaseModel):
    """A consolidated point-in-time summary of the project's 'Truth'."""
    snapshot_id: str
    project_id: str
    summary: str
    key_findings: List[str] = Field(default_factory=list)
    timestamp: float
    
    class Config:
        from_attributes = True
