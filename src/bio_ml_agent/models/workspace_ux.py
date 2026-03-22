from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from bio_ml_agent.models.domain import (
    Project as WorkspaceProject,
    Mission as ProjectMission,
    Artifact as ProjectArtifact,
    Comment,
    ReviewThread,
    ProjectTruthSnapshot,
    ProjectMemoryItem,
    ProjectDashboardSummary,
    WorkspaceMode,
    ProjectState,
    ArtifactReviewStatus,
    ArtifactType,
    StepStatus,
    RiskLevel,
    AgentRole,
    TaskType,
    MissionPriority,
    ResourceIntensity,
    TimelineEventType
)

class TimelineEvent(BaseModel):
    event_id: str
    project_id: str
    event_type: TimelineEventType = Field(default=TimelineEventType.INFO)
    message: str
    timestamp: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    agent_name: Optional[str] = None

# Legacy models replaced by Canonical Domain Core imports above.

