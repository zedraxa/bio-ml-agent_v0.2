from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from bio_ml_agent.models.workspace_ux import AgentRole, TaskType, RiskLevel

class MissionPackStep(BaseModel):
    """Detailed blueprint for a step within a Mission Pack."""
    step_id: str
    title: str
    description: str
    task_type: TaskType
    assigned_agent: AgentRole
    preferred_agent_id: Optional[str] = None
    depends_on: List[str] = Field(default_factory=list)
    requires_approval: bool = False
    risk_level: RiskLevel = RiskLevel.LOW
    expected_artifact_type: str = "REPORT"

class MissionPack(BaseModel):
    """A named, capability-based mission scenario (Orchestration Pack)."""
    pack_id: str
    name: str
    description: str
    capabilities: List[str] = Field(default_factory=list)
    steps: List[MissionPackStep] = Field(default_factory=list)
    final_outputs: List[str] = Field(default_factory=list)
    version: str = "1.0.0"
