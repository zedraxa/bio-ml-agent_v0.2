from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from bio_ml_agent.models.workspace_ux import AgentRole, TaskType

class AgentTier(str, Enum):
    """Classification for agent maturity and deployment readiness."""
    STABLE = "stable"           # Production-ready, high reliability
    ACTIVE = "active"           # Currently in use, undergoing refinements
    BETA = "beta"               # New features, testing required
    EXPERIMENTAL = "experimental" # Visionary/Research agents, potential instability

class AgentCapability(str, Enum):
    """Specific capabilities agents can possess."""
    BROWSER_NAV = "browser_navigation"
    CODE_EXEC = "code_execution"
    DATA_CLEAN = "data_cleaning"
    LIT_SEARCH = "literature_search"
    MODEL_TRAIN = "model_training"
    MOLECULAR_SIM = "molecular_simulation"
    XAI = "explainable_ai"
    REPORT_GEN = "report_generation"

class AgentRegistryEntry(BaseModel):
    """Metadata for an agent family in the Unified Lifecycle."""
    agent_id: str
    role: AgentRole
    tier: AgentTier = Field(default=AgentTier.EXPERIMENTAL)
    primary_task_type: TaskType = TaskType.ANALYZE 
    path: Optional[str] = None # Relative path to the implementation
    mission_pack: Optional[str] = None # Targeted mission pack
    version: str = "1.0.0"
    description: str
    capabilities: List[AgentCapability] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    last_heartbeat: Optional[float] = None
    is_enabled: bool = True

class AgentHeartbeat(BaseModel):
    """Runtime status update from an active agent."""
    agent_id: str
    timestamp: float
    status: str = "online" # online, busy, offline, error
    current_mission_id: Optional[str] = None
    resource_usage: Dict[str, float] = Field(default_factory=dict)
