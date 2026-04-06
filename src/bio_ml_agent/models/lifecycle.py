from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

try:
    from bio_ml_agent.models.workspace_ux import AgentRole, TaskType
except ImportError:
    from enum import Enum as _Enum

    class AgentRole(str, _Enum):
        RESEARCH_AGENT = "research_agent"

    class TaskType(str, _Enum):
        ANALYZE = "analyze"


# ── Data Lifecycle Enums & Models ──


class DataStage(str, Enum):
    """Stage of data in the processing pipeline."""
    RAW = "raw"
    CLEANED = "cleaned"
    TRANSFORMED = "transformed"
    FEATURE_ENGINEERED = "feature_engineered"
    SPLIT = "split"
    AUGMENTED = "augmented"


class ModelStatus(str, Enum):
    """Status of a model artifact in the registry."""
    CANDIDATE = "candidate"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class LineageNodeType(str, Enum):
    """Type of node in the data lineage graph."""
    DATASET = "dataset"
    MODEL = "model"
    EXPERIMENT = "experiment"
    ARTIFACT = "artifact"
    PIPELINE = "pipeline"


class DataVersion(BaseModel):
    """Tracks a specific version of a dataset."""
    version_id: str
    dataset_name: str
    stage: DataStage = DataStage.RAW
    storage_artifact_id: str = ""
    row_count: int = 0
    schema_hash: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    created_by_run_id: str = ""


class ExperimentRun(BaseModel):
    """Tracks a single experiment execution."""
    run_id: str
    experiment_name: str
    params: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    artifact_ids: List[str] = Field(default_factory=list)
    start_time: str = Field(default_factory=lambda: datetime.now().isoformat())
    status: str = "running"


class ModelArtifact(BaseModel):
    """Registered model artifact in the model registry."""
    model_id: str
    name: str
    version: str = "0.1.0"
    framework: str = "unknown"
    status: ModelStatus = ModelStatus.CANDIDATE
    storage_path: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class LineageNode(BaseModel):
    """A node in the data/model lineage graph."""
    node_id: str
    type: LineageNodeType = LineageNodeType.DATASET
    reference_id: str = ""
    parents: List[str] = Field(default_factory=list)


class ReproBundle(BaseModel):
    """Reproducibility bundle for an experiment run."""
    bundle_id: str
    experiment_run_id: str
    required_data_version_ids: List[str] = Field(default_factory=list)
    environment_yaml_url: str = ""
    entrypoint_script: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    is_verified: bool = False


# ── Agent Lifecycle Models ──

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
