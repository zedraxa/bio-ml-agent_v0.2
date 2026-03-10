from enum import Enum
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field

# --- GÖREV 1: Data Versioning ---
class DataStage(str, Enum):
    RAW = "raw"
    CLEANED = "cleaned"
    FEATURE = "feature"
    TRAIN = "train"
    EVAL = "eval"
    TEST = "test"

class DataVersion(BaseModel):
    """Veri kümesi versiyon kaydı."""
    version_id: str
    dataset_name: str
    stage: DataStage
    storage_artifact_id: str # StorageArtifact.artifact_id
    row_count: Optional[int] = None
    schema_hash: str
    created_at: str
    created_by_run_id: str

# --- GÖREV 2: Experiment Tracking ---
class ExperimentRun(BaseModel):
    """Bireysel deney çalışması."""
    run_id: str
    experiment_name: str
    params: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    artifact_ids: List[str] = Field(default_factory=list)
    prompt_version_id: Optional[str] = None
    tool_versions: Dict[str, str] = Field(default_factory=dict)
    start_time: str
    end_time: Optional[str] = None
    status: str = Field(default="running")

# --- GÖREV 3: Model Registry ---
class ModelStatus(str, Enum):
    DRAFT = "draft"
    CANDIDATE = "candidate"
    CHAMPION = "champion"
    ARCHIVED = "archived"

class ModelArtifact(BaseModel):
    """Model varlığı ve metadata."""
    model_id: str
    name: str
    version: str
    framework: str # Örn: "PyTorch", "TensorFlow", "Scikit-Learn"
    status: ModelStatus = Field(default=ModelStatus.DRAFT)
    metrics_at_creation: Dict[str, float] = Field(default_factory=dict)
    storage_path: str
    created_at: str

# --- GÖREV 4 & 5: Lineage & Repro ---
class LineageNodeType(str, Enum):
    DATA_VERSION = "data_version"
    CODE_VERSION = "code_version"
    PROMPT_VERSION = "prompt_version"
    EXPERIMENT_RUN = "experiment_run"
    MODEL = "model"

class LineageNode(BaseModel):
    """Soyağacı düğümü."""
    node_id: str
    type: LineageNodeType
    reference_id: str # İlgili modelin kendi id'si
    parents: List[str] = Field(default_factory=list, description="Bağlı olduğu üst düğüm id'leri")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ReproBundle(BaseModel):
    """Deney reprodüksiyon paketi."""
    bundle_id: str
    experiment_run_id: str
    required_data_version_ids: List[str]
    environment_yaml_url: str
    entrypoint_script: str
    created_at: str
    is_verified: bool = Field(default=False)
