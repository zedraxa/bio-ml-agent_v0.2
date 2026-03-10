from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# --- GÖREV 1: Unified Dashboard State ---
class ControlPlaneState(BaseModel):
    """Tüm sistem birimlerinin anlık durumunu tutan entegre panel modeli."""
    active_runs: int = 0
    active_browser_sessions: int = 0
    provisioned_cloud_workspaces: int = 0
    running_bio_pipelines: int = 0
    system_health_score: float = 1.0 # 0.0 - 1.0 arası

# --- GÖREV 2: Unified Run Graph ---
class StepSource(str, Enum):
    LOCAL_EXEC = "local_exec"
    REMOTE_NODE = "remote_node"
    BROWSER_AGENT = "browser_agent"
    SWARM_DELEGATE = "swarm_delegate"
    BIO_PIPELINE = "bio_pipeline"

class UnifiedRunGraphNode(BaseModel):
    """Farklı mecralarda çalışan adımların ortak grafik düğümü."""
    node_id: str
    parent_node_id: Optional[str] = None
    step_source: StepSource
    action_name: str
    status: str # "success", "failed", "running"
    duration_ms: int
    artifacts_produced: List[str] = Field(default_factory=list)

class UnifiedRunGraph(BaseModel):
    run_id: str
    nodes: List[UnifiedRunGraphNode] = Field(default_factory=list)

# --- GÖREV 3: Cross-module Identity ---
class GlobalIdentity(BaseModel):
    """Servisler arası tutarlı kimlik (Identity)."""
    global_user_id: str
    active_project_id: str
    tenant_id: str
    roles: List[str]
    api_key_hash: Optional[str] = None

# --- GÖREV 4: Artifact Lineage (Graph) ---
class ArtifactLineageNode(BaseModel):
    """Bir çıktının (Rapor, Model) soy ağacını gösteren düğüm."""
    artifact_id: str
    artifact_type: str # "report", "model", "dataset"
    produced_by_run_id: str
    derived_from_artifact_ids: List[str] = Field(default_factory=list) # Hangi veriden/modelden üretildi?

class ArtifactLineage(BaseModel):
    root_artifact_id: str
    lineage_nodes: List[ArtifactLineageNode] = Field(default_factory=list)

# --- GÖREV 5: End-to-end Policy Layer ---
class GlobalPolicy(BaseModel):
    """Uçtan uca sistem kısıt paketini temsil eden ana politika modeli."""
    policy_id: str
    require_human_approval_for_external_tx: bool = True
    max_budget_usd_per_run: float = 5.0
    secret_access_level: str = "restricted"
    artifact_retention_days: int = 30
    allow_public_publishing: bool = False
