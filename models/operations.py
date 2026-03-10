from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# --- GÖREV 1: Cost Ledger ---
class CostEntry(BaseModel):
    """Detaylı maliyet kaydı (Financial Ledger)."""
    entry_id: str
    timestamp: str
    user_id: str
    project_id: str
    resource_type: str # "gpt-4", "gpu-hour", "storage-gb"
    cost_amount: float
    currency: str = "USD"
    metadata: Dict[str, Any] = Field(default_factory=dict) # model_name, tool_name

# --- GÖREV 2: Quota Engine ---
class QuotaLimit(BaseModel):
    """Kaynak kullanım limitleri."""
    quota_id: str
    tenant_id: str
    daily_token_limit: int
    monthly_gpu_limit_hours: float
    max_concurrent_browsers: int
    storage_quota_gb: float
    used_tokens_today: int = 0
    used_gpu_hours_month: float = 0.0

# --- GÖREV 3: FinOps Recommendations ---
class FinOpsRecommendation(BaseModel):
    """Maliyet optimizasyon önerisi."""
    rec_id: str
    project_id: str
    category: str # "compute_offload", "model_resize", "storage_tier"
    suggestion: str # "Lokal Llama kullanılarak $50 tasarruf edilebilir."
    potential_savings: float
    confidence_score: float

# --- GÖREV 4: Incident Playbooks ---
class IncidentSeverity(str, Enum):
    P0 = "p0" # Critical çöküş
    P1 = "p1" # Major sorun
    P2 = "p2" # Minor / Yavaşlık

class IncidentPlaybook(BaseModel):
    """Olay müdahale senaryosu."""
    playbook_id: str
    component: str # "qdrant", "redis", "api"
    trigger_condition: str # "latency > 500ms"
    severity: IncidentSeverity
    remediation_steps: List[str]
    automated_fix_script_path: Optional[str] = None
