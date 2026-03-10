from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# --- GÖREV 1 & 2: Core Roles & Contracts ---

class AgentContract(BaseModel):
    """Her uzman ajanın yetki sınırlarını ve davranış kalıplarını belirleyen sözleşme."""
    role_id: str
    name: str
    do_list: List[str] = Field(..., description="Ajanın yapması beklenen temel eylemler")
    dont_list: List[str] = Field(..., description="Ajanın kaçınması gereken eylemler (Kırmızı çizgiler)")
    tool_allowlist: List[str] = Field(default_factory=list, description="Erişebileceği araç ID'leri")
    output_schema: Dict[str, Any] = Field(default_factory=dict, description="Çıktı JSON şeması")

# --- GÖREV 3: Handoff Protocol ---

class HandoffPayload(BaseModel):
    """Ajanlar arası iş devri sırasında aktarılan zengin veri paketi."""
    source_agent_id: str
    target_agent_id: str
    partial_results: Dict[str, Any] = Field(default_factory=dict, description="Şu ana kadar elde edilen bulgular")
    open_questions: List[str] = Field(default_factory=list, description="Hâlâ cevaplanması gereken sorular")
    attached_artifacts: List[str] = Field(default_factory=list, description="Devredilen dosya/artifact ID'leri")
    priority_level: int = Field(default=3, ge=1, le=5)

# --- GÖREV 4: Budget-aware Delegation ---

class ModelStrength(str, Enum):
    LOCAL_TINY = "local_tiny"     # Örn: Llama-3B
    LOCAL_STRONG = "local_strong" # Örn: Llama-70B
    CLOUD_FAST = "cloud_fast"     # Örn: Gemini Flash
    CLOUD_POWERFUL = "cloud_powerful" # Örn: Gemini Pro / GPT-4

class BudgetPolicy(BaseModel):
    """Görevin maliyet ve model gücü bazlı kısıtları."""
    max_cost_usd: float = Field(default=0.0)
    cost_ceiling_reached: bool = Field(default=False)
    preferred_strength: ModelStrength = Field(default=ModelStrength.LOCAL_STRONG)
    currency: str = Field(default="USD")

class ModelRoutingConfig(BaseModel):
    """Görevin karmaşıklığına göre hangi modele gideceğinin kararı."""
    task_complexity_score: float = Field(..., ge=0.0, le=1.0)
    forced_model: Optional[str] = None
    fallback_allowed: bool = Field(default=True)
