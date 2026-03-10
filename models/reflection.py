from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# --- GÖREV 1: Failure Taxonomy ---
class FailureCategory(str, Enum):
    PLANNING_FAILURE = "planning_failure"
    TOOL_MISUSE = "tool_misuse"
    HALLUCINATED_FILE = "hallucinated_file"
    STALE_SELECTOR = "stale_selector"
    INVALID_CODE_PATCH = "invalid_code_patch"
    WEAK_CITATION = "weak_citation"
    TIMEOUT = "timeout"
    API_ERROR = "api_error"

class FailureRecord(BaseModel):
    """Hata kaydı ve analizi."""
    failure_id: str
    run_id: str
    category: FailureCategory
    error_message: str
    trace_context: Optional[str] = None # İlgili span_id
    root_cause_analysis: str
    user_correction: Optional[str] = None

# --- GÖREV 2 & 3: Adaptation & Correction ---
class RefinementRule(BaseModel):
    """Hata sonrası çıkarılan yeni davranış kuralı."""
    rule_id: str
    scope: str # "global", "domain:bio", "tool:browser"
    pattern: str # Hata paterni açıklaması
    correction: str # Düzeltilmiş davranış kuralı
    source_failure_ids: List[str]
    confidence: float = Field(ge=0.0, le=1.0)
    is_active: bool = Field(default=True)

# --- GÖREV 4 & 5: Pattern Mining & Reports ---
class ReflectionReport(BaseModel):
    """Haftalık/Dönemlik öz-yansıma raporu."""
    report_id: str
    period_start: str
    period_end: str
    top_failure_patterns: List[Dict[str, Any]]
    new_rules_derived: int
    quality_trend: Dict[str, float] # accuracy over time
    recommendations: List[str]
