from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# --- GÖREV 1 & 4: Golden Scenarios & Demo Flows ---
class ScenarioDifficulty(str, Enum):
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    PRODUCTION_GRADE = "production_grade"

class EndToEndScenario(BaseModel):
    """Uçtan uca sistem testi veya altın demo senaryosu."""
    scenario_id: str
    title: str # Örn: "DICOM Ingest -> Segmentation -> Report"
    difficulty: ScenarioDifficulty
    required_services: List[str] # ["browser", "gpu_node", "qdrant"]
    entry_prompt: str
    expected_final_artifact_type: str # "scientific_report"
    is_demo_flow: bool = False

class DemoFlow(EndToEndScenario):
    """Yatırımcı veya laboratuvar sunumları için kurgulanmış akış."""
    target_audience: str # "investor", "researcher", "student"
    demo_script_notes: str

# --- GÖREV 2: Chaos Engineering ---
class ChaosAction(str, Enum):
    BROWSER_CRASH = "browser_crash"
    NODE_RESET = "node_reset"
    DB_LATENCY_INJECTION = "db_latency"
    SECRET_EXPIRATION = "secret_expiry"
    APPROVAL_TIMEOUT = "approval_timeout"

class ChaosTestConfig(BaseModel):
    """Sistemin hatalara dayanıklılığını test eden kaos senaryosu."""
    test_id: str
    base_scenario_id: str
    injected_chaos: ChaosAction
    injection_delay_seconds: int
    expected_system_behavior: str # "System should pause and request HITL"

# --- GÖREV 3: Recovery Validation ---
class RecoveryValidation(BaseModel):
    """Kurtarma işlemlerinin tutarlılığını onaylayan veri modeli."""
    validation_id: str
    chaos_test_id: str
    state_resumed_successfully: bool
    data_loss_detected: bool = False
    duplicate_billing_detected: bool = False
    recovery_time_ms: int
