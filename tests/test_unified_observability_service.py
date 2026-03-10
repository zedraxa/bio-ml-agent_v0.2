import pytest
from datetime import datetime
from services.unified_observability_service import observability_service
from models.unified_observability import (
    SpanType, CostCategory, SystemActionType
)

def test_observability_timelines():
    tl = observability_service.start_run_timeline("run-100")
    assert tl.run_id == "run-100"
    
    span = observability_service.add_span(
        "run-100", SpanType.THINKING, "Planning Node"
    )
    assert span.name == "Planning Node"
    
    observability_service.end_span("run-100", span.span_id, {"res": "ok"})
    assert span.status == "ok"
    assert observability_service._timelines["run-100"].total_duration_ms >= 0

def test_observability_failure_replay():
    session = observability_service.record_browser_failure(
        "run-200", "Timeout on click", []
    )
    assert session.failure_reason == "Timeout on click"

def test_observability_cost_trace():
    c1 = observability_service.track_cost(
        "run-300", CostCategory.LLM_TOKEN, "openai", 0.05, "100 tok"
    )
    c2 = observability_service.track_cost(
        "run-300", CostCategory.COMPUTE_GPU, "aws", 1.20, "1 hr"
    )
    
    trace = observability_service.generate_project_billing(
        project_id="proj-3", run_ids=["run-300"]
    )
    assert trace.total_cost_usd == 1.25

def test_observability_audits():
    audit = observability_service.log_user_action(
        "user-7", SystemActionType.PROJECT_CREATE, "proj-4"
    )
    assert audit.target_resource_id == "proj-4"
    
    rec = observability_service.generate_secret_receipt(
        "DB_PASS", "run-500", "DB Migration", agent="coder"
    )
    assert rec.accessed_by_agent == "coder"
