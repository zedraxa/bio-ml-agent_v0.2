import pytest
from datetime import datetime, timezone
from bio_ml_agent.models.unified_observability import (
    SpanType, TraceSpan, RunTimeline,
    BrowserTimelineEvent, FailureReplaySession,
    CostCategory, CloudCostEntry, ProjectBillingTrace,
    SystemActionType, UserAuditLog,
    SecretAccessReceipt
)

def test_trace_span_and_timeline():
    span = TraceSpan(
        span_id="s1",
        run_id="r1",
        span_type=SpanType.TOOL_EXECUTION,
        name="Search Web",
        started_at=datetime.now(timezone.utc)
    )
    timeline = RunTimeline(
        timeline_id="tl1",
        run_id="r1",
        spans=[span],
        created_at=datetime.now(timezone.utc)
    )
    assert len(timeline.spans) == 1
    assert timeline.spans[0].span_type == SpanType.TOOL_EXECUTION

def test_failure_replay_session():
    event = BrowserTimelineEvent(
        event_id="e1",
        run_id="r2",
        url="https://example.com",
        action_type="click",
        timestamp=datetime.now(timezone.utc)
    )
    session = FailureReplaySession(
        replay_id="rep1",
        run_id="r2",
        failure_reason="Element not found",
        events=[event],
        recorded_at=datetime.now(timezone.utc)
    )
    assert session.failure_reason == "Element not found"
    assert session.events[0].action_type == "click"

def test_cloud_cost_trace():
    entry = CloudCostEntry(
        cost_id="c1",
        run_id="r3",
        category=CostCategory.LLM_TOKEN,
        resource_provider="openai",
        amount_usd=0.05,
        usage_metric="1000 tokens",
        timestamp=datetime.now(timezone.utc)
    )
    trace = ProjectBillingTrace(
        trace_id="b1",
        project_id="p1",
        total_cost_usd=0.05,
        entries=[entry],
        generated_at=datetime.now(timezone.utc)
    )
    assert trace.total_cost_usd == 0.05
    assert trace.entries[0].resource_provider == "openai"

def test_user_audit_and_receipts():
    audit = UserAuditLog(
        audit_id="a1",
        user_id="u1",
        action_type=SystemActionType.TASK_APPROVE,
        target_resource_id="task-99",
        timestamp=datetime.now(timezone.utc)
    )
    assert audit.action_type == SystemActionType.TASK_APPROVE

    receipt = SecretAccessReceipt(
        receipt_id="rec1",
        secret_name="AWS_KEY",
        run_id="r4",
        justification="For S3 Upload",
        accessed_by_agent="coder_node",
        accessed_at=datetime.now(timezone.utc)
    )
    assert receipt.secret_name == "AWS_KEY"
    assert receipt.is_authorized is True
