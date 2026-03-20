import pytest
from datetime import datetime

# Import Singleton Services from previous phases
from bio_ml_agent.services.unified_storage_service import storage_service
from bio_ml_agent.services.unified_execution_service import execution_service
from bio_ml_agent.services.unified_observability_service import observability_service

# Import required Enums/Models
from bio_ml_agent.models.unified_storage import CacheItemStatus, ArtifactTier
from bio_ml_agent.models.unified_execution import ExecutionRole, ResourceRequirement, NodeLocation, HandoffChannel, HandoffStatus
from bio_ml_agent.models.unified_observability import SpanType, CostCategory, SystemActionType

def test_release_candidate_e2e_flow():
    """
    Bu test fonksiyonu Ajan Platformu'nun tüm parçalarının (Storage, Execution, Observability) 
    birlikte uçtan uca çalışabildiğini doğrular (RC Validation).
    """
    project_id = "rc-proj-1"
    
    # 1. Storage Plane - Local & Remote Workspace Initialization
    cache = storage_service.sync_local_cache(project_id, "/workspace/rc", 5000)
    assert cache.status == CacheItemStatus.SYNCED
    art = storage_service.register_artifact(project_id, "rc-model.pt", "s3-bucket")
    assert art.tier == ArtifactTier.HOT
    
    # 2. Observability Plane - Start Run Timeline & Cost Tracing
    tl = observability_service.start_run_timeline("run-rc-1")
    span_think = observability_service.add_span("run-rc-1", SpanType.THINKING, "Plan RC Workflow")
    observability_service.end_span("run-rc-1", span_think.span_id, {"res": "planned"})
    assert tl.total_duration_ms >= 0

    cost = observability_service.track_cost("run-rc-1", CostCategory.LLM_TOKEN, "openai", 0.02, "500 tokens")
    assert cost.amount_usd == 0.02
    
    # 3. Execution Plane - Graph Creation & Hybrid Node Routing
    graph = execution_service.create_run_graph(project_id)
    node_coder = execution_service.add_node_to_graph(graph.graph_id, ExecutionRole.CODER, "Write script")
    assert graph.nodes[node_coder.node_id].role == ExecutionRole.CODER
    
    # Cloud Offload Validation (High RAM Requirement -> REMOTE CPU)
    req = ResourceRequirement(min_ram_gb=128, requires_gpu=False)
    decision = execution_service.route_task_to_environment("task-rc-1", {"do": "heavy_process"}, req)
    assert decision.selected_environment == NodeLocation.REMOTE_CPU
    
    # 4. Omnichannel HITL (Approvals & WhatsApp Adapter Validation)
    handoff = execution_service.suspend_for_human_approval(
        "task-rc-1", HandoffChannel.WHATSAPP, "Need budget approval", {"cost": 50}
    )
    assert handoff.status == HandoffStatus.PENDING
    
    # Simulate user approving via WhatsApp
    resolved = execution_service.resume_handoff(handoff.handoff_id, HandoffStatus.APPROVED, {"msg": "Approve"})
    assert resolved.status == HandoffStatus.APPROVED
    
    # 5. Rollback & Recovery Validation (Browser Failure Replay)
    obs_span = observability_service.add_span("run-rc-1", SpanType.BROWSER_ACTION, "Nav to AWS")
    session = observability_service.record_browser_failure("run-rc-1", "Captcha blocked", [])
    observability_service.end_span("run-rc-1", obs_span.span_id, error="Captcha blocked")
    assert session.failure_reason == "Captcha blocked"
    
    # 6. Audit & Approvals (Per-user Audit)
    audit = observability_service.log_user_action("user-rc-1", SystemActionType.TASK_APPROVE, "task-rc-1")
    assert audit.action_type == SystemActionType.TASK_APPROVE
    
    # Check Billing Guardrails
    billing = observability_service.generate_project_billing(project_id, ["run-rc-1"])
    assert billing.total_cost_usd > 0
