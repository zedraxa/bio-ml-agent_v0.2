import pytest
from datetime import datetime
from services.unified_execution_service import execution_service
from models.unified_execution import (
    ExecutionRole, NodeLocation, HandoffChannel,
    HandoffStatus, ResourceRequirement
)

def test_execution_service_graph_creation():
    graph = execution_service.create_run_graph("p-1")
    assert graph.project_id == "p-1"
    
    node = execution_service.add_node_to_graph(
        graph.graph_id, ExecutionRole.PLANNER, "Test Plan"
    )
    assert node.role == ExecutionRole.PLANNER
    
    fetched = execution_service.get_graph(graph.graph_id)
    assert len(fetched.nodes) == 1

def test_execution_service_routing():
    req = ResourceRequirement(min_ram_gb=64, requires_gpu=False)
    decision = execution_service.route_task_to_environment(
        task_id="t-1", payload={"data": "test"}, requirements=req
    )
    assert decision.selected_environment == NodeLocation.REMOTE_CPU
    assert decision.requirements.min_ram_gb == 64

def test_execution_service_handoff():
    req = execution_service.suspend_for_human_approval(
        task_id="t-2",
        channel=HandoffChannel.WEB_UI,
        reason="Needs Review",
        context={"diff": "..."}
    )
    assert req.status == HandoffStatus.PENDING
    
    resolved = execution_service.resume_handoff(
        req.handoff_id, HandoffStatus.APPROVED, {"note": "looks fine"}
    )
    assert resolved.status == HandoffStatus.APPROVED
    assert resolved.resolution_data["note"] == "looks fine"
