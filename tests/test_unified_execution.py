import pytest
from datetime import datetime, timezone
from bio_ml_agent.models.unified_execution import (
    ExecutionRole, ExecutionNode, UnifiedRunGraph, GraphEdge,
    NodeLocation, HybridExecutionTask,
    HandoffChannel, HandoffStatus, HandoffRequest,
    ResourceRequirement, ExecutionRouterDecision
)

def test_unified_run_graph():
    node1 = ExecutionNode(
        node_id="n1",
        role=ExecutionRole.PLANNER,
        task_description="Plan the pipeline"
    )
    node2 = ExecutionNode(
        node_id="n2",
        role=ExecutionRole.BIO_ANALYST,
        task_description="Analyze the genome",
        dependencies=["n1"]
    )
    edge = GraphEdge(source_node_id="n1", target_node_id="n2")

    graph = UnifiedRunGraph(
        graph_id="g1",
        project_id="p1",
        nodes={"n1": node1, "n2": node2},
        edges=[edge],
        created_at=datetime.now(timezone.utc)
    )

    assert len(graph.nodes) == 2
    assert graph.nodes["n2"].dependencies[0] == "n1"
    assert graph.status == "initialized"

def test_hybrid_execution_task():
    task = HybridExecutionTask(
        task_id="t1",
        assigned_location=NodeLocation.REMOTE_GPU,
        payload={"model": "pytorch-resnet"}
    )
    assert task.assigned_location == NodeLocation.REMOTE_GPU
    assert task.payload["model"] == "pytorch-resnet"

def test_handoff_request():
    req = HandoffRequest(
        handoff_id="h1",
        related_task_id="t1",
        requested_channel=HandoffChannel.WHATSAPP,
        reason="Please approve this $50 cloud spending",
        context_data={"cost": 50},
        requested_at=datetime.now(timezone.utc)
    )
    assert req.status == HandoffStatus.PENDING
    assert req.requested_channel == "whatsapp"

def test_execution_router_decision():
    req = ResourceRequirement(requires_gpu=True, min_ram_gb=16)
    decision = ExecutionRouterDecision(
        decision_id="d1",
        task_id="t1",
        requirements=req,
        selected_environment=NodeLocation.REMOTE_GPU,
        reasoning="Task needs GPU, local has no GPU.",
        timestamp=datetime.now(timezone.utc)
    )
    assert decision.selected_environment == NodeLocation.REMOTE_GPU
    assert decision.requirements.requires_gpu is True
