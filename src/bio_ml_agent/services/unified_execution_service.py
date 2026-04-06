from typing import Dict, List, Optional
from datetime import datetime, timezone
import uuid
import logging

from bio_ml_agent.models.unified_execution import (
    UnifiedRunGraph, ExecutionNode, ExecutionRole, GraphEdge,
    NodeLocation, HybridExecutionTask,
    HandoffRequest, HandoffChannel, HandoffStatus,
    ExecutionRouterDecision, ResourceRequirement
)

logger = logging.getLogger(__name__)

class UnifiedExecutionService:
    def __init__(self):
        # 1. Run Graph Management
        self._graphs: Dict[str, UnifiedRunGraph] = {}
        # 2. Hybrid Tasks
        self._hybrid_tasks: Dict[str, HybridExecutionTask] = {}
        # 3. Handoff Requests (HITL)
        self._handoffs: Dict[str, HandoffRequest] = {}
        # 4. Router Decisions
        self._router_decisions: Dict[str, ExecutionRouterDecision] = {}

    # --- 1. Run Graph Core ---
    def create_run_graph(self, project_id: str) -> UnifiedRunGraph:
        graph_id = f"graph-{uuid.uuid4().hex[:8]}"
        graph = UnifiedRunGraph(
            graph_id=graph_id,
            project_id=project_id,
            created_at=datetime.now(timezone.utc),
            status="initialized"
        )
        self._graphs[graph_id] = graph
        return graph

    def add_node_to_graph(self, graph_id: str, role: ExecutionRole, desc: str, dependencies: List[str] = None) -> ExecutionNode:
        if graph_id not in self._graphs:
            raise ValueError(f"Graph {graph_id} not found.")

        node_id = f"node-{uuid.uuid4().hex[:6]}"
        node = ExecutionNode(
            node_id=node_id,
            role=role,
            task_description=desc,
            dependencies=dependencies or [],
            status="pending"
        )
        self._graphs[graph_id].nodes[node_id] = node

        # Auto-create edges from dependencies
        for dep in (dependencies or []):
            if dep in self._graphs[graph_id].nodes:
                edge = GraphEdge(source_node_id=dep, target_node_id=node_id)
                self._graphs[graph_id].edges.append(edge)

        return node

    def get_graph(self, graph_id: str) -> Optional[UnifiedRunGraph]:
        return self._graphs.get(graph_id)

    # --- 2. Hybrid Execution Routing ---
    def route_task_to_environment(self, task_id: str, payload: dict, requirements: ResourceRequirement) -> ExecutionRouterDecision:
        # Simple heuristic routing algorithm
        location = NodeLocation.LOCAL
        reason = "Default to local"

        if requirements.requires_gpu:
            location = NodeLocation.REMOTE_GPU
            reason = "Task explicitly requires GPU"
        elif requirements.min_ram_gb > 32:
            location = NodeLocation.REMOTE_CPU
            reason = f"High RAM requirement ({requirements.min_ram_gb}GB)"

        decision_id = f"route-{uuid.uuid4().hex[:6]}"
        decision = ExecutionRouterDecision(
            decision_id=decision_id,
            task_id=task_id,
            requirements=requirements,
            selected_environment=location,
            reasoning=reason,
            timestamp=datetime.now(timezone.utc)
        )
        self._router_decisions[decision_id] = decision

        # Save payload as a Hybrid Task
        task = HybridExecutionTask(
            task_id=task_id,
            assigned_location=location,
            payload=payload,
            started_at=datetime.now(timezone.utc)
        )
        self._hybrid_tasks[task_id] = task
        logger.info(f"Routed task {task_id} to {location.value}. Reason: {reason}")

        return decision

    # --- 3. Omnichannel HITL & Handoff ---
    def suspend_for_human_approval(self, task_id: str, channel: HandoffChannel, reason: str, context: dict) -> HandoffRequest:
        handoff_id = f"hitl-{uuid.uuid4().hex[:6]}"
        req = HandoffRequest(
            handoff_id=handoff_id,
            related_task_id=task_id,
            requested_channel=channel,
            reason=reason,
            context_data=context,
            requested_at=datetime.now(timezone.utc)
        )
        self._handoffs[handoff_id] = req
        logger.warning(f"Task {task_id} suspended. Waiting for {channel.value} human approval.")
        return req

    def resume_handoff(self, handoff_id: str, action: HandoffStatus, resolution_data: dict = None) -> HandoffRequest:
        if handoff_id not in self._handoffs:
            raise ValueError(f"Handoff request {handoff_id} not found.")

        req = self._handoffs[handoff_id]
        if req.status != HandoffStatus.PENDING:
            raise ValueError("Handoff is already resolved.")

        req.status = action
        req.resolution_data = resolution_data
        req.resolved_at = datetime.now(timezone.utc)

        # Notify task resume (mock logic)
        logger.info(f"Handoff {handoff_id} resolved with status {action.value}. Resuming task {req.related_task_id}.")
        return req

# Singleton Service
execution_service = UnifiedExecutionService()
