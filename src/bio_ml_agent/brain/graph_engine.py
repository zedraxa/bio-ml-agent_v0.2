from typing import List, Dict, Any, Set
import logging
from .models import (
    AgentGraph, 
    AgentNode, 
    MissionStep, 
    StepStatus
)
from .feature_flags import FEATURE_CONTROLLER

logger = logging.getLogger(__name__)

# Capability tracking for graph construction
AGENT_CAPABILITIES = {
    "researcher": {"model_tier": 2, "can_do": ["search", "read", "summarize"]},
    "data_engineer": {"model_tier": 1, "can_do": ["clean", "transform", "sql"]},
    "ml_expert": {"model_tier": 3, "can_do": ["train", "evaluate", "optimize"]},
    "academic_expert": {"model_tier": 2, "can_do": ["write", "cite", "review"]},
    "critic": {"model_tier": 3, "can_do": ["validate", "flag", "score"]},
    "microscopy_agent": {"model_tier": 2, "can_do": ["segment", "identify", "extract"]},
    "in_silico_expert": {"model_tier": 3, "can_do": ["docking", "alignment", "simulation"]},
    "coding_agent": {"model_tier": 2, "can_do": ["debug", "refactor", "test"]},
}

class MissionGraphEngine:
    """
    Part VI: Mission Graph Engine Module.
    
    Orchestrates the creation and topological management of agent 
    interaction graphs (DAGs).
    """

    def build_graph(self, mission_id: str, steps: List[MissionStep]) -> AgentGraph:
        """Build a DAG of agents from the mission steps."""
        agent_map: Dict[str, AgentNode] = {}
        
        for step in steps:
            # Axis H4: Check Agent Feature Flag
            agent_role = step.assigned_agent
            role_val = agent_role.value if hasattr(agent_role, 'value') else str(agent_role)
            agent_feature = f"agent.{role_val}"
            
            if not FEATURE_CONTROLLER.is_enabled(agent_feature):
                logger.error(f"[GraphEngine:H4] AGENT BLOCKED: '{role_val}' is disabled.")
                step.status = StepStatus.CANCELLED
                continue

            agent_id = f"{role_val}_{step.step_id}"
            caps = AGENT_CAPABILITIES.get(role_val, {"model_tier": 2, "can_do": []})
            
            node = AgentNode(
                agent_id=agent_id,
                role=agent_role,
                model_tier=caps.get("model_tier", 2),
                capabilities=caps.get("can_do", []),
            )
            agent_map[step.step_id] = node
        
        # Wire up edges based on step dependencies
        for step in steps:
            if step.step_id not in agent_map:
                continue
                
            node = agent_map[step.step_id]
            for dep_id in step.depends_on:
                if dep_id in agent_map:
                    upstream_node = agent_map[dep_id]
                    node.upstream.append(upstream_node.agent_id)
                    upstream_node.downstream.append(node.agent_id)
        
        # Build execution order (topological layers)
        execution_order = self.topological_sort(steps)
        total_time = sum(s.estimated_duration_seconds for s in steps)
        
        return AgentGraph(
            mission_id=mission_id,
            nodes=list(agent_map.values()),
            execution_order=execution_order,
            total_estimated_seconds=total_time,
        )

    def topological_sort(self, steps: List[MissionStep]) -> List[List[str]]:
        """Sort steps into parallel execution layers."""
        completed: Set[str] = set()
        layers: List[List[str]] = []
        remaining = [s for s in steps if s.status != StepStatus.CANCELLED]
        
        while remaining:
            ready = [s for s in remaining if all(d in completed for d in s.depends_on)]
            
            if not ready:
                # Handle remaining or errors
                if remaining:
                    layers.append([s.step_id for s in remaining])
                break
            
            layer = [s.step_id for s in ready]
            layers.append(layer)
            completed.update(layer)
            remaining = [s for s in remaining if s.step_id not in completed]
        
        return layers
