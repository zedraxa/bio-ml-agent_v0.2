from bio_ml_agent.legacy.ultra_agent.orchestration.langgraph.hitl import check_hitl_policy  # noqa: F401

# hitl_node alias for forward compatibility
hitl_node = check_hitl_policy

__all__ = ["check_hitl_policy", "hitl_node"]
