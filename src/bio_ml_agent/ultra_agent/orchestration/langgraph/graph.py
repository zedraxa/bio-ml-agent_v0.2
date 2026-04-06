"""
LangGraph orchestration graph — with graceful fallback if langgraph is not installed.
"""
import logging

log = logging.getLogger("bio_ml_agent")

def build_graph():
    """
    Builds the Plan → Execute → Verify → Artifact graph topology.
    Falls back to a no-op stub when the langgraph library is not available.
    """
    try:
        from bio_ml_agent.legacy.ultra_agent.orchestration.langgraph.graph import build_graph as _build_graph
        return _build_graph()
    except ImportError:
        log.warning("langgraph not installed — using passthrough stub graph.")
        return _StubGraph()


class _StubGraph:
    """Minimal stub that immediately reaches the 'artifact' node so agent_core falls through to _tool_loop."""

    def stream(self, state):
        yield {"artifact": state}
