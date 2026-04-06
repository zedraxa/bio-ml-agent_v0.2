"""Temporal workflows — re-exported from legacy; requires temporalio to be installed."""
try:
    from bio_ml_agent.legacy.ultra_agent.orchestration.temporal_workflows.workflows import (
        VirtualScreeningWorkflow,
        AgentWorkspaceIndexingWorkflow,
    )
    __all__ = ["VirtualScreeningWorkflow", "AgentWorkspaceIndexingWorkflow"]
except ImportError:
    # temporalio not installed — expose stub classes so imports don't crash at module load
    import logging
    logging.getLogger("bio_ml_agent").warning(
        "temporalio not installed — VirtualScreeningWorkflow is a stub."
    )

    class VirtualScreeningWorkflow:  # type: ignore[no-redef]
        pass

    class AgentWorkspaceIndexingWorkflow:  # type: ignore[no-redef]
        pass

    __all__ = ["VirtualScreeningWorkflow", "AgentWorkspaceIndexingWorkflow"]
