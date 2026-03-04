import logging
from datetime import timedelta
from typing import Dict, Any

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from ultra_agent.orchestration.temporal_workflows.activities import index_workspace_activity

log = logging.getLogger("bio_ml_agent")

@workflow.defn
class AgentWorkspaceIndexingWorkflow:
    """
    S3-1 & S3-3: Workspace'i indekslemek için tasarlanmış basit Temporal workflow.
    """
    
    # S3-2: Cancel / Signal yetenekleri
    def __init__(self):
        self._is_cancelled = False
        
    @workflow.signal
    def cancel_workflow(self) -> None:
        self._is_cancelled = True
        log.info("Workflow iptal (Kill-Switch) sinyali aldı!")

    @workflow.run
    async def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        log.info("Indexing Workflow Başladı.")
        
        if self._is_cancelled:
            return {"status": "cancelled", "reason": "Kill-switch triggered before activity"}
            
        result = await workflow.execute_activity(
            index_workspace_activity,
            params,
            schedule_to_close_timeout=timedelta(minutes=5),
        )
        
        if self._is_cancelled:
            return {"status": "cancelled", "reason": "Kill-switch triggered after activity"}
            
        return result
