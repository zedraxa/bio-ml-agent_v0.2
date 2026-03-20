import asyncio
import logging
from temporalio.client import Client
from temporalio.worker import Worker

from bio_ml_agent.ultra_agent.orchestration.temporal_workflows.activities import index_workspace_activity, run_virtual_screening_activity
from bio_ml_agent.ultra_agent.orchestration.temporal_workflows.workflows import AgentWorkspaceIndexingWorkflow, VirtualScreeningWorkflow

log = logging.getLogger("bio_ml_agent")

async def main():
    # Geliştirme/test aşamasındaki lokal Temporal ortamına bağlanın.
    try:
        from bio_ml_agent.utils.config import get_config
        app_config = get_config()
        # Temporal cluster address could be in config too, using default for now
        client = await Client.connect("localhost:7233")
    except Exception as e:
        log.error(f"Lokal Temporal ağı kurulamadı (Docker çalışmıyor olabilir): {e}")
        return

    # İşçiyi (worker) hazırla
    worker = Worker(
        client,
        task_queue="bio-ml-queue",
        workflows=[AgentWorkspaceIndexingWorkflow, VirtualScreeningWorkflow],
        activities=[index_workspace_activity, run_virtual_screening_activity],
    )
    
    log.info("Temporal Worker başlatılıyor. (bio-ml-queue)")
    await worker.run()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
