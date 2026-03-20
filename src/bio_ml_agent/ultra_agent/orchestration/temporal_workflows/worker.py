import asyncio
import logging
from temporalio.client import Client
from temporalio.worker import Worker

from bio_ml_agent.ultra_agent.orchestration.temporal_workflows.activities import index_workspace_activity
from bio_ml_agent.ultra_agent.orchestration.temporal_workflows.workflows import AgentWorkspaceIndexingWorkflow

log = logging.getLogger("bio_ml_agent")

async def main():
    # Geliştirme/test aşamasındaki lokal Temporal ortamına bağlanın.
    try:
        client = await Client.connect("localhost:7233")
    except Exception as e:
        log.error(f"Lokal Temporal ağı kurulamadı (Docker çalışmıyor olabilir): {e}")
        return

    # İşçiyi (worker) hazırla
    worker = Worker(
        client,
        task_queue="bio_ml_agent_queue",
        workflows=[AgentWorkspaceIndexingWorkflow],
        activities=[index_workspace_activity],
    )
    
    log.info("Temporal Worker başlatılıyor. (bio_ml_agent_queue)")
    await worker.run()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
