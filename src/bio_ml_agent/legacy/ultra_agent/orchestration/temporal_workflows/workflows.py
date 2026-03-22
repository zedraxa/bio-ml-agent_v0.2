import logging
from datetime import timedelta
from typing import Dict, Any

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from bio_ml_agent.ultra_agent.orchestration.temporal_workflows.activities import (
        index_workspace_activity,
        run_virtual_screening_activity,
    )

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


@workflow.defn
class VirtualScreeningWorkflow:
    """Pillar 4-1: Sanal Tarama (Virtual Screening) Temporal Workflow.

    Bir hedef protein ve kimyasal kütüphane verildiğinde,
    otonom olarak en iyi ilaç adaylarını raporlar.

    Kullanım:
        client = await TemporalClient.connect("localhost:7233")
        result = await client.execute_workflow(
            VirtualScreeningWorkflow.run,
            {
                "target_protein": "1CRN",
                "smiles_library": ["CCO", "CC(=O)O", ...],
                "max_candidates": 5,
            },
            id="vs-run-001",
            task_queue="bio-ml-queue",
        )
    """

    def __init__(self):
        self._is_cancelled = False
        self._progress: Dict[str, Any] = {"phase": "idle", "pct": 0}

    @workflow.signal
    def cancel_screening(self) -> None:
        self._is_cancelled = True
        log.info("Virtual Screening iptal sinyali aldı!")

    @workflow.signal
    def update_progress(self, progress: Dict[str, Any]) -> None:
        """Live Telemetry: İlerleme durumunu günceller."""
        self._progress = progress

    @workflow.query
    def get_progress(self) -> Dict[str, Any]:
        """Live Telemetry: Anlık ilerleme sorgulama."""
        return self._progress

    @workflow.run
    async def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        log.info(f"Virtual Screening Workflow başladı: {params.get('target_protein', 'N/A')}")

        self._progress = {"phase": "screening", "pct": 10}

        if self._is_cancelled:
            return {"status": "cancelled", "reason": "Kill-switch before screening"}

        self._progress = {"phase": "lipinski_filtering", "pct": 30}

        # Virtual Screening Activity'yi çalıştır
        result = await workflow.execute_activity(
            run_virtual_screening_activity,
            params,
            schedule_to_close_timeout=timedelta(minutes=30),
            heartbeat_timeout=timedelta(minutes=5),
        )

        if self._is_cancelled:
            return {"status": "cancelled", "reason": "Kill-switch after screening"}

        self._progress = {"phase": "completed", "pct": 100}
        return result
