# services/agent/orchestration.py
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Generator, Any
from pathlib import Path

from bio_ml_agent.core.config import SYSTEM_PROMPT
from bio_ml_agent.core.message_normalizer import MessageNormalizer
from bio_ml_agent.ultra_agent.control.router import LLMRouter
from bio_ml_agent.ultra_agent.observability.metrics import metrics as otel_metrics
from bio_ml_agent.core.agent_core import AgentCore

log = logging.getLogger("bio_ml_agent")

async def submit_temporal_job(user_msg: str, session_id: str, workspace: Path):
    """Temporal kümesine uzun süreli görev iletir."""
    try:
        from temporalio.client import Client
        from bio_ml_agent.ultra_agent.orchestration.temporal_workflows.workflows import VirtualScreeningWorkflow

        client = await Client.connect("localhost:7233")
        target_protein = "1CRN" # Default
        words = user_msg.split()
        for w in words:
            if len(w) == 4 and w.isalnum() and not w.isalpha() and not w.isdigit():
                target_protein = w.upper()
                break

        run_id = f"vs-run-{session_id}-{datetime.now().strftime('%M%S')}"

        handle = await client.start_workflow(
            VirtualScreeningWorkflow.run,
            {
                "target_protein": target_protein,
                "smiles_library": ["CCO", "CC(=O)O", "c1ccccc1"],
                "max_candidates": 5,
                "workspace": str(workspace)
            },
            id=run_id,
            task_queue="bio-ml-queue",
        )
        return run_id, target_protein
    except Exception as e:
        log.error(f"Temporal submission failed: {e}")
        raise

def get_routing_decision(user_msg: str) -> Dict:
    """LLMRouter kullanarak model ve zorluk seviyesi belirler."""
    router = LLMRouter()
    return router.route_request(SYSTEM_PROMPT, user_msg)

def prepare_agent_core(config, project_name: str) -> AgentCore:
    """AgentCore instance'ını yapılandırır."""
    return AgentCore(config, project_name=project_name)

def handle_temporal_trigger(user_msg: str) -> bool:
    """Mesajın Temporal iş akışını tetikleyip tetiklemeyeceğini kontrol eder."""
    msg_low = user_msg.lower()
    return any(kw in msg_low for kw in ["sanal tarama", "virtual screening", "vs run"])
