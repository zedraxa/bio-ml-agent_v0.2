import random
import logging
from typing import List, Optional, Any
from .models import MissionPlan, StepStatus
from .persistence import MISSION_STORE

logger = logging.getLogger("bio_ml_agent")

class ChaosMonkey:
    """
    Axis F: Injects failures into the mission execution flow 
    to test the resilience of the Unified Brain.
    """
    def __init__(self, failure_rate: float = 0.2):
        self.failure_rate = failure_rate
        self.last_failed_step_id: Optional[str] = None
        
    def should_fail(self) -> bool:
        """Determines if the next step should fail based on the failure rate."""
        return random.random() < self.failure_rate

    def inject_failure(self, plan: MissionPlan, step_index: int):
        """Forces a step to fail and records the failure."""
        if step_index >= len(plan.steps):
            return
            
        step = plan.steps[step_index]
        logger.warning(f"[Chaos Monkey] Injecting failure into step: {step.step_id} ({step.description})")
        
        step.status = StepStatus.FAILED
        step.metadata["error_message"] = "[Chaos Monkey] Simulated agent failure."
        self.last_failed_step_id = step.step_id
        
        # Save state to trigger replan logic in the orchestrator
        MISSION_STORE.save(plan)
        
    def simulate_latency(self):
        """Simulates API or network latency."""
        import time
        lat = random.uniform(0.5, 2.0)
        logger.info(f"[Chaos Monkey] Simulating {lat:.2f}s latency...")
        time.sleep(lat)

    def simulate_stuck_agent(self, plan: MissionPlan, step_id: str):
        """F5: Simulates an agent that hangs in RUNNING state."""
        step = next((s for s in plan.steps if s.step_id == step_id), None)
        if not step: return
        step.status = StepStatus.RUNNING 
        logger.warning(f"[Chaos Monkey] Simulating STUCK agent for {step_id}")

    def simulate_missing_artifact(self, plan: MissionPlan, step_id: str):
        """F5: Simulates a missing required input artifact."""
        step = next((s for s in plan.steps if s.step_id == step_id), None)
        if not step: return
        step.input_artifacts = [] # Empty inputs even if depends_on succeeded
        logger.warning(f"[Chaos Monkey] Simulating MISSING ARTIFACT for {step_id}")

    def simulate_low_confidence(self, plan: MissionPlan, step_id: str):
        """F5: Simulates an agent result with low confidence."""
        step = next((s for s in plan.steps if s.step_id == step_id), None)
        if not step: return
        step.metadata["confidence_score"] = 0.2 # Critical failure threshold
        logger.warning(f"[Chaos Monkey] Simulating LOW CONFIDENCE for {step_id}")

    def simulate_conflict(self, graph: Any, artifact_id: str):
        """F5/E4: Simulates a version conflict on a specific artifact."""
        if artifact_id not in graph.nodes: return
        
        orig_node = graph.nodes[artifact_id]
        from .agent_contract import ArtifactRecord, ArtifactReviewStatus
        
        conflict_rec = ArtifactRecord(
            artifact_id=f"{artifact_id}_conflict",
            artifact_type=orig_node.record.artifact_type,
            title=f"Divergent {orig_node.record.title}",
            producer_agent_id="adversarial_agent",
            producer_agent_role="critic",
            mission_id=orig_node.record.mission_id,
            project_id=orig_node.record.project_id,
            parent_artifact_id=orig_node.record.parent_artifact_id,
            status=ArtifactReviewStatus.DRAFT
        )
        graph.add_artifact(conflict_rec)
        logger.warning(f"[Chaos Monkey] Simulating VERSION CONFLICT for {artifact_id}")
