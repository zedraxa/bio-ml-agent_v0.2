import os
import json
import logging
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path
from ..models import MissionPlan, MissionStep, StepStatus, MissionSnapshot, ProjectState

logger = logging.getLogger("bio_ml_agent.brain.recovery")

class RecoveryManager:
    """
    Axis G: Reliability and Recovery Layer (R6-5).
    Manages checkpoints, retry policies, and state resumption.
    """
    def __init__(self, checkpoint_dir: str = "", policy_path: str = ""):
        self.checkpoint_dir = checkpoint_dir or str(Path(__file__).parent.parent / "checkpoint_store")
        self.policy_path = policy_path or str(Path(__file__).parent.parent / "retry_policies.yaml")
        self.policies = self.load_policies(self.policy_path)
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def load_policies(self, path: str) -> Dict[str, Any]:
        """Loads retry policies from YAML."""
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load retry policies: {e}")
            return {"policies": {}, "default_policy": {"max_retries": 1}}

    def create_checkpoint(self, mission_id: str, plan: MissionPlan, project: ProjectState) -> str:
        """Saves a full state snapshot to the checkpoint store."""
        snapshot = MissionSnapshot(
            snapshot_id=f"snap_{mission_id}_{int(os.times()[4])}",
            mission_id=mission_id,
            project_id=project.project_id,
            plan_snapshot=plan,
            project_snapshot=project
        )
        
        path = os.path.join(self.checkpoint_dir, f"{mission_id}_latest.json")
        with open(path, 'w') as f:
            f.write(snapshot.model_dump_json())
        
        logger.info(f"[Recovery:G1] Checkpoint saved: {path}")
        return path

    def resume_mission(self, mission_id: str) -> Optional[MissionSnapshot]:
        """Loads the latest checkpoint for a mission."""
        path = os.path.join(self.checkpoint_dir, f"{mission_id}_latest.json")
        if not os.path.exists(path):
            return None
            
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                return MissionSnapshot(**data)
        except Exception as e:
            logger.error(f"Failed to resume mission {mission_id}: {e}")
            return None

    def get_retry_policy(self, role: str) -> Dict[str, Any]:
        """Returns the specific retry policy for an agent role."""
        policies = self.policies.get("policies", {})
        return policies.get(role.lower(), self.policies.get("default_policy", {}))

    def should_retry(self, role: str, current_retries: int) -> bool:
        """Determines if an agent should retry based on the policy."""
        policy = self.get_retry_policy(role)
        return current_retries < policy.get("max_retries", 0)
