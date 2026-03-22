import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from .models import MissionPlan, StepStatus, ProjectState
from .persistence import MISSION_STORE, PROJECT_STORE
from .artifact_graph import ArtifactGraph
from .agent_contract import ArtifactRecord

logger = logging.getLogger("bio_ml_agent")

class RegressionIssue(BaseModel):
    issue_type: str # 'missing_step', 'agent_mismatch', 'artifact_count_mismatch', 'status_regression'
    description: str
    original_value: Any
    new_value: Any

class RegressionReport(BaseModel):
    mission_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_regression: bool
    issues: List[RegressionIssue] = Field(default_factory=list)

class MissionSnapshot(BaseModel):
    snapshot_id: str
    mission_plan: MissionPlan
    artifact_graph: ArtifactGraph
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ScenarioReplaySystem:
    """
    Axis F2: Scenario Replay System.
    Records and replays missions to ensure stability and quality across versions.
    """
    
    def __init__(self, storage_dir: str = "/tmp/bio_ml_agent/snapshots"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, snapshot_id: str) -> Path:
        return self.storage_dir / f"{snapshot_id}.json"

    def record_snapshot(self, mission_id: str, graph: ArtifactGraph) -> str:
        """Saves a 'Golden' version of a mission and its outputs."""
        plan = MISSION_STORE.load(mission_id)
        if not plan:
            raise ValueError(f"Mission {mission_id} not found in store.")
            
        snapshot_id = f"snap_{mission_id}_{int(datetime.now(timezone.utc).timestamp())}"
        snapshot = MissionSnapshot(
            snapshot_id=snapshot_id,
            mission_plan=plan,
            artifact_graph=graph
        )
        
        path = self._get_path(snapshot_id)
        with open(path, "w") as f:
            f.write(snapshot.model_dump_json(indent=2))
            
        logger.info(f"[Axis F2] Snapshot recorded: {snapshot_id}")
        return snapshot_id

    def load_snapshot(self, snapshot_id: str) -> Optional[MissionSnapshot]:
        path = self._get_path(snapshot_id)
        if not path.exists():
            return None
        with open(path, "r") as f:
            return MissionSnapshot(**json.load(f))

    def compare(self, original: MissionSnapshot, current_plan: MissionPlan, current_graph: ArtifactGraph) -> RegressionReport:
        """Compares a current run against a golden snapshot."""
        issues = []
        
        # 1. Step Comparison
        orig_steps = original.mission_plan.steps
        curr_steps = current_plan.steps
        
        if len(orig_steps) != len(curr_steps):
            issues.append(RegressionIssue(
                issue_type="step_count_mismatch",
                description="Number of steps in the plan has changed.",
                original_value=len(orig_steps),
                new_value=len(curr_steps)
            ))
            
        # 2. Agent Assignment Comparison
        for i, (o_step, c_step) in enumerate(zip(orig_steps, curr_steps)):
            if o_step.assigned_agent != c_step.assigned_agent:
                issues.append(RegressionIssue(
                    issue_type="agent_mismatch",
                    description=f"Step {i} ({o_step.title}) assigned to different agent.",
                    original_value=o_step.assigned_agent,
                    new_value=c_step.assigned_agent
                ))
            if o_step.status == StepStatus.COMPLETED and c_step.status != StepStatus.COMPLETED:
                issues.append(RegressionIssue(
                    issue_type="status_regression",
                    description=f"Step {i} was completed in golden run but failed/incomplete now.",
                    original_value=o_step.status,
                    new_value=c_step.status
                ))

        # 3. Artifact Count Comparison
        orig_art_count = len(original.artifact_graph.nodes)
        curr_art_count = len(current_graph.nodes)
        
        if orig_art_count != curr_art_count:
            issues.append(RegressionIssue(
                issue_type="artifact_count_mismatch",
                description="Total number of artifacts produced has changed.",
                original_value=orig_art_count,
                new_value=curr_art_count
            ))

        return RegressionReport(
            mission_id=original.mission_plan.mission_id,
            is_regression=len(issues) > 0,
            issues=issues
        )
