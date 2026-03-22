import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List
from .models import MissionPlan, ProjectState

class MissionPersistence:
    """Handles persistent storage and retrieval of Mission Plans."""
    
    def __init__(self, storage_dir: str = "/tmp/bio_ml_agent/missions"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_path(self, mission_id: str) -> Path:
        return self.storage_dir / f"{mission_id}.json"
        
    def save(self, plan: MissionPlan) -> str:
        """Saves the mission plan to disk."""
        path = self._get_path(plan.mission_id)
        with open(path, "w") as f:
            f.write(plan.model_dump_json(indent=2))
        return str(path)
        
    def load(self, mission_id: str) -> Optional[MissionPlan]:
        """Loads a mission plan from disk."""
        path = self._get_path(mission_id)
        if not path.exists():
            return None
        with open(path, "r") as f:
            data = json.load(f)
            return MissionPlan(**data)
            
    def list_missions(self) -> List[str]:
        """Lists all persistent mission IDs."""
        return [f.stem for f in self.storage_dir.glob("*.json")]

    def save_snapshot(self, snapshot: "MissionSnapshot") -> str:
        """Saves a versioned snapshot of the mission."""
        checkpoint_dir = self.storage_dir / "checkpoints" / snapshot.mission_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{snapshot.timestamp.strftime('%Y%m%d_%H%M%S')}_{snapshot.snapshot_id}.json"
        path = checkpoint_dir / filename
        with open(path, "w") as f:
            f.write(snapshot.model_dump_json(indent=2))
        return str(path)

    def get_latest_snapshot(self, mission_id: str) -> Optional["MissionSnapshot"]:
        """Retrieves the most recent snapshot for a mission."""
        checkpoint_dir = self.storage_dir / "checkpoints" / mission_id
        if not checkpoint_dir.exists():
            return None
            
        snapshots = sorted(checkpoint_dir.glob("*.json"))
        if not snapshots:
            return None
            
        latest = snapshots[-1]
        with open(latest, "r") as f:
            data = json.load(f)
            # Late import to avoid circularity if needed, 
            # but we already import MissionPlan/ProjectState.
            # mission_models is actually better.
            from .models import MissionSnapshot
            return MissionSnapshot(**data)

class ProjectPersistence:
    """Handles persistent storage and retrieval of Project States."""
    
    def __init__(self, storage_dir: str = "/tmp/bio_ml_agent/projects"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_path(self, project_id: str) -> Path:
        return self.storage_dir / f"{project_id}.json"
        
    def save(self, project: ProjectState) -> str:
        """Saves the project state to disk."""
        path = self._get_path(project.project_id)
        with open(path, "w") as f:
            f.write(project.model_dump_json(indent=2))
        return str(path)
        
    def load(self, project_id: str) -> Optional[ProjectState]:
        """Loads a project state from disk."""
        path = self._get_path(project_id)
        if not path.exists():
            return None
        with open(path, "r") as f:
            data = json.load(f)
            return ProjectState(**data)

class SyncController:
    """Manages synchronization between local and remote states."""
    
    @staticmethod
    def sync_project(local: ProjectState, remote: ProjectState) -> ProjectState:
        """
        Merges two project states. 
        Strategy: Higher version wins. If versions match, newest timestamp wins.
        """
        if remote.version > local.version:
            return remote
        elif remote.version == local.version:
            if remote.last_updated > local.last_updated:
                return remote
        
        # Local is newer or same
        local.last_synced_at = datetime.now(timezone.utc)
        return local

    @staticmethod
    def sync_mission(local: MissionPlan, remote: MissionPlan) -> MissionPlan:
        """Merges two mission plans using versioning."""
        if remote.version > local.version:
            return remote
        
        local.last_synced_at = datetime.now(timezone.utc)
        return local

# Global persistence instances
MISSION_STORE = MissionPersistence()
PROJECT_STORE = ProjectPersistence()
SYNC_ENGINE = SyncController()
