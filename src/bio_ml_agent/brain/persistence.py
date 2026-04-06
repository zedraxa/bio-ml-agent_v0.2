import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Any
import logging

from .models import MissionPlan, ProjectState, MissionSnapshot
from bio_ml_agent.db.session import SessionLocal
from bio_ml_agent.db.models import MissionDB, ProjectDB
from bio_ml_agent.models.domain import MissionStatus
from bio_ml_agent.models.workspace_ux import StepStatus

logger = logging.getLogger("bio_ml_agent.persistence")

class MissionPersistence:
    """Handles persistent storage and retrieval of Mission Plans."""

    def __init__(self, storage_dir: str = "data/missions"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, mission_id: str) -> Path:
        return self.storage_dir / f"{mission_id}.json"

    def save(self, plan: MissionPlan) -> str:
        """Saves the mission plan. DB is primary truth, disk as audit/fallback."""
        # 1. DB Truth Layer (Primary)
        db = SessionLocal()
        db_synced = False
        try:
            db_mission = db.query(MissionDB).filter(MissionDB.mission_id == plan.mission_id).first()
            if not db_mission:
                db_mission = MissionDB(
                    mission_id=plan.mission_id,
                    project_id=plan.project_id,
                    title=plan.title,
                    objective=plan.objective,
                    created_at=plan.created_at.timestamp() if hasattr(plan.created_at, "timestamp") else plan.created_at
                )
                db.add(db_mission)

            # Sync metadata and plan
            db_mission.status = plan.steps[-1].status if plan.steps else MissionStatus.PENDING
            db_mission.progress_percentage = int((len([s for s in plan.steps if s.status == StepStatus.COMPLETED]) / len(plan.steps)) * 100) if plan.steps else 0
            db_mission.full_plan_json = plan.model_dump()
            db_mission.assigned_agents = list(set(s.assigned_agent for s in plan.steps))
            db_mission.updated_at = datetime.now().timestamp()

            db.commit()
            db_synced = True
            logger.info(f"[Persistence] Mission {plan.mission_id} saved as Truth to DB.")
        except Exception as e:
            logger.error(f"[Persistence] CRITICAL: Failed to save mission {plan.mission_id} to DB: {e}")
            db.rollback()
        finally:
            db.close()

        # 2. Disk Fail-safe / Audit Log
        path = self._get_path(plan.mission_id)
        try:
            with open(path, "w") as f:
                f.write(plan.model_dump_json(indent=2))
        except Exception as e:
            logger.warning(f"[Persistence] Audit log write failed for {plan.mission_id}: {e}")

        return str(path)

    def load(self, mission_id: str) -> Optional[MissionPlan]:
        """Loads a mission plan. DB preferred, disk as fallback."""
        db = SessionLocal()
        try:
            db_mission = db.query(MissionDB).filter(MissionDB.mission_id == mission_id).first()
            if db_mission and db_mission.full_plan_json:
                return MissionPlan(**db_mission.full_plan_json)
        except Exception as e:
            logger.error(f"[Persistence] DB load failed for {mission_id}: {e}")
        finally:
            db.close()

        # Disk Fallback
        path = self._get_path(mission_id)
        if path.exists():
            with open(path, "r") as f:
                data = json.load(f)
                return MissionPlan(**data)
        return None

    def get_plan(self, mission_id: str) -> Optional[MissionPlan]:
        """Brain-specific Alias for load."""
        return self.load(mission_id)

    def list_missions(self) -> List[str]:
        """Lists all persistent mission IDs. DB preferred."""
        db = SessionLocal()
        try:
            return [m.mission_id for m in db.query(MissionDB).all()]
        except Exception as e:
            logger.error(f"[Persistence] Failed to list missions from DB: {e}")
            return [f.stem for f in self.storage_dir.glob("*.json")]
        finally:
            db.close()
        return [] # Default fallback

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

    def __init__(self, storage_dir: str = "data/projects"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, project_id: str) -> Path:
        return self.storage_dir / f"{project_id}.json"

    def save(self, project: ProjectState) -> str:
        """Saves the project state. DB is primary truth."""
        # 1. DB Truth Layer (Primary)
        db = SessionLocal()
        try:
            db_project = db.query(ProjectDB).filter(ProjectDB.project_id == project.project_id).first()
            if not db_project:
                db_project = ProjectDB(
                    project_id=project.project_id,
                    name=project.name,
                    created_at=datetime.now().timestamp(),
                    updated_at=datetime.now().timestamp()
                )
                db.add(db_project)

            db_project.brain_state_json = project.model_dump()
            db_project.updated_at = datetime.now().timestamp()
            db_project.active_mission_id = project.active_mission_id

            db.commit()
            logger.info(f"[Persistence] Project {project.project_id} saved as Truth to DB.")
        except Exception as e:
            logger.error(f"[Persistence] CRITICAL: Failed to save project {project.project_id} to DB: {e}")
            db.rollback()
        finally:
            db.close()

        # 2. Disk fail-safe / Audit
        path = self._get_path(project.project_id)
        try:
            with open(path, "w") as f:
                f.write(project.model_dump_json(indent=2))
        except Exception as e:
            logger.warning(f"[Persistence] Audit log write failed for project {project.project_id}: {e}")

        return str(path)

    def load(self, project_id: str) -> Optional[ProjectState]:
        """Loads a project state. DB preferred."""
        db = SessionLocal()
        try:
            db_project = db.query(ProjectDB).filter(ProjectDB.project_id == project_id).first()
            if db_project and db_project.brain_state_json:
                return ProjectState(**db_project.brain_state_json)
        except Exception as e:
            logger.error(f"[Persistence] DB load failed for project {project_id}: {e}")
        finally:
            db.close()

        # Disk fallback
        path = self._get_path(project_id)
        if path.exists():
            with open(path, "r") as f:
                data = json.load(f)
                return ProjectState(**data)
        return None

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
