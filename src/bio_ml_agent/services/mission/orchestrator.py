import logging
import uuid
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

from bio_ml_agent.models.mission_pack import MissionPack, MissionPackStep
from bio_ml_agent.models.domain import MissionStatus, ArtifactReviewStatus, TaskType, AgentRole
from bio_ml_agent.brain.models import MissionPlan, MissionStep, StepStatus, ProjectState, MissionSnapshot, MissionTelemetry
from bio_ml_agent.services.agent_registry import agent_registry
from bio_ml_agent.db.session import SessionLocal
from bio_ml_agent.db.models import MissionDB, MissionStepDB, ArtifactDB, ProjectMemoryDB, CommentDB, ReviewThreadDB, ProjectDB
from bio_ml_agent.brain.recovery_manager import RecoveryManager

logger = logging.getLogger(__name__)

class MissionOrchestrator:
    """
    Axis D3: Real-time Orchestrator for Mission Packs.
    Executes a graph of standardized agents based on a MissionPack blueprint.
    """
    
    def __init__(self, config):
        self.config = config
        self.recovery = RecoveryManager()

    def resume_pack(self, mission_id: str) -> bool:
        """Hydrates a mission from checkpoint and resumes execution."""
        snapshot = self.recovery.resume_mission(mission_id)
        if not snapshot:
            logger.error(f"Failed to resume mission {mission_id}: Checkpoint not found.")
            return False

        plan = snapshot.plan_snapshot
        project_id = snapshot.project_id

        # Load the original pack to get the full step list
        from bio_ml_agent.services.mission_pack_registry import mission_pack_registry
        pack_id = getattr(plan, "pack_id", None)
        if not pack_id:
            logger.error(f"Mission {mission_id} does not have an associated pack_id.")
            return False

        pack = mission_pack_registry.get_pack(pack_id)
        if not pack:
            # Reconstruct a minimal MissionPack from the checkpoint plan steps so
            # we can resume even when the pack is not registered (e.g. ad-hoc missions).
            logger.warning(
                f"Pack {pack_id!r} not in registry — reconstructing from checkpoint."
            )
            pack = MissionPack(
                pack_id=pack_id,
                name=getattr(plan, "title", pack_id),
                description=getattr(plan, "objective", ""),
                steps=[
                    MissionPackStep(
                        step_id=s.step_id,
                        title=s.title,
                        description=s.description,
                        task_type=s.task_type or TaskType.ANALYZE,
                        assigned_agent=s.assigned_agent or AgentRole.RESEARCH_AGENT,
                        depends_on=getattr(s, "depends_on", []),
                    )
                    for s in plan.steps
                ],
            )

        logger.info(f"🔄 Resuming Mission: {mission_id} (Pack: {pack_id})")
        self._update_mission_status(mission_id, MissionStatus.RUNNING)
        if plan.telemetry:
            plan.telemetry.status = MissionStatus.RUNNING

        # Resume loop
        self._run_loop(mission_id, pack, project_id, plan)
        return True

    def _create_plan_from_pack(self, mission_id: str, project_id: str, pack: MissionPack, user_prompt: str) -> MissionPlan:
        """Create a MissionPlan from a MissionPack blueprint."""
        return MissionPlan(
            mission_id=mission_id,
            project_id=project_id,
            pack_id=pack.pack_id,
            user_prompt=user_prompt,
            title=pack.name,
            objective=pack.description,
            steps=[MissionStep(
                step_id=s.step_id,
                title=s.title,
                description=s.description,
                task_type=s.task_type or TaskType.ANALYZE,
                assigned_agent=s.assigned_agent or AgentRole.RESEARCH_AGENT,
                status=StepStatus.PENDING
            ) for s in pack.steps],
            telemetry=MissionTelemetry(mission_id=mission_id, start_time=time.time(), status=MissionStatus.RUNNING)
        )

    def execute_pack(self, pack: MissionPack, project_id: str, user_prompt: str) -> str:
        """Starts the execution of a Mission Pack and returns the mission_id."""
        mission_id = f"msn-{uuid.uuid4().hex[:6]}"

        # 1. Ensure parent project exists, then register mission in DB
        with SessionLocal() as db:
            # Upsert project so the FK constraint is satisfied
            existing_project = db.query(ProjectDB).filter(ProjectDB.project_id == project_id).first()
            if not existing_project:
                db.add(ProjectDB(project_id=project_id, name=project_id, status="active"))
            new_mission = MissionDB(
                mission_id=mission_id,
                project_id=project_id,
                title=pack.name,
                objective=pack.description,
                status=MissionStatus.RUNNING,
                created_at=time.time(),
                updated_at=time.time()
            )
            db.add(new_mission)
            db.commit()

        logger.info(f"🚀 Mission Started: {mission_id} (Pack: {pack.pack_id})")
        
        plan = self._create_plan_from_pack(mission_id, project_id, pack, user_prompt)
        
        # 2. Run Execution Loop (In a real system, this would be async/background)
        self._run_loop(mission_id, pack, project_id, plan)
        
        return mission_id

    def _run_loop(self, mission_id: str, pack: MissionPack, project_id: str, plan: MissionPlan):
        completed_steps = set()
        
        # Hydrate from DB in case we are resuming
        with SessionLocal() as db:
            db_steps = db.query(MissionStepDB).filter(MissionStepDB.mission_id == mission_id, MissionStepDB.action_type == "COMPLETED").all()
            for ds in db_steps:
                # step_id in DB is f"{mission_id}-{pack_step.step_id}"
                original_id = ds.step_id.replace(f"{mission_id}-", "")
                completed_steps.add(original_id)

        for pack_step in pack.steps:
            if pack_step.step_id in completed_steps:
                continue

            # Check dependencies
            if not all(dep in completed_steps for dep in pack_step.depends_on):
                logger.warning(f"Skipping {pack_step.step_id} - dependencies not met.")
                continue

            # Execute Step
            try:
                self._execute_step(mission_id, pack_step, project_id, plan)
                completed_steps.add(pack_step.step_id)
                
                # Checkpoint after each step
                project = ProjectState(project_id=project_id, name="Project Workspace")
                self.recovery.create_checkpoint(mission_id, plan, project)
                
                # Check if we should stop for approval
                with SessionLocal() as db:
                    m = db.query(MissionDB).filter(MissionDB.mission_id == mission_id).first()
                    if m and m.status == MissionStatus.WAITING:
                        logger.info(f"⏸️ Loop Paused for Approval: {mission_id}")
                        return
                
            except Exception as e:
                logger.error(f"❌ Step Failed: {pack_step.step_id} | Error: {e}", exc_info=True)
                self._update_mission_status(mission_id, MissionStatus.FAILED)
                if plan.telemetry:
                    plan.telemetry.status = MissionStatus.FAILED
                return

        self._update_mission_status(mission_id, MissionStatus.COMPLETED)
        if plan.telemetry:
            plan.telemetry.status = MissionStatus.COMPLETED
            plan.telemetry.end_time = time.time()
            plan.telemetry.total_duration_seconds = plan.telemetry.end_time - plan.telemetry.start_time
            
        logger.info(f"✅ Mission Completed: {mission_id}")

    def _execute_step(self, mission_id: str, pack_step: MissionPackStep, project_id: str, plan: MissionPlan):
        logger.info(f"🏃 Executing Step: {pack_step.title} ({pack_step.assigned_agent.value})")
        
        # 0. Get parent artifacts for lineage
        with SessionLocal() as db:
            # Simple lineage: all artifacts already in this mission are potential parents of next steps
            # In a more advanced version, we'd use pack_step.depends_on to filter
            parents = db.query(ArtifactDB).filter(ArtifactDB.mission_id == mission_id).all()
            lineage_parents = [
                art_id for p in parents
                if (art_id := getattr(p, "artifact_id", None)) is not None
            ]

        # 1. Log Step Start
        with SessionLocal() as db:
            db_step = MissionStepDB(
                step_id=f"{mission_id}-{pack_step.step_id}",
                mission_id=mission_id,
                agent_name=pack_step.preferred_agent_id or "Auto",
                agent_role=pack_step.assigned_agent.value,
                action_type="RUNNING",
                content=f"Starting: {pack_step.title}",
                timestamp=time.time()
            )
            db.add(db_step)
            db.commit()

        # 2. Load Agent
        agent_id = pack_step.preferred_agent_id
        if not agent_id:
            available = agent_registry.find_by_role(pack_step.assigned_agent)
            if not available:
                raise RuntimeError(f"No agent found for role {pack_step.assigned_agent}")
            agent_id = available[0].agent_id

        agent = agent_registry.get_agent_instance(agent_id, self.config)
        
        # 3. Standardized Lifecycle Call
        context = {
            "mission_id": mission_id,
            "project_id": project_id,
            "step_id": pack_step.step_id,
            "title": pack_step.title,
            "description": pack_step.description,
            "goal": pack_step.description
        }
        agent.perceive(context)
        
        # In this simple loop, we execute the first plan item
        steps = agent.plan(pack_step.description)
        for s in steps:
            agent.act(s)
            
        agent.verify("") 
        result = agent.summarize() # Returns AgentResult
        
        # 4. Process Outputs (Artifacts, Memory, Reviews)
        # Handle Artifacts
        created_artifact_ids = []
        for artifact_path in getattr(result, "artifacts", []):
            art_id = self._save_artifact(mission_id, project_id, agent_id, artifact_path, 
                                        pack_step.expected_artifact_type, lineage_parents)
            created_artifact_ids.append(art_id)

        # Handle Memory Items (if present in result.data or metadata)
        if isinstance(result.data, dict) and "memory_items" in result.data:
            for mem in result.data["memory_items"]:
                self._save_memory(project_id, mission_id, mem)

        # Handle Critiques/Comments (Review Items)
        if hasattr(result, "self_comments") and result.self_comments:
            for comment in result.self_comments:
                # If these are self-critiques, we might want to attach them to the produced artifacts
                for art_id in created_artifact_ids:
                    self._add_comment(art_id, comment, agent_id)

        # 5. Log Step Completion
        with SessionLocal() as db:
            db_step = db.query(MissionStepDB).filter(MissionStepDB.step_id == f"{mission_id}-{pack_step.step_id}").first()
            if db_step:
                db_step.action_type = "COMPLETED"
                db_step.content = result.message
                db_step.confidence = result.confidence if isinstance(result.confidence, (int, float)) else 0.8
                db_step.metadata_json = {"evidence": [str(e) for e in result.evidence]}
                db_step.timestamp = time.time()
                db.commit()

        # 6. Handle Approval Gates & Review Threads
        if pack_step.requires_approval or pack_step.task_type == "verify":
            logger.info(f"⏸️ Step {pack_step.step_id} REQUIRES REVIEW. Opening Threads.")
            for art_id in created_artifact_ids:
                self._open_review_thread(art_id)
            self._update_mission_status(mission_id, MissionStatus.WAITING)

        # 7. Update in-memory plan step status and create checkpoint
        for ps in plan.steps:
            if ps.step_id == pack_step.step_id:
                ps.status = StepStatus.COMPLETED
                break
        project = ProjectState(project_id=plan.project_id, name="Project Workspace")
        self.recovery.create_checkpoint(mission_id, plan, project)

    def _save_artifact(self, mission_id: str, project_id: str, agent_id: str, 
                       path: str, category: str, parents: List[str]) -> str:
        """Saves produced artifact to DB with lineage."""
        with SessionLocal() as db:
            art_id = f"art-{uuid.uuid4().hex[:6]}"
            new_art = ArtifactDB(
                artifact_id=art_id,
                project_id=project_id,
                mission_id=mission_id,
                created_by=agent_id,
                title=f"Result: {path.split('/')[-1]}",
                category=category,
                file_type=path.split('.')[-1] if '.' in path else "txt",
                status=ArtifactReviewStatus.DRAFT, # Start as draft
                lineage_parents=parents,
                created_at=time.time(),
                updated_at=time.time()
            )
            db.add(new_art)
            db.commit()
            return art_id

    def _save_memory(self, project_id: str, mission_id: str, mem_data: Dict):
        """Saves a memory item to DB."""
        with SessionLocal() as db:
            mem_id = f"mem-{uuid.uuid4().hex[:6]}"
            new_mem = ProjectMemoryDB(
                memory_id=mem_id,
                project_id=project_id,
                category=mem_data.get("category", "finding"),
                title=mem_data.get("title", "Insight"),
                content=mem_data.get("content", ""),
                importance=mem_data.get("importance", 1),
                created_at=time.time()
            )
            db.add(new_mem)
            db.commit()

    def _add_comment(self, artifact_id: str, comment_data: Any, agent_id: str):
        """Adds a comment/critique to an artifact."""
        with SessionLocal() as db:
            cid = f"cmt-{uuid.uuid4().hex[:6]}"
            # comment_data could be a string or a dict/object
            content = comment_data if isinstance(comment_data, str) else str(comment_data)
            new_cmt = CommentDB(
                comment_id=cid,
                artifact_id=artifact_id,
                author=agent_id,
                content=content,
                timestamp=time.time()
            )
            db.add(new_cmt)
            db.commit()

    def _open_review_thread(self, artifact_id: str):
        """Creates a ReviewThread for an artifact."""
        with SessionLocal() as db:
            # Check if artifact exists and set status to REVIEW_NEEDED
            art = db.query(ArtifactDB).filter(ArtifactDB.artifact_id == artifact_id).first()
            if art:
                art.status = ArtifactReviewStatus.REVIEW_NEEDED
                
                tid = f"thrd-{uuid.uuid4().hex[:6]}"
                new_thread = ReviewThreadDB(
                    thread_id=tid,
                    artifact_id=artifact_id,
                    status="open",
                    created_at=time.time(),
                    updated_at=time.time()
                )
                db.add(new_thread)
                db.commit()

    def _update_mission_status(self, mission_id: str, status: MissionStatus):
        with SessionLocal() as db:
            mission = db.query(MissionDB).filter(MissionDB.mission_id == mission_id).first()
            if mission:
                mission.status = status
                mission.updated_at = time.time()
                db.commit()
