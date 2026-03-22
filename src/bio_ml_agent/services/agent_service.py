import os
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Generator, Union

# Proje kökü importları
from bio_ml_agent.utils.config import load_config
from bio_ml_agent.core.config import AgentConfig, SYSTEM_PROMPT
from bio_ml_agent.core.conversation import generate_session_id, save_conversation, load_conversation

# Sub-module imports
from bio_ml_agent.services.agent.orchestration import (
    get_routing_decision, 
    prepare_agent_core, 
    handle_temporal_trigger,
    submit_temporal_job
)
from bio_ml_agent.services.agent.memory_context import get_compressed_context
from bio_ml_agent.services.agent.execution_policy import (
    is_action_request, 
    needs_approval, 
    format_tool_output
)
from bio_ml_agent.services.agent.project_lifecycle import ensure_project_context
from bio_ml_agent.models.domain import MissionStatus

log = logging.getLogger("bio_ml_agent")

class AgentService:
    """Ajanın UI'den bağımsız (headless) olarak çalışmasını sağlayan core servis katmanı.
    Geleneksel monolithic yapıdan modüler yapıya (Facade Pattern) dönüştürülmüştür."""
    
    def __init__(self, model: str = "", workspace: str = "", timeout: int = 0, max_steps: int = 0):
        app_config = load_config()
        self.config = AgentConfig(
            model=model or app_config.agent.model,
            workspace=Path(workspace or app_config.workspace.base_dir).expanduser().resolve(),
            timeout=timeout or app_config.agent.timeout,
            max_steps=max_steps or app_config.agent.max_steps,
            history_dir=Path(app_config.history.directory).expanduser().resolve(),
            approval_mode=1,
        )
        self.config.workspace.mkdir(parents=True, exist_ok=True)
        self.session_id = generate_session_id()
        self.session_metadata = {"created_at": datetime.now().isoformat()}
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        self.project_name: Optional[str] = None
        self.project_root: Optional[Path] = None

        self.approval_mode: int = 1
        self.approval_interval: int = 5
        self.checkpoint_step: int = 50
        self._current_step: int = 0
        self._paused: bool = False
        self._plan_approved: bool = False
        self.swarm_enabled: bool = False
        self.mission_id: Optional[str] = None
        self.batch_size: int = 0
        self.batch_completed: int = 0

    def set_session(self, session_id: str, messages: List[Dict], metadata: Optional[Dict] = None):
        """Mevcut bir oturumu geri yükle."""
        self.session_id = session_id
        self.messages = messages
        if metadata:
            self.session_metadata = metadata
            proj_name = metadata.get("project_name")
            proj_path = metadata.get("project_path")
            self.mission_id = metadata.get("mission_id")
            if proj_name:
                self.project_name = proj_name
                if proj_path:
                    self.project_root = Path(proj_path)
                elif hasattr(self.config, 'workspace') and self.config.workspace:
                    self.project_root = self.config.workspace / proj_name
                
                if self.project_root:
                    self.project_root.mkdir(parents=True, exist_ok=True)
                os.environ["AGENT_PROJECT"] = proj_name
                log.info("📁 Proje bağlamı geri yüklendi: %s", proj_name)

    def reset_session(self):
        """Oturumu sıfırla."""
        self.session_id = generate_session_id()
        self.session_metadata = {"created_at": datetime.now().isoformat()}
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.project_name = None
        self.project_root = None
        os.environ.pop("AGENT_PROJECT", None)

    def run_mission_pack(self, pack_id: str, user_prompt: str) -> str:
        """Phase 3: Executes a named Mission Pack (Orchestration context)."""
        from bio_ml_agent.services.mission_pack_registry import mission_pack_registry
        from bio_ml_agent.services.mission.orchestrator import MissionOrchestrator
        
        pack = mission_pack_registry.get_pack(pack_id)
        if not pack:
            raise ValueError(f"Mission Pack {pack_id} not found.")

        orchestrator = MissionOrchestrator(self.config)
        # We need a project_id. Ensure one exists.
        if not self.project_name:
            self.project_name = "default_mission_project"
        
        # Ensure project exists in DB and FS
        ensure_project_context(user_prompt, self.session_id, self.config.workspace, self.project_name)
        
        # In a real system, we'd look up the numeric project_id from DB
        from bio_ml_agent.db.session import SessionLocal
        from bio_ml_agent.db.models import ProjectDB
        with SessionLocal() as db:
            proj = db.query(ProjectDB).filter(ProjectDB.name == self.project_name).first()
            project_id = proj.project_id if proj else "proj-default"

        mid = orchestrator.execute_pack(pack, project_id, user_prompt)
        self.mission_id = mid
        return mid

    def process_message(self, user_msg: str, files: Optional[List[str]] = None) -> Generator[Dict[str, Any], None, None]:
        """Mesaj mantığını işler ve olayları dışarı stream eder."""
        intent_override = None

        if user_msg == "_DEVAM_ET_":
            self._paused = False
            if self.approval_mode == 4 and not self._plan_approved:
                self._plan_approved = True
            intent_override = "TOOL_LOOP"
            self.messages.append({"role": "user", "content": "Kullanıcı onay verdi. Devam et, bir sonraki adıma geç."})
        else:
            self._paused = False
            self._current_step = 0
        
        # 1. Project Lifecycle
        proj_info = ensure_project_context(user_msg, self.session_id, self.config.workspace, self.project_name)
        if proj_info:
            self.project_name = proj_info["project_name"]
            self.project_root = Path(proj_info["project_path"])
            self.session_metadata.update(proj_info)

        # 2. Memory Context & Normalization
        from bio_ml_agent.core.message_normalizer import MessageNormalizer
        mem_context = get_compressed_context(user_msg, self.config.model, self.project_name, self.session_id)
        base_text = f"{mem_context}\n\n[Mevcut Görev/Soru]:\n{user_msg}" if mem_context else user_msg
        
        provider = "openai"
        if self.config.model.startswith("gemini"): provider = "gemini"
        elif self.config.model.startswith("claude"): provider = "anthropic"
        
        std_msg = MessageNormalizer.normalize_input({"text": base_text, "files": files or []}, role="user")
        self.messages.append(MessageNormalizer.to_provider_format(std_msg, provider=provider))

        # 3. Orchestration & Routing
        routing = get_routing_decision(user_msg)
        self.config.model = routing["primary_model"]
        yield {"type": "status", "content": f"Model yönlendirildi: {self.config.model} (Zorluk: {routing['complexity_level']})"}

        # Metrics
        workflow_id = f"wf-{self.session_id}-{int(datetime.now().timestamp())}"
        from bio_ml_agent.ultra_agent.observability.metrics import metrics as otel_metrics
        otel_metrics.register_workflow(workflow_id, "process_message")

        # Temporal Check
        if handle_temporal_trigger(user_msg):
            yield {"type": "status", "content": "Temporal Cluster'a bağlanılıyor..."}
            try:
                import asyncio
                run_id, protein = asyncio.run(submit_temporal_job(user_msg, self.session_id, self.config.workspace))
                yield {"type": "assistant", "content": f"🚀 Hedef {protein} için Sanal Tarama Temporal kümesinde başlatıldı. ID: `{run_id}`"}
                return
            except Exception as e:
                yield {"type": "error", "content": f"Temporal hatası: {e}. Yerel analiz ile devam ediliyor..."}

        # 4. Agent Core Execution
        core = prepare_agent_core(self.config, self.project_name or "scratch_project")
        event_generator = core.route_task(
            user_msg=user_msg,
            messages=self.messages,
            session_id=self.session_id,
            session_metadata=self.session_metadata,
            intent_override=intent_override,
        )

        for event in event_generator:
            # Phase R5-3: Log Mission Step if mission is active
            if self.mission_id:
                self._handle_mission_telemetry(event)
                # D3: Real-time Intervention Check
                intervention = self.check_interventions()
                if intervention:
                    if intervention == "SIMPLER_MODE":
                        self.log_mission_step("System", "intervention", "Switching to Simpler Mode as requested.")
                    elif intervention == "FORCE_CRITIC":
                         # Force critic logic placeholder
                         pass
                
                # D5: Explainable Progress (Human-readable)
                event_type = event.get("type")
                if event_type == "thought":
                    # Extract a short progress snippet from thought
                    thought_text = event.get("content", "")
                    if "Searching" in thought_text or "Analyzing" in thought_text or "Scanning" in thought_text:
                        self.set_readable_progress(thought_text[:60] + "...")
                
            event_type = event.get("type")
            if event_type == "tool_output":
                tool = event.get("tool", "")
                output = event.get("output", "")
                event["formatted"] = format_tool_output(tool, output)
                
                # S8-2/S5-3: Background Job Telemetry Registration
                if tool == "BACKGROUND_JOB" and "Task ID:" in output:
                    try:
                        # Extract Task ID: 123...
                        task_id = output.split("Task ID:")[1].strip().split("\n")[0]
                        otel_metrics.register_workflow(
                            task_id, 
                            "background_job", 
                            {"parent_wf": workflow_id, "session_id": self.session_id}
                        )
                    except Exception as te:
                        log.warning(f"Telemetry registration error for background job: {te}")
                
                self._current_step += 1
                if needs_approval(self._current_step, self.approval_mode, self.approval_interval, self.checkpoint_step, self._plan_approved, tool):
                    self._paused = True
                    reason = f"Onay bekleniyor (adım {self._current_step})" # Simplified reason for now
                    yield {
                        "type": "approval_required",
                        "step": self._current_step,
                        "content": f"⏸️ {reason}\n\nDevam etmek için 'Devam Et' butonuna basın.",
                    }
                    return
            if event_type == "assistant" and self.mission_id:
                self.create_notification(
                    "MISSION_COMPLETE", 
                    "Mission Finished", 
                    f"Agent {self.session_metadata.get('active_agent', 'Swarm')} has completed the mission.",
                    action_url=f"#missions"
                )
            yield event
        
        # 5. Final/Step Checkpoint
        self.save_checkpoint()

    def save_checkpoint(self):
        """Phase 0: Saves current state as a DB-first checkpoint."""
        if not self.project_name:
            return
        
        try:
            # 1. Save conversation history (for now keeping disk history for large context)
            save_conversation(self.config.history_dir, self.session_id, self.messages, self.session_metadata)
            
            # 2. Sync checkpoint to DB (Primary)
            from bio_ml_agent.db.session import SessionLocal
            from bio_ml_agent.db.models import ProjectDB
            
            checkpoint = {
                "session_id": self.session_id,
                "project_name": self.project_name,
                "timestamp": datetime.now().isoformat(),
                "message_count": len(self.messages),
                "mission_id": self.mission_id
            }
            
            with SessionLocal() as db:
                proj = db.query(ProjectDB).filter(ProjectDB.name == self.project_name).first()
                if proj:
                    proj.checkpoint_json = checkpoint
                    proj.updated_at = datetime.now().timestamp()
                    db.commit()
                    log.info("📊 Checkpoint synced to DB: %s", self.project_name)
            
            # 3. Disk Fail-safe (Audit only)
            if self.project_root:
                (self.project_root / "checkpoint.json").write_text(
                    json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            # 4. Check for Active Mission to Resume
            if self.mission_id:
                from bio_ml_agent.services.mission.orchestrator import MissionOrchestrator
                from bio_ml_agent.db.models import MissionDB
                from bio_ml_agent.models.domain import MissionStatus
                
                with SessionLocal() as db:
                    m = db.query(MissionDB).filter(MissionDB.mission_id == self.mission_id).first()
                    if m and m.status in [MissionStatus.RUNNING, MissionStatus.WAITING]:
                        log.info(f"🔄 Interrupted mission detected: {self.mission_id}. Triggering recovery...")
                        orch = MissionOrchestrator(self.config)
                        # In a real system, this would be a background task
                        # orch.resume_pack(self.mission_id) 
            
        except Exception as e:
            log.error("❌ Checkpoint Sync Error: %s", e)

    def recover_last_session(self) -> bool:
        """Phase 0: Recovers the last active project/session from the DB."""
        try:
            from bio_ml_agent.db.session import SessionLocal
            from bio_ml_agent.db.models import ProjectDB
            
            with SessionLocal() as db:
                # Get the most recently accessed/updated project
                latest_proj = db.query(ProjectDB).order_by(ProjectDB.updated_at.desc()).first()
                if not latest_proj or not latest_proj.checkpoint_json:
                    log.info("ℹ️ No active DB checkpoints found for recovery.")
                    return False
                
                ckpt = latest_proj.checkpoint_json
                sid = ckpt["session_id"]
                mid = ckpt.get("mission_id")
                
                # Check for mission restoration
                if mid:
                    from bio_ml_agent.services.mission.orchestrator import MissionOrchestrator
                    orch = MissionOrchestrator(self.config)
                    # Attempt to resume
                    if orch.resume_pack(mid):
                        log.info(f"✅ Mission {mid} successfully rehydrated and resumed.")
                        self.mission_id = mid
                
                # Load messages from history (Disk is still our message store due to size)
                messages, metadata = load_conversation(self.config.history_dir, sid)
                
                # Restore state
                self.set_session(sid, messages, metadata)
                self.mission_id = ckpt.get("mission_id")
                
                log.info("♻️ Session recovered from DB: %s (Project: %s)", sid, latest_proj.name)
                return True
        except Exception as e:
            log.warning("⚠️ DB-driven recovery failed: %s. Falling back to disk scan...", e)
            return self._recover_from_disk()

    def _recover_from_disk(self) -> bool:
        """Legacy disk-based recovery fallback."""
        try:
            projects = [d for d in self.config.workspace.iterdir() if d.is_dir() and (d / "project.json").exists()]
            if not projects: return False
            latest_project = max(projects, key=lambda d: d.stat().st_mtime)
            checkpoint_path = latest_project / "checkpoint.json"
            if checkpoint_path.exists():
                data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                sid = data["session_id"]
            else:
                data = json.loads((latest_project / "project.json").read_text(encoding="utf-8"))
                sid = data["session_id"]
            messages, metadata = load_conversation(self.config.history_dir, sid)
            self.set_session(sid, messages, metadata)
            return True
        except Exception:
            return False

    def check_job_status(self, task_id: str) -> Dict[str, Any]:
        """Redis/RQ üzerinden asenkron görevin durumunu sorgula."""
        try:
            from redis import Redis
            from rq.job import Job
            
            app_config = load_config()
            redis_conn = Redis(
                host=app_config.redis.host,
                port=app_config.redis.port,
                password=app_config.redis.password or None,
                db=app_config.redis.db
            )
            
            job = Job.fetch(task_id, connection=redis_conn)
            
            return {
                "task_id": task_id,
                "status": job.get_status(),
                "progress": job.meta.get("progress", "Bekleniyor..."),
                "result": job.result if job.is_finished else None,
                "error": job.exc_info if job.is_failed else None
            }
        except Exception as e:
            log.error(f"Job status check error: {e}")
            return {"task_id": task_id, "status": "error", "message": str(e)}

    def log_project_event(self, event_type: str, message: str, metadata: Optional[Dict] = None):
        """B3: Proje timeline'ına profesyonel bir olay kaydeder."""
        if not self.project_name: return
        
        try:
            from bio_ml_agent.db.session import SessionLocal
            from bio_ml_agent.db.models import TimelineEventDB, ProjectDB
            import uuid
            import time

            with SessionLocal() as db:
                proj = db.query(ProjectDB).filter(ProjectDB.name == self.project_name).first()
                if not proj: return

                event_id = f"evt-{uuid.uuid4().hex[:6]}"
                new_event = TimelineEventDB(
                    event_id=event_id,
                    project_id=proj.project_id,
                    event_type=event_type,
                    message=message,
                    timestamp=time.time(),
                    metadata_json=metadata or {},
                    agent_name=self.session_metadata.get("active_agent", "System")
                )
                db.add(new_event)
                db.commit()
        except Exception as e:
            log.warning(f"Timeline event logging error: {e}")

    def update_project_state(self, state: str):
        """B4: Proje durumunu (milestone) günceller."""
        if not self.project_name: return
        
        try:
            from bio_ml_agent.db.session import SessionLocal
            from bio_ml_agent.db.models import ProjectDB
            
            with SessionLocal() as db:
                proj = db.query(ProjectDB).filter(ProjectDB.name == self.project_name).first()
                if proj:
                    proj.state = state
                    db.commit()
                    self.log_project_event("STATE_CHANGE", f"Project state transitioned to: {state}")
        except Exception as e:
            log.warning(f"Project state update error: {e}")

    def log_mission_step(self, agent_name: str, action_type: str, content: str, 
                         thought: Optional[str] = None, metadata: Optional[Dict] = None,
                         agent_role: Optional[str] = None, confidence: Optional[float] = None):
        """Phase R5-3: Görev adımını kalıcı DB'ye kaydeder."""
        if not self.mission_id: return
        
        try:
            from bio_ml_agent.db.session import SessionLocal
            from bio_ml_agent.db.models import MissionStepDB
            import uuid
            import time

            with SessionLocal() as db:
                sid = f"stp-{uuid.uuid4().hex[:6]}"
                new_step = MissionStepDB(
                    step_id=sid,
                    mission_id=self.mission_id,
                    agent_name=agent_name,
                    agent_role=agent_role,
                    action_type=action_type,
                    content=str(content),
                    thought=thought,
                    confidence=confidence,
                    metadata_json=metadata or {},
                    timestamp=time.time()
                )
                db.add(new_step)
                db.commit()
                
                # Phase R5-5: Notify on Error or Approval
                if action_type == "error":
                     self.create_notification("STUCK", "Agent Needs Help", f"{agent_name} encountered an error: {content[:50]}...", action_url="#missions")
                     
        except Exception as e:
            log.warning(f"Mission step logging error: {e}")

    def create_notification(self, ntype: str, title: str, message: str, action_url: Optional[str] = None):
        """Phase R5-5: Sistemsel bildirim oluşturur."""
        try:
            from bio_ml_agent.db.session import SessionLocal
            from bio_ml_agent.db.models import NotificationDB
            import time
            import uuid
            
            with SessionLocal() as db:
                new_notif = NotificationDB(
                    id=f"not-{uuid.uuid4().hex[:6]}",
                    type=ntype,
                    title=title,
                    message=message,
                    action_url=action_url,
                    timestamp=time.time()
                )
                db.add(new_notif)
                db.commit()
        except Exception as e:
            log.warning(f"Notification creation error: {e}")

    def set_readable_progress(self, text: str):
        """D5: Anlamlı ilerleme cümlesini günceller."""
        if not self.mission_id: return
        try:
            from bio_ml_agent.db.session import SessionLocal
            from bio_ml_agent.db.models import MissionDB
            with SessionLocal() as db:
                mission = db.query(MissionDB).filter(MissionDB.mission_id == self.mission_id).first()
                if mission:
                    mission.readable_progress = text
                    db.commit()
        except Exception as e:
            log.warning(f"Progress update error: {e}")

    def check_interventions(self) -> Optional[str]:
        """D3: Bekleyen müdahaleyi okur ve döner."""
        if not self.mission_id: return None
        try:
            from bio_ml_agent.db.session import SessionLocal
            from bio_ml_agent.db.models import MissionDB
            with SessionLocal() as db:
                mission = db.query(MissionDB).filter(MissionDB.mission_id == self.mission_id).first()
                if mission and mission.intervention_requested:
                    action = mission.intervention_requested
                    mission.intervention_requested = None # Consume it
                    db.commit()
                    return action
        except Exception as e:
            log.warning(f"Intervention check error: {e}")
        return None

    def set_batch_info(self, size: int, completed: int):
        """D4: Toplu işlem bilgisini günceller."""
        self.batch_size = size
        self.batch_completed = completed
        if self.mission_id:
            try:
                from bio_ml_agent.db.session import SessionLocal
                from bio_ml_agent.db.models import MissionDB
                with SessionLocal() as db:
                    m = db.query(MissionDB).filter(MissionDB.mission_id == self.mission_id).first()
                    if m:
                        m.batch_size = size
                        m.batch_completed = completed
                        db.commit()
            except Exception as e:
                log.warning(f"Batch info update error: {e}")

    def _handle_mission_telemetry(self, event: Dict[str, Any]):
        """Event tipine göre mission_id üzerinden telemetri kaydı yapar."""
        etype = event.get("type")
        agent = self.session_metadata.get("active_agent", "System")
        role = self.session_metadata.get("active_agent_role")
        conf = event.get("confidence")
        
        # Batch telemetry extraction
        if "batch_size" in event:
            self.batch_size = event["batch_size"]
        if "batch_completed" in event:
            self.batch_completed = event["batch_completed"]

        if etype == "thought":
            self.log_mission_step(agent, "THOUGHT", event.get("content", ""), thought=event.get("content"), agent_role=role, confidence=conf)
        elif etype == "tool_use":
            self.log_mission_step(agent, "TOOL_USE", event.get("tool", ""), metadata={"input": event.get("input")}, agent_role=role, confidence=conf)
        elif etype == "tool_output":
            self.log_mission_step(agent, "RESULT", event.get("tool", ""), metadata={"output": event.get("output")}, agent_role=role, confidence=conf)
        elif etype == "error":
            self.log_mission_step(agent, "ERROR", event.get("content", ""), agent_role=role)
        elif etype == "assistant":
            self.log_mission_step(agent, "FINAL_ANSWER", event.get("content", ""), agent_role=role, confidence=conf)

        # Sync mission progress and batch stats
        if self.mission_id:
            try:
                from bio_ml_agent.db.session import SessionLocal
                from bio_ml_agent.db.models import MissionDB
                with SessionLocal() as db:
                    m = db.query(MissionDB).filter(MissionDB.mission_id == self.mission_id).first()
                    if m:
                        if "progress" in event:
                            m.progress_percentage = event["progress"]
                        m.status = MissionStatus.RUNNING
                        m.batch_size = self.batch_size
                        m.batch_completed = self.batch_completed
                        db.commit()
            except Exception as e:
                log.warning(f"Mission sync error: {e}")
