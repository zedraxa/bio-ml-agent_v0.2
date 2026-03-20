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

    def set_session(self, session_id: str, messages: List[Dict], metadata: Optional[Dict] = None):
        """Mevcut bir oturumu geri yükle."""
        self.session_id = session_id
        self.messages = messages
        if metadata:
            self.session_metadata = metadata
            proj_name = metadata.get("project_name")
            proj_path = metadata.get("project_path")
            if proj_name:
                self.project_name = proj_name
                self.project_root = Path(proj_path) if proj_path else self.config.workspace / proj_name
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
            yield event
        
        # 5. Final/Step Checkpoint
        self.save_checkpoint()

    def save_checkpoint(self):
        """Mevcut durumu bir checkpoint olarak kaydet."""
        if not self.project_root:
            return
        
        try:
            # Conversation geçmişini kaydet
            save_conversation(self.config.history_dir, self.session_id, self.messages, self.session_metadata)
            
            # Proje klasörü altına checkpoint.json bırak
            checkpoint = {
                "session_id": self.session_id,
                "project_name": self.project_name,
                "timestamp": datetime.now().isoformat(),
                "message_count": len(self.messages)
            }
            (self.project_root / "checkpoint.json").write_text(
                json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            log.debug("💾 Checkpoint kaydedildi: %s", self.project_name)
        except Exception as e:
            log.error("❌ Checkpoint kaydetme hatası: %s", e)

    def recover_last_session(self) -> bool:
        """En son aktif olan projeyi bul ve geri yükle."""
        try:
            # En yeni proje klasörünü bul (Hızlı ama basit yöntem)
            projects = [d for d in self.config.workspace.iterdir() if d.is_dir() and (d / "project.json").exists()]
            if not projects:
                return False
            
            latest_project = max(projects, key=lambda d: d.stat().st_mtime)
            
            # Checkpoint varsa oradan al, yoksa project.json'dan
            checkpoint_path = latest_project / "checkpoint.json"
            if checkpoint_path.exists():
                data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                sid = data["session_id"]
            else:
                data = json.loads((latest_project / "project.json").read_text(encoding="utf-8"))
                sid = data["session_id"]

            messages, metadata = load_conversation(self.config.history_dir, sid)
            self.set_session(sid, messages, metadata)
            log.info("♻️ Oturum otomatik kurtarıldı: %s", sid)
            return True
        except Exception as e:
            log.warning("⚠️ Otomatik kurtarma başarısız: %s", e)
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
