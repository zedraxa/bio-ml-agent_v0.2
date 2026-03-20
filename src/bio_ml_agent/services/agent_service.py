# services/agent_service.py
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Generator

# Proje kökü importları
from bio_ml_agent.utils.config import load_config
from bio_ml_agent.core.config import AgentConfig, SYSTEM_PROMPT
from bio_ml_agent.core.conversation import generate_session_id

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

    def set_session(self, session_id: str, messages: List[Dict], metadata: Dict = None):
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

    def process_message(self, user_msg: str, files: List[str] = None) -> Generator[Dict[str, Any], None, None]:
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
                event["formatted"] = format_tool_output(tool, event.get("output", ""))
                
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
