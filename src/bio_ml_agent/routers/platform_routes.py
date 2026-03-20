import uuid
import json
import asyncio
from typing import Optional, List, Dict, Any
from urllib import request as urllib_request, error as urllib_error
from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect, Header
from bio_ml_agent.utils.config import get_config
from bio_ml_agent.services.agent_service import AgentService

# Import Pydantic models
from bio_ml_agent.models.remote_gateway import AuthSession, AuthMethod, RemoteSessionRegistry, StreamEvent, EventStreamType
from bio_ml_agent.models.remote_client import DashboardSummaryTemplate, MobileClientAction
from bio_ml_agent.models.remote_storage import CloudArtifactItem, SharedProjectAccess, ProjectAccessRole
from bio_ml_agent.models.remote_browser import BrowserTakeoverEvent, RiskActionApprovalState
from bio_ml_agent.models.cloud_offload import ExecutionTarget, JobClassification, RuntimePackage
from bio_ml_agent.models.cloud_workspace import WorkspaceSnapshot

config = get_config()

async def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    expected_key = config.security.api_key
    if not expected_key:
        return # Security disabled
    
    # Websocket requests can't easily pass custom headers in frontend Native WS API (they pass it via query or subprotocols). 
    # For now we handle basic header auth.
    if x_api_key != expected_key:
        raise HTTPException(status_code=403, detail="Yetersiz veya geçersiz API Key.")

router = APIRouter(dependencies=[Depends(verify_api_key)])

# DUMMY IN-MEMORY DATABASES
_users: Dict[str, Any] = {}
_projects: Dict[str, Any] = {}
_workspaces: Dict[str, Any] = {}
_artifacts: Dict[str, Any] = {}
_runs: Dict[str, Any] = {}
_notifications: List[Dict[str, Any]] = []

# WEBSOCKET BAĞLANTI YÖNETİCİSİ (State Sync)
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
        self.active_connections[session_id].append(websocket)

    def disconnect(self, websocket: WebSocket, session_id: str):
        if session_id in self.active_connections:
            self.active_connections[session_id].remove(websocket)
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]

    async def broadcast_to_session(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            for connection in self.active_connections[session_id]:
                await connection.send_json(message)

manager = ConnectionManager()

# --- 1) Auth & User Endpoints ---
@router.post("/auth/login", response_model=AuthSession, tags=["1. Auth & AuthZ"])
async def login():
    """Kullanıcı yetkilendirmesi ve session oluşturma (Mock)"""
    return AuthSession(
        session_id=f"sess-{uuid.uuid4().hex[:8]}",
        user_id="usr-mock",
        auth_method=AuthMethod.EMAIL_PASSWORD,
        access_token="mock_token",
        refresh_token="mock_refresh",
        expires_at="2026-12-31T23:59:59"
    )

@router.get("/users/me", tags=["1. Auth & AuthZ"])
async def get_current_user():
    return {"user_id": "usr-mock", "name": "Unified User", "role": "admin"}


# --- 2) Project Endpoints ---
@router.post("/projects", tags=["2. Projects"])
async def create_project(name: str, description: Optional[str] = ""):
    project_id = f"prj-{uuid.uuid4().hex[:8]}"
    proj = {"id": project_id, "name": name, "description": description, "status": "active"}
    _projects[project_id] = proj
    return proj

@router.get("/projects", tags=["2. Projects"])
async def list_projects():
    return list(_projects.values())

@router.post("/projects/{project_id}/invite", response_model=SharedProjectAccess, tags=["2. Projects"])
async def project_invite(project_id: str, access: SharedProjectAccess):
    if project_id not in _projects:
        raise HTTPException(status_code=404, detail="Project not found")
    return access


# --- 3) Workspace Endpoints ---
@router.get("/workspaces/{workspace_id}", response_model=WorkspaceSnapshot, tags=["3. Workspaces"])
async def get_workspace(workspace_id: str):
    """Bulut veya lokal çalışma alanının güncel snapshot'ını döner."""
    return WorkspaceSnapshot(
        snapshot_id=workspace_id,
        project_id="prj-mock",
        files_hash="hash_mock",
        is_synced=True,
        commit_message="Initial sync"
    )

@router.post("/workspaces/sync", tags=["3. Workspaces"])
async def sync_workspace(snapshot: WorkspaceSnapshot):
    _workspaces[snapshot.snapshot_id] = snapshot
    return {"status": "synced"}


# --- 4) Run (Agent Execution) Endpoints ---
@router.post("/runs", tags=["4. Runs"])
async def start_run(project_id: str, prompt: str):
    run_id = f"run-{uuid.uuid4().hex[:8]}"
    _runs[run_id] = {"id": run_id, "project": project_id, "prompt": prompt, "status": "running"}
    return _runs[run_id]

@router.get("/runs/{run_id}", tags=["4. Runs"])
async def get_run_status(run_id: str):
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail="Run not found")
    return _runs[run_id]

@router.post("/runs/{run_id}/events/emit", tags=["4. Runs"])
async def emit_live_run_event(run_id: str, event: StreamEvent):
    """Web/Mobil arayüzlere SSE/WS üzerinden canlı akacak stream logları."""
    # Olayı doğrudan WS ile mevcut bağlı tüm istemcilere at (Redis pub/sub entegrasyonu da gelebilir)
    event_dict = event.dict()
    asyncio.create_task(manager.broadcast_to_session(event.session_id, event_dict))
    return {"status": "emitted", "event_id": event.event_id, "run": run_id}

@router.websocket("/ws/events/{session_id}")
async def websocket_event_endpoint(websocket: WebSocket, session_id: str):
    """Tüm istemcilerin (Web/Mobil) kendi özel kanallarına bağlanarak canlı akışı izlediği WS Ucu."""
    await manager.connect(websocket, session_id)
    try:
        while True:
            # Client'dan ping gelirse veya input gelirse alınabilir. Şu an dinlemede.
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, session_id)

@router.get("/sessions/{session_id}", response_model=RemoteSessionRegistry, tags=["4. Runs"])
async def get_remote_session(session_id: str):
    """Aktif koşuların ve onay bekleyen işlemlerin state brokering işlemleri"""
    return RemoteSessionRegistry(
        registry_id=f"reg-{session_id}",
        run_id="run-mock",
        last_connected_at="2026-03-10T12:00:00Z"
    )


# --- 5) Approval (Human-in-the-Loop) Endpoints ---
@router.post("/approvals/action", response_model=RiskActionApprovalState, tags=["5. Approvals"])
async def handle_risk_action_approval(action: RiskActionApprovalState):
    """Yüksek riskli (shell, write, payment) eylemler için insan onayı."""
    action.is_human_approved = True  # Otomatik onay mock
    action.human_reviewer_id = "usr-human-1"
    return action

@router.post("/approvals/browser/takeover", tags=["5. Approvals"])
async def trigger_browser_takeover(event: BrowserTakeoverEvent):
    """Ajanın takıldığı yerde tarayıcı kontrolünü insanın devralması (Takeover)"""
    return {"status": "takeover_granted", "session_id": event.session_id}


# --- 6) Artifact Endpoints ---
@router.get("/artifacts/{artifact_id}/manifest", tags=["6. Artifacts"])
async def get_artifact_manifest(artifact_id: str):
    """Object Storage üzerindeki dosyanın meta ve indirme bilgilerini sunar."""
    if artifact_id not in _artifacts:
        _artifacts[artifact_id] = {"artifact_id": artifact_id, "status": "available", "url": f"s3://bio-ml-bucket/artifacts/{artifact_id}"}
    return _artifacts[artifact_id]


# --- 7) Notification Endpoints ---
@router.post("/notifications/emit", tags=["7. Notifications"])
async def send_notification(user_id: str, title: str, message: str):
    """Tüm kanallara atılacak ortak bildirim (Push, Email, WhatsApp vb.)"""
    notif = {"id": str(uuid.uuid4()), "user": user_id, "title": title, "msg": message, "read": False}
    _notifications.append(notif)
    return {"status": "sent", "notif_id": notif["id"]}

@router.get("/notifications/mine", tags=["7. Notifications"])
async def get_my_notifications():
    return _notifications[-10:] if len(_notifications) >= 10 else _notifications


# --- 8) Cloud Execution Target Endpoints ---
@router.post("/cloud-execution/job", tags=["8. Cloud Execution"])
async def dispatch_cloud_job(classification: JobClassification, package: RuntimePackage):
    """İşleri Cloud GPU / CPU node'larına yollayan execution router."""
    return {"status": "dispatched", "target": classification.target.value, "env_vars": len(package.env_vars)}

@router.post("/chat/async", tags=["Chat"])
async def chat_async(payload: Dict[str, Any]):
    """WhatsApp veya diğer kanallardan gelen asenkron mesaj işleyici."""
    session_id = payload.get("session_id", "default")
    message = payload.get("message", "")
    channel = payload.get("channel", "unknown")
    callback_url = payload.get("callback_url")
    callback_payload = payload.get("callback_payload", {})

    if not message:
        raise HTTPException(status_code=400, detail="message alanı zorunludur.")

    async def _run_chat() -> None:
        service = AgentService()
        service.session_id = session_id
        service.session_metadata["channel"] = channel

        final_text = ""
        try:
            for event in service.process_message(message):
                if event.get("type") in {"assistant", "chunk"}:
                    final_text += event.get("content", "")
        except Exception:
            if callback_url:
                error_body = dict(callback_payload)
                error_body["text"] = "Sistemsel hata: Mesaj işlenirken beklenmeyen bir problem oluştu."
                await _post_callback(callback_url, error_body)
            return

        if callback_url:
            body = dict(callback_payload)
            body["text"] = (final_text or "İşlem tamamlandı, ancak yanıt metni üretilemedi.").strip()
            await _post_callback(callback_url, body)

    async def _post_callback(url: str, body: Dict[str, Any]) -> None:
        def _do_post() -> None:
            req = urllib_request.Request(
                url=url,
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib_request.urlopen(req, timeout=10):
                    pass
            except urllib_error.URLError:
                pass

        await asyncio.to_thread(_do_post)

    asyncio.create_task(_run_chat())

    return {"status": "accepted", "session_id": session_id}

@router.get("/dashboard/summary", response_model=DashboardSummaryTemplate, tags=["Platform Dashboard"])
async def get_dashboard_summary():
    """Hafif istemciler (Mobil, Web) için birleştirilmiş platform özeti."""
    return DashboardSummaryTemplate(
        active_projects_count=len(_projects),
        running_jobs_count=len([r for r in _runs.values() if r["status"] == "running"]),
        pending_approvals_count=0,
        total_storage_used_mb=1250.5,
        estimated_cost_usd=5.20
    )
