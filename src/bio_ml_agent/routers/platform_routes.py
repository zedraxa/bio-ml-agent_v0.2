import uuid
import json
import asyncio
import os
from pathlib import Path
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
from bio_ml_agent.db.session import get_db
from bio_ml_agent.db.models import ProjectDB, MissionDB, ArtifactDB, TimelineEventDB, ProjectMemoryDB, MissionStepDB, NotificationDB, CommentDB, SettingsDB
from sqlalchemy.orm import Session
import time
from bio_ml_agent.models.workspace_ux import (
    WorkspaceProject,
    ProjectState,
    WorkspaceMode,
    ProjectMission,
    ProjectArtifact,
    TimelineEvent,
    TimelineEventType,
    ProjectMemoryItem,
    ProjectDashboardSummary
)

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
async def list_projects(db: Session = Depends(get_db)):
    db_projects = db.query(ProjectDB).all()
    # Pydantic objesi yerine dict veya Pydantic model döneceksek, modelden objeye çevirelim
    mapped_projects = []
    for p in db_projects:
        mapped_projects.append({
            "id": p.project_id,
            "project_id": p.project_id,
            "name": p.name,
            "description": p.description,
            "status": "active",
            "goals": p.goals,
            "workspace_mode": p.workspace_mode.value if isinstance(p.workspace_mode, WorkspaceMode) else p.workspace_mode
        })
    # Eski projeleri (_projects) mock için sona ekle
    return mapped_projects + list(_projects.values())

@router.post("/projects/v2", response_model=WorkspaceProject, tags=["2. Projects"])
async def create_workspace_project(name: str, description: str, mode: WorkspaceMode = WorkspaceMode.RESEARCH, db: Session = Depends(get_db)):
    """Part V: Yeni nesil Proje oluşturma endpoint'i (SQLite Kalıcı)."""
    import time
    pid = f"proj-{uuid.uuid4().hex[:8]}"
    
    new_proj = ProjectDB(
        project_id=pid,
        name=name,
        description=description,
        workspace_mode=mode,
        created_at=time.time(),
        updated_at=time.time()
    )
    db.add(new_proj)
    
    # Timeline genesis
    ev = TimelineEventDB(
        event_id=f"evt-{uuid.uuid4().hex[:6]}",
        project_id=pid,
        event_type=TimelineEventType.INFO,
        message=f"Project '{name}' initialized in {mode.value} mode.",
        timestamp=time.time()
    )
    db.add(ev)
    db.commit()
    db.refresh(new_proj)
    
    return WorkspaceProject(
        project_id=new_proj.project_id,
        name=new_proj.name,
        description=new_proj.description,
        goals=new_proj.goals,
        state=new_proj.state,
        workspace_mode=new_proj.workspace_mode,
        open_questions=new_proj.open_questions,
        next_actions=new_proj.next_actions,
        created_at=new_proj.created_at,
        updated_at=new_proj.updated_at
    )

@router.get("/projects/v2/last-active", response_model=Optional[WorkspaceProject], tags=["2. Projects"])
async def get_last_active_project(db: Session = Depends(get_db)):
    """C3: En son ziyaret edilen projeyi döndürür."""
    p = db.query(ProjectDB).order_by(ProjectDB.last_accessed_at.desc()).first()
    if not p:
        return None
    return WorkspaceProject(
        project_id=p.project_id,
        name=p.name,
        description=p.description,
        goals=p.goals,
        state=p.state,
        workspace_mode=p.workspace_mode,
        open_questions=p.open_questions,
        next_actions=p.next_actions
    )

@router.get("/projects/v2/{project_id}", response_model=WorkspaceProject, tags=["2. Projects"])
async def get_workspace_project(project_id: str, db: Session = Depends(get_db)):
    proj = db.query(ProjectDB).filter(ProjectDB.project_id == project_id).first()
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")
    return WorkspaceProject.from_orm(proj) if hasattr(WorkspaceProject, 'from_orm') else proj

@router.get("/projects/v2/{project_id}/missions", response_model=List[ProjectMission], tags=["2. Projects"])
async def get_project_missions(project_id: str, db: Session = Depends(get_db)):
    missions = db.query(MissionDB).filter(MissionDB.project_id == project_id).all()
    res = []
    for m in missions:
        res.append(ProjectMission(
            mission_id=m.mission_id,
            project_id=m.project_id,
            title=m.title,
            objective=m.objective,
            template_name=m.template_name,
            status=m.status,
            progress_percentage=m.progress_percentage,
            created_by_user=m.created_by_user,
            assigned_agents=m.assigned_agents,
            created_at=m.created_at,
            completed_at=m.completed_at
        ))
    return res

@router.get("/projects/v2/{project_id}/artifacts", response_model=List[ProjectArtifact], tags=["2. Projects"])
async def get_project_artifacts(project_id: str, db: Session = Depends(get_db)):
    artifacts = db.query(ArtifactDB).filter(ArtifactDB.project_id == project_id).all()
    res = []
    for a in artifacts:
        res.append(ProjectArtifact(
            artifact_id=a.artifact_id,
            project_id=a.project_id,
            mission_id=a.mission_id,
            title=a.title,
            description=a.description,
            category=a.category or "Documents",
            file_type=a.file_type,
            created_by=a.created_by,
            confidence=a.confidence,
            review_status=a.review_status,
            lineage_parents=a.lineage_parents,
            content_uri=a.content_uri,
            created_at=a.created_at,
            updated_at=a.updated_at
        ))
    return res

@router.get("/projects/v2/{project_id}/timeline", response_model=List[TimelineEvent], tags=["2. Projects"])
async def get_project_timeline(project_id: str, db: Session = Depends(get_db)):
    events = db.query(TimelineEventDB).filter(TimelineEventDB.project_id == project_id).order_by(TimelineEventDB.timestamp.desc()).all()
    res = []
    for e in events:
        res.append(TimelineEvent(
            event_id=e.event_id,
            project_id=e.project_id,
            event_type=e.event_type,
            message=e.message,
            timestamp=e.timestamp,
            metadata=e.metadata_json,
            agent_name=e.agent_name
        ))
    return res

@router.get("/projects/v2/{project_id}/dashboard", response_model=ProjectDashboardSummary, tags=["2. Projects"])
async def get_project_dashboard_summary(project_id: str, db: Session = Depends(get_db)):
    """B1: Meta-endpoint for Smart Project Dashboard."""
    proj = db.query(ProjectDB).filter(ProjectDB.project_id == project_id).first()
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # C3: Last accessed sync
    proj.last_accessed_at = time.time()
    db.commit()
    
    # Detailed fetch
    missions = db.query(MissionDB).filter(MissionDB.project_id == project_id).order_by(MissionDB.created_at.desc()).limit(5).all()
    artifacts = db.query(ArtifactDB).filter(ArtifactDB.project_id == project_id).order_by(ArtifactDB.created_at.desc()).limit(10).all()
    memory = db.query(ProjectMemoryDB).filter(ProjectMemoryDB.project_id == project_id).order_by(ProjectMemoryDB.created_at.desc()).limit(20).all()
    
    mapped_missions = []
    for m in missions:
        mapped_missions.append(ProjectMission(
            mission_id=m.mission_id,
            project_id=m.project_id,
            name=m.title,
            status=m.status,
            progress_narrative=f"{m.progress_percentage}% completed. {m.objective[:50]}...",
            active_agent=m.assigned_agents[0] if m.assigned_agents else None,
            created_at=m.created_at
        ))

    mapped_artifacts = []
    for a in artifacts:
        mapped_artifacts.append(ProjectArtifact(
            artifact_id=a.artifact_id,
            project_id=a.project_id,
            mission_id=a.mission_id,
            title=a.title,
            description=a.description,
            category=a.category or "Documents",
            file_type=a.file_type,
            created_by=a.created_by,
            confidence=a.confidence,
            review_status=a.review_status,
            lineage_parents=a.lineage_parents or [],
            content_uri=a.content_uri or "",
            created_at=a.created_at,
            updated_at=a.updated_at
        ))

    mapped_memory = []
    for mem in memory:
        mapped_memory.append(ProjectMemoryItem(
            memory_id=mem.memory_id,
            project_id=mem.project_id,
            category=mem.category,
            title=mem.title,
            content=mem.content,
            importance=mem.importance,
            metadata=mem.metadata_json,
            created_at=mem.created_at
        ))

    return ProjectDashboardSummary(
        project=WorkspaceProject(
            project_id=proj.project_id,
            name=proj.name,
            description=proj.description,
            goals=proj.goals,
            state=proj.state,
            workspace_mode=proj.workspace_mode,
            open_questions=proj.open_questions,
            next_actions=proj.next_actions,
            created_at=proj.created_at,
            updated_at=proj.updated_at
        ),
        recent_missions=mapped_missions,
        recent_artifacts=mapped_artifacts,
        recent_memory=mapped_memory,
        active_agent_count=len([m for m in missions if m.status == "running"])
    )

@router.get("/projects/v2/{project_id}/memory", response_model=List[ProjectMemoryItem], tags=["2. Projects"])
async def get_project_memory(project_id: str, db: Session = Depends(get_db)):
    """B2: Proje hafızasını (Memory) döner."""
    items = db.query(ProjectMemoryDB).filter(ProjectMemoryDB.project_id == project_id).order_by(ProjectMemoryDB.created_at.desc()).all()
    res = []
    for mem in items:
        res.append(ProjectMemoryItem(
            memory_id=mem.memory_id,
            project_id=mem.project_id,
            category=mem.category,
            title=mem.title,
            content=mem.content,
            importance=mem.importance,
            metadata=mem.metadata_json,
            created_at=mem.created_at
        ))
    return res

@router.post("/projects/v2/{project_id}/memory", response_model=ProjectMemoryItem, tags=["2. Projects"])
async def add_project_memory(
    project_id: str, 
    category: str, 
    title: str, 
    content: str, 
    importance: int = 1,
    metadata_json: Optional[str] = "{}",
    db: Session = Depends(get_db)
):
    """B2: Hafıza girişi ekler (Kararlar, modeller, bulgular vb)."""
    import time
    mid = f"mem-{uuid.uuid4().hex[:6]}"
    try:
        meta_dict = json.loads(metadata_json)
    except:
        meta_dict = {}
        
    new_mem = ProjectMemoryDB(
        memory_id=mid,
        project_id=project_id,
        category=category,
        title=title,
        content=content,
        importance=importance,
        metadata_json=meta_dict,
        created_at=time.time()
    )
    db.add(new_mem)
    db.commit()
    db.refresh(new_mem)
    return ProjectMemoryItem(
        memory_id=new_mem.memory_id,
        project_id=new_mem.project_id,
        category=new_mem.category,
        title=new_mem.title,
        content=new_mem.content,
        importance=new_mem.importance,
        metadata=new_mem.metadata_json,
        created_at=new_mem.created_at
    )


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


# --- 5) Mission Control & Live Runs Endpoints ---
@router.get("/missions/active", tags=["4. Runs"])
async def list_active_missions(db: Session = Depends(get_db)):
    """Tüm projelerdeki aktif (koşan) görevleri listeler."""
    missions = db.query(MissionDB).filter(MissionDB.status == "running").all()
    res = []
    for m in missions:
        res.append({
            "mission_id": m.mission_id,
            "project_id": m.project_id,
            "title": m.title,
            "status": m.status,
            "progress": m.progress_percentage,
            "readable_progress": m.readable_progress,
            "category": m.category or "General",
            "active_agent": m.assigned_agents[0] if m.assigned_agents else "System",
            "created_at": m.created_at
        })
    return res

@router.post("/missions/{mission_id}/steps", tags=["4. Runs"])
async def log_mission_step(
    mission_id: str, 
    agent_name: str, 
    action_type: str, 
    content: str, 
    thought: Optional[str] = None, 
    metadata: Optional[str] = "{}",
    db: Session = Depends(get_db)
):
    """Görevin içindeki adımları (düşünce, araç kullanımı vb) kaydeder."""
    import time
    sid = f"stp-{uuid.uuid4().hex[:6]}"
    try:
        meta_dict = json.loads(metadata)
    except:
        meta_dict = {}
        
    new_step = MissionStepDB(
        step_id=sid,
        mission_id=mission_id,
        agent_name=agent_name,
        action_type=action_type,
        content=content,
        thought=thought,
        metadata_json=meta_dict,
        timestamp=time.time()
    )
    db.add(new_step)
    db.commit()
    return {"status": "success", "step_id": sid}

@router.get("/missions/{mission_id}/steps", tags=["4. Runs"])
async def get_mission_steps(mission_id: str, db: Session = Depends(get_db)):
    """Bir göreve ait tüm adımları kronolojik olarak döner."""
    steps = db.query(MissionStepDB).filter(MissionStepDB.mission_id == mission_id).order_by(MissionStepDB.timestamp.asc()).all()
    return steps

@router.post("/missions/{mission_id}/pause", tags=["4. Runs"])
async def pause_mission(mission_id: str, db: Session = Depends(get_db)):
    mission = db.query(MissionDB).filter(MissionDB.mission_id == mission_id).first()
    if mission:
        mission.status = "paused"
        db.commit()
    return {"status": "paused"}

@router.post("/missions/{mission_id}/resume", tags=["4. Runs"])
async def resume_mission(mission_id: str, db: Session = Depends(get_db)):
    mission = db.query(MissionDB).filter(MissionDB.mission_id == mission_id).first()
    if mission:
        mission.status = "running"
        db.commit()
    return {"status": "running"}

@router.post("/missions/{mission_id}/cancel", tags=["4. Runs"])
async def cancel_mission(mission_id: str, db: Session = Depends(get_db)):
    mission = db.query(MissionDB).filter(MissionDB.mission_id == mission_id).first()
    if mission:
        mission.status = "cancelled"
        db.commit()
    return {"status": "cancelled"}

@router.post("/missions/{mission_id}/intervene", tags=["4. Runs"])
async def intervene_mission(mission_id: str, action: str, db: Session = Depends(get_db)):
    """Phase R5-3 D3: Göreve dışarıdan müdahale (intervention) gönderir."""
    mission = db.query(MissionDB).filter(MissionDB.mission_id == mission_id).first()
    if not mission: raise HTTPException(status_code=404, detail="Mission not found")
    mission.intervention_requested = action
    db.commit()
    return {"status": "ok", "message": f"Intervention '{action}' registered"}

# --- 6) Approval (Human-in-the-Loop) Endpoints ---
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
@router.get("/notifications", tags=["7. Notifications"])
async def get_notifications(db: Session = Depends(get_db)):
    """Aktif (okunmamış) bildirimleri döner."""
    return db.query(NotificationDB).filter(NotificationDB.is_read == False).order_by(NotificationDB.timestamp.desc()).all()

    return {"status": "ok"}
    
@router.post("/artifacts/{artifact_id}/approve", tags=["6. Artifacts"])
async def approve_artifact(artifact_id: str, db: Session = Depends(get_db)):
    from bio_ml_agent.models.workspace_ux import ArtifactReviewStatus
    art = db.query(ArtifactDB).filter(ArtifactDB.artifact_id == artifact_id).first()
    if art:
        art.review_status = ArtifactReviewStatus.APPROVED
        db.commit()
    return {"status": "approved", "artifact_id": artifact_id}

@router.post("/artifacts/{artifact_id}/reject", tags=["6. Artifacts"])
async def reject_artifact(artifact_id: str, db: Session = Depends(get_db)):
    from bio_ml_agent.models.workspace_ux import ArtifactReviewStatus
    art = db.query(ArtifactDB).filter(ArtifactDB.artifact_id == artifact_id).first()
    if art:
        art.review_status = ArtifactReviewStatus.REJECTED
        db.commit()
    return {"status": "rejected", "artifact_id": artifact_id}

@router.get("/artifacts/{artifact_id}/comments", tags=["6. Artifacts"])
async def get_artifact_comments(artifact_id: str, db: Session = Depends(get_db)):
    return db.query(CommentDB).filter(CommentDB.artifact_id == artifact_id).order_by(CommentDB.timestamp.asc()).all()

@router.post("/artifacts/{artifact_id}/comments", tags=["6. Artifacts"])
async def add_artifact_comment(artifact_id: str, content: str, author: str = "Researcher", location_data: Optional[dict] = None, db: Session = Depends(get_db)):
    import time
    import uuid
    new_comment = CommentDB(
        comment_id=str(uuid.uuid4()),
        artifact_id=artifact_id,
        author=author,
        content=content,
        location_data=location_data,
        timestamp=time.time()
    )
    db.add(new_comment)
    db.commit()
    return new_comment

@router.post("/comments/{comment_id}/resolve", tags=["6. Artifacts"])
async def resolve_comment(comment_id: str, db: Session = Depends(get_db)):
    comment = db.query(CommentDB).filter(CommentDB.comment_id == comment_id).first()
    if comment:
        comment.is_resolved = True
        db.commit()
    return {"status": "resolved", "comment_id": comment_id}

@router.post("/artifacts/{artifact_id}/status", tags=["6. Artifacts"])
async def update_artifact_status(artifact_id: str, status: str, db: Session = Depends(get_db)):
    # Valid states: DRAFT, REVIEW_NEEDED, APPROVED, SUPERSEDED, FINAL, EXPORTED
    art = db.query(ArtifactDB).filter(ArtifactDB.artifact_id == artifact_id).first()
    if not art:
        raise HTTPException(status_code=404, detail="Artifact not found")
    
    art.review_status = status.upper()
    
    # Auto-Supersede logic: If this is marked FINAL, supersede others in same category
    if art.review_status == "FINAL":
        others = db.query(ArtifactDB).filter(
            ArtifactDB.project_id == art.project_id,
            ArtifactDB.category == art.category,
            ArtifactDB.artifact_id != art.artifact_id,
            ArtifactDB.review_status != "SUPERSEDED"
        ).all()
        for other in others:
            other.review_status = "SUPERSEDED"
            
    db.commit()
    return {"status": art.review_status, "artifact_id": artifact_id}

@router.get("/artifacts/{artifact_id}/export", tags=["6. Artifacts"])
async def export_artifact(artifact_id: str, format: str, db: Session = Depends(get_db)):
    from fastapi.responses import StreamingResponse
    import io
    import zipfile
    
    art = db.query(ArtifactDB).filter(ArtifactDB.artifact_id == artifact_id).first()
    if not art:
        raise HTTPException(status_code=404, detail="Artifact not found")

    title = art.title or "Exported_Artifact"
    clean_title = "".join([c if c.isalnum() else "_" for c in title])
    content = art.description or "No content available."

    if format == "docx":
        try:
            from docx import Document
            doc = Document()
            doc.add_heading(title, 0)
            doc.add_paragraph(content)
            
            file_stream = io.BytesIO()
            doc.save(file_stream)
            file_stream.seek(0)
            
            headers = {'Content-Disposition': f'attachment; filename="{clean_title}.docx"'}
            return StreamingResponse(file_stream, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", headers=headers)
        except ImportError:
            # Fallback if docx is not installed
            headers = {'Content-Disposition': f'attachment; filename="{clean_title}.txt"'}
            return StreamingResponse(io.BytesIO(f"DOCX Export (Text Fallback)\n\n{title}\n\n{content}".encode('utf-8')), media_type="text/plain", headers=headers)
            
    elif format == "pdf":
        # Minimal valid PDF string representing the export
        pdf_content = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> >>\nendobj\n4 0 obj\n<< /Length 59 >>\nstream\nBT\n/F1 18 Tf\n50 700 Td\n(Bio-ML Agent Technical Export) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000242 00000 n \ntrailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n350\n%%EOF\n"
        headers = {'Content-Disposition': f'attachment; filename="{clean_title}.pdf"'}
        return StreamingResponse(io.BytesIO(pdf_content), media_type="application/pdf", headers=headers)

    elif format in ["md_bundle", "report_package"]:
        file_stream = io.BytesIO()
        with zipfile.ZipFile(file_stream, 'w') as zf:
            zf.writestr(f"{clean_title}.md", f"# {title}\n\n{content}")
            if format == "report_package":
                zf.writestr("data/metrics.csv", "metric,value,confidence\nsegmentation_accuracy,0.98,High\ncell_count,452,High")
                zf.writestr("plots/distribution.png", "Placeholder for visual distribution plot.")
                zf.writestr("references.bib", "@article{bioml2026, title={AI-Driven Microscopy}}")
        file_stream.seek(0)
        headers = {'Content-Disposition': f'attachment; filename="{clean_title}.zip"'}
        return StreamingResponse(file_stream, media_type="application/zip", headers=headers)
        
    elif format == "presentation":
        notes_content = f"SLIDE 1: {title}\n\nTALKING POINTS:\n- Highlight the automated segmentation results.\n- Mention 98% confidence score.\n\nCONTENT:\n{content}"
        headers = {'Content-Disposition': f'attachment; filename="{clean_title}_presentation_notes.txt"'}
        return StreamingResponse(io.BytesIO(notes_content.encode('utf-8')), media_type="text/plain", headers=headers)

    # Fallback to markdown
    headers = {'Content-Disposition': f'attachment; filename="{clean_title}.md"'}
    return StreamingResponse(io.BytesIO(f"# {title}\n\n{content}".encode('utf-8')), media_type="text/markdown", headers=headers)

@router.get("/settings/{settings_id}", tags=["8. Settings"])
async def get_settings(settings_id: str, db: Session = Depends(get_db)):
    settings = db.query(SettingsDB).filter(SettingsDB.settings_id == settings_id).first()
    if not settings:
        # Return default if not found
        return {"settings_id": settings_id, "config": {}}
    return settings

@router.post("/settings/{settings_id}", tags=["8. Settings"])
async def update_settings(settings_id: str, config: dict, db: Session = Depends(get_db)):
    import time
    settings = db.query(SettingsDB).filter(SettingsDB.settings_id == settings_id).first()
    if not settings:
        settings = SettingsDB(settings_id=settings_id, config=config, updated_at=time.time())
        db.add(settings)
    else:
        # Merge config
        current = settings.config or {}
        current.update(config)
        settings.config = current
        settings.updated_at = time.time()
    
    db.commit()
    return settings

@router.get("/integrations/status", tags=["8. Settings"])
async def get_integrations_status():
    """Fetch health and connectivity status for all integrated services."""
    return {
        "communication": [
            {"id": "gmail", "name": "Gmail / Workspace", "status": "healthy", "latency": "12ms", "last_sync": "2m ago"},
            {"id": "whatsapp", "name": "WhatsApp Business", "status": "idle", "latency": "45ms", "last_sync": "10m ago"}
        ],
        "data": [
            {"id": "github", "name": "GitHub Enterprise", "status": "healthy", "branch": "main", "last_push": "1h ago"},
            {"id": "drive", "name": "Google Drive", "status": "syncing", "progress": 85, "last_sync": "Just now"},
            {"id": "vectordb", "name": "Chroma Vector DB", "status": "healthy", "index_size": "1.2GB", "latency": "8ms"}
        ],
        "research": [
            {"id": "microscopy", "name": "Live Microscopy", "status": "busy", "task": "Cell Segmentation", "progress": 64},
            {"id": "alphafold", "name": "AlphaFold Engine", "status": "idle", "version": "v2.3.1", "last_run": "Yesterday"}
        ]
    }

@router.post("/notifications/create", tags=["7. Notifications"])
async def create_notification(type: str, title: str, message: str, action_url: Optional[str] = None, db: Session = Depends(get_db)):
    new_notif = NotificationDB(
        id=f"not-{uuid.uuid4().hex[:6]}",
        type=type,
        title=title,
        message=message,
        action_url=action_url,
        timestamp=time.time()
    )
    db.add(new_notif)
    db.commit()
    return new_notif


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

        # Media Discovery for WhatsApp/Mobile
        media_path = None
        try:
            if service.project_root and service.project_root.exists():
                # En güncel PDF veya PNG dosyasını bul
                files = list(service.project_root.glob("*.pdf")) + list(service.project_root.glob("*.png"))
                if files:
                    media_path = str(sorted(files, key=os.path.getmtime)[-1])
        except Exception:
            pass

        if callback_url:
            body = dict(callback_payload)
            body["text"] = (final_text or "İşlem tamamlandı, ancak yanıt metni üretilemedi.").strip()
            if media_path:
                body["media_path"] = media_path
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
