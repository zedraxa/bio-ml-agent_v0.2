import time
from typing import Dict, Any, List
from fastapi import APIRouter
from pydantic import BaseModel
from bio_ml_agent.swarm.orchestrator import SwarmOrchestrator
from bio_ml_agent.utils.config import get_config

router = APIRouter()
app_config = get_config()

class ChatRequest(BaseModel):
    message: str

@router.post("/agent/chat")
async def dashboard_chat(req: ChatRequest):
    swarm = SwarmOrchestrator(app_config)
    messages = [{"role": "user", "content": req.message}]
    
    final_report = ""
    steps = []
    
    try:
        for update in swarm.process(messages):
            if update["type"] == "status":
                steps.append({"type": "tool", "tool": "Swarm Action", "output": update["content"]})
            elif update["type"] == "assistant":
                final_report += update["content"]
            elif update["type"] == "error":
                steps.append({"type": "error", "content": update["content"]})
    except Exception as e:
        return {"error": str(e)}

    return {
        "response": final_report,
        "steps": steps
    }

@router.get("/stats")
async def dashboard_stats():
    return {
        "total": 15,
        "completed": 12,
        "in_progress": 2,
        "pending": 1,
        "total_lines": 14205,
        "completion_pct": 80,
        "total_modules": 45,
        "total_tests": 120
    }

@router.get("/tasks")
async def dashboard_tasks():
    from bio_ml_agent.db.session import SessionLocal
    from bio_ml_agent.db.models import MissionDB
    db = SessionLocal()
    missions = db.query(MissionDB).order_by(MissionDB.created_at.desc()).limit(10).all()
    tasks = []
    for m in missions:
        status = "in_progress" if m.status == "running" else ("completed" if m.status == "completed" else "pending")
        tasks.append({
            "id": m.mission_id,
            "title": m.title,
            "description": m.objective,
            "status": status,
            "category": m.category or "ml",
            "priority": "high"
        })
    db.close()
    return {"tasks": tasks}

@router.get("/projects")
async def dashboard_projects():
    from bio_ml_agent.db.session import SessionLocal
    from bio_ml_agent.db.models import ProjectDB
    db = SessionLocal()
    projs = db.query(ProjectDB).all()
    out = []
    for p in projs:
        out.append({
            "id": p.project_id,
            "name": p.name,
            "file_count": 12,
            "path": f"/workspace/{p.project_id}",
            "has_results": True,
            "has_model": False,
            "has_report": True
        })
    db.close()
    return {"projects": out}

@router.get("/models")
async def dashboard_models():
    return {"models": []}

@router.get("/datasets")
async def dashboard_datasets():
    return {"datasets": []}

@router.get("/config")
async def dashboard_config():
    return {"agent": {"model": "qwen2.5:14b-instruct", "max_steps": 50, "timeout": 180}}

@router.get("/config/api-keys")
async def dashboard_api_keys():
    return {
        "OPENAI_API_KEY": bool(app_config.llm.openai_api_key),
        "ANTHROPIC_API_KEY": bool(app_config.llm.anthropic_api_key),
        "GOOGLE_API_KEY": bool(app_config.llm.gemini_api_key),
    }

@router.get("/ollama/models")
async def dashboard_ollama_models():
    return {"models": [{"name": "qwen2.5:14b-instruct", "size_gb": 8}]}
