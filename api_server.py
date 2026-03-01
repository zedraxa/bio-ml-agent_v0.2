import os
import json
import logging
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
from pydantic import BaseModel, Field
from fastapi import FastAPI, BackgroundTasks, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# Logger Ayarı
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
import uvicorn # type: ignore
from redis import Redis  # type: ignore
from rq import Queue  # type: ignore
from utils.config import get_config

# FastAPI Uygulaması
app = FastAPI(
    title="Bio-ML Enterprise API",
    description="Bio-ML Agent V6 - Derin Öğrenme, AutoML ve Otonom Araştırma REST API'si",
    version="6.0.0"
)

# CORS Ayarları (Tüm kaynaklara açık - geliştirme amaçlı)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from redis import Redis
from rq import Queue
from utils.config import get_config

# Config & Redis Queue
config = get_config()
redis_conn = Redis(
    host=config.redis.host,
    port=config.redis.port,
    db=config.redis.db,
    password=config.redis.password or None
)
task_queue = Queue('agent_tasks', connection=redis_conn)

# ── Pydantic Modelleri (Veri Doğrulama) ──

class TrainCNNRequest(BaseModel):
    dataset_path: str = Field(..., description="Eğitim verilerinin bulunduğu dizin (örn: data/raw/brain_mri)")
    preset: str = Field(..., description="Medikal preset (brain_mri, chest_xray, vb.)")
    architecture: str = Field(default="resnet18", description="CNN mimarisi (resnet18, efficientnet_b0, vb.)")
    epochs: int = Field(default=10, description="Eğitim epoch sayısı", ge=1, le=100)

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    message: str
    result: Optional[Dict[str, Any]] = None

@app.post("/api/v1/agent/train_cnn", status_code=status.HTTP_202_ACCEPTED, tags=["Eğitim"])
async def trigger_cnn_training(req: TrainCNNRequest):
    """
    Derin Öğrenme modülünü asenkron olarak tetikler ve bir görev ID'si döner.
    İşlem arka planda devam eder, durumu /api/v1/agent/status/{task_id} ile sorgulayabilirsiniz.
    """
    import uuid
    session_id = f"api_sess_{uuid.uuid4().hex[:8]}"
    
    # Prompt hazırlığı
    prompt = (
        f"Lütfen yetenekli bir yapay zeka mühendisi olarak davran. Kullanıcı, {req.dataset_path} dizinindeki "
        f"görüntü veya veriler ile, {req.architecture} mimarisini kullanarak {req.preset} konfigürasyonunda "
        f"{req.epochs} epoch süren bir Deep Learning/CNN eğitimi yapmanı istiyor.\n\n"
        f"Bunun için PYTHON aracını kullan. Eğittiğin modelin çıktılarını "
        f"ve sonuçlarını results/api_tasks/ altına kaydet ve başarısını raporla."
    )
    
    # RQ'ya Gönder
    job = task_queue.enqueue(
        "job_worker.execute_agent_job",
        session_id=session_id,
        prompt=prompt,
        model=config.agent.model,
        timeout=config.agent.timeout,
        max_steps=config.agent.max_steps,
        job_timeout=config.agent.timeout + 120  # İşlem uzun sürebileceğinden queue limitini arttırıyoruz
    )
    
    return {
        "task_id": job.id, 
        "message": "Eğitim görevi arka planda (Redis MQ) başlatıldı.",
        "status_url": f"/api/v1/agent/status/{job.id}"
    }

@app.post("/api/v1/rag/index", status_code=status.HTTP_202_ACCEPTED, tags=["RAG"])
async def trigger_rag_indexing():
    """
    Tüm workspace dizinindeki desteklenen dosyaları (PDF, DOCX, TXT, PY vb.) asenkron olarak RAG için indeksler.
    İşlem arka planda devam eder, durumu /api/v1/agent/status/{task_id} ile sorgulayabilirsiniz.
    """
    job = task_queue.enqueue(
        "job_worker.index_documents_job",
        job_timeout=600  # İndeksleme büyük projelerde uzun sürebilir (10 dk limit)
    )
    
    return {
        "task_id": job.id,
        "message": "RAG İndeksleme görevi başlatıldı.",
        "status_url": f"/api/v1/agent/status/{job.id}"
    }

@app.get("/api/v1/agent/status/{task_id}", response_model=TaskStatusResponse, tags=["Görevler"])
async def get_task_status(task_id: str):
    """RQ üzerinde çalışan arka plan görev durumunu sorgular."""
    from rq.job import Job
    from rq.exceptions import NoSuchJobError
    
    try:
        job = Job.fetch(task_id, connection=redis_conn)
    except NoSuchJobError:
        raise HTTPException(status_code=404, detail="Görev bulunamadı.")
        
    status_map = {
        "queued": "pending",
        "started": "running",
        "finished": "completed",
        "failed": "error",
        "deferred": "pending",
        "canceled": "error",
        "stopped": "error",
    }
    
    mapped_status = status_map.get(job.get_status(), "unknown")
    progress_msg = job.meta.get("progress", "Görev sıraya alındı, başlatılması bekleniyor...")
    error_msg = job.meta.get("error")
    
    if error_msg:
        progress_msg = f"Hata: {error_msg}"
        
    result_data = None
    if mapped_status == "completed":
        result_data = {"agent_report": job.meta.get("result_summary", "")}
        
    return TaskStatusResponse(
        task_id=task_id,
        status=mapped_status,
        message=progress_msg,
        result=result_data
    )


# Sunucuyu doğrudan başlatmak için
if __name__ == "__main__":
    import uvicorn # type: ignore
    uvicorn.run("api_server:app", host="0.0.0.0", port=8001, reload=True)
