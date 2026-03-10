import os
import json
import logging
import asyncio
import uuid
import time
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

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request

# Rate Limiter
limiter = Limiter(key_func=get_remote_address)
from redis import Redis  # type: ignore
from rq import Queue  # type: ignore
from utils.config import get_config

# FastAPI Uygulaması
app = FastAPI(
    title="Bio-ML Enterprise API",
    description="Bio-ML Agent V6 - Derin Öğrenme, AutoML ve Otonom Araştırma REST API'si",
    version="6.0.0"
)

# SlowAPI Limit Handler Ayarı
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS Ayarları (Tüm kaynaklara açık - geliştirme amaçlı olan '*' yerine kısıtlı default yapıldı)
origins = os.environ.get("API_ALLOW_ORIGINS", "http://localhost:5050,http://127.0.0.1:5050,http://localhost:8001,http://127.0.0.1:8001").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Correlation ID & Latency Middleware
@app.middleware("http")
async def add_process_time_header(request, call_next):
    correlation_id = request.headers.get("X-Correlation-ID") or uuid.uuid4().hex[:8]
    # Local context or logging extra can be used here
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Correlation-ID"] = correlation_id
    response.headers["X-Process-Time"] = f"{process_time:.4f}s"
    logging.info(f"REQ {correlation_id} | {request.method} {request.url.path} | Time: {process_time:.4f}s | Status: {response.status_code}")
    return response

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

class ClinicalDataRequest(BaseModel):
    data: Dict[str, Any] = Field(..., description="Klinik veri yükü (JSON formatında)")
    model_override: Optional[str] = Field(default=None, description="Analizde kullanılacak LLM modeli")

# Security Helpers
from fastapi import Request, Depends

async def verify_api_key(request: Request):
    expected_key = config.security.api_key
    if not expected_key:
        return # Güvenlik kapalı
        
    api_key = request.headers.get("X-API-Key")
    if api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Geçersiz veya eksik API Key."
        )

@app.post("/api/v1/agent/train_cnn", 
          status_code=status.HTTP_202_ACCEPTED, 
          tags=["Eğitim"],
          dependencies=[Depends(verify_api_key)])
@limiter.limit("5/minute")
async def trigger_cnn_training(request: Request, req: TrainCNNRequest):
    """
    Derin Öğrenme modülünü asenkron olarak tetikler ve bir görev ID'si döner.
    İşlem arka planda devam eder, durumu /api/v1/agent/status/{task_id} ile sorgulayabilirsiniz.
    """
    import uuid
    u_hex = str(uuid.uuid4().hex)
    session_id = f"api_sess_{u_hex[:8]}"
    
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

@app.post("/api/v1/rag/index", 
          status_code=status.HTTP_202_ACCEPTED, 
          tags=["RAG"],
          dependencies=[Depends(verify_api_key)])
@limiter.limit("3/minute")
async def trigger_rag_indexing(request: Request):
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

@app.post("/api/v1/webhook/clinical_data", 
          status_code=status.HTTP_202_ACCEPTED, 
          tags=["Webhook"],
          dependencies=[Depends(verify_api_key)])
@limiter.limit("20/minute")
async def clinical_data_webhook(request: Request, req: ClinicalDataRequest):
    """
    Dış sistemlerden (hastane, IoT) gelen klinik verileri alır ve arka planda Swarm analiz sürecini başlatır.
    İşlem arka planda devam eder, durumu /api/v1/agent/status/{task_id} ile sorgulayabilirsiniz.
    """
    import uuid
    u_hex = str(uuid.uuid4().hex)
    session_id = f"webhook_{u_hex[:8]}"
    
    # RQ'ya Gönder
    job = task_queue.enqueue(
        "job_worker.execute_swarm_job",
        session_id=session_id,
        payload_data=req.data,
        model_override=req.model_override,
        job_timeout=600  # Swarm uzun sürebilir
    )
    
    return {
        "task_id": job.id, 
        "message": "Klinik veri başarıyla alındı. Swarm analiz pipeline'ı arka planda başlatıldı.",
        "status_url": f"/api/v1/agent/status/{job.id}"
    }

@app.get("/api/v1/agent/status/{task_id}", response_model=TaskStatusResponse, tags=["Görevler"])
@limiter.limit("60/minute")
async def get_task_status(request: Request, task_id: str):
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
