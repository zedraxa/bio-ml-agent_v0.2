import os
import json
import logging
import asyncio
import uuid
import time
from typing import Dict, Any, Optional, List
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field
from fastapi import FastAPI, BackgroundTasks, HTTPException, status, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
import httpx

# Logger Ayarı
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
import uvicorn # type: ignore
try:
    from redis import Redis  # type: ignore
except ImportError:
    Redis = None  # type: ignore[assignment,misc]
try:
    from rq import Queue  # type: ignore
except ImportError:
    Queue = None  # type: ignore[assignment,misc]
from bio_ml_agent.utils.config import get_config

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    limiter = Limiter(key_func=get_remote_address)
except ImportError:
    # slowapi optional — rate limiting disabled when not installed
    limiter = None  # type: ignore[assignment]
    _rate_limit_exceeded_handler = None  # type: ignore[assignment]
    RateLimitExceeded = None  # type: ignore[assignment,misc]
    def get_remote_address(request):  # type: ignore[misc]
        return getattr(request.client, "host", "unknown")

# FastAPI Uygulaması
app = FastAPI(
    title="Bio-ML Enterprise API",
    description="Bio-ML Agent V6 - Derin Öğrenme, AutoML ve Otonom Araştırma REST API'si",
    version="6.0.0"
)

@app.on_event("startup")
def startup_event():
    from bio_ml_agent.db.session import engine, Base
    from bio_ml_agent.db.models import ProjectDB  # import to register metadata
    Base.metadata.create_all(bind=engine)
    logging.info("SQLite Workspace Database initialized.")

# Canonical Routers
from bio_ml_agent.routers.platform_routes import router as platform_router

app.include_router(platform_router, prefix="/api/v1/platform")

# SlowAPI Limit Handler Ayarı
if limiter is not None:
    app.state.limiter = limiter
if RateLimitExceeded is not None and _rate_limit_exceeded_handler is not None:
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

# Static Files & Frontend SPA Support
static_path = Path(__file__).resolve().parent.parent / "static"
if not static_path.exists():
    static_path.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

@app.get("/", tags=["UI"])
async def serve_spa():
    """Yeni nesil Unified Workspace UI'yı (Part V) servis eder."""
    spa_path = static_path / "v2" / "index.html"
    if spa_path.exists():
        return FileResponse(spa_path)
    return JSONResponse({"status": "UI initialized", "message": "Unified Workspace index.html not found. Creating it now..."}, status_code=202)

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

class ClinicalDataRequest(BaseModel):
    data: Dict[str, Any] = Field(..., description="Klinik veri yükü (JSON formatında)")
    model_override: Optional[str] = Field(default=None, description="Analizde kullanılacak LLM modeli")

# Security Helpers
# The following imports are redundant as they are already present above.
# from fastapi import Request, Depends
import hmac
import hashlib

async def verify_api_key(request: Request):
    expected_key = config.security.api_key
    if not expected_key:
        return # Güvenlik kapalı
        
    api_key = request.headers.get("X-API-Key")
    if api_key != expected_key and api_key != config.gateway.secret_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Geçersiz veya eksik API Key."
        )

async def verify_webhook_signature(request: Request):
    """HMAC SHA256 Webhook imza doğrulaması"""
    expected_secret = config.security.webhook_secret
    if not expected_secret:
        return # Eğer webhook secret girilmemişse imza kontrolü pas geçilir
        
    signature = request.headers.get("X-Webhook-Signature")
    if not signature:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-Webhook-Signature başlığı eksik."
        )

    # Payload'u binary olarak oku
    payload = await request.body()
    
    expected_mac = hmac.new(expected_secret.encode(), payload, hashlib.sha256).hexdigest()
    expected_sig = f"sha256={expected_mac}"
    
    if not hmac.compare_digest(expected_sig, signature):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="G# Legacy Placeholder Endpoints (REMOVED) - Logic moved to MissionOrchestrator and Platform Routes"
        )

@app.post("/api/v1/agent/train_cnn",
          status_code=status.HTTP_202_ACCEPTED,
          tags=["Eğitim"])
async def trigger_cnn_training(req: TrainCNNRequest):
    """
    Derin Öğrenme modülünü asenkron olarak tetikler ve bir görev ID'si döner.
    """
    import uuid as _uuid
    task_id = f"cnn_{_uuid.uuid4().hex[:8]}"
    if task_queue is not None:
        try:
            task_queue.enqueue(
                "job_worker.execute_agent_job",
                session_id=task_id,
                prompt=f"Train {req.architecture} on {req.dataset_path}",
                job_timeout=600,
            )
        except Exception:
            pass
    return {
        "task_id": task_id,
        "message": "Eğitim görevi başlatıldı.",
        "status_url": f"/api/v1/agent/status/{task_id}",
    }


#           status_code=status.HTTP_202_ACCEPTED,
#           tags=["Eğitim"],
#           dependencies=[Depends(verify_api_key)])
# @limiter.limit("5/minute")
# async def trigger_cnn_training(request: Request, req: TrainCNNRequest):
#     """
#     Derin Öğrenme modülünü asenkron olarak tetikler ve bir görev ID'si döner.
#     İşlem arka planda devam eder, durumu /api/v1/agent/status/{task_id} ile sorgulayabilirsiniz.
#     """
#     import uuid
#     u_hex = str(uuid.uuid4().hex)
#     session_id = f"api_sess_{u_hex[:8]}"
    
#     # Prompt hazırlığı
#     prompt = (
#         f"Lütfen yetenekli bir yapay zeka mühendisi olarak davran. Kullanıcı, {req.dataset_path} dizinindeki "
#         f"görüntü veya veriler ile, {req.architecture} mimarisini kullanarak {req.preset} konfigürasyonunda "
#         f"{req.epochs} epoch süren bir Deep Learning/CNN eğitimi yapmanı istiyor.\n\n"
#         f"Bunun için PYTHON aracını kullan. Eğittiğin modelin çıktılarını "
#         f"ve sonuçlarını results/api_tasks/ altına kaydet ve başarısını raporla."
#     )
    
#     # RQ'ya Gönder
#     job = task_queue.enqueue(
#         "job_worker.execute_agent_job",
#         session_id=session_id,
#         prompt=prompt,
#         model=config.agent.model,
#         timeout=config.agent.timeout,
#         max_steps=config.agent.max_steps,
#         job_timeout=config.agent.timeout + 120  # İşlem uzun sürebileceğinden queue limitini arttırıyoruz
#     )
    
#     return {
#         "task_id": job.id, 
#         "message": "Eğitim görevi arka planda (Redis MQ) başlatıldı.",
#         "status_url": f"/api/v1/agent/status/{job.id}"
#     }

# @app.post("/api/v1/rag/index", 
#           status_code=status.HTTP_202_ACCEPTED, 
#           tags=["RAG"],
#           dependencies=[Depends(verify_api_key)])
# @limiter.limit("3/minute")
# async def trigger_rag_indexing(request: Request):
#     """
#     Tüm workspace dizinindeki desteklenen dosyaları (PDF, DOCX, TXT, PY vb.) asenkron olarak RAG için indeksler.
#     İşlem arka planda devam eder, durumu /api/v1/agent/status/{task_id} ile sorgulayabilirsiniz.
#     """
#     job = task_queue.enqueue(
#         "job_worker.index_documents_job",
#         job_timeout=600  # İndeksleme büyük projelerde uzun sürebilir (10 dk limit)
#     )
    
#     return {
#         "task_id": job.id,
#         "message": "RAG İndeksleme görevi başlatıldı.",
#         "status_url": f"/api/v1/agent/status/{job.id}"
#     }

# @app.get("/api/v1/agent/hitl/pending", tags=["Onay"])
# async def get_pending_hitl():
#     """Bekleyen tüm HITL onay isteklerini getirir."""
#     keys = redis_conn.keys("hitl:request:*")
#     requests = []
#     for k in keys:
#         data = redis_conn.get(k)
#         if data:
#             requests.append(json.loads(data))
#     return {"pending_requests": requests}

# @app.post("/api/v1/agent/hitl/{approval_id}", tags=["Onay"], dependencies=[Depends(verify_api_key)])
# @limiter.limit("20/minute")
# async def process_hitl_approval(request: Request, approval_id: str, payload: dict):
#     """
#     Bekleyen HITL isteğine onay (veya ret) gönderir.
#     Örnek payload: {"approved": true}
#     """
#     approved = payload.get("approved", False)
#     res_key = f"hitl:response:{approval_id}"
#     req_key = f"hitl:request:{approval_id}"
    
#     req_data = redis_conn.get(req_key)
#     if not req_data:
#         raise HTTPException(status_code=404, detail="Onay isteği bulunamadı veya zaman aşımına uğramış.")
        
#     res_data = {"approved": approved, "timestamp": time.time()}
#     redis_conn.setex(res_key, 300, json.dumps(res_data))
    
#     status_str = "Onaylandı" if approved else "Reddedildi"
#     return {"message": f"İşlem {approval_id} başarıyla {status_str}."}

# @app.post("/api/v1/webhook/clinical_data", 
#           status_code=status.HTTP_202_ACCEPTED, 
#           tags=["Webhook"],
#           dependencies=[Depends(verify_api_key)])
# @limiter.limit("20/minute")
# async def clinical_data_webhook(request: Request, req: ClinicalDataRequest):
#     """
#     Dış sistemlerden (hastane, IoT) gelen klinik verileri alır ve arka planda Swarm analiz sürecini başlatır.
#     İşlem arka planda devam eder, durumu /api/v1/agent/status/{task_id} ile sorgulayabilirsiniz.
#     """
#     import uuid
#     u_hex = str(uuid.uuid4().hex)
#     session_id = f"webhook_{u_hex[:8]}"
    
#     # RQ'ya Gönder
#     job = task_queue.enqueue(
#         "job_worker.execute_swarm_job",
#         session_id=session_id,
#         payload_data=req.data,
#         model_override=req.model_override,
#         job_timeout=600  # Swarm uzun sürebilir
#     )
    
#     return {
#         "task_id": job.id, 
#         "message": "Klinik veri başarıyla alındı. Swarm analiz pipeline'ı arka planda başlatıldı.",
#         "status_url": f"/api/v1/agent/status/{job.id}"
#     }

# Task Status logic moved to platform routes
# @app.get("/api/v1/agent/status/{task_id}", response_model=TaskStatusResponse, tags=["Görevler"])
# @limiter.limit("60/minute")
# async def get_task_status(request: Request, task_id: str):
#     """RQ üzerinde çalışan arka plan görev durumunu sorgular."""
#     from rq.job import Job
#     from rq.exceptions import NoSuchJobError
    
#     try:
#         job = Job.fetch(task_id, connection=redis_conn)
#     except NoSuchJobError:
#         raise HTTPException(status_code=404, detail="Görev bulunamadı.")
        
#     status_map = {
#         "queued": "pending",
#         "started": "running",
#         "finished": "completed",
#         "failed": "error",
#         "deferred": "pending",
#         "canceled": "error",
#         "stopped": "error",
#     }
    
#     mapped_status = status_map.get(job.get_status(), "unknown")
#     progress_msg = job.meta.get("progress", "Görev sıraya alındı, başlatılması bekleniyor...")
#     error_msg = job.meta.get("error")
    
#     if error_msg:
#         progress_msg = f"Hata: {error_msg}"
        
#     result_data = None
#     if mapped_status == "completed":
#         result_data = {"agent_report": job.meta.get("result_summary", "")}
        
#     return TaskStatusResponse(
#         task_id=task_id,
#         status=mapped_status,
#         message=progress_msg,
#         result=result_data
#     )


@app.get("/health", tags=["Sistem"])
async def health_check():
    """Sistem sağlık kontrolü (Docker Healthcheck için)"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "6.0.0",
        "mode": "enterprise"
    }


# ── Gözlemlenebilirlik (Observability) ────────────────

@app.get("/api/v1/observability/metrics", tags=["Gözlemlenebilirlik"])
async def get_metrics():
    """Sistem maliyet ve kullanım metriklerini döndürür."""
    from bio_ml_agent.services.observability.metrics import metrics
    return metrics.get_cost_report()
@app.get("/api/v1/observability/audit", tags=["Gözlemlenebilirlik"])
async def get_audit_logs(limit: int = 50):
    """Kritik eylemlerin denetim günlüklerini döndürür."""
    from bio_ml_agent.ultra_agent.observability.audit_trail import AuditTrailLogger
    # api_server configuration'ı global 'config' nesnesinden alıyor
    logger = AuditTrailLogger(workspace=Path(config.workspace.base_dir))

# ── Gateway & Proxy Endpoints ──

async def _do_proxy(target_url: str, request: Request):
    """S8-5: Yardımcı proxy fonksiyonu."""
    async with httpx.AsyncClient() as client:
        method = request.method
        headers = dict(request.headers)
        headers.pop("host", None) # Host çakışmasını önle
        
        # Orijinal gövdeyi (body) al
        content = await request.body()
        
        try:
            resp = await client.request(
                method,
                target_url,
                headers=headers,
                content=content,
                params=request.query_params,
                timeout=30.0
            )
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=dict(resp.headers)
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Upstream service error: {str(e)}")

@app.api_route("/api/v1/gateway/mlflow/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_mlflow(path: str, request: Request, _=Depends(verify_api_key)):
    if not config.gateway.enabled:
        raise HTTPException(status_code=403, detail="Gateway modu kapalı.")
    target = f"{config.gateway.mlflow_url}/{path}"
    return await _do_proxy(target, request)

@app.api_route("/api/v1/gateway/qdrant/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_qdrant(path: str, request: Request, _=Depends(verify_api_key)):
    if not config.gateway.enabled:
        raise HTTPException(status_code=403, detail="Gateway modu kapalı.")
    target = f"{config.gateway.qdrant_url}/{path}"
    return await _do_proxy(target, request)

# Sunucuyu doğrudan başlatmak için
def main():
    import uvicorn # type: ignore
    uvicorn.run("bio_ml_agent.api.api_server:app", host="0.0.0.0", port=8001, reload=True)

if __name__ == "__main__":
    main()
