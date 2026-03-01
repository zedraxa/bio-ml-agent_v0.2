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
logger = logging.getLogger(__name__)

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

# ── Görev Durumu Veritabanı (Simülasyon için bellekte) ──
# Gerçek dünyada Redis veya PostgreSQL kullanılmalıdır.
background_tasks_db: Dict[str, Dict[str, Any]] = {}

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

# ── Arka Plan Görev Fonksiyonları ──

def run_cnn_training_task(task_id: str, req: TrainCNNRequest):
    """
    Arka planda CNN eğitimini yürütür ve durumu günceller.
    Ayrıca AgentService kullanarak derin öğrenme kodunu otonom olarak (python aracılığıyla) 
    ya da doğrudan deep_learning metodlarıyla çalıştırabilecek bir agent session başlatır.
    """
    logger.info(f"[Task {task_id}] Derin Öğrenme süreci AgentService ile başlatılıyor...")
    background_tasks_db[task_id]["status"] = "running"
    background_tasks_db[task_id]["message"] = f"Ajan {req.architecture} mimarisini kuruyor..."
    
    try:
        from services.agent_service import AgentService
        
        # Çıktı klasörünü göreve özel oluştur
        output_dir = f"results/api_tasks/{task_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        service = AgentService(model="qwen2.5:7b-instruct", timeout=300, max_steps=15)
        
        prompt = (
            f"Lütfen yetenekli bir yapay zeka mühendisi olarak davran. Kullanıcı, {req.dataset_path} dizinindeki "
            f"görüntü veya veriler ile, {req.architecture} mimarisini kullanarak {req.preset} konfigürasyonunda "
            f"{req.epochs} epoch süren bir Deep Learning/CNN eğitimi yapmanı istiyor.\n\n"
            f"Bunun için PYTHON aracını kullan. `deep_learning.quick_train_cnn` vb scriptleri kullanabilirsin. Eğittiğin modelin çıktılarını "
            f"ve sonuçlarını `{output_dir}` klasörüne kaydet ve başarısını raporla."
        )

        for event in service.process_message(prompt):
            ev_type = event.get("type")
            if ev_type == "status":
                background_tasks_db[task_id]["message"] = event.get("content", "")
            elif ev_type == "tool_start":
                background_tasks_db[task_id]["message"] = f"Ajan {event.get('tool')} aracını çalıştırıyor..."
        
        # Başarı durumu kaydı
        last_message = ""
        if service.messages and service.messages[-1].get("role") == "assistant":
            last_message = service.messages[-1].get("content", "")

        logger.info(f"[Task {task_id}] Başarıyla tamamlandı.")
        background_tasks_db[task_id]["status"] = "completed"
        background_tasks_db[task_id]["message"] = "Model ajan tarafından başarıyla eğitildi ve rapor üretildi."
        background_tasks_db[task_id]["result"] = {"agent_report": last_message}
        
    except Exception as e:
        logger.error(f"[Task {task_id}] HATA: {str(e)}")
        background_tasks_db[task_id]["status"] = "failed"
        background_tasks_db[task_id]["message"] = str(e)



# ── REST API Uç Noktaları ──

@app.get("/", tags=["Sistem"])
async def root():
    """Sistemin çalışıp çalışmadığını kontrol eder."""
    return {"status": "ok", "message": "Bio-ML Enterprise API Çalışıyor", "version": "6.0.0"}

@app.post("/api/v1/agent/train_cnn", status_code=status.HTTP_202_ACCEPTED, tags=["Eğitim"])
async def trigger_cnn_training(req: TrainCNNRequest, background_tasks: BackgroundTasks):
    """
    Derin Öğrenme modülünü asenkron olarak tetikler ve bir görev ID'si döner.
    İşlem arka planda devam eder, durumu /api/v1/agent/status/{task_id} ile sorgulayabilirsiniz.
    """
    import uuid
    task_id = f"task_{uuid.uuid4().hex[:8]}"
    
    # Başlangıç durumu
    background_tasks_db[task_id] = {
        "status": "pending",
        "message": "Görev sıraya alındı, başlatılması bekleniyor...",
        "request": req.model_dump()
    }
    
    # Arka plana at
    background_tasks.add_task(run_cnn_training_task, task_id, req)
    
    return {
        "task_id": task_id, 
        "message": "Eğitim görevi arka planda başlatıldı.",
        "status_url": f"/api/v1/agent/status/{task_id}"
    }

@app.get("/api/v1/agent/status/{task_id}", response_model=TaskStatusResponse, tags=["Görevler"])
async def get_task_status(task_id: str):
    """Arka planda çalışan ML eğitiminin veya XAI raporunun güncel durumunu sorgular."""
    if task_id not in background_tasks_db:
        raise HTTPException(status_code=404, detail="Görev bulunamadı.")
        
    task_info = background_tasks_db[task_id]
    return TaskStatusResponse(
        task_id=task_id,
        status=task_info["status"],
        message=task_info["message"],
        result=task_info.get("result")
    )


# Sunucuyu doğrudan başlatmak için
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
