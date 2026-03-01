import os
import sys
import uuid
import json
import logging
from redis import Redis
from rq import Queue, Worker, get_current_job
from datetime import datetime
from pathlib import Path

# Proje dizinini path'e ekle
sys.path.insert(0, str(Path(__file__).resolve().parent))

from services.agent_service import AgentService
from utils.config import load_config
from rag_engine import RAGEngine

# Ayarları yükle
config = load_config()

# Redis Bağlantısı (RQ için)
redis_conn = Redis(
    host=config.redis.host,
    port=config.redis.port,
    db=config.redis.db,
    password=config.redis.password or None
)

# Servisi initialize edelim
task_queue = Queue('agent_tasks', connection=redis_conn)

# Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] Worker: %(message)s')
log = logging.getLogger("worker")

def execute_agent_job(session_id: str, prompt: str, model: str, timeout: int, max_steps: int):
    """
    Kuyruktan (Redis) alınıp background (Arkaplan) olarak yürütülen asıl Worker prosesi.
    Bu metod `AgentService`i çağırıp status güncellemelerini işler ve bitince `job.meta` kayıt eder.
    """
    job = get_current_job()
    if job:
        job.meta['progress'] = 'Başlıyor...'
        job.save_meta()

    try:
        service = AgentService(model=model, timeout=timeout, max_steps=max_steps)
        # Mevcut bir konuşma geçmişi çekilebilir, şimdilik sıfırdan başlatalım
        service.reset_session()
        service.session_id = session_id
        
        final_answer = ""
        
        log.info(f"[Job {job.id if job else 'local'}] Ajan servisi {session_id} için başlatıldı.")

        for event in service.process_message(prompt):
            ev_type = event.get("type")
            status_text = event.get("content", "")
            
            if ev_type == "status":
                log.info(f"Durum: {status_text}")
                if job:
                    job.meta['progress'] = status_text
                    job.save_meta()
            elif ev_type == "tool_start":
                tool_name = event.get("tool")
                if job:
                    job.meta['progress'] = f"Ajan {tool_name} aracını kullanıyor..."
                    job.save_meta()
            elif ev_type == "chunk":
                pass
        
        # Başarı
        if service.messages and service.messages[-1]["role"] == "assistant":
            final_answer = service.messages[-1]["content"]

        if job:
            job.meta['progress'] = 'P0 Tamamlandı - Sonuç Kaydediliyor'
            job.meta['result_summary'] = final_answer
            job.meta['updated_at'] = datetime.now().isoformat()
            job.save_meta()

        return {"status": "completed", "result": final_answer}

    except Exception as e:
        log.error(f"[Job {job.id if job else 'local'}] HATA: {e}")
        if job:
            job.meta['progress'] = 'Hata Oluştu'
            job.meta['error'] = str(e)
            job.save_meta()
        raise e

def index_documents_job():
    """
    Tüm workspace'i asenkron olarak RAG için indeksleyen RQ görevi.
    """
    job = get_current_job()
    if job:
        job.meta['progress'] = 'İndeksleme başlıyor...'
        job.save_meta()
        
    try:
        workspace_path = Path(config.workspace.base_dir).expanduser().resolve()
        rag = RAGEngine(workspace_dir=workspace_path)
        
        log.info(f"[Job {job.id if job else 'local'}] RAG İndekslemesi başlatıldı: {workspace_path}")
        
        count = rag.index_workspace()
        
        if job:
            job.meta['progress'] = f'İndeksleme tamamlandı: {count} dosya.'
            job.meta['doc_count'] = count
            job.save_meta()
            
        return {"status": "completed", "doc_count": count}
    except Exception as e:
        log.error(f"[Job {job.id if job else 'local'}] RAG HATA: {e}")
        if job:
            job.meta['progress'] = 'İndeksleme Hatası'
            job.meta['error'] = str(e)
            job.save_meta()
        raise e

if __name__ == '__main__':
    log.info("RQ Worker başlatılıyor... (Kuyruk listeleniyor: agent_tasks)")
    worker = Worker([task_queue], connection=redis_conn)
    worker.work()
