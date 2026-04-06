import os
import sys
import uuid
import json
import logging
from redis import Redis
from rq import Queue, Worker, get_current_job
from datetime import datetime
from pathlib import Path

# Platform Veri Modelleri (Faz 5A & 5B)
from bio_ml_agent.models.remote_gateway import StreamEvent, EventStreamType
from bio_ml_agent.models.cloud_offload import CheckpointResumeStrategy

from bio_ml_agent.services.agent_service import AgentService
from swarm.orchestrator import SwarmOrchestrator
from bio_ml_agent.core.config import AgentConfig
from bio_ml_agent.utils.config import load_config
# Legacy RAG siliniyor, yeni tool tabanlı sisteme geçiliyor.
# from legacy.rag_engine import RAGEngine
from bio_ml_agent.core.tools import index_workspace

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
        # Checkpoint/Resume Policy (Bulut/Kapanma anı toleransları için)
        checkpoint_policy = CheckpointResumeStrategy()

        service = AgentService(model=model, timeout=timeout, max_steps=max_steps)

        # ── Session Hydration (Eğer checkpoint varsa oradan devam et) ──
        if job and job.meta.get("messages_checkpoint"):
            service.set_session(
                session_id=session_id,
                messages=job.meta.get("messages_checkpoint"),
                metadata=job.meta.get("session_metadata")
            )
            log.info(f"[Job {job.id}] Ajan oturumu {session_id} checkpoint'ten beslendi (Hydrated).")
        else:
            service.reset_session()
            service.session_id = session_id

        final_answer = ""

        log.info(f"[Job {job.id if job else 'local'}] Ajan servisi {session_id} için başlatıldı.")

        for event in service.process_message(prompt):
            ev_type = event.get("type")
            status_text = event.get("content", "")

            structured_event = None
            if ev_type == "status":
                log.info(f"Durum: {status_text}")
                structured_event = StreamEvent(
                    event_id=uuid.uuid4().hex[:8],
                    run_id=job.id if job else "local-run",
                    event_type=EventStreamType.AGENT_THINKING,
                    payload={"message": status_text},
                    timestamp=datetime.now().isoformat()
                )
            elif ev_type == "tool_start":
                tool_name = event.get("tool")
                log.info(f"Araç {tool_name} tetiklendi...")
                structured_event = StreamEvent(
                    event_id=uuid.uuid4().hex[:8],
                    run_id=job.id if job else "local-run",
                    event_type=EventStreamType.TOOL_START,
                    payload={"tool_name": tool_name, "message": f"Araç devrede: {tool_name}"},
                    timestamp=datetime.now().isoformat()
                )

            # ── Heartbeat & Structured Event Update ──
            if job and structured_event:
                job.meta['last_event'] = structured_event.model_dump()
                job.meta['progress'] = structured_event.payload.get("message", "")
                job.meta['last_heartbeat'] = datetime.now().isoformat()

                # Olası kesintilere karşı aralıklı Checkpoint Save (Context Kaybını Önle)
                if service.messages and len(service.messages) % checkpoint_policy.snapshot_interval_steps == 0:
                     job.meta['messages_checkpoint'] = service.messages
                     job.meta['session_metadata'] = service.session_metadata

                job.save_meta()

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

def execute_swarm_job(session_id: str, payload_data: dict, model_override: str = None):
    """
    Webhook'tan gelen klinik veriyi diske kaydedip, Bio-ML Swarm Pipeline üzerinden
    analiz eden asenkron Arka Plan Worker'ı.
    """
    job = get_current_job()
    if job:
        job.meta['progress'] = 'Swarm Pipeline Başlatılıyor...'
        job.save_meta()

    try:
        # 1. Veriyi çalışma dizinine (workspace) JSON olarak kaydet
        cfg = load_config()
        workspace = Path(cfg.workspace.base_dir).expanduser().resolve()
        data_dir = workspace / "data" / "webhook_inbox"
        data_dir.mkdir(parents=True, exist_ok=True)

        file_path = data_dir / f"clinical_data_{session_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload_data, f, indent=4, ensure_ascii=False)

        # 2. Config ve Orkestratör Hazırlığı
        if model_override:
            cfg.agent.model = model_override

        orchestrator = SwarmOrchestrator(cfg)
        log.info(
            "[Job %s] Swarm Orkestratör başlatıldı (Session: %s). Dosya: %s",
            job.id if job else 'local', session_id, file_path
        )

        if job:
            job.meta['progress'] = 'Veri Sisteme Alındı. Ajanlar (Data/ML/Bio) veri analizi yapıyor...'
            job.save_meta()

        # 3. Yapay zeka sistemini trigger'la
        # "tümör", "kanser" vb pipeline'ı zorlamak için "kanser pipeline" kelimeleri eklendi
        task_prompt = (
            f"Şu yoldaki JSON verisini oku: {file_path}. "
            "Bu veriyi temizle, model kur, ve kanser pipeline analizinden geçirip biyolojik sonuç çıkar."
        )
        messages = [{"role": "user", "content": task_prompt}]

        final_report = orchestrator.process(messages)

        # 4. Başarılı Bitiş
        if job:
            job.meta['progress'] = 'Pipeline Tamamlandı - Klinik Rapor Hazır.'
            job.meta['result_summary'] = final_report
            job.meta['updated_at'] = datetime.now().isoformat()
            job.save_meta()

        return {"status": "completed", "result": final_report}

    except Exception as e:
        log.error(f"[Job {job.id if job else 'local'}] SWARM HATA: {e}")
        if job:
            job.meta['progress'] = 'Swarm Pipeline Hatası'
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
        log.info(
            "[Job %s] RAG İndekslemesi başlatıldı (Yeni Tool tabanlı): %s",
            job.id if job else 'local', workspace_path
        )

        # Yeni index_workspace tool'unu kullan (payload boş ise tüm workspace'i tarar)
        result_msg = index_workspace("", workspace_path)

        if job:
            job.meta['progress'] = f'İndeksleme bitti: {result_msg}'
            job.meta['result_summary'] = result_msg
            job.save_meta()

        return {"status": "completed", "message": result_msg}
    except Exception as e:
        log.error(f"[Job {job.id if job else 'local'}] RAG HATA: {e}")
        if job:
            job.meta['progress'] = 'İndeksleme Hatası'
            job.meta['error'] = str(e)
            job.save_meta()
        raise e

def main():
    log.info("RQ Worker başlatılıyor... (Kuyruk listeleniyor: agent_tasks)")
    worker = Worker([task_queue], connection=redis_conn)
    worker.work()

if __name__ == '__main__':
    main()
