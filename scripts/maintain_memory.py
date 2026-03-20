import os
import sys
import logging
import argparse

# PATH ayarı: Ana dizini ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bio_ml_agent.ultra_agent.memory.qdrant_store import QdrantMemoryStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger("memory_maintenance")

def run_maintenance(host="localhost", port=6333):
    log.info("Semantik Hafıza Bakım Görevi Başlatılıyor...")
    try:
        store = QdrantMemoryStore(host=host, port=port)
        if not store.enabled:
            log.error("Qdrant Store aktif değil. Bakım yapılamaz.")
            return
            
        store.maintenance()
        log.info("Hafıza bakımı başarıyla tamamlandı.")
    except Exception as e:
        log.error(f"Bakım sırasında hata oluştu: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantik Hafıza Bakım Betiği")
    parser.add_argument("--host", default="localhost", help="Qdrant host")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant port")
    
    args = parser.parse_args()
    run_maintenance(host=args.host, port=args.port)
