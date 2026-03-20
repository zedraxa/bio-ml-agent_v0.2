import os
import sys
import logging
import argparse

# PATH ayarı: Ana dizini ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bio_ml_agent.ultra_agent.memory.evaluator import MemoryEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger("memory_evaluation")

def run_evaluation(query: str, mock_memories: bool = False):
    log.info("Semantik Hafıza Değerlendirme Betiği Başlatılıyor...")
    
    evaluator = MemoryEvaluator(model_name="gemini-2.0-flash")
    
    if mock_memories:
        # Test amaçlı yapay anılar
        memories = [
            {
                "content": "Python'da Qdrant kullanarak semantik arama yapmak için qdrant_client kütüphanesi kullanılır.",
                "summary": "Qdrant Python Client kullanımı"
            },
            {
                "content": "Kullanıcı kahvesini şekersiz içmeyi seviyor.",
                "summary": "Kullanıcı tercihi"
            },
            {
                "content": "Vektör veritabanlarında cos-sim, cosine similarity anlamına gelir ve -1 ile 1 arasında değer alır.",
                "summary": "Cosine similarity açıklaması"
            }
        ]
        log.info(f"Yapay (Mock) anılar üzerinden değerlendirme yapılıyor. Sorgu: '{query}'")
    else:
        # Gerçek Qdrant bağlantısı
        from bio_ml_agent.ultra_agent.memory.qdrant_store import QdrantMemoryStore
        store = QdrantMemoryStore()
        if not store.enabled:
            log.error("Qdrant Store aktif değil. Değerlendirme yapılamaz.")
            return
            
        memories = store.search_memory(query, limit=5, project_filter="test-project")
        log.info(f"Qdrant'tan {len(memories)} anı getirildi. Sorgu: '{query}'")

    results = evaluator.evaluate_recall(query, memories)
    
    print("\n" + "="*50)
    print(" DEĞERLENDİRME SONUÇLARI (LLM Hakemliği)")
    print("="*50)
    print(f"Toplam Değerlendirilen Anı : {results.get('total_evaluated', len(memories))}")
    print(f"Hit Rate (En az 1 yararlı): {results.get('hit_rate', 0)}")
    print(f"Precision@k (Doğruluk)  : {results.get('precision@k', 0)}")
    print(f"Pollution (Kirlilik Oranı): {results.get('pollution_rate', 0)}")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantik Hafıza Değerlendirme Betiği (Faz 9)")
    parser.add_argument("--query", type=str, required=True, help="Değerlendirilecek test sorgusu")
    parser.add_argument("--mock", action="store_true", help="Gerçek veritabanı yerine sahte anılarla test et")
    
    args = parser.parse_args()
    run_evaluation(query=args.query, mock_memories=args.mock)
