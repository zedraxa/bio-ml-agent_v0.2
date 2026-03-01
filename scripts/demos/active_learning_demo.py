"""
active_learning_demo.py
-----------------------
Sprint 11 (Continuous Learning ve Streaming) özelliklerini test eden demo script.
Amaç:
1. Redis Stream'ine sahte (IoT/Sensör karakterli) hasta verileri basar (Producer).
2. Aynı anda arkaplanda çalışan `ActiveLearningWorker` bu verileri yakalayıp 
   eski verilerle birleştirir.
3. Model baştan eğitilir ve eski production modeliyle kıyaslanır.
"""

import sys
import time
import threading
import random
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from data_streams.kafka_redis_consumer import StreamConsumer
from swarm.active_learning_worker import ActiveLearningWorker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("bio_ml_agent")

def run_background_worker():
    """Tüketici (Consumer) - Gerçekte bu Docker içinde ayrı bir container veya daemon process olarak yaşar."""
    worker = ActiveLearningWorker(stream_name="demo_sensor_stream")
    # batch_size=5 diyoruz ki, 5 yeni hasta geldiği an modeli yeniden eğitmeye başlasın
    worker.start_listening(batch_size=5)

def simulate_iot_sensors():
    """Üretici (Producer) - Sensörlerden/Hastanelerden saniye saniye akan veriyi simüle eder."""
    producer = StreamConsumer("demo_sensor_stream", "active_learning_group", "test_producer")
    
    log.info("💉 Sensör Simülasyonu Başladı: Her saniye 1 yeni hasta verisi akıyor...")
    
    # Şeker hastası (Outcome=1) özelliklerine sahip 5 yapay veri basacağız
    # Böylece yeni veri setinin bias'ı değişecek ve retrain tetiklenecek
    for i in range(1, 6):
        mock_data = {
            "Pregnancies": random.randint(0, 5),
            "Glucose": random.randint(150, 200), # Yüksek şeker
            "BloodPressure": random.randint(70, 90),
            "SkinThickness": random.randint(20, 40),
            "Insulin": random.randint(100, 300),
            "BMI": random.uniform(28.0, 40.0), # Yüksek BMI
            "DiabetesPedigreeFunction": random.uniform(0.5, 1.5),
            "Age": random.randint(40, 70),
            "Outcome": 1 # Kesin diyabetli profili veriyoruz ki model öğrensin
        }
        
        producer.publish_test_message(mock_data)
        time.sleep(1) # Sensör gecikmesi simülasyonu
        
    log.info("🏁 5 adet IoT veri akışı tamamlandı. Sensör sustu.")

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 BIO-ML AGENT: ACTIVE LEARNING & STREAMING DEMO")
    print("="*50 + "\n")
    
    # 1. Arkaplan worker'ını Thread olarak ayağa kaldır
    worker_thread = threading.Thread(target=run_background_worker, daemon=True)
    worker_thread.start()
    
    # Worker'ın grubunu kurması için ufak bir bekleme
    time.sleep(2)
    
    # 2. Sensör verilerini basmaya başla
    simulate_iot_sensors()
    
    # 3. Worker'ın (Thread) modeli retrain etmesini bekleyelim 
    # (Kafka/Redis'ten 5 veriyi çekip MLFlow'a bağlanması vs 10-15 saniye sürer)
    log.info("⏳ Ana process, Arkaplan Active Learning Eğitiminin bitmesini bekliyor... (20sn)")
    time.sleep(20)
    
    print("\n✅ Demo senaryosu tamamlandı. Terminal Loglarını inceleyerek yeni modelin 'Production' olup olmadığına bakabilirsiniz.")
