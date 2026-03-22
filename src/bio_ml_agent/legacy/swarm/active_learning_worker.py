"""
active_learning_worker.py
-------------------------
Arkaplanda çalışan sürekli öğrenme (Continuous Learning) servisi.
Redis Streams (veya DB) dinleyicisi üzerinden yeni veriler geldiğinde tetiklenir, 
modeli (random forest, log-reg vb.) eski veri + yeni veri ile retrain eder, 
MLFlow üzerinde skoru eskisinden iyiyse Production'a alır.
"""

import sys
import time
import os
from pathlib import Path
import pandas as pd
import logging

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_streams.db_connector import DBConnector
from data_streams.kafka_redis_consumer import StreamConsumer
from utils.model_compare import compare_models, deploy_if_better_than_production
from utils.preprocessor import quick_preprocess

log = logging.getLogger("bio_ml_agent")

class ActiveLearningWorker:
    def __init__(self, db_url=None, stream_name="sensor_stream"):
        self.db = DBConnector(db_url)
        self.stream = StreamConsumer(stream_name, "active_learning_group", "al_worker_1")
        
        # Gerçek eğitim verisi (Mock senaryo için base datayı yüklüyoruz)
        self.base_data_path = Path("workspace/data/raw/diabetes.csv")
        self.target_col = "Outcome"
        self.model_registry_name = "Diabetes_Prediction_Model"

    def process_new_batch(self, batch_messages: list[dict]):
        """Stream'den gelen yeni IoT/Sensör verilerini alıp retrain'i tetikler"""
        log.info(f"🔄 Active Learning Tetiklendi! Gelen yeni vaka sayısı: {len(batch_messages)}")
        
        # 1. Base datayı ve yeni datayı birleştir
        if not self.base_data_path.exists():
            log.error("Ana eğitim verisi bulunamadı. Retrain atlanıyor.")
            return

        df_base = pd.read_csv(self.base_data_path)
        df_new = pd.DataFrame(batch_messages)
        
        # Stream'den sayısal gelmeyen verileri cast edelim (Redis datayı string tutar)
        for col in df_new.columns:
            try:
                df_new[col] = pd.to_numeric(df_new[col])
            except:
                pass

        # İki veriyi alt alta birleştir (Gelecekte eski verilerin belli bir kısmı silinebilir - Sliding Window)
        df_combined = pd.concat([df_base, df_new], ignore_index=True).dropna(subset=[self.target_col])
        log.info(f"📊 Yeni eğitim veri seti boyutu: {len(df_combined)} satır")

        # 2. Veri Hazırlığı
        X = df_combined.drop(columns=[self.target_col])
        y = df_combined[self.target_col]
        X_clean = quick_preprocess(X.values, scale=True, pca=0)
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_clean, y.values, test_size=0.2, random_state=42)

        # 3. Model Eğitimi
        log.info("🤖 Yeni veriyle modeller eğitiliyor...")
        comparator, results = compare_models(
            X_train, X_test, y_train, y_test,
            task_type="classification",
            output_dir="workspace/results/active_learning",
            cv_folds=3 # Hız için
        )

        best_model_result = comparator.results[0]
        
        # 4. Deployment Check (MLFlow Üzerinden Kıyaslama)
        success = deploy_if_better_than_production(
            new_model_result=best_model_result,
            new_model_path=Path("workspace/results/active_learning/best_model.pkl"),
            model_name_for_registry=self.model_registry_name,
            primary_metric="accuracy"
        )
        
        if success:
            log.info("🚀 Yeni model canlıya alındı. Base dataset güncelleniyor...")
            # Yeni model Production'a geçtiyse, yeni verileri Base setine kalıcı olarak kaydet
            df_combined.to_csv(self.base_data_path, index=False)
        else:
            log.info("ℹ️ Mevcut production model hala daha iyi performans gösteriyor.")

    def start_listening(self, batch_size=10):
        log.info("⏱️ Active Learning Worker başlatıldı. Dinleniyor...")
        # Stream'i bloklayarak dinle (Batch yakalandığında callback çalışır)
        self.stream.listen(batch_size=batch_size, callback=self.process_new_batch, poll_timeout_ms=5000)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    worker = ActiveLearningWorker()
    worker.start_listening(batch_size=5) # Demo için düşük tutuldu
