"""
Bio-ML Agent Demo: Breast Cancer Classification with MLflow Tracking.

Bu script, ajanın veri seti kataloğunu kullanarak bir model eğitmesini,
veri versiyonlamasını yapmasını ve sonuçları MLflow'a kaydetmesini gösterir.
"""

import os
import sys
from pathlib import Path

# Proje kök dizinini ekle
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

from dataset_catalog import load_dataset, get_dataset_version
from mlflow_tracker import get_shared_tracker
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def run_demo():
    print("🚀 Bio-ML Agent: Breast Cancer Demo Başlatılıyor...")
    
    # 1. Veri Setini Yükle
    print("📦 Veri seti kataloğundan 'breast_cancer' yükleniyor...")
    X, y, target_names = load_dataset("breast_cancer")
    dataset_hash = get_dataset_version("breast_cancer", workspace=root_dir)
    
    # 2. MLflow Tracker Başlat
    tracker = get_shared_tracker()
    
    with tracker.start_run(run_name="Breast_Cancer_Demo_Run"):
        print(f"📊 MLflow Run Aktif: {tracker._active_run if tracker._using_mlflow else 'JSON Fallback'}")
        
        # 3. Parametreleri Logla
        params = {
            "n_estimators": 100,
            "max_depth": 5,
            "random_state": 42,
            "test_size": 0.2
        }
        tracker.log_params(params)
        tracker.set_tag("dataset.hash", dataset_hash)
        tracker.set_tag("demo_mode", "true")
        
        # 4. Veriyi Böl
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=params["test_size"], random_state=params["random_state"]
        )
        
        # 5. Model Eğit
        print("🧠 Model eğitiliyor (RandomForest)...")
        clf = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=params["random_state"]
        )
        clf.fit(X_train, y_train)
        
        # 6. Tahmin ve Metrikler
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"✅ Eğitim Tamamlandı. Doğruluk (Accuracy): {acc:.4f}")
        
        tracker.log_metrics({"accuracy": acc})
        
        # 7. Modeli Kaydet (Opsiyonel)
        tracker.log_model(clf, "random_forest_model")
        
        print("\n📝 Sınıflandırma Raporu:")
        print(classification_report(y_test, y_pred, target_names=target_names))

    print("\n🏁 Demo Başarıyla Tamamlandı.")
    if not tracker._using_mlflow:
        print(f"💡 Not: MLflow kurulu olmadığı için sonuçlar şuraya kaydedildi: {tracker.fallback_dir}")

if __name__ == "__main__":
    run_demo()
