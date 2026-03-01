"""
Bio-ML Agent Demo: EEG Signal Classification (Simulated).

Bu script, biyomühendislik odaklı sinyal işleme ve sınıflandırma senaryosunu gösterir.
Sentetik EEG verisi üreterek "Nöbet (Seizure)" tespiti yapar.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Proje kök dizinini ekle
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

from mlflow_tracker import get_shared_tracker
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

def generate_synthetic_eeg(n_samples=200):
    """Sentetik EEG sinyalleri üretir (Frekansta basit farklar)."""
    t = np.linspace(0, 1, 100) # 1 saniyelik sinyal, 100 örnek
    
    X = []
    y = []
    
    for i in range(n_samples):
        if i % 2 == 0:
            # Normal: Düşük frekanslı alfa/beta dalgaları
            sig = np.sin(2 * np.pi * 10 * t) + np.random.normal(0, 0.5, 100)
            target = 0 # Normal
        else:
            # Nöbet: Yüksek frekanslı, yüksek genlikli "spikes"
            sig = 2 * np.sin(2 * np.pi * 30 * t) + np.random.normal(0, 0.8, 100)
            target = 1 # Seizure
        
        X.append(sig)
        y.append(target)
        
    return np.array(X), np.array(y)

def run_demo():
    print("🚀 Bio-ML Agent: EEG Signal Demo Başlatılıyor...")
    
    # 1. Veri Üret
    X, y = generate_synthetic_eeg()
    print(f"📊 {len(X)} EEG örneği üretildi (100 özellik/zaman-adımı).")
    
    # 2. Tracker Başlat
    tracker = get_shared_tracker()
    
    with tracker.start_run(run_name="EEG_Seizure_Detection"):
        # 3. Parametreler
        params = {"kernel": "rbf", "C": 1.0, "gamma": "scale"}
        tracker.log_params(params)
        tracker.set_tag("bio_task", "eeg_classification")
        
        # 4. Veri Böl
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        
        # 5. Model Eğit
        print("🧠 SVM Modeli eğitiliyor (Seizure Detection)...")
        clf = SVC(**params)
        clf.fit(X_train, y_train)
        
        # 6. Sonuçlar
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"✅ Başarı Oranı: {acc:.4f}")
        
        tracker.log_metrics({"accuracy": acc})
        
        cm = confusion_matrix(y_test, y_pred)
        print("\n📈 Karmaşıklık Matrisi (Seizure Detection):")
        print(cm)

    print("\n🏁 EEG Demo Tamamlandı.")

if __name__ == "__main__":
    run_demo()
