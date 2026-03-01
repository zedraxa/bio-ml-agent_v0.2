"""
diabetes_xai_demo.py
--------------------
Bu betik, uçtan uca bir makine öğrenmesi sürecini işletir ve ardından 
otomatik olarak SHAP/LIME (XAI) süreçlerini test eder. 

Çalıştırma:
python scripts/demos/diabetes_xai_demo.py
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Proje kök dizinini ekle
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xai_engine import XAIEngine

def main():
    print("=== XAI Demo (Diabetes Dataset) Başlatılıyor ===")
    
    # 1. Veri Yükleme
    data_path = Path("data/raw/diabetes.csv")
    if not data_path.exists():
        print(f"Hata: {data_path} bulunamadı. Lütfen önce veri setini indirin.")
        print("Örn: <PYTHON> komutu ile veri indirebilirsiniz.")
        return

    df = pd.read_csv(data_path)
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    feature_names = X.columns.tolist()

    # Eğitim/Test Ayrımı
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Model Eğitimi (Black Box Model)
    print("Model eğitiliyor (Random Forest)...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"Model Accuracy: {acc:.4f}")

    # 3. YENİ ÖZELLİK: Otomatik XAI Entegrasyonu Testi
    print("\nSHAP ve LIME (XAI) süreçleri başlatılıyor...")
    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    xai = XAIEngine(model, X_train, feature_names=feature_names, task_type="classification")
    
    # SHAP
    try:
        xai.generate_shap_summary(X_test, output_dir=str(output_dir), max_display=10)
        print("✔ SHAP grafiği oluşturuldu: results/plots/shap_summary_plot.png")
    except Exception as e:
        print(f"❌ SHAP Hatası: {str(e)}")

    # LIME
    try:
        # Test setinden ilk hastayı al
        instance = X_test.iloc[0]
        xai.explain_instance_lime(instance, output_dir=str(output_dir))
        print("✔ LIME raporu oluşturuldu: results/plots/lime_explanation.html")
    except Exception as e:
        print(f"❌ LIME Hatası: {str(e)}")

    print("\n=== XAI Demo Tamamlandı ===")

if __name__ == "__main__":
    main()
