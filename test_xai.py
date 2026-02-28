import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xai_engine import XAIEngine

def test_xai():
    print("🚀 XAI Motoru Testi Başlatılıyor...")
    
    # 1. Dummy Veri Üret
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
    feature_names = ["Age", "Blood_Pressure", "Cholesterol", "BMI", "Glucose"]
    class_names = ["Healthy", "Sick"]
    
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # 2. Dummy Model Eğit
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_df, y)
    
    # 3. XAI Motorunu Başlat
    xai = XAIEngine(model, X_df, feature_names=feature_names, class_names=class_names)
    
    out_dir = Path("results/test_xai")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 4. SHAP Plot Üret
    print("⏳ SHAP Summary Plot Üretiliyor...")
    X_sample = X_df.iloc[:20]
    shap_res = xai.generate_shap_summary(X_sample, output_dir=str(out_dir))
    print(f"✅ SHAP Sonucu: {shap_res}")
    
    # 5. LIME Raporu Üret
    print("⏳ LIME HTML Raporu Üretiliyor...")
    lime_res = xai.explain_instance_lime(X_sample.iloc[0], output_dir=str(out_dir))
    if "error" in lime_res:
        print(f"❌ LIME Hatası: {lime_res['error']}")
    else:
        print(f"✅ LIME Sonucu: HTML Kaydedildi -> {lime_res.get('html_path')}")
        
    print("🎯 Test Tamamlandı!")

if __name__ == "__main__":
    test_xai()
