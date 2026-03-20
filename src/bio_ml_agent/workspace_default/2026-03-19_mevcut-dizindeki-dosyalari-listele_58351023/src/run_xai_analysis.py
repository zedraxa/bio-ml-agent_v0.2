import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xai_engine import XAIEngine

def main():
    """
    Loads the best model and runs SHAP analysis to generate
    feature importance and summary plots.
    """
    # Kaydedilmiş en iyi modeli yükle
    model_path = 'results/best_model.pkl'
    best_model = joblib.load(model_path)
    print(f"En iyi model başarıyla yüklendi: {best_model.named_steps['model'].__class__.__name__}")

    # İşlenmiş veriyi yükle
    data_path = 'data/processed/processed_heart.csv'
    df = pd.read_csv(data_path)
    
    # Özellikler (X) ve hedef (y) olarak ayır
    X = df.drop('target', axis=1)
    y = df['target']
    feature_names = X.columns.tolist()
    
    # Veriyi eğitim ve test setlerine ayır (XAI engine'e eğitim verisi gerekir)
    X_train, X_test, _, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # XAI motorunu başlat
    xai = XAIEngine(
        model=best_model,
        training_data=X_train,
        feature_names=feature_names,
        task_type="classification"
    )

    # Test seti üzerinde SHAP özet grafiklerini oluştur
    xai.generate_shap_summary(
        data_to_explain=X_test,
        output_dir="results/plots",
        max_display=13 # Toplam 13 özellik var
    )

if __name__ == "__main__":
    main()