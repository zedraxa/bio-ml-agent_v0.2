import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from utils.visualize import MLVisualizer

def main():
    """
    Loads the best model, evaluates it on the test set, and generates
    a comprehensive set of visualizations.
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
    
    # Veriyi eğitim ve test setlerine ayır (aynı random_state ile)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Görselleştirme sınıfını başlat
    visualizer = MLVisualizer(output_dir="results/plots")

    # Tüm grafikleri oluştur ve kaydet
    visualizer.plot_all(
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        df=df # Korelasyon matrisi için tüm dataframe
    )

if __name__ == "__main__":
    main()