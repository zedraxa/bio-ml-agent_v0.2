import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from utils.xai_engine import XAIEngine

def main():
    print("--- Açıklanabilir Yapay Zeka (XAI) Analizi Başladı ---")

    # 1. Veri setini yükle ve ön işle
    print("Veri yükleniyor ve ön işleniyor...")
    file_path = 'data/raw/wdbc.data'
    column_names = [
        'id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
        'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
        'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
        'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
        'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]
    df = pd.read_csv(file_path, header=None, names=column_names)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    
    feature_cols = [col for col in column_names if col not in ['id', 'diagnosis']]
    X = df[feature_cols]
    y = df['diagnosis']

    # Veriyi eğitim ve test setlerine ayır (eğitim ile tutarlı olması için)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 2. En iyi modeli yükle
    model_path = 'results/best_model.pkl'
    print(f"En iyi model yükleniyor: {model_path}")
    best_model_pipeline = joblib.load(model_path)

    # 3. XAI Motorunu Başlat
    # LIME ve SHAP'ın arka plan verisi olarak ölçeklenmemiş (orijinal) veriyi kullanması daha doğrudur.
    # Pipeline, tahmin sırasında veriyi kendisi ölçekleyecektir.
    xai_engine = XAIEngine(
        model=best_model_pipeline,
        training_data=X_train,
        feature_names=feature_cols,
        class_names=['Benign (B)', 'Malignant (M)']
    )

    # 4. SHAP Özet Grafiği Oluştur
    # Test setinin bir alt kümesi üzerinde SHAP analizi yapmak hesaplama süresini kısaltabilir.
    xai_engine.generate_shap_summary(X_test, output_dir="results/plots", max_display=15)

    # 5. Tek bir örnek için LIME Raporu Oluştur
    # Açıklamak için test setinden ilk örneği seçelim.
    instance_to_explain = X_test.iloc[[0]]
    print(f"\nTest setinden ilk örnek LIME ile açıklanıyor...")
    xai_engine.explain_instance_lime(instance_to_explain.squeeze(), output_dir="results/plots", num_features=10)

    print("\n--- XAI Analizi Başarıyla Tamamlandı ---")

if __name__ == '__main__':
    main()
