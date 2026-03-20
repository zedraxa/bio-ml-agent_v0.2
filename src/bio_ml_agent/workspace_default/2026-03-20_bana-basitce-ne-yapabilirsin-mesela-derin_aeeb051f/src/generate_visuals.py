import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from utils.visualize import MLVisualizer

def main():
    print("--- Görselleştirme Süreci Başladı ---")

    # 1. Veri setini yükle
    print("Veri yükleniyor...")
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

    # 2. Veriyi ön işle
    print("Veri ön işleniyor...")
    # 'diagnosis' sütununu sayısal formata dönüştür (M=1, B=0)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    
    # Özellik (X) ve hedef (y) değişkenlerini ayır
    feature_cols = [col for col in column_names if col not in ['id', 'diagnosis']]
    X = df[feature_cols]
    y = df['diagnosis']

    # Veriyi eğitim ve test setlerine ayır (train.py ile aynı random_state kullanarak tutarlılığı sağla)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. En iyi modeli yükle
    model_path = 'results/best_model.pkl'
    print(f"En iyi model yükleniyor: {model_path}")
    best_model_pipeline = joblib.load(model_path)

    # 4. Görselleştiriciyi başlat ve tüm grafikleri oluştur
    viz = MLVisualizer(output_dir="results/plots")
    
    # Ham DataFrame'i de (orijinal 'diagnosis' sütunu ile) gönderiyoruz
    # Class distribution ve correlation matrix gibi grafikler için gereklidir.
    df_for_plots = df.copy()
    df_for_plots['diagnosis'] = y # Sayısallaştırılmış y'yi atıyoruz
    
    viz.plot_all(
        model=best_model_pipeline,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_cols,
        df=df_for_plots,
        class_names=['Benign (B)', 'Malignant (M)']
    )

    print("\n--- Görselleştirme Süreci Başarıyla Tamamlandı ---")

if __name__ == '__main__':
    main()
