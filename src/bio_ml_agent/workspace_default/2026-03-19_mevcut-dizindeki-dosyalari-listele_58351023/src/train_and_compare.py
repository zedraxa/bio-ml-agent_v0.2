import pandas as pd
import os
from sklearn.model_selection import train_test_split
from utils.model_compare import compare_models

def main():
    """
    Loads the processed data, splits it, runs a multi-model comparison,
    and saves the results.
    """
    # Veri setini yükle
    data_path = 'data/processed/processed_heart.csv'
    df = pd.read_csv(data_path)

    # Özellikler (X) ve hedef (y) olarak ayır
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Veriyi eğitim ve test setlerine ayır
    # stratify=y, eğitim ve test setlerindeki hedef sınıf dağılımını korur.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Veri başarıyla yüklendi ve bölündü.")
    print(f"Eğitim seti boyutu: {X_train.shape}")
    print(f"Test seti boyutu: {X_test.shape}")

    # Model karşılaştırmasını çalıştır
    # Bu fonksiyon modelleri (LR, RF, GB, SVC, KNN) eğitir, değerlendirir,
    # karşılaştırma raporu ve en iyi modeli kaydeder.
    output_dir = "results/"
    
    # compare_models fonksiyonu StandardScaler içeren bir pipeline kullanır.
    comparator, results_df = compare_models(
        X_train, X_test, y_train, y_test,
        task_type="classification",
        output_dir=output_dir
    )

    print("\n--- Model Karşılaştırma Sonuçları ---")
    print(results_df.to_markdown(index=False))

    print(f"\nKarşılaştırma raporu '{output_dir}comparison_report.md' dosyasına kaydedildi.")
    print(f"JSON sonuçları '{output_dir}comparison_results.json' dosyasına kaydedildi.")
    print(f"En iyi model '{output_dir}best_model.pkl' olarak kaydedildi.")

if __name__ == "__main__":
    main()