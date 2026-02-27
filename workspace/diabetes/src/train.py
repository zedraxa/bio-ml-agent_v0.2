import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os

# Proje kök dizinini Python yoluna ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_catalog import load_dataset
from utils.model_compare import compare_models
from utils.visualize import MLVisualizer


def main():
    print("Meme Kanseri Veri Seti Sınıflandırma Projesi Başlıyor...")

    # 1. Veri setini yükle
    print("Veri seti yükleniyor: breast_cancer")
    X, y, feature_names = load_dataset('breast_cancer')
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y

    print(f"Veri seti yüklendi. Özellik sayısı: {X.shape[1]}, Örnek sayısı: {X.shape[0]}")
    print("Hedef değişken sınıfları (0: Malignant, 1: Benign):", pd.Series(y).value_counts())

    # Hedef sınıf isimlerini tanımla
    class_names = ["Malignant", "Benign"]

    # 2. Veriyi eğitim ve test setlerine ayır
    print("Veri eğitim ve test setlerine ayrılıyor (test_size=0.2, random_state=42).")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Eğitim seti boyutu: {X_train.shape[0]} örnek")
    print(f"Test seti boyutu: {X_test.shape[0]} örnek")

    # X_train, X_test'i DataFrame'e dönüştürerek feature_names'i koru
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    y_train_series = pd.Series(y_train, name='target')
    y_test_series = pd.Series(y_test, name='target')


    # 3. Modelleri karşılaştır
    print("\n--- Modeller Karşılaştırılıyor ---")
    best_model_pipeline, results_df = compare_models(X_train_df, X_test_df, y_train_series, y_test_series,
                                                     task_type="classification", output_dir="results/")

    print("\nModel Karşılaştırma Sonuçları:")
    print(results_df.to_markdown(index=False))

    if best_model_pipeline:
        print(f"\nEn İyi Model (Test ROC AUC'ye Göre): {best_model_pipeline.named_steps['model'].__class__.__name__}")
    else:
        print("\nHiçbir model eğitilemedi veya en iyi model belirlenemedi.")
        return

    # 4. Görselleştirmeleri oluştur
    if best_model_pipeline:
        viz = MLVisualizer(output_dir="results/plots")
        viz.plot_all(best_model_pipeline, X_train_df, X_test_df, y_train_series, y_test_series,
                     feature_names=feature_names, df=df, class_names=class_names)
    else:
        print("En iyi model bulunamadığı için görselleştirmeler oluşturulamadı.")

    print("\nMeme Kanseri Sınıflandırma Projesi Tamamlandı.")

if __name__ == "__main__":
    main()