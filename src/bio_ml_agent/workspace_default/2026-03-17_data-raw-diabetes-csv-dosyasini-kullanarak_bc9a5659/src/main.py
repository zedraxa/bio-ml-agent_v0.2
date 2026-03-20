import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.preprocessor import DataPreprocessor, analyze_data_quality
from src.utils.model_compare import compare_models
from src.utils.visualize import MLVisualizer
from src.xai_engine import XAIEngine
import os
import joblib

def main():
    # 1. Veri Yükleme
    print("Veri yükleniyor...")
    try:
        df = pd.read_csv("data/raw/diabetes.csv")
    except FileNotFoundError:
        print("Hata: data/raw/diabetes.csv dosyası bulunamadı. Lütfen dosyanın doğru yolda olduğundan emin olun.")
        return

    # Veri setini tanıyalım
    print(df.head())
    print(df.info())

    # Hedef ve Özellik Sütunlarını Ayırma
    # 'Outcome' genellikle diyabet tahmininde hedef sütundur.
    if 'Outcome' in df.columns:
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
    else:
        print("Hata: 'Outcome' sütunu bulunamadı. Lütfen hedef sütun adını kontrol edin.")
        return

    feature_cols = X.columns.tolist()

    # 2. Veri Kalitesi Analizi
    print("\nVeri Kalitesi Analizi:")
    quality_report = analyze_data_quality(df)
    print(quality_report)
    with open("results/data_quality_report.md", "w") as f:
        f.write("# Veri Kalitesi Raporu\n\n")
        f.write(quality_report)
    print("Veri kalite raporu 'results/data_quality_report.md' dosyasına kaydedildi.")

    # 3. Eğitim ve Test Setlerine Ayırma
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nEğitim seti boyutu: {X_train_raw.shape}")
    print(f"Test seti boyutu: {X_test_raw.shape}")

    # 4. Veri Ön İşleme (DataPreprocessor ile)
    print("\nVeri ön işleme başlatılıyor...")
    preprocessor = DataPreprocessor(
        impute_strategy="median",
        scale_method="standard",
        detect_outliers="iqr", # Aykırı değer tespiti için IQR metodunu kullan
        remove_outliers=True,  # Aykırı değerleri kaldır
        pca_components=None    # PCA uygulanmayacak
    )
    
    # Fit ve Transform ile eğitim verisini işle, aykırı değerleri kaldır
    X_train_processed, y_train_processed = preprocessor.fit_transform(X_train_raw.copy(), y_train.copy())
    X_test_processed = preprocessor.transform(X_test_raw.copy())

    print("\nÖn İşleme Özeti:")
    print(preprocessor.summary_text())

    # Ensure processed data has column names for consistency
    if not isinstance(X_train_processed, pd.DataFrame):
        X_train_processed = pd.DataFrame(X_train_processed, columns=feature_cols)
    if not isinstance(X_test_processed, pd.DataFrame):
        X_test_processed = pd.DataFrame(X_test_processed, columns=feature_cols)

    # 5. Model Eğitimi ve Karşılaştırma
    print("\nModel eğitimi ve karşılaştırma başlatılıyor...")
    comparator, results, best_model_name = compare_models(
        X_train_processed, X_test_processed, y_train_processed, y_test,
        task_type="classification",
        output_dir="results/"
    )
    print(f"\nEn iyi model: {best_model_name}")
    
    # En iyi modeli yükle
    best_model_pipeline = joblib.load(os.path.join("results/", "best_model.pkl"))

    # 6. Görselleştirmeler
    print("\nGörselleştirmeler oluşturuluyor...")
    viz = MLVisualizer(output_dir="results/plots")
    
    # Class names should be inferred from y_train or explicitly passed
    class_names = [str(c) for c in sorted(y.unique())]
    
    viz.plot_all(best_model_pipeline, X_train_processed, X_test_processed, y_train_processed, y_test,
                 feature_names=feature_cols, df=X_train_processed.join(y_train_processed.rename('Outcome')), class_names=class_names)

    # 7. Açıklanabilir Yapay Zeka (XAI)
    print("\nAçıklanabilir Yapay Zeka (XAI) analizleri başlatılıyor...")
    xai = XAIEngine(best_model_pipeline, X_train_processed, feature_names=feature_cols,
                    task_type="classification", class_names=class_names)
    
    shap_top_features = xai.generate_shap_summary(X_test_processed, output_dir="results/plots")
    xai.explain_instance_lime(X_test_processed.iloc[0], output_dir="results/plots", filename="lime_instance_0.png")

    # 8. Raporlama
    print("\nRaporlama oluşturuluyor...")
    # Add XAI summary to the main report
    with open("results/comparison_report.md", "a") as f:
        f.write("\n\n## Görselleştirmeler\n")
        f.write("Aşağıdaki görselleştirmeler, modelin performansı ve veri özellikleri hakkında ek bilgiler sağlar:\n")
        f.write("- [Karışıklık Matrisi (Ham)](plots/confusion_matrix_raw.png)\n")
        f.write("- [Karışıklık Matrisi (Normalize Edilmiş)](plots/confusion_matrix_normalized.png)\n")
        f.write("- [ROC Eğrisi](plots/roc_curve.png)\n")
        f.write("- [Özellik Önemleri](plots/feature_importance.png)\n")
        f.write("- [Korelasyon Matrisi](plots/correlation_matrix.png)\n")
        f.write("- [Eğitim Sınıf Dağılımı](plots/class_distribution_train.png)\n")
        f.write("- [Test Sınıf Dağılımı](plots/class_distribution_test.png)\n")

        f.write("\n## Klinik Karar Özeti (SHAP Analizine Göre)\n")
        f.write("SHAP analizine göre, diyabet tahmini üzerinde en etkili özellikler şunlardır:\n")
        for index, row in shap_top_features.iterrows():
            f.write(f"- **{row['feature']}**: Modelin tahminlerine önemli katkı sağlayan bir özelliktir. SHAP değeri, bu özelliğin modelin çıktısını ne yönde ve ne kadar etkilediğini gösterir.\n")
        f.write("\nDaha fazla detay için [SHAP Özet Grafiği](plots/shap_summary.png) ve ilk test örneği için [LIME Açıklaması](plots/lime_instance_0.png) incelenebilir.\n")

    # README.md oluştur
    with open("README.md", "w") as f:
        f.write("# Diyabet Tahmin Projesi\n\n")
        f.write("Bu proje, `data/raw/diabetes.csv` dosyasındaki verileri kullanarak diyabeti tahmin etmek için bir makine öğrenimi modeli oluşturur, eğitir ve değerlendirir.\n\n")
        f.write("## Kurulum\n\n")
        f.write("1. Gerekli kütüphaneleri yükleyin:\n")
        f.write("```bash\n")
        f.write("pip install -r requirements.txt\n")
        f.write("```\n\n")
        f.write("2. `data/raw/diabetes.csv` dosyasının projenin `data/raw/` dizini altında bulunduğundan emin olun.\n\n")
        f.write("## Projeyi Çalıştırma\n\n")
        f.write("Ana scripti çalıştırın:\n")
        f.write("```bash\n")
        f.write("python src/main.py\n")
        f.write("```\n\n")
        f.write("Bu komut:\n")
        f.write("- Veriyi yükler ve ön işler.\n")
        f.write("- Birden fazla sınıflandırma modelini eğitir ve karşılaştırır.\n")
        f.write("- En iyi modeli `results/best_model.pkl` olarak kaydeder.\n")
        f.write("- Çeşitli görselleştirmeler oluşturur ve `results/plots/` dizinine kaydeder.\n")
        f.write("- SHAP ve LIME kullanarak model kararlarını açıklar.\n")
        f.write("- Kapsamlı bir raporu `results/comparison_report.md` dosyasına kaydeder.\n\n")
        f.write("## Sonuçlar\n\n")
        f.write("Eğitim ve değerlendirme sonuçları `results/comparison_report.md` dosyasında bulunabilir. Görselleştirmeler `results/plots/` dizinindedir.\n")
        f.write("En iyi model, `results/best_model.pkl` dosyasında kaydedilmiştir ve `src/utils/model_compare.py` içinde belirtilen kullanım talimatlarına göre yüklenebilir.\n")

    print("Proje başarıyla tamamlandı. Raporlar ve görseller 'results/' klasöründe, README ise ana dizinde.")

if __name__ == "__main__":
    main()