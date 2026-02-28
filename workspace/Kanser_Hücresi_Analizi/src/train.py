import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import sys

# utils modüllerini sys.path'e ekleyin
sys.path.append(os.path.abspath('scratch_project/utils'))

from model_compare import compare_models
from visualize import MLVisualizer
from preprocessor import DataPreprocessor, analyze_data_quality
from hyperparameter_optimizer import optimize_model

# Veri yolu
DATA_PATH = 'scratch_project/data/raw/breast_cancer_wisconsin.csv'
OUTPUT_DIR = 'scratch_project/results'
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')

def run_experiment():
    print("--- Kanser Hücresi Analizi Projesi Başlatılıyor ---")

    # 1. Veri Yükleme ve Sütun Adlarını Belirleme
    print(f"Veri yükleniyor: {DATA_PATH}")
    # Veri setinin sütun adları yok, bu yüzden manuel olarak ekleyeceğiz
    # Kaynak: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
    column_names = [
        'id', 'diagnosis', 
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]
    df = pd.read_csv(DATA_PATH, header=None, names=column_names)

    # İlk sütun olan 'id' ve son sütun olan 'Unnamed: 32' (eğer varsa) düşürülmeli
    df = df.drop(columns=['id'])
    if 'Unnamed: 32' in df.columns: # Bazı durumlarda boş bir sütun gelebilir
        df = df.drop(columns=['Unnamed: 32'])

    print("Veri başarıyla yüklendi ve sütun adları ayarlandı.")
    print("Veri setinin ilk 5 satırı:")
    print(df.head())
    print("\nVeri seti bilgileri:")
    df.info()

    # 2. Hedef Değişkeni Kodlama ('diagnosis': M=Malignant, B=Benign)
    le = LabelEncoder()
    df['diagnosis'] = le.fit_transform(df['diagnosis']) # M=1, B=0 olarak kodlanacak
    print(f"\nTeşhis sınıfları kodlandı: {list(le.classes_)} -> {list(le.transform(le.classes_))}")

    # 3. Özellikleri ve Hedef Değişkeni Ayırma
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    feature_cols = X.columns.tolist()

    # 4. Veri Kalitesi Analizi
    print("\n--- Veri Kalitesi Raporu ---")
    quality_report = analyze_data_quality(df)
    print(quality_report)

    # 5. Eğitim ve Test Setlerine Ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nEğitim seti boyutu: {X_train.shape}")
    print(f"Test seti boyutu: {X_test.shape}")

    # 6. Veri Ön İşleme (DataPreprocessor kullanarak)
    print("\n--- Veri Ön İşleme Başlatılıyor ---")
    preprocessor = DataPreprocessor(
        impute_strategy="median",
        scale_method="standard",
        detect_outliers="iqr", # IQR kullanarak aykırı değerleri tespit et
        remove_outliers=True,  # Tespit edilen aykırı değerleri kaldır
        pca_components=None    # PCA şu an için uygulanmıyor
    )
    X_train_processed, y_train_processed = preprocessor.fit_transform(X_train, y_train)
    X_test_processed = preprocessor.transform(X_test)
    print(preprocessor.summary_text())

    # Ensure feature names are maintained if not using PCA
    if preprocessor.pca is None:
        X_train_processed.columns = feature_cols
        X_test_processed.columns = feature_cols
    
    # 7. Model Karşılaştırması
    print("\n--- Model Karşılaştırması Başlatılıyor ---")
    comparator, results_df = compare_models(
        X_train_processed, X_test_processed, y_train_processed, y_test,
        task_type="classification",
        output_dir=OUTPUT_DIR
    )
    print("\nModel Karşılaştırma Sonuçları:")
    print(results_df.to_markdown(index=False))

    best_model = comparator.best_model
    best_model_name = comparator.best_model_name
    print(f"\nEn iyi model: {best_model_name}")

    # 8. En İyi Model İçin Hiperparametre Optimizasyonu (İsteğe bağlı)
    print(f"\n--- En İyi Model ({best_model_name}) İçin Hiperparametre Optimizasyonu Başlatılıyor ---")
    # Sadece en iyi modelin türüne göre parametre ızgarası belirliyoruz
    # Not: Pipeline ile modelin adlandırması 'model__' ön eki ile yapılır
    
    optimized_model, best_params, cv_results = optimize_model(
        X_train_processed, y_train_processed, 
        model_name=best_model_name.replace('Classifier', '').replace('Regressor', ''), # model_compare'daki isimlerle eşleşmeli
        task_type="classification",
        method="random", n_iter=20, # Rastgele arama ile 20 iterasyon
        output_dir=OUTPUT_DIR
    )
    print(f"\nOptimize Edilmiş En İyi Model ({best_model_name}) Parametreleri: {best_params}")

    # 9. Görselleştirmeler (Optimize Edilmiş Model Üzerinden)
    print("\n--- Görselleştirmeler Oluşturuluyor ---")
    viz = MLVisualizer(output_dir=PLOTS_DIR)
    
    # Feature names for visualization should correspond to X_train_processed columns
    # If PCA was applied, feature_cols would be PCA_Component_X
    current_feature_names = X_train_processed.columns.tolist()

    viz.plot_all(optimized_model, X_train_processed, X_test_processed, y_train_processed, y_test,
                 feature_names=current_feature_names, df=df.drop('diagnosis', axis=1), # Orijinal veri çerçevesini geçirin
                 task_type="classification")

    # 10. Sonuç Raporu ve README Oluşturma
    print("\n--- Sonuç Raporu ve README Oluşturuluyor ---")
    create_final_report(results_df, optimized_model, best_params, PLOTS_DIR, best_model_name)
    create_readme(results_df, best_model_name, best_params)

    print("\n--- Kanser Hücresi Analizi Projesi Tamamlandı ---")

def create_final_report(results_df, optimized_model, best_params, plots_dir, best_model_name):
    report_content = []
    report_content.append("# Kanser Hücresi Analizi Sonuç Raporu\n")
    report_content.append("Bu rapor, Wisconsin Meme Kanseri (Tanısal) veri setini kullanarak meme kanseri teşhisi için makine öğrenimi modellerinin performansını detaylandırmaktadır.\n\n")

    report_content.append("## 1. Veri Seti\n")
    report_content.append("Veri seti, çekirdek özelliklerinden elde edilen 30 adet sayısal özellik ile hücre çekirdeğinin dijitalleştirilmiş bir görüntüsünden elde edilmiştir. Hedef değişken, teşhis ('M' için Malign, 'B' için Benign) şeklindedir.\n\n")

    report_content.append("## 2. Ön İşleme Adımları\n")
    report_content.append("- Eksik değerler medyan ile doldurulmuştur.\n")
    report_content.append("- IQR yöntemi kullanılarak aykırı değerler tespit edilmiş ve eğitim veri setinden çıkarılmıştır.\n")
    report_content.append("- Özellikler StandardScaler kullanılarak standartlaştırılmıştır.\n\n")

    report_content.append("## 3. Model Karşılaştırması\n")
    report_content.append("Çeşitli sınıflandırma modelleri eğitilmiş ve 5 katlı çapraz doğrulama ile değerlendirilmiştir. Performans, Accuracy, Precision, Recall, F1 Score ve ROC AUC metrikleri üzerinden ölçülmüştür.\n\n")
    report_content.append("### Karşılaştırma Tablosu\n")
    report_content.append(results_df.to_markdown(index=False))
    report_content.append(f"\n\nKarşılaştırma sonucunda **{best_model_name}** en iyi performansı gösteren model olarak belirlenmiştir.\n\n")

    report_content.append(f"## 4. Hiperparametre Optimizasyonu ({best_model_name})\n")
    report_content.append(f"En iyi model olan **{best_model_name}** için Random Search ile hiperparametre optimizasyonu yapılmıştır.\n")
    report_content.append("### Optimize Edilmiş Parametreler:\n")
    for param, value in best_params.items():
        report_content.append(f"- `{param}`: `{value}`\n")
    report_content.append("\n")

    report_content.append("## 5. Görselleştirmeler\n")
    report_content.append("Aşağıdaki grafikler, optimize edilmiş en iyi modelin performansını ve veri özelliklerini daha ayrıntılı incelemektedir:\n\n")
    report_content.append(f"- **Confusion Matrix (Normalleştirilmiş ve Normal)**: Modelin doğru ve yanlış sınıflandırmalarını gösterir. ([{os.path.join(plots_dir, 'confusion_matrix.png')}], [{os.path.join(plots_dir, 'normalized_confusion_matrix.png')}] tesislerinde bulunabilir)\n")
    report_content.append(f"- **ROC Eğrisi**: Modelin farklı karar eşiklerinde True Positive Rate (TPR) ve False Positive Rate (FPR) arasındaki dengeyi gösterir. ([{os.path.join(plots_dir, 'roc_curve.png')}] tesisinde bulunabilir)\n")
    
    # Feature importance sadece belirli modeller için geçerlidir
    if hasattr(optimized_model.named_steps['model'], 'feature_importances_') or hasattr(optimized_model.named_steps['model'], 'coef_'):
        report_content.append(f"- **Özellik Önem Derecesi**: Modelin karar verirken hangi özelliklere daha çok güvendiğini gösterir. ([{os.path.join(plots_dir, 'feature_importance.png')}] tesisinde bulunabilir)\n")
    
    report_content.append(f"- **Korelasyon Matrisi**: Özellikler arasındaki ilişkiyi gösteren bir ısı haritası. ([{os.path.join(plots_dir, 'correlation_matrix.png')}] tesisinde bulunabilir)\n")
    report_content.append(f"- **Öğrenme Eğrisi**: Modelin eğitim boyutuyla performansının nasıl değiştiğini gösterir, aşırı uyum ve eksik uyum sorunlarını anlamaya yardımcı olur. ([{os.path.join(plots_dir, 'learning_curve.png')}] tesisinde bulunabilir)\n")
    report_content.append(f"- **Sınıf Dağılımı**: Hedef değişkenin sınıflarının veri setindeki dağılımını gösterir. ([{os.path.join(plots_dir, 'class_distribution_bar.png')}], [{os.path.join(plots_dir, 'class_distribution_donut.png')}] tesislerinde bulunabilir)\n\n")

    report_content.append("## 6. Modelin Kullanımı\n")
    
    # Düzeltme burada: Inner f-string'i ayrı bir değişken olarak tanımlıyoruz
    model_filename = f"best_optimized_{best_model_name.lower().replace(' ', '')}.pkl"
    model_full_path = os.path.join(OUTPUT_DIR, model_filename)

    report_content.append(f"Eğitilmiş ve optimize edilmiş en iyi model **{best_model_name}**, `{model_full_path}` adresine kaydedilmiştir.\n")
    report_content.append("Bu modeli yeni veriler üzerinde tahmin yapmak için aşağıdaki gibi yükleyebilir ve kullanabilirsiniz:\n\n")
    report_content.append("```python\n")
    report_content.append("import pandas as pd\n")
    report_content.append("from utils.model_loader import load_and_predict\n")
    report_content.append("from utils.preprocessor import DataPreprocessor\n\n")
    
    # Düzeltme burada: model_full_path değişkenini kullanıyoruz
    report_content.append(f"model_path = '{model_full_path}'\n")
    report_content.append("# Yeni verilerinizi yükleyin veya oluşturun\n")
    report_content.append("# X_new_raw = pd.DataFrame(...)\n\n")
    report_content.append("# Modeli eğitirken kullanılan aynı ön işleme adımlarını uygulayın (eğer varsa)\n")
    report_content.append("# preprocessor = DataPreprocessor(impute_strategy='median', scale_method='standard', detect_outliers='iqr', remove_outliers=True)\n")
    report_content.append("# preprocessor.fit(X_train_original_full_dataset, y_train_original_full_dataset) # Modeli eğitmek için kullanılan tüm veriyi kullanarak fit etmelisiniz\n")
    report_content.append("# X_new_processed = preprocessor.transform(X_new_raw)\n\n")
    report_content.append("# Tahmin yap\n")
    report_content.append("# predictions = load_and_predict(model_path, X_new_processed)\n")
    report_content.append("# print(predictions)\n")
    report_content.append("```\n")

    with open(os.path.join(OUTPUT_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.writelines(report_content)
    print(f"Final raporu kaydedildi: {os.path.join(OUTPUT_DIR, 'report.md')}")

def create_readme(results_df, best_model_name, best_params):
    readme_content = []
    readme_content.append("# Kanser Hücresi Analizi Projesi\n\n")
    readme_content.append("Bu proje, Wisconsin Meme Kanseri (Tanısal) veri setini kullanarak meme kanseri teşhisi için makine öğrenimi modelleri geliştirmeyi ve karşılaştırmayı amaçlamaktadır.\n\n")

    readme_content.append("## Kurulum\n")
    readme_content.append("Projeyi kurmak için aşağıdaki adımları izleyin:\n")
    readme_content.append("```bash\n")
    readme_content.append("git clone <repo-url>\n")
    readme_content.append("cd <project-directory>\n")
    readme_content.append("pip install -r scratch_project/requirements.txt\n")
    readme_content.append("```\n\n")

    readme_content.append("## Veri Seti\n")
    readme_content.append("Kullanılan veri seti [UCI Machine Learning Repository - Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) adresinden temin edilmiştir. Veri seti, bir hücre çekirdeğinin dijitalleştirilmiş bir görüntüsünden hesaplanan 30 adet gerçek değerli özellik içerir ve teşhis (iyi huylu/kötü huylu) hedef değişkenidir.\n\n")

    readme_content.append("## Proje Yapısı\n")
    readme_content.append("```\n")
    readme_content.append("scratch_project/\n")
    readme_content.append("├── data/\n")
    readme_content.append("│   └── raw/\n")
    readme_content.append("│       └── breast_cancer_wisconsin.csv\n")
    readme_content.append("├── src/\n")
    readme_content.append("│   └── train.py\n")
    readme_content.append("├── utils/\n")
    readme_content.append("│   ├── model_compare.py\n")
    readme_content.append("│   ├── visualize.py\n")
    readme_content.append("│   ├── preprocessor.py\n")
    readme_content.append("│   ├── model_loader.py\n")
    readme_content.append("│   └── hyperparameter_optimizer.py\n")
    readme_content.append("├── results/\n")
    readme_content.append("│   ├── plots/              # Oluşturulan görseller buraya kaydedilir\n")
    readme_content.append("│   ├── best_model.pkl      # Karşılaştırmadan elde edilen en iyi model\n")
    readme_content.append("│   ├── best_optimized_*.pkl # Hiperparametre optimizasyonundan elde edilen en iyi model\n")
    readme_content.append("│   ├── comparison_results.json # Tüm modellerin karşılaştırma sonuçları\n")
    readme_content.append("│   ├── comparison_report.md # Model karşılaştırma raporu\n")
    readme_content.append("│   ├── cv_results_*.csv    # Hiperparametre optimizasyonu çapraz doğrulama sonuçları\n")
    readme_content.append("│   └── report.md           # Detaylı proje raporu\n")
    readme_content.append("└── requirements.txt\n")
    readme_content.append("```\n\n")

    readme_content.append("## Kullanım\n")
    readme_content.append("Projeyi çalıştırmak ve modelleri eğitmek için `train.py` betiğini çalıştırın:\n")
    readme_content.append("```bash\n")
    readme_content.append("python scratch_project/src/train.py\n")
    readme_content.append("```\n")
    readme_content.append("Bu betik, veri önişleme yapacak, birden çok modeli eğitecek, karşılaştıracak, en iyi model üzerinde hiperparametre optimizasyonu uygulayacak ve sonuçları `results/` dizinine kaydedecektir.\n\n")

    readme_content.append("## Sonuçlar\n")
    readme_content.append("Model karşılaştırma sonuçları ve optimize edilmiş model detayları `results/report.md` dosyasında bulunabilir. Görselleştirmeler `results/plots/` dizininde mevcuttur.\n")
    readme_content.append("Özetle, yapılan karşılaştırmalar ve optimizasyonlar sonucunda **{}** modelinin en iyi performansı gösterdiği tespit edilmiştir.\n".format(best_model_name))
    readme_content.append("Optimize edilmiş parametreler:\n")
    for param, value in best_params.items():
        readme_content.append(f"- `{param}`: `{value}`\n")
    readme_content.append("\n")

    readme_content.append("## Katkıda Bulunma\n")
    readme_content.append("Geri bildirimler ve katkılar memnuniyetle karşılanır.\n\n")

    readme_content.append("## Lisans\n")
    readme_content.append("Bu proje MIT Lisansı altında lisanslanmıştır.\n")

    # Düzeltme burada: model_full_path değişkenini kullanıyoruz
    model_filename = f"best_optimized_{best_model_name.lower().replace(' ', '')}.pkl"
    model_full_path = os.path.join(OUTPUT_DIR, model_filename)
    readme_content.append(f"Eğitilmiş ve optimize edilmiş en iyi model **{best_model_name}**, `{model_full_path}` adresine kaydedilmiştir.\n")


    with open("scratch_project/README.md", "w", encoding="utf-8") as f:
        f.writelines(readme_content)
    print(f"README.md dosyası kaydedildi: scratch_project/README.md")

if __name__ == "__main__":
    run_experiment()