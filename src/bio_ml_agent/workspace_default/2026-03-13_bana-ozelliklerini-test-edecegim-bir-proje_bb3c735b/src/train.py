import sys
sys.path.append('.') # Proje kök dizinini sys.path'e ekle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import joblib
import os

from utils.model_compare import compare_models
from utils.visualize import MLVisualizer
from xai_engine import XAIEngine

def main():
    # 1. Veri Yükleme
    data_path = "data/raw/diabetes.csv"
    
    # Veri setinin manuel olarak yüklendiğini varsayarak,
    # eğer yoksa bir hata mesajı verelim.
    if not os.path.exists(data_path):
        print(f"Hata: {data_path} bulunamadı. Lütfen Pima Indian Diabetes veri setini manuel olarak bu konuma indirin.")
        print("Örnek URL: https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv")
        print("İndirdikten sonra dosya adını 'diabetes.csv' olarak kaydedin.")
        return

    column_names = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
        "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
    ]
    df = pd.read_csv(data_path, names=column_names)

    # 0 değerleri olan sütunları kontrol et ve gerektiğinde median/mean ile doldur
    # Bu veri setinde 0 değerleri bazı sütunlar için eksik değer anlamına gelebilir.
    # 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'
    # Bu sütunlardaki 0'ları NaN ile değiştirip median ile dolduralım.
    cols_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_to_impute] = df[cols_to_impute].replace(0, np.nan)
    for col in cols_to_impute:
        df[col] = df[col].fillna(df[col].median())

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # 2. Veriyi Eğitim ve Test Setlerine Ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Özellik isimleri
    feature_names = X.columns.tolist()

    # 3. Model Tanımlama ve Karşılaştırma
    print("Modeller Eğitiliyor ve Karşılaştırılıyor (compare_models kullanılarak)...")
    best_model_pipeline, comparison_results_df = compare_models(
        X_train, X_test, y_train, y_test,
        task_type="classification",
        output_dir="results/"
    )
    
    # Rapor için sonuçları HTML veya Markdown tablosuna dönüştür
    comparison_table_md = comparison_results_df.to_markdown()

    print(f"\nEn iyi model: {best_model_pipeline.named_steps['model'].__class__.__name__}")

    # 4. Görselleştirme
    print("\nGörselleştirmeler Oluşturuluyor (MLVisualizer kullanılarak)...")
    viz = MLVisualizer(output_dir="results/plots")
    viz.plot_all(best_model_pipeline, X_train, X_test, y_train, y_test,
                 feature_names=feature_names, df=df.copy(), # df.copy() for safety
                 model_name=best_model_pipeline.named_steps['model'].__class__.__name__)

    # 5. Açıklanabilir Yapay Zeka (XAI)
    print("\nXAI Analizi Yapılıyor (XAIEngine kullanılarak)...")
    # SHAP için X_train'i scaler'dan geçirmeden kullanıyoruz, çünkü XAIEngine içinde bu işlem yapılıyor.
    # Ancak KernelExplainer için X_train'in örneklenmiş bir kısmı kullanılır.
    # TreeExplainer için scaler uygulanmış X_test'i bekler.
    
    # XAIEngine init'i için X_train_df yerine X_train dataframe'ini göndermeliyiz
    xai = XAIEngine(best_model_pipeline, X_train, feature_names=feature_names, task_type="classification")
    top_shap_features = xai.generate_shap_summary(X_test, output_dir="results/plots", max_display=10)
    lime_explanation = xai.explain_instance_lime(X_test.iloc[0], output_dir="results/plots", filename="lime_explanation_instance_0.png")

    # Klinik Karar Özeti
    clinical_decision_summary = "### Klinik Karar Özeti\n\n"
    clinical_decision_summary += f"En iyi model olan **{best_model_pipeline.named_steps['model'].__class__.__name__}** kullanılarak yapılan analizde, SHAP değerlerine göre diyabet riskini etkileyen en önemli özellikler şunlardır:\n"
    for feature, importance in top_shap_features:
        clinical_decision_summary += f"- **{feature}**: Modelin kararlarında önemli bir etkiye sahip.\n"
    clinical_decision_summary += "\nBu özellikler, klinik uygulamalarda hastalığın erken teşhisi ve risk faktörlerinin belirlenmesi için yol gösterici olabilir."


    # 6. Raporlama
    print("\nRaporlar Oluşturuluyor...")
    with open("results/comparison_report.md", "w") as f:
        f.write("# Makine Öğrenimi Model Karşılaştırma Raporu\n\n")
        f.write("## 1. Giriş\n")
        f.write("Bu rapor, Pima Indian Diabetes veri seti üzerinde çeşitli makine öğrenimi modellerinin performansını değerlendirmektedir. Amaç, diyabet teşhisi için en uygun modeli belirlemek ve modelin kararlarını açıklayan içgörüler sunmaktır.\n\n")
        
        f.write("## 2. Veri Seti\n")
        f.write("Kullanılan veri seti, Pima Indian Diabetes veri setidir. Bu veri seti, Pima Kızılderililerindeki diyabet teşhisi ile ilgili sekiz klinik özelliği (örn. hamilelik sayısı, glikoz seviyesi, kan basıncı, BMI) ve bir çıktı değişkeni (diyabet varlığı/yokluğu) içermektedir.\n")
        f.write("Veri ön işleme adımı olarak, 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI' sütunlarındaki 0 değerleri NaN ile değiştirilmiş ve ardından medyan değerlerle doldurulmuştur.\n\n")
        
        f.write("## 3. Metodoloji\n")
        f.write("Veri seti %80 eğitim ve %20 test olmak üzere ikiye ayrılmıştır. Her model, `StandardScaler` ile birlikte bir `Pipeline` içinde eğitilmiştir. Modellerin performansını değerlendirmek için Accuracy, Precision, Recall, F1-Score ve ROC-AUC metrikleri kullanılmıştır. Ayrıca 5-katlı çapraz doğrulama uygulanmıştır.\n\n")
        
        f.write("## 4. Model Karşılaştırma Sonuçları\n")
        f.write("Aşağıdaki tabloda eğitilen modellerin test seti üzerindeki performans metrikleri ve çapraz doğrulama skorları gösterilmiştir:\n\n")
        f.write(comparison_table_md)
        f.write("\n\nYukarıdaki tabloya göre, en yüksek ROC-AUC değerine sahip model **" + best_model_pipeline.named_steps['model'].__class__.__name__ + "** olarak belirlenmiştir.\n\n")
        
        f.write("## 5. Görselleştirmeler\n")
        f.write("Seçilen en iyi model için aşağıdaki görselleştirmeler oluşturulmuştur:\n")
        f.write("- **Confusion Matrix (Normal & Normalize Edilmiş):** `results/plots/best_model_confusion_matrix.png`, `results/plots/best_model_confusion_matrix_normalized.png`\n")
        f.write("- **ROC Eğrisi:** `results/plots/best_model_roc_curve.png`\n")
        f.write("- **Özellik Önem Derecesi/Katsayıları:** `results/plots/best_model_feature_importance.png` veya `results/plots/best_model_feature_coefficients.png`\n")
        f.write("- **Veri Seti Korelasyon Matrisi:** `results/plots/correlation_matrix.png`\n")
        f.write("- **Test Seti Sınıf Dağılımı:** `results/plots/class_distribution_bar.png`, `results/plots/class_distribution_donut.png`\n\n")

        f.write("## 6. Açıklanabilir Yapay Zeka (XAI)\n")
        f.write("Modelin kararlarını anlamak için SHAP ve LIME yöntemleri kullanılmıştır:\n")
        f.write("- **SHAP Summary Plot:** `results/plots/shap_summary_plot.png` (Genel özellik önemini gösterir)\n")
        f.write("- **LIME Instance Açıklaması:** `results/plots/lime_explanation_instance_0.png` ve `results/plots/lime_explanation_instance_0.txt` (Belirli bir örnek için modelin neden o tahmini yaptığını gösterir)\n\n")
        f.write(clinical_decision_summary)
        f.write("\n\n")

        f.write("## 7. Model Kullanımı\n")
        f.write("Eğitilmiş en iyi model `results/best_model.pkl` dosyasında kaydedilmiştir. Model aşağıdaki gibi yüklenebilir ve yeni veriler üzerinde tahminler yapmak için kullanılabilir:\n\n")
        f.write("```python\n")
        f.write("import joblib\n")
        f.write("import pandas as pd\n\n")
        f.write("best_model = joblib.load('results/best_model.pkl')\n\n")
        f.write("# Yeni veri örneği (sütun isimleri eğitimdekiyle aynı olmalı)\n")
        f.write("new_data = pd.DataFrame([[6, 148, 72, 35, 0, 33.6, 0.627, 50]], columns=[\n")
        f.write("    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',\n")
        f.write("    'BMI', 'DiabetesPedigreeFunction', 'Age'\n")
        f.write("])\n\n")
        f.write("prediction = best_model.predict(new_data)\n")
        f.write("prediction_proba = best_model.predict_proba(new_data)[:, 1]\n\n")
        f.write("print(f'Tahmin: {prediction[0]} (0: Diyabet Yok, 1: Diyabet Var)')\n")
        f.write("print(f'Diyabet Var olma Olasılığı: {prediction_proba[0]:.4f}')\n")
        f.write("```\n")

    with open("README.md", "w") as f:
        f.write("# Pima Indian Diyabet Tahmini Projesi\n\n")
        f.write("Bu proje, Pima Indian Diabetes veri setini kullanarak diyabet teşhisi için makine öğrenimi modelleri geliştirmeyi ve karşılaştırmayı amaçlamaktadır. Proje, veri ön işleme, model eğitimi, performans değerlendirme, görselleştirme ve açıklanabilir yapay zeka (XAI) adımlarını içermektedir.\n\n")
        f.write("## Kurulum\n")
        f.write("1. Depoyu klonlayın:\n")
        f.write("   `git clone <repo_url>`\n")
        f.write("   `cd <project_directory>`\n")
        f.write("2. Gerekli kütüphaneleri yükleyin:\n")
        f.write("   `pip install -r requirements.txt`\n\n")
        f.write("## Veri Seti\n")
        f.write(f"Pima Indian Diabetes veri seti manuel olarak `data/raw/diabetes.csv` konumuna indirilmelidir. Örnek bir indirme linki: https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv\n\n")
        f.write("## Proje Yapısı\n")
        f.write("```\n")
        f.write(". # Proje Kök Dizini\n")
        f.write("├── data/\n")
        f.write("│   └── raw/             # Ham veri seti\n")
        f.write("├── results/\n")
        f.write("│   ├── plots/           # Görselleştirmeler ve XAI çıktıları\n")
        f.write("│   ├── best_model.pkl   # Eğitilmiş en iyi model\n")
        f.write("│   ├── comparison_results.json # Model karşılaştırma sonuçları\n")
        f.write("│   └── comparison_report.md # Detaylı proje raporu\n")
        f.write("├── src/\n")
        f.write("│   └── train.py         # Ana eğitim ve değerlendirme scripti\n")
        f.write("├── utils/\n")
        f.write("│   ├── model_compare.py # Model karşılaştırma fonksiyonları\n")
        f.write("│   └── visualize.py     # Görselleştirme fonksiyonları\n")
        f.write("├── xai_engine.py        # Açıklanabilir AI (SHAP, LIME) fonksiyonları\n")
        f.write("├── requirements.txt     # Python bağımlılıkları\n")
        f.write("└── README.md            # Bu README dosyası\n")
        f.write("```\n\n")
        f.write("## Kullanım\n")
        f.write("Tüm pipeline'ı çalıştırmak için:\n")
        f.write("```bash\n")
        f.write("python src/train.py\n")
        f.write("```\n")
        f.write("Bu komut, modelleri eğitecek, performanslarını karşılaştıracak, çeşitli görselleştirmeler ve XAI çıktıları üretecek ve `results/` dizininde bir rapor oluşturacaktır.\n\n")
        f.write("## Rapor\n")
        f.write("Detaylı analiz sonuçları ve klinik karar özeti `results/comparison_report.md` dosyasında bulunabilir.\n\n")
        f.write("## Geliştirici Notu\n")
        f.write("Bu proje, genel bir makine öğrenimi projesinin tüm temel adımlarını göstermektedir. Daha fazla iyileştirme için hiperparametre optimizasyonu, gelişmiş özellik mühendisliği veya farklı model mimarileri denenebilir.\n")
    print("Raporlar 'results/comparison_report.md' ve 'README.md' olarak oluşturuldu.")

if __name__ == "__main__":
    main()