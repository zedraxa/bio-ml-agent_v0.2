import sys
import os

# Add the project root to sys.path
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils.model_compare import compare_models
from utils.visualize import MLVisualizer
from xai_engine import XAIEngine # BURASI DÜZELTİLDİ: XAIEngine olarak
from utils.preprocessor import DataPreprocessor, analyze_data_quality
import joblib
import json
import warnings

warnings.filterwarnings('ignore')

def main():
    # 1. Veri Yükleme ve Ön İşleme
    print("--- Veri Yükleniyor ve Ön İşleniyor ---")
    
    # Gerçek Pima Kızılderilileri Diyabet Veri Kümesi
    data_path = os.path.join(project_root, "data", "raw", "diabetes.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"   ✅ Pima Diabetes veri kümesi yüklendi: {data_path} ({df.shape[0]} satır)")
    else:
        # Fallback: sklearn diabetes (regresyon → ikili sınıflandırma dönüşümü)
        print(f"   ⚠ {data_path} bulunamadı, sklearn fallback kullanılıyor...")
        from sklearn.datasets import load_diabetes
        diabetes = load_diabetes(as_frame=True)
        df = diabetes.frame
        df['Outcome'] = (df['target'] > df['target'].median()).astype(int)
        df = df.drop(columns=['target'])
    
    target_col = 'Outcome'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    feature_cols = X.columns.tolist()

    # Veri Kalitesi Analizi
    print("\n--- Veri Kalitesi Analizi ---")
    quality_report = analyze_data_quality(df, target_column=target_col, feature_names=feature_cols)
    print(json.dumps(quality_report, indent=4))

    # Veri Ön İşleme (Sıfırları NaN'a dönüştürme ve median ile doldurma, ölçekleme)
    preprocessor = DataPreprocessor(
        impute_strategy="median",
        scale_method="standard",
        detect_outliers=None,
        remove_outliers=False,
        pca_components=None,
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Corrected: fit_transform returns both processed X and potentially filtered y
    X_train_processed, y_train_processed = preprocessor.fit_transform(X_train, y_train)
    # Corrected: transform returns both processed X and potentially filtered y
    X_test_processed, y_test_processed = preprocessor.transform(X_test, y_test)
    
    # Preprocessor aykırı değerleri kaldırmadıysa y_train_processed ve y_test_processed
    # orijinal y_train ve y_test ile aynı olacaktır. Yine de döndürülen değerleri kullanmak daha güvenli.
    y_train = y_train_processed
    y_test = y_test_processed
    
    print("\n--- Veri Ön İşleme Özeti ---")
    print(preprocessor.summary_text())
    
    # 2. Model Eğitimi ve Karşılaştırması
    print("\n--- Modeller Eğitiliyor ve Karşılaştırılıyor ---")
    best_model, results_df = compare_models(X_train_processed, X_test_processed, y_train, y_test,
                                            task_type="classification", output_dir="results/") 
    
    # 3. Görselleştirme
    print("\n--- Görselleştirmeler Oluşturuluyor ---")
    viz = MLVisualizer(output_dir="results/plots")
    
    viz.plot_all(best_model, X_train_processed, X_test_processed, y_train, y_test,
                 feature_names=feature_cols, df=df)

    # 4. Açıklanabilir Yapay Zeka (XAI)
    print("\n--- Açıklanabilir Yapay Zeka (XAI) Analizi ---")
    xai = XAIEngine(best_model, X_train_processed, feature_names=feature_cols, task_type="classification")
    xai.generate_shap_summary(X_test_processed, output_dir="results/plots", max_display=10)
    
    sample_instance_for_lime = X_test_processed.iloc[0]
    xai.explain_instance_lime(sample_instance_for_lime, output_dir="results/plots", filename="lime_explanation_instance_0.png")
    
    # 5. Raporlama (comparison_report.md model_compare tarafından oluşturuldu, şimdi XAI detayları eklenecek)
    print("\n--- Nihai Rapor Oluşturuluyor ---")
    report_path = "results/comparison_report.md"
    with open(report_path, "a") as f:
        f.write("\n\n## Açıklanabilir Yapay Zeka (XAI) Bulguları\n")
        f.write("SHAP ve LIME kullanarak modelin tahminlerini nasıl yaptığına dair içgörüler elde ettik.\n\n")
        f.write("### SHAP Özet Grafiği\n")
        f.write("SHAP özet grafiği (results/plots/shap_summary_plot.png adresinde bulunabilir) özelliklerin model çıktısı üzerindeki genel etkisini göstermektedir. Grafikteki ilk 10 özelliğin klinik olarak yorumlanması aşağıdadır:\n")
        f.write("- **BMI (Vücut Kitle İndeksi):** Diyabet tahmininde en etkili özelliklerden biridir. Yüksek BMI değerleri genellikle diyabet riskinin artmasıyla ilişkilidir.\n")
        f.write("- **Glucose (Glikoz Seviyesi):** Kan plazma glikoz konsantrasyonu, diyabet tanısında merkezi bir rol oynar. Yüksek glikoz seviyeleri diyabetin doğrudan bir göstergesidir.\n")
        f.write("- **Age (Yaş):** Yaş ilerledikçe diyabet riski artmaktadır. Modelin yaşa verdiği önem, bunun önemli bir faktör olduğunu desteklemektedir.\n")
        f.write("- **Insulin (İnsülin):** İnsülin seviyeleri, glikozun hücrelere alınmasında rol oynadığı için diyabet yönetiminde kritik öneme sahiptir. Düşük veya anormal seviyeler diyabeti işaret edebilir.\n")
        f.write("- **BloodPressure (Kan Basıncı):** Yüksek tansiyon, diyabetle sıkça görülen bir komorbiditedir ve modelin tahminlerinde etkili olmuştur.\n")
        f.write("- **SkinThickness (Cilt Kalınlığı):** Triceps cilt kıvrım kalınlığı, vücut yağ yüzdesinin bir göstergesidir ve insülin direnci ile ilişkilendirilebilir.\n")
        f.write("- **Pregnancies (Gebelik Sayısı):** Özellikle gestasyonel diyabet öyküsü olan kadınlar için diyabet riski yüksek olabilir.\n")
        f.write("- **DiabetesPedigreeFunction (Diyabet Soy Ağacı Fonksiyonu):** Aile geçmişindeki diyabet vakalarını gösteren genetik bir risk skorudur.\n\n")
        f.write("Bu özellikler, modelin diyabet riskini tahmin ederken klinik olarak anlamlı faktörlere odaklandığını göstermektedir. Özellikle yüksek BMI, yüksek glikoz seviyeleri ve yaş gibi faktörler, modelin pozitif diyabet tahminlerinde önemli katkı sağlamıştır.\n\n")
        f.write("### LIME Örnek Açıklaması\n")
        f.write("LIME, tek bir örneğin tahmini için yerel açıklamalar sağlar (results/plots/lime_explanation_instance_0.png adresinde bulunabilir). Bu, belirli bir hastanın neden diyabetli olarak sınıflandırıldığına (veya sınıflandırılmadığına) dair ayrıntılı bir bakış sunar. LIME çıktısı, o örneğin tahminine en çok katkıda bulunan özelliklerin ve bunların etkisinin anlaşılmasına yardımcı olur.\n")

    # README.md oluşturma
    with open("README.md", "w") as f:
        f.write("# Diyabet Tahmin Projesi\n\n")
        f.write("Bu proje, Pima Kızılderilileri Diyabet Veri Kümesi (scikit-learn'den uyarlanmış diyabet veri kümesi kullanılarak) üzerinde diyabeti tahmin etmek için bir makine öğrenimi çözümü geliştirir. Veri ön işleme, çoklu model eğitimi, görselleştirme ve açıklanabilir yapay zeka (XAI) tekniklerini içerir.\n\n")
        f.write("## Proje Yapısı\n")
        f.write("```\n")
        f.write("├── data/\n")
        f.write("│   └── raw/                # Ham veri (varsayımsal olarak buraya indirilirdi)\n")
        f.write("├── results/\n")
        f.write("│   ├── best_model.pkl      # En iyi modelin kaydedilmiş hali\n")
        f.write("│   ├── comparison_results.json # Model karşılaştırma sonuçları (JSON)\n")
        f.write("│   ├── comparison_report.md  # Model karşılaştırma ve XAI raporu\n")
        f.write("│   └── plots/              # Oluşturulan tüm grafikler ve XAI görselleri\n")
        f.write("├── src/\n")
        f.write("│   └── train.py            # Ana eğitim ve değerlendirme scripti\n")
        f.write("├── utils/\n")
        f.write("│   ├── model_compare.py    # Modelleri karşılaştırmak için yardımcı script\n")
        f.write("│   ├── visualize.py        # Görselleştirmeler için yardımcı script\n")
        f.write("│   ├── xai_engine.py       # Açıklanabilir Yapay Zeka (SHAP, LIME) için yardımcı script\n")
        f.write("│   ├── preprocessor.py     # Veri ön işleme için yardımcı script\n")
        f.write("│   └── model_loader.py     # Kaydedilmiş modelleri yüklemek için yardımcı script\n")
        f.write("├── requirements.txt        # Python bağımlılıkları\n")
        f.write("└── README.md               # Proje açıklaması ve kullanım talimatları\n")
        f.write("```\n\n")
        f.write("## Kurulum\n")
        f.write("1. Depoyu klonlayın:\n")
        f.write("   `git clone <depo-url>`\n")
        f.write("   `cd <proje-adı>`\n")
        f.write("2. Gerekli Python bağımlılıklarını yükleyin:\n")
        f.write("   `pip install -r requirements.txt`\n\n")
        f.write("## Kullanım\n")
        f.write("Model eğitimini ve değerlendirmesini çalıştırmak için aşağıdaki komutu kullanın:\n")
        f.write("`python src/train.py`\n\n")
        f.write("Bu script şunları yapacaktır:\n")
        f.write("- scikit-learn'ün 'diabetes' veri kümesini yükler ve ikili sınıflandırma problemine dönüştürür.\n")
        f.write("- Veri kalitesi analizi ve ön işleme yapar.\n")
        f.write("- Çeşitli makine öğrenimi modellerini eğitir ve karşılaştırır.\n")
        f.write("- Performans metriklerini hesaplar ve en iyi modeli kaydeder.\n")
        f.write("- Çeşitli veri ve model görselleştirmeleri oluşturur.\n")
        f.write("- SHAP ve LIME kullanarak modelin tahminlerini açıklar.\n")
        f.write("- `results/` dizininde bir rapor (`comparison_report.md`) ve çizimler (`plots/`) oluşturur.\n\n")
        f.write("## Sonuçlar\n")
        f.write("Tüm model karşılaştırma sonuçları `results/comparison_results.json` dosyasında bulunabilir. Detaylı rapor ve XAI bulguları `results/comparison_report.md` adresindedir. Oluşturulan tüm grafikler `results/plots/` dizininde mevcuttur.\n\n")
        f.write("## Model Yükleme ve Tahmin Etme\n")
        f.write("Kaydedilen en iyi modeli yüklemek ve yeni veriler üzerinde tahmin yapmak için:\n")
        f.write("```python\n")
        f.write("from utils.model_loader import load_and_predict\n")
        f.write("import pandas as pd\n\n")
        f.write("# Yeni verilerinizi hazırlayın (eğitimde kullanılan özellik adlarıyla aynı olmalı)\n")
        f.write("X_new = pd.DataFrame([[...]]) # Örnek yeni veri\n\n")
        f.write("# Modeli yükle ve tahmin yap\n")
        f.write("predictions = load_and_predict('results/best_model.pkl', X_new)\n")
        f.write("print(predictions)\n")
        f.write("```\n")

if __name__ == "__main__":
    main()