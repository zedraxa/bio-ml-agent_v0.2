# Proje Planı: Pima Kızılderilileri Diyabet Tahmini

## 1. Giriş
Bu proje, Pima Kızılderilileri Diyabet Veri Seti'ni kullanarak diyabetin erken teşhisi için makine öğrenimi modellerini geliştirmeyi, karşılaştırmayı ve değerlendirmeyi amaçlamaktadır.

## 2. Veri Seti Seçimi
*   **Adı**: Pima Kızılderilileri Diyabet Veri Seti
*   **Kaynak**: UCI Machine Learning Repository veya Kaggle
*   **Hedef**: 'Outcome' sütununu tahmin etmek (0: Diyabet yok, 1: Diyabet var)
*   **Özellikler**: Hamilelik sayısı, glikoz konsantrasyonu, kan basıncı, cilt kalınlığı, insülin seviyesi, BMI, diyabet soy ağacı fonksiyonu, yaş.

## 3. Adımlar
1.  **Proje Yapısı Oluşturma**: `data/raw`, `src`, `results`, `results/plots` dizinlerini oluştur.
2.  **Veri İndirme**: Pima Kızılderilileri Diyabet Veri Seti'ni `data/raw/diabetes.csv` konumuna indir.
3.  **Bağımlılıklar**: `requirements.txt` dosyasını oluştur ve gerekli kütüphaneleri belirt.
4.  **Veri Ön İşleme (src/preprocess.py)**:
    *   Veriyi yükle.
    *   Sıfır değerleri (anlamsız olanları) uygun stratejilerle doldur (örn. medyan).
    *   Veriyi eğitim ve test setlerine ayır.
    *   Veri kalitesi analizi yap (`utils.preprocessor.analyze_data_quality`).
    *   Veriyi ölçeklendir (`StandardScaler`).
5.  **Model Eğitimi ve Karşılaştırma (src/train.py)**:
    *   En az 3 sınıflandırma modeli tanımla (örn. Lojistik Regresyon, Random Orman, Gradient Boosting).
    *   Her model için `StandardScaler` ile `Pipeline` oluştur.
    *   `utils.model_compare.compare_models` kullanarak 5 katlı çapraz doğrulama ile modelleri eğit ve karşılaştır.
    *   Doğruluk, hassasiyet, duyarlılık, F1 skoru ve ROC AUC metriklerini hesapla.
    *   En iyi modeli belirle ve `results/best_model.pkl` olarak kaydet.
6.  **Hiperparametre Optimizasyonu (src/train.py)** (Opsiyonel ama gösterim için faydalı):
    *   Belirlenen en iyi model için `utils.hyperparameter_optimizer.optimize_model` kullanarak basit bir hiperparametre optimizasyonu yap.
7.  **Görselleştirme (src/train.py)**:
    *   `utils.visualize.MLVisualizer` kullanarak aşağıdaki grafikleri oluştur ve `results/plots/` dizinine kaydet:
        *   Karışıklık Matrisi (normal ve normalize edilmiş)
        *   ROC Eğrisi
        *   Özellik Önem Derecesi (feature importance)
        *   Korelasyon Matrisi
        *   Sınıf Dağılımı
8.  **Açıklanabilir Yapay Zeka (XAI) (src/train.py)**:
    *   `xai_engine.XAIEngine` kullanarak en iyi model için SHAP özet grafiği ve LIME örnek açıklaması oluştur.
    *   Grafikleri `results/plots/` dizinine kaydet.
9.  **Raporlama ve Belgeleme**:
    *   `results/comparison_report.md` dosyasında model karşılaştırma tablosu, seçilen en iyi model, performans metrikleri, Klinik Karar Özeti (XAI çıktısı ile) ve görselleştirmelere referans içeren ayrıntılı bir rapor oluştur.
    *   `README.md` dosyasında projenin amacı, nasıl çalıştırılacağı ve çıktıların ne anlama geldiği hakkında genel bir bilgi ver.