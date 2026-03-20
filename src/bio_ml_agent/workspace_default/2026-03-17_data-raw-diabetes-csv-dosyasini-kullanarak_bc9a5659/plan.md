# Diyabet Tahmin Projesi Planı

1.  **Proje Yapısı ve Gereksinimler:**
    *   Gerekli dizinleri (data/, src/, results/) oluştur.
    *   `requirements.txt` dosyasını oluştur.
2.  **Veri Yükleme ve Ön İşleme:**
    *   `data/raw/diabetes.csv` dosyasını Pandas ile yükle.
    *   `analyze_data_quality` kullanarak veri kalitesini analiz et.
    *   `DataPreprocessor` kullanarak eksik değerleri işle, aykırı değerleri tespit et/kaldır ve özellikleri ölçeklendir.
    *   Veriyi eğitim ve test setlerine ayır (X, y ayırımı, train_test_split).
3.  **Model Eğitimi ve Karşılaştırma:**
    *   `utils.model_compare.compare_models` kullanarak LogisticRegression, RandomForestClassifier ve GradientBoostingClassifier modellerini eğit ve 5-kat çapraz doğrulama ile karşılaştır.
    *   En iyi modeli belirle ve kaydet.
4.  **Model Görselleştirme:**
    *   `utils.visualize.MLVisualizer` kullanarak Karışıklık Matrisi, ROC Eğrisi, Özellik Önemleri, Korelasyon Matrisi, Sınıf Dağılımı gibi grafikleri oluştur ve `results/plots/` dizinine kaydet.
5.  **Açıklanabilir Yapay Zeka (XAI):**
    *   `xai_engine.XAIEngine` kullanarak en iyi model için SHAP özet grafiği oluştur ve `results/plots/` dizinine kaydet.
    *   SHAP analizinden elde edilen en önemli özelliklere dayanarak `report.md` içine "Klinik Karar Özeti" bölümü ekle.
6.  **Raporlama:**
    *   `results/report.md` dosyasını oluştur; model karşılaştırma tablosunu, en iyi modelin metriklerini, görselleştirmelere referansları ve XAI özetini içerecek şekilde hazırla.
    *   Projenin kurulumu ve çalıştırılması talimatlarını içeren `README.md` dosyasını oluştur.