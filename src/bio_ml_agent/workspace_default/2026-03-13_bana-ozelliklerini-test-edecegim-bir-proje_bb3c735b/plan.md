# Proje Planı

Bu proje, makine öğrenimi pipeline'ının temel adımlarını uygulayarak, çeşitli ML modellerinin performansını karşılaştıracak ve en iyi modeli belirleyecektir. Ayrıca, modelin kararlarını anlamak için görselleştirmeler ve açıklanabilir yapay zeka (XAI) teknikleri kullanılacaktır.

## Adımlar:

1.  **Veri Seti Seçimi ve İndirme:**
    *   Yaygın ve iyi bilinen bir tıbbi veya biyolojik veri seti bulunacak (örneğin, Diyabet, Kalp Hastalığı).
    *   Veri seti `data/raw/` dizinine indirilecek.
2.  **Proje Yapısı ve Bağımlılıklar:**
    *   Temel proje dizinleri (`src/`, `data/`, `results/`, `notebooks/`) oluşturulacak.
    *   `requirements.txt` dosyası oluşturulacak ve gerekli kütüphaneler eklenecek.
    *   Yardımcı araçlar için `utils/` ve `xai_engine.py` dosyaları oluşturulacak.
3.  **Veri Yükleme ve Ön İşleme:**
    *   Veri seti yüklenecek ve temel bir hızlı kalite kontrolü yapılacak.
    *   Gerekirse eksik değerler ele alınacak ve kategorik değişkenler kodlanacak.
    *   Veri eğitim ve test setlerine ayrılacak.
4.  **Çoklu Model Karşılaştırması:**
    *   En az 3 farklı sınıflandırma/regresyon modeli (örneğin, Logistic Regression, RandomForest, GradientBoosting) StandardScaler ile bir Pipeline içinde eğitilecek.
    *   Her model için 5-katlı çapraz doğrulama uygulanacak.
    *   Modellerin performans metrikleri (accuracy, precision, recall, f1, roc_auc) hesaplanacak ve karşılaştırma tablosu oluşturulacak.
    *   En iyi model belirlenecek ve kaydedilecek.
5.  **Görselleştirme:**
    *   En iyi model için çeşitli performans grafikleri oluşturulacak:
        *   Confusion Matrix (normalleştirilmiş ve normal)
        *   ROC Eğrisi
        *   Özellik Önem Derecesi (Feature Importance)
        *   Korelasyon Matrisi (heatmap)
        *   Sınıf Dağılımı
    *   Tüm görselleştirmeler `results/plots/` dizinine kaydedilecek.
6.  **Açıklanabilir Yapay Zeka (XAI):**
    *   Belirlenen en iyi model için SHAP (SHapley Additive exPlanations) değerleri hesaplanacak.
    *   SHAP özeti (summary plot) ve tek bir örnek için LIME (Local Interpretable Model-agnostic Explanations) açıklama grafikleri oluşturulacak.
    *   XAI çıktıları `results/plots/` dizinine kaydedilecek.
    *   Modelin kararlarını açıklayan kısa bir "Klinik Karar Özeti" raporun içinde sunulacak.
7.  **Raporlama:**
    *   `results/comparison_report.md` dosyası oluşturulacak. Bu rapor, kullanılan veri setini, metodolojiyi, model karşılaştırma sonuçlarını (tablo dahil), görselleştirmelere referansları ve XAI bulgularını (özellikle Klinik Karar Özeti) içerecek.
    *   `README.md` dosyası oluşturularak projenin genel bir özeti ve nasıl çalıştırılacağına dair talimatlar sağlanacak.