# Proje Planı: Diyabet Tahmin Modeli

Bu proje, Pima Kızılderilileri Diyabet Veri Kümesi'ni kullanarak bir diyabet sınıflandırma modeli geliştirmeyi ve performansını değerlendirmeyi amaçlamaktadır. Proje, veri ön işleme, çoklu model karşılaştırması, görselleştirme ve açıklanabilir yapay zeka (XAI) adımlarını içerecektir.

## Adımlar:

1.  **Proje Yapısını Oluşturma:**
    *   Gerekli dizinleri (`data/raw`, `src`, `results`, `results/plots`) oluşturma.
    *   `requirements.txt` dosyasını oluşturma.
    *   Gerekli yardımcı araçları (`utils/model_compare.py`, `utils/visualize.py`, `xai_engine.py`, `utils/preprocessor.py`, `utils/model_loader.py`) oluşturma.

2.  **Veri Toplama:**
    *   Pima Kızılderilileri Diyabet Veri Kümesi'ni bir URL'den (`data/raw/diabetes.csv`) indirme.

3.  **Veri Ön İşleme ve Analiz:**
    *   `src/train.py` dosyasını oluşturma.
    *   Veriyi yükleme ve genel bir kalite analizi yapma.
    *   Eksik değerleri (genellikle 0 olarak kodlanmış) belirleme ve uygun bir stratejiyle doldurma (örneğin medyan).
    *   Özellikleri ölçekleme (StandardScaler).
    *   Veri setini eğitim ve test setlerine ayırma.

4.  **Model Eğitimi ve Karşılaştırması:**
    *   En az 3 farklı sınıflandırma modeli (Logistic Regression, RandomForestClassifier, GradientBoostingClassifier) eğiteceğim.
    *   `utils.model_compare.compare_models` kullanarak her modelin performansını 5 katlı çapraz doğrulama ile değerlendirme.
    *   Doğruluk (accuracy), kesinlik (precision), hatırlama (recall), F1 skoru ve ROC AUC gibi metrikleri hesaplama.
    *   Sonuçları `results/comparison_results.json` ve `results/comparison_report.md` dosyalarına kaydetme.
    *   En iyi performans gösteren modeli belirleme ve `results/best_model.pkl` olarak kaydetme.

5.  **Veri Görselleştirme:**
    *   `utils.visualize.MLVisualizer` kullanarak çeşitli grafikler oluşturma:
        *   Sınıf Dağılımı (Bar ve Donut grafikleri)
        *   Korelasyon Matrisi (Isı haritası)
        *   En iyi model için Karışıklık Matrisi (normalize edilmiş ve normal)
        *   ROC Eğrisi
        *   Özellik Önem Derecesi (Feature Importance)
    *   Tüm görselleştirmeleri `results/plots/` dizinine `.png` formatında kaydetme.

6.  **Açıklanabilir Yapay Zeka (XAI):**
    *   `xai_engine.XAIEngine` kullanarak en iyi model için SHAP (SHapley Additive exPlanations) değerlerini hesaplama ve görselleştirme.
    *   SHAP özet grafiği ve örnek LIME açıklamaları oluşturma.
    *   XAI sonuçlarını `results/plots/` dizinine `.png` formatında kaydetme.

7.  **Raporlama:**
    *   Proje sonuçlarını, model karşılaştırmalarını, önemli özellik çıkarımlarını ve XAI analizlerini içeren kapsamlı bir `results/comparison_report.md` raporu oluşturma.
    *   `Klinik Karar Özeti` bölümünde XAI bulgularını açıklama.
    *   Proje hakkında genel bilgi, kurulum ve kullanım talimatlarını içeren bir `README.md` dosyası oluşturma.