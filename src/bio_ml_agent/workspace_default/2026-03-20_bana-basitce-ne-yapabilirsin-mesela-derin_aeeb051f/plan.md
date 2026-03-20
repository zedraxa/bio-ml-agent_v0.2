# Meme Kanseri Teşhisi Projesi Planı

Bu proje, Wisconsin Meme Kanseri veri setini kullanarak bir hastanın tümörünün iyi huylu (benign) mu yoksa kötü huylu (malignant) mu olduğunu tahmin eden bir makine öğrenmesi modeli geliştirmeyi amaçlamaktadır.

**Adımlar:**

1.  **Proje Yapısını Oluştur:** `data/raw`, `src`, `results/plots` gibi gerekli klasörleri oluştur.
2.  **Veri Setini İndir:** UCI Machine Learning Repository'den Wisconsin Breast Cancer (Diagnostic) veri setini indir.
3.  **Gereksinimleri Belirle:** Proje için gerekli Python kütüphanelerini (`scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `shap`) içeren bir `requirements.txt` dosyası oluştur ve yükle.
4.  **Model Eğitimi ve Karşılaştırması:** `src/train.py` script'i oluşturulacak. Bu script:
    *   Veriyi yükleyecek ve hazırlayacak.
    *   Veriyi eğitim ve test setlerine ayıracak.
    *   Lojistik Regresyon, Random Forest ve Gradient Boosting modellerini içeren bir karşılaştırma yapacak.
    *   Karşılaştırma sonuçlarını `results/comparison_report.md` ve `results/comparison_results.json` olarak kaydedecek.
    *   En iyi modeli `results/best_model.pkl` olarak kaydedecek.
5.  **Görselleştirme ve Açıklanabilirlik (XAI):** `src/generate_report_assets.py` script'i oluşturulacak. Bu script:
    *   En iyi modeli ve veriyi yükleyecek.
    *   Korelasyon matrisi, karışıklık matrisi, ROC eğrisi ve özellik önemi gibi grafikleri oluşturup `results/plots/` altına kaydedecek.
    *   SHAP kullanarak modelin kararlarını açıklayan özet grafiği oluşturacak ve `results/plots/` altına kaydedecek.
6.  **Nihai Raporu Oluştur:** Tüm bulguları, model performans tablosunu, grafikleri ve SHAP analizine dayalı "Klinik Karar Özeti"ni içeren bir `report.md` dosyası oluştur.
7.  **README Oluştur:** Projeyi ve nasıl çalıştırılacağını açıklayan bir `README.md` dosyası oluştur.
---