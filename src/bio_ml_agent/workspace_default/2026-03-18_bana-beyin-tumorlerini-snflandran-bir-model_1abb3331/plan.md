# Proje Planı: Beyin Tümörü Sınıflandırma Modeli

## Hedef
MRI görüntüleri kullanarak beyin tümörlerini (tümörlü/tümörsüz veya farklı tümör tipleri) sınıflandıran bir derin öğrenme modeli eğitmek.

## Adımlar

1.  **Veri Seti Bulma ve İndirme**:
    *   Kaggle veya benzeri platformlarda beyin tümörü MRI veri setlerini arayacağız.
    *   Uygun bir veri seti bulunduğunda, onu `data/raw/` dizinine indireceğiz.

2.  **Proje Yapısını ve Bağımlılıkları Oluşturma**:
    *   `requirements.txt` dosyasını oluşturacağız.
    *   Gerekli dizinleri (`src/`, `data/raw/`, `results/`, `results/plots/`) oluşturacağız.

3.  **Veri Ön İşleme ve Hazırlık**:
    *   İndirilen veri setini `deep_learning` modülünün beklediği formata (sınıf bazlı alt dizinler) dönüştüreceğiz.
    *   Veri setini eğitim ve doğrulama kümelerine ayıracağız.

4.  **Derin Öğrenme Modeli Eğitimi**:
    *   `deep_learning.quick_train_cnn` fonksiyonunu kullanarak önceden eğitilmiş bir CNN (örneğin, ResNet18 veya EfficientNet) modelini eğiteceğiz. `brain_mri` presetini kullanacağız.
    *   Birden fazla mimariyi karşılaştırmak için `compare_architectures` da kullanabiliriz.

5.  **Model Değerlendirme ve Görselleştirme**:
    *   Eğitim süreci sonunda elde edilen metrikleri (doğruluk, hassasiyet, duyarlılık, F1 skoru, ROC AUC) kaydedeceğiz.
    *   `MLVisualizer` kullanarak öğrenme eğrileri, karışıklık matrisi ve ROC eğrileri gibi grafikleri oluşturup `results/plots/` altına kaydedeceğiz.

6.  **Açıklanabilir Yapay Zeka (XAI)**:
    *   Eğitilen modelin kararlarını anlamak için `XAIEngine` kullanarak SHAP veya LIME gibi yöntemlerle açıklayıcı görseller oluşturacağız.
    *   `Klinik Karar Özeti` bölümünde bu bulguları açıklayacağız.

7.  **Raporlama ve Belgeleme**:
    *   Projenin çıktılarını ve bulgularını içeren detaylı bir `report.md` dosyası oluşturacağız.
    *   Modelin nasıl kullanılacağına dair talimatlar içeren bir `README.md` dosyası yazacağız.