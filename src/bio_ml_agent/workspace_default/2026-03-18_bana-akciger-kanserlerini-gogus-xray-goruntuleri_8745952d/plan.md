# Akciğer Kanseri Sınıflandırma Projesi Planı

## 1. Veri Kümesi Araştırma ve İndirme
*   Göğüs röntgeni görüntülerinden akciğer kanseri sınıflandırması için uygun bir açık kaynak veri kümesi bul. (Örn: Kaggle, NIH Chest X-ray Dataset).
*   Veri kümesini `data/raw/` dizinine indir.
*   Veri yapısını `deep_learning` araç setinin beklediği formata göre düzenle (sınıf bazlı alt klasörler).

## 2. Proje Yapısı ve Bağımlılıklar
*   Temel proje dizinlerini oluştur: `data/raw/`, `src/`, `results/`.
*   `requirements.txt` dosyasını oluştur ve gerekli kütüphaneleri ekle (torch, torchvision, scikit-learn, matplotlib, seaborn, shap, lime vb.).

## 3. Derin Öğrenme Modeli Eğitimi
*   `deep_learning` aracını kullanarak önceden eğitilmiş (transfer öğrenme) en az 3 farklı CNN mimarisi (örneğin ResNet18, EfficientNetB0, DenseNet121) ile modelleri eğit.
*   Her model için eğitim metriklerini kaydet.
*   `deep_learning.compare_architectures` fonksiyonunu kullanarak modelleri karşılaştır.

## 4. Model Değerlendirme ve Görselleştirme
*   En iyi performans gösteren modeli belirle.
*   `MLVisualizer` kullanarak aşağıdaki görselleştirmeleri oluştur ve `results/plots/` dizinine kaydet:
    *   Karışıklık Matrisi (Confusion Matrix)
    *   ROC Eğrisi
    *   Sınıf Dağılımı
*   `XAIEngine` kullanarak en iyi model için SHAP özet grafiği oluştur ve `results/plots/` dizinine kaydet.
*   Örnek bir vaka için LIME açıklaması oluştur.

## 5. Raporlama ve Sonuçlar
*   `results/comparison_results.json` ve `results/best_model.pkl` dosyalarını kaydet (otomatik olarak `deep_learning` aracı tarafından yapılacaktır).
*   `report.md` dosyasını oluştur:
    *   Kullanılan veri kümesinin açıklaması.
    *   Eğitilen modeller ve performans metriklerinin karşılaştırmalı tablosu.
    *   Görselleştirmelere referanslar.
    *   XAI (SHAP) çıktılarına dayanarak "Klinik Karar Özeti"ni yaz.
    *   En iyi modelin nasıl yükleneceği ve kullanılacağına dair talimatlar.
*   `README.md` dosyasını oluştur.