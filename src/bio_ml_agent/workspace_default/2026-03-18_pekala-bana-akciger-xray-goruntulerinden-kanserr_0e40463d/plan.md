# Akciğer X-ray Kanser Sınıflandırma Projesi Planı

## 1. Veri Seti Araştırması ve İndirme
*   Akciğer X-ray görüntüleri üzerinde kanser (malign/benign) sınıflandırması için uygun bir genel kullanıma açık veri seti bul.
*   Veri setini `data/raw/` dizinine indir ve organize et. Derin öğrenme araçları için sınıf bazlı alt dizin yapısı tercih edilir (örn: `data/raw/malignant`, `data/raw/benign`, `data/raw/no_cancer`).

## 2. Proje Yapısını Oluşturma
*   Gerekli dizinleri oluştur: `src/`, `data/raw/`, `results/`, `results/plots/`.
*   Python bağımlılıklarını içeren `requirements.txt` dosyasını oluştur.

## 3. Veri Ön İşleme ve Yükleme
*   İndirilen veri setini `MedicalCNN` veya `quick_train_cnn` aracının beklediği formata (sınıf alt dizinleri) getir. Gerekirse veri setini eğitim, doğrulama ve test kümelerine ayır.

## 4. Model Eğitimi (Derin Öğrenme)
*   Önceden eğitilmiş Evrişimsel Sinir Ağları (CNN) kullanarak transfer öğrenimi uygulayacağız.
*   `deep_learning.quick_train_cnn` aracını veya `MedicalCNN` sınıfını kullanarak birden fazla mimariyi (örneğin ResNet18, EfficientNet_B0) deneyeceğiz.
*   Model eğitimini başlat ve eğitim sürecini izle.

## 5. Model Değerlendirme ve Görselleştirme
*   Eğitilen modelin performansını test veri seti üzerinde değerlendir (doğruluk, hassasiyet, duyarlılık, F1 skoru, ROC AUC).
*   `utils.visualize.MLVisualizer` kullanarak karışıklık matrisi, ROC eğrisi gibi önemli grafikler oluştur ve `results/plots/` dizinine kaydet.

## 6. Açıklanabilir Yapay Zeka (XAI)
*   Modelin kararlarını anlamak için SHAP/LIME (veya Grad-CAM gibi CNN'e özel yöntemler) kullan.
*   `xai_engine.XAIEngine` aracılığıyla SHAP özet grafikleri oluştur ve `results/plots/` dizinine kaydet.
*   Elde edilen açıklamalardan yola çıkarak "Klinik Karar Özeti" hazırla.

## 7. Raporlama
*   Projenin özetini, kullanılan veri setini, model mimarisini, eğitim parametrelerini, performans metriklerini, görselleştirmeleri ve XAI çıktılarını içeren kapsamlı bir `results/comparison_report.md` dosyası oluştur.
*   Projenin nasıl çalıştırılacağını ve modelin nasıl kullanılacağını açıklayan bir `README.md` dosyası hazırla.