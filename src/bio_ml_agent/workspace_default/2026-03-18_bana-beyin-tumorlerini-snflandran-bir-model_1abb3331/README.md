# Beyin Tümörü Sınıflandırma Projesi (Geleneksel ML Yaklaşımı)

Bu proje, sistem ortamındaki kısıtlamalar nedeniyle simüle edilmiş bir veri seti üzerinde geleneksel makine öğrenimi modellerini kullanarak beyin tümörlerini sınıflandırmayı amaçlamaktadır.

## Proje Yapısı

.
├── data/
│   └── raw/
│       └── simulated_brain_tumor_features.csv # Simüle edilmiş özellik veri seti
├── results/
│   ├── best_model.pkl           # Eğitilmiş en iyi model (SVC)
│   ├── comparison_results.json  # Tüm modellerin karşılaştırma sonuçları
│   ├── comparison_report.md     # Model karşılaştırma raporu
│   ├── classification_report_simulated.txt # En iyi modelin sınıflandırma raporu
│   └── plots/                   # Görselleştirme çıktıları
│       ├── confusion_matrix.png
│       ├── normalized_confusion_matrix.png
│       ├── roc_curve.png
│       ├── class_distribution.png
│       ├── feature_importance.png
│       ├── shap_summary_plot.png
│       └── lime_explanation_instance.png
├── utils/                       # Yardımcı Python modülleri
│   ├── model_compare.py
│   ├── visualize.py
│   └── model_loader.py
├── xai_engine.py                # Açıklanabilir AI modülü
├── deep_learning.py             # (Kullanılmadı, PyTorch hatası nedeniyle atlandı)
├── requirements.txt             # Proje bağımlılıkları
├── plan.md                      # Proje geliştirme planı
└── report.md                    # Proje sonuç raporu

## Kurulum

1.  **Gereksinimleri Yükle**:

    pip install -r requirements.txt

## Veri Seti

Sistem kısıtlamaları nedeniyle gerçek bir MRI veri seti indirilememiştir. Bunun yerine, 4 sınıfı temsil eden 20 özellikli 1000 örnekten oluşan sentetik bir tabular veri seti (`data/raw/simulated_brain_tumor_features.csv`) kullanılmıştır.

## Model Eğitimi ve Karşılaştırması

Proje kapsamında `LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`, `SVC` ve `KNeighborsClassifier` modelleri karşılaştırılmıştır. En iyi performansı **SVC** modeli göstermiştir.

Tüm model karşılaştırma sonuçları `results/comparison_results.json` ve `results/comparison_report.md` dosyalarında bulunabilir.

## Sonuçlar ve Değerlendirme

Modelin performans metrikleri ve çeşitli görselleştirmeler (karışıklık matrisi, ROC eğrisi, sınıf dağılımı, özellik önemi) `results/` ve `results/plots/` dizinlerinde bulunmaktadır. Detaylı sınıflandırma raporu ve klinik karar özeti `report.md` dosyasında yer almaktadır.

## Açıklanabilir Yapay Zeka (XAI)

Modelin karar verme sürecini anlamak için SHAP ve LIME yöntemleri kullanılmıştır. Bu görseller ve klinik karar özeti `report.md` dosyasında bulunabilir.

*   `shap_summary_plot.png`: SHAP ile özellik önem özet grafiği.
*   `lime_explanation_instance.png`: LIME ile tek bir örnek için açıklama.

## Model Kullanımı

Eğitilmiş en iyi model (`results/best_model.pkl`) ile yeni bir veri örneği üzerinde tahmin yapmak için `report.md` dosyasındaki "Model Kullanımı" bölümüne bakınız.