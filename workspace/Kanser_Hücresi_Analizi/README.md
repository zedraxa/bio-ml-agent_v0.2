# Kanser Hücresi Analizi Projesi

Bu proje, Wisconsin Meme Kanseri (Tanısal) veri setini kullanarak meme kanseri teşhisi için makine öğrenimi modelleri geliştirmeyi ve karşılaştırmayı amaçlamaktadır.

## Kurulum
Projeyi kurmak için aşağıdaki adımları izleyin:
```bash
git clone <repo-url>
cd <project-directory>
pip install -r scratch_project/requirements.txt
```

## Veri Seti
Kullanılan veri seti [UCI Machine Learning Repository - Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) adresinden temin edilmiştir. Veri seti, bir hücre çekirdeğinin dijitalleştirilmiş bir görüntüsünden hesaplanan 30 adet gerçek değerli özellik içerir ve teşhis (iyi huylu/kötü huylu) hedef değişkenidir.

## Proje Yapısı
```
scratch_project/
├── data/
│   └── raw/
│       └── breast_cancer_wisconsin.csv
├── src/
│   └── train.py
├── utils/
│   ├── model_compare.py
│   ├── visualize.py
│   ├── preprocessor.py
│   ├── model_loader.py
│   └── hyperparameter_optimizer.py
├── results/
│   ├── plots/              # Oluşturulan görseller buraya kaydedilir
│   ├── best_model.pkl      # Karşılaştırmadan elde edilen en iyi model
│   ├── best_optimized_*.pkl # Hiperparametre optimizasyonundan elde edilen en iyi model
│   ├── comparison_results.json # Tüm modellerin karşılaştırma sonuçları
│   ├── comparison_report.md # Model karşılaştırma raporu
│   ├── cv_results_*.csv    # Hiperparametre optimizasyonu çapraz doğrulama sonuçları
│   └── report.md           # Detaylı proje raporu
└── requirements.txt
```

## Kullanım
Projeyi çalıştırmak ve modelleri eğitmek için `train.py` betiğini çalıştırın:
```bash
python scratch_project/src/train.py
```
Bu betik, veri önişleme yapacak, birden çok modeli eğitecek, karşılaştıracak, en iyi model üzerinde hiperparametre optimizasyonu uygulayacak ve sonuçları `results/` dizinine kaydedecektir.

## Sonuçlar
Model karşılaştırma sonuçları ve optimize edilmiş model detayları `results/report.md` dosyasında bulunabilir. Görselleştirmeler `results/plots/` dizininde mevcuttur.
Özetle, yapılan karşılaştırmalar ve optimizasyonlar sonucunda **RandomForest** modelinin en iyi performansı gösterdiği tespit edilmiştir.
Optimize edilmiş parametreler:
- `model__n_estimators`: `50`
- `model__min_samples_split`: `2`
- `model__max_depth`: `10`

## Katkıda Bulunma
Geri bildirimler ve katkılar memnuniyetle karşılanır.

## Lisans
Bu proje MIT Lisansı altında lisanslanmıştır.
Eğitilmiş ve optimize edilmiş en iyi model **RandomForest**, `scratch_project/results/best_optimized_randomforest.pkl` adresine kaydedilmiştir.
