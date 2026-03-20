# Diyabet Tahmin Projesi

Bu proje, Pima Kızılderilileri Diyabet Veri Kümesi (scikit-learn'den uyarlanmış diyabet veri kümesi kullanılarak) üzerinde diyabeti tahmin etmek için bir makine öğrenimi çözümü geliştirir. Veri ön işleme, çoklu model eğitimi, görselleştirme ve açıklanabilir yapay zeka (XAI) tekniklerini içerir.

## Proje Yapısı
```
├── data/
│   └── raw/                # Ham veri (varsayımsal olarak buraya indirilirdi)
├── results/
│   ├── best_model.pkl      # En iyi modelin kaydedilmiş hali
│   ├── comparison_results.json # Model karşılaştırma sonuçları (JSON)
│   ├── comparison_report.md  # Model karşılaştırma ve XAI raporu
│   └── plots/              # Oluşturulan tüm grafikler ve XAI görselleri
├── src/
│   └── train.py            # Ana eğitim ve değerlendirme scripti
├── utils/
│   ├── model_compare.py    # Modelleri karşılaştırmak için yardımcı script
│   ├── visualize.py        # Görselleştirmeler için yardımcı script
│   ├── xai_engine.py       # Açıklanabilir Yapay Zeka (SHAP, LIME) için yardımcı script
│   ├── preprocessor.py     # Veri ön işleme için yardımcı script
│   └── model_loader.py     # Kaydedilmiş modelleri yüklemek için yardımcı script
├── requirements.txt        # Python bağımlılıkları
└── README.md               # Proje açıklaması ve kullanım talimatları
```

## Kurulum
1. Depoyu klonlayın:
   `git clone <depo-url>`
   `cd <proje-adı>`
2. Gerekli Python bağımlılıklarını yükleyin:
   `pip install -r requirements.txt`

## Kullanım
Model eğitimini ve değerlendirmesini çalıştırmak için aşağıdaki komutu kullanın:
`python src/train.py`

Bu script şunları yapacaktır:
- scikit-learn'ün 'diabetes' veri kümesini yükler ve ikili sınıflandırma problemine dönüştürür.
- Veri kalitesi analizi ve ön işleme yapar.
- Çeşitli makine öğrenimi modellerini eğitir ve karşılaştırır.
- Performans metriklerini hesaplar ve en iyi modeli kaydeder.
- Çeşitli veri ve model görselleştirmeleri oluşturur.
- SHAP ve LIME kullanarak modelin tahminlerini açıklar.
- `results/` dizininde bir rapor (`comparison_report.md`) ve çizimler (`plots/`) oluşturur.

## Sonuçlar
Tüm model karşılaştırma sonuçları `results/comparison_results.json` dosyasında bulunabilir. Detaylı rapor ve XAI bulguları `results/comparison_report.md` adresindedir. Oluşturulan tüm grafikler `results/plots/` dizininde mevcuttur.

## Model Yükleme ve Tahmin Etme
Kaydedilen en iyi modeli yüklemek ve yeni veriler üzerinde tahmin yapmak için:
```python
from utils.model_loader import load_and_predict
import pandas as pd

# Yeni verilerinizi hazırlayın (eğitimde kullanılan özellik adlarıyla aynı olmalı)
X_new = pd.DataFrame([[...]]) # Örnek yeni veri

# Modeli yükle ve tahmin yap
predictions = load_and_predict('results/best_model.pkl', X_new)
print(predictions)
```
