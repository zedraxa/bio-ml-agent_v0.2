# Pima Indian Diyabet Tahmini Projesi

Bu proje, Pima Indian Diabetes veri setini kullanarak diyabet teşhisi için makine öğrenimi modelleri geliştirmeyi ve karşılaştırmayı amaçlamaktadır. Proje, veri ön işleme, model eğitimi, performans değerlendirme, görselleştirme ve açıklanabilir yapay zeka (XAI) adımlarını içermektedir.

## Kurulum
1. Depoyu klonlayın:
   `git clone <repo_url>`
   `cd <project_directory>`
2. Gerekli kütüphaneleri yükleyin:
   `pip install -r requirements.txt`

## Veri Seti
Pima Indian Diabetes veri seti manuel olarak `data/raw/diabetes.csv` konumuna indirilmelidir. Örnek bir indirme linki: https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv

## Proje Yapısı
```
. # Proje Kök Dizini
├── data/
│   └── raw/             # Ham veri seti
├── results/
│   ├── plots/           # Görselleştirmeler ve XAI çıktıları
│   ├── best_model.pkl   # Eğitilmiş en iyi model
│   ├── comparison_results.json # Model karşılaştırma sonuçları
│   └── comparison_report.md # Detaylı proje raporu
├── src/
│   └── train.py         # Ana eğitim ve değerlendirme scripti
├── utils/
│   ├── model_compare.py # Model karşılaştırma fonksiyonları
│   └── visualize.py     # Görselleştirme fonksiyonları
├── xai_engine.py        # Açıklanabilir AI (SHAP, LIME) fonksiyonları
├── requirements.txt     # Python bağımlılıkları
└── README.md            # Bu README dosyası
```

## Kullanım
Tüm pipeline'ı çalıştırmak için:
```bash
python src/train.py
```
Bu komut, modelleri eğitecek, performanslarını karşılaştıracak, çeşitli görselleştirmeler ve XAI çıktıları üretecek ve `results/` dizininde bir rapor oluşturacaktır.

## Rapor
Detaylı analiz sonuçları ve klinik karar özeti `results/comparison_report.md` dosyasında bulunabilir.

## Geliştirici Notu
Bu proje, genel bir makine öğrenimi projesinin tüm temel adımlarını göstermektedir. Daha fazla iyileştirme için hiperparametre optimizasyonu, gelişmiş özellik mühendisliği veya farklı model mimarileri denenebilir.
