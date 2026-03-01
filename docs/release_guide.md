# Bio-ML Agent Release Strategy & Guide

Bu doküman, Bio-ML Agent projesinin versiyonlama, paketleme ve yayınlama süreçlerini açıklar.

## 🏷️ Semantic Versioning (SemVer)

Proje `MAJOR.MINOR.PATCH` formatını takip eder:
- **MAJOR:** Geriye dönük uyumsuz büyük mimari değişiklikler (v1.0.0 gibi).
- **MINOR:** Geriye dönük uyumlu yeni özellikler (Sprint tamamlamaları).
- **PATCH:** Hata düzeltmeleri ve küçük iyileştirmeler.

## 🌿 Dal (Branch) Yapısı

- `main`: Stabil, her zaman yayına hazır kod.
- `develop`: Aktif geliştirme dalı. Özellikler buraya merge edilir.
- `feature/*`: Yeni özellikler için geçici dallar.

## 🚀 Yayınlama Adımları (Release Process)

1. **Versiyon Güncelleme:** `pyproject.toml` içindeki `version` alanını güncelleyin.
2. **Değişiklik Günlüğü (Changelog):** `CHANGELOG.md` dosyasına (veya `walkthrough.md` özetine) yeni versiyon bilgilerini ekleyin.
3. **MLflow Doğrulaması:** Kritik ML özelliklerinin MLflow logları üzerinden doğruluğunu kontrol edin.
4. **Build:** `python -m build` ile `sdist` ve `wheel` paketlerini oluşturun.
5. **Tagleme:** `git tag -a v0.5.0 -m "Sprint 5: ML Lifecycle Release"`

## 🧪 Reproducibility Checklist (Tekrar Üretilebilirlik)

Bir yayından önce şu kontrolleri yapın:
- [ ] `VERSION_DATASET` tüm örnek veri setleri için çalışıyor mu?
- [ ] MLflow Run ID'leri raporlarda görünüyor mu?
- [ ] `all` opsiyonel bağımlılıkları temiz bir venv'de kurulabiliyor mu?
- [ ] `eval_bench.py` skorları kabul edilebilir seviyede mi?

## 🤖 CI/CD Entegrasyonu (Gelecek Planı)

- GitHub Actions üzerinden otomatik `pytest` ve `lint` kontrolü.
- PyPI'ya otomatik yayınlama (sadece stabil release'ler için).
