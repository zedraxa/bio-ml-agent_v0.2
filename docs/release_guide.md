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
2. **Değişiklik Günlüğü (Changelog):** `walkthrough.md` (Artifact) özetine yeni versiyon bilgilerini ekleyin.
3. **CI Doğrulaması:** GitHub Actions üzerindeki tüm testlerin (Lint, Unit, Smoke) geçtiğinden emin olun.
4. **Build:** `python3 -m build` ile paketleri oluşturun.
5. **Tagleme:** `git tag -a v0.1.0-clean -m "Phase 4: Productization Release"`

## 🧪 Reproducibility Checklist (Tekrar Üretilebilirlik)

Bir yayından önce şu kontrolleri yapın:
- [ ] `VERSION_DATASET` tüm örnek veri setleri için çalışıyor mu?
- [ ] MLflow Run ID'leri raporlarda görünüyor mu?
- [ ] `all` opsiyonel bağımlılıkları temiz bir venv'de kurulabiliyor mu?
- [ ] `eval_bench.py` skorları kabul edilebilir seviyede mi?

- [x] GitHub Actions üzerinden otomatik `pytest` ve `ruff` lint kontrolü.
- [ ] PyPI'ya otomatik yayınlama (Gelecek planı).
