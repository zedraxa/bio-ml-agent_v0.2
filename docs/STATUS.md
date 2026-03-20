# 📦 Proje Durumu ve Dal (Branch) Yapısı

Bio-ML Agent gelişim sürecini yönetmek için aşağıdaki dal yapısını kullanırız:

## 🌿 Ana Dallar

### `main`
- **Amaç:** Kararlı (stable), test edilmiş ve yayına hazır kod.
- **Durum:** En güncel özellikler (Deep Research dahil) buradadır.
- **Kullanım:** Son kullanıcılar ve demo sunumları için bu dalı kullanın.

### `experimental`
- **Amaç:** Yeni özelliklerin ilk denendiği, "kırılabilir" alan.
- **Durum:** Geliştirme süreci bittiğinde `main` dalına merge edilir.
- **Kullanım:** Yeni bir araç veya ajan yeteneği eklerken bu dal üzerinde çalışın.

## 🛠️ Mevcut Geliştirme Durumu (v0.1.0-clean)

- **Tamamlananlar:**
  - **Mimari:** Modüler UI ve Servis katmanı ayrıştırıldı (`src/bio_ml_agent`).
  - **QA:** `ruff` linting ve `pytest` tabanlı tam test suite (Smoke/Integration) kuruldu.
  - **CI:** GitHub Actions üzerine Lint+Test+Secret Scan pipeline eklendi.
  - **Demo:** Golden Path (Breast Cancer) senaryosu `examples/` altına eklendi.
  - **Core:** Checkpoint destekli otomatik oturum kurtarma (Auto-recovery) eklendi.

- **Devam Edenler (Faz 5):**
  - RAG + Qdrant (Bilimsel döküman arama ve indexleme)
  - Redis Background Worker (Uzun soluklu analizler için kuyruk sistemi)

---
*Son Güncelleme: 20 Mart 2026 (v0.1.0-clean)*
