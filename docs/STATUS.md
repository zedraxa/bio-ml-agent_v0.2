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

## 🛠️ Mevcut Geliştirme Durumu (v0.2.x)

- **Tamamlananlar:**
  - Deep Research (Iterative Web Research)
  - Core Tool Integration
  - Message Normalization
  - Docker Compose Setup

- **Devam Edenler (Faz 1-2):**
  - Kurulum stabilizasyonu (healthchecks)
  - Mimari sadeleştirme (web_ui.py modularization)
  - Kapsamlı test suite (Smoke/Integration)

---
*Son Güncelleme: 20 Mart 2026*
