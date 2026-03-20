# 🗺️ Bio-ML Agent — Geliştirme & Ürünleşme Yol Haritası (V2)

Bu plan, Bio-ML Agent'ı "çalışan bir prototip" seviyesinden "güven veren, profesyonel bir ürün" seviyesine taşımak için hazırlanmıştır.

---

## 🛠️ Faz 0 — Merge Blocker Temizliği
**Öncelik:** Hemen  
**Amaç:** Repoyu dışarıdan bakıldığında güven veren hale getirmek.

### Yapılacaklar:
- [x] **Repo Temizliği:** `browser_artifacts/`, `local_node/`, log/db çıktıları, üretilen raporlar (`research/`), trace/video dosyaları ve geçici binary'ler repodan çıkarılsın.
- [x] **.gitignore Sertleştir:** Workspace, qdrant data, mlflow db, audit/log, screenshots, videos, temp exports, node/vendor benzeri tüm klasörler net bir şekilde ignore edilsin.
- [x] **README Revizyonu:** Kurulum, tek komutla çalıştırma, mimari özet, ekran görüntüleri ve "hangi mod ne işe yarıyor" bölümleri yeniden düzenlensin.
- [x] **docs/STATUS.md:** Branch'lerin amacını açıklayan bir durum dokümanı ekle. (Experimental'de ne tamam, ne yarım, ne demo amaçlı açıkça yazılsın).

**Tamamlanma Ölçütü:**
- [x] Repo boyutu ciddi oranda düşer. (Not: 9GB'lık alan sadece yerel `.venv` dizinindedir ve `.gitignore` kapsamındadır. Tracked dosyalar temizlenmiştir.)
- [x] GitHub ana sayfası temiz görünür.
- [x] Neyin ürün, neyin çıktı dosyası olduğu netleşir.

---

## 🧱 Faz 1 — Çalışabilirlik ve Kurulum Stabilizasyonu
**Öncelik:** P0  
**Amaç:** "Bende çalışıyor" seviyesinden çıkmak.

### Yapılacaklar:
- [x] **Docker Healthcheck:** Dockerfile içindeki `/health` kontrolünü standardize et veya mevcut endpoint'lere göre güncelle.
- [ ] **Resmi Başlatma Yolu:** İki net senaryo tanımla:
    1. **Minimal Local Mode:** En az bağımlılıkla hızlı başlangıç.
    2. **Full Docker Mode:** Tüm servislerle (MLflow, Qdrant vb.) tam kapasite.
- [ ] **Hardcoded Path Temizliği:** Web UI içindeki Node yolları gibi makineye özel tanımları config/env tabanlı hale getir.
- [ ] **Deep Health Check:** `bio-ml-agent check` komutunu; import, env, port ve opsiyonel servis kontrollerini içerecek şekilde güçlendir.

**Tamamlanma Ölçütü:**
- Temiz bir makinede sadece README izlenerek sistem ayağa kalkar.
- Senaryolar birbirine karışmaz.

---

## 📐 Faz 2 — Mimari Sadeleştirme
**Öncelik:** P1  
**Amaç:** Doğru fikri koruyup karmaşıklığı azaltmak.

### Yapılacaklar:
- [ ] **Dizayn Standartlaştırma:** Çalışan kodun tamamını `src/bio_ml_agent` altına topla, kök dizinde sadece ince launcher dosyaları kalsın.
- [ ] **Legacy Politikası:** `legacy/` klasörü için net bir temizlik/referans takvimi belirle.
- [ ] **web_ui.py Modularization:** Dev dosyayı şu modüllere böl:
    - `ui/chat_handlers.py`
    - `ui/session_handlers.py`
    - `ui/whatsapp.py`
    - `ui/data_explorer.py`
- [ ] **AgentService Sınırları:** Routing, memory compression, project bootstrap gibi görevleri alt modüllere ayır:
    - `orchestration`, `memory_context`, `execution_policy`, `project_lifecycle`.
- [ ] **Capability Alignment:** Dokümandaki yetenek matrisi ile koddaki özellikleri eşle.

**Tamamlanma Ölçütü:**
- Yeni bir geliştirici 15-20 dakikada akışı anlar.
- UI / Service / Core sınırları netleşir.

---

## 🧪 Faz 3 — Test ve Kalite Güvence
**Öncelik:** P1  
**Amaç:** "Bozuldu mu?" sorusunu otomatik cevaplayabilmek.

### Yapılacaklar:
- [ ] **Smoke Tests:** Uygulama import oluyor mu? Proje klasörü oluşuyor mu? Temel tool'lar çalışıyor mu?
- [ ] **Integration Tests:** `AgentService` -> Proje bootstrap, Session save/load, UI message flow.
- [ ] **Mocked Backend:** Gerçek API key gerektirmeden tool döngüsünü test eden mock sistemleri.
- [ ] **Sert CI:** Lint + Unit + Smoke + Secret Scan adımlarını içeren CI pipeline.

**Tamamlanma Ölçütü:**
- Main branch'e merge öncesi minimum güvenlik ağı oluşur.

---

## 💎 Faz 4 — Ürünleşme ve Demo Kalitesi
**Öncelik:** P2  
**Amaç:** Sunum başarısını ve profesyonelliği artırmak.

### Yapılacaklar:
- [ ] **Golden Path Demo:** Tek veri seti ve prompt ile; analiz, eğitim, XAI ve rapor çıktısını uçtan uca gösteren senaryo.
- [ ] **Isolated Demo Workspace:** Repo içine veri düşürmeyen, `examples/` altındaki temiz örnekler.
- [ ] **Görsel README:** Ekran görüntüleri, GIF'ler veya kısa tanıtım videoları.
- [ ] **Değer Önerisi (Differentiation):** Bioengineering odaklı farkları (local-first, explainability vb.) net vurgula.
- [ ] **Sürümleme (Versioning):** `v0.x.x-clean` gibi anlamlı tag sistemine geç.

---

## 🚀 Faz 5 — İleri Özellikler (Gerçekçi Sıralama)
**Öncelik:** P3  
**Amaç:** Servisleri doğru sırayla sisteme dahil etmek.

**Uygulama Sırası:**
1. Core Chat + Tools + Project Saving
2. RAG + Vector Store (Qdrant)
3. Background Jobs / Queue (Redis)
4. Audit / Observability
5. Gateway / Remote Mode
6. WhatsApp Gateway
7. Temporal / Long Workflows
8. Multi-agent Swarm Deepening

---
> **Not:** Önce çekirdek değer önerisi kusursuz olmalı, sonra yan kanallar (WhatsApp vb.) gelmeli.
