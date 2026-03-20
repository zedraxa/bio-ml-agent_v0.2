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
- [x] **Resmi Başlatma Yolu:** İki net senaryo tanımla:
    1. **Minimal Local Mode:** En az bağımlılıkla hızlı başlangıç.
    2. **Full Docker Mode:** Tüm servislerle (MLflow, Qdrant vb.) tam kapasite.
- [x] **Hardcoded Path Temizliği:** Web UI içindeki Node yolları gibi makineye özel tanımları config/env tabanlı hale getir.
- [x] **Sağlam Check Komutu:** `bio-ml-agent check` (veya `check_setup.py`) komutunu .env, python sürümü ve kritik bağımlılıkları denetleyecek şekilde güçlendir.

**Tamamlanma Ölçütü:**
- Temiz bir makinede sadece README izlenerek sistem ayağa kalkar.
- Senaryolar birbirine karışmaz.

---

## 📐 Faz 2 — Mimari Sadeleştirme
**Öncelik:** P1  
**Durum:** ✅ Tamamlandı  

### Yapılacaklar:
- [x] **Dizayn Standartlaştırma:** Çalışan kodun tamamı `src/bio_ml_agent` altına toplandı.
- [x] **Legacy Politikası:** `legacy/` klasörü dışındaki tüm core dosyalar modernize edildi.
- [x] **web_ui.py Modularization:** UI bileşenleri `ui/` altına (chat, session, whatsapp, explorer) ayrıştırıldı.
- [x] **AgentService Sınırları:** Routing ve lifecycle görevleri `services/agent/` alt modüllerine bölündü.

---

## 🧪 Faz 3 — Test ve Kalite Güvence
**Öncelik:** P1  
**Durum:** ✅ Tamamlandı  

### Yapılacaklar:
- [x] **Smoke Tests:** `tests/test_smoke.py` güncellendi.
- [x] **Integration Tests:** `tests/test_agent_integration.py` ile uçtan uca akış doğrulandı.
- [x] **Mocked Backend:** API anahtarı gerektirmeyen `MockBackend` eklendi.
- [x] **Sert CI:** GitHub Actions üzerine Lint + Test + Gitleaks + Docker pipeline kuruldu.

---

## 💎 Faz 4 — Ürünleşme ve Demo Kalitesi
**Öncelik:** P2  
**Durum:** ✅ Tamamlandı  

### Yapılacaklar:
- [x] **Golden Path Demo:** `examples/golden_path_demo` ile uçtan uca senaryo hazırlandı.
- [x] **Isolated Demo Workspace:** Demoların ana sistemi etkilememesi sağlandı.
- [x] **Değer Önerisi:** Biyomühendislik odağı ve XAI yetenekleri vurgulandı.
- [x] **Sürümleme:** `v0.1.0-clean` etiketleme sistemine geçildi.

---

## 🚀 Faz 5 — İleri Özellikler (Gerçekçi Sıralama)
**Öncelik:** P3  
**Durum:** ✅ Tamamlandı  

**Uygulama Sırası:**
1. [x] Core Chat + Tools + Project Saving (Auto-recovery & Checkpoints)
2. [x] RAG + Vector Store (Qdrant Entegrasyonu)
3. [x] Background Jobs / Queue (Redis)
4. [x] Audit / Observability
    - [x] Kritik eylem denetimi (BASH, WRITE_FILE vb.)
    - [x] Arka plan iş telemetry'si (Redis job linkleme)
    - [x] Merkezi LLM maliyet takibi (Unified OTel metrics)
    - [x] Observability API uç noktaları
5. [x] Gateway / Remote Mode (Proxy ve Auth katmanı)
6. [x] WhatsApp Gateway (Kullanıcı etkileşimi için)
7. [x] Temporal / Uzun Süreli İş Akışları
8. [x] Multi-agent Swarm Geliştirme (Deepening)
