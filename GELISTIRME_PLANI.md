# 🗺️ Bio-ML Agent — İyileştirme & Ürünleşme Yol Haritası

> **Tarih:** 1 Mart 2026  
> **Mevcut Durum:** v3 — Çekirdek (P0) ve Sprint 1-3 (Enterprise RAG/API) tamamlandı. Sprint 4 (Model Intelligence & DX) çalışmaları başlatıldı.
> **Yeni Hedef:** Bio-ML Agent'ı "çalışan ve etkileyici demo" seviyesinden çıkarıp, kurulabilir, test edilebilir, güvenilir, ölçeklenebilir ve topluluk dostu açık kaynak ürün seviyesine taşımak.

## Başarı Kriterleri
- Temiz kurulumla tek komutta ayağa kalkma
- UI / API / WhatsApp / CLI arasında ortak çekirdek mantık (AgentService)
- Provider bağımsız multimodal mesaj modeli (MessageNormalizer)
- Kalıcı görev yönetimi ve izlenebilirlik (Job Queue)
- Geniş doküman/RAG kapsaması (PDF, DOCX)
- Güvenli plugin/tool çalıştırma modeli (Sandbox)
- Release, CI ve dokümantasyon disiplininin oturması

---

## 🔴 P0 — Stabilizasyon ve Ürün Çekirdeği
*Amaç: Kırılgan bağlantıları kaldırmak, kurulum ve entegrasyonları sağlamlaştırmak.*

### 1) Public Branch Senkronu ve Release Hijyeni (Tamamlandı)
- [x] main branch'in gerçekten güncel olduğundan emin ol.
- [x] README, repo tree, requirements ve tracked file durumunu doğrula.
- [x] İlk temiz durum için release yayınla (v0.3.0-alpha).
- *Bitti kriteri:* public GitHub görünümünde `.env`/`config.yaml` tracked değil; README gerçek durumu anlatıyor.

### 2) AgentService Çekirdeğini Çıkar
- Ortak iş akışını `web_ui.py` içinden ayırıp `agent_service.py` benzeri tek bir servis katmanına taşı.
- CLI, Gradio, FastAPI, WhatsApp bu servis katmanını kullansın.
- *Bitti kriteri:* hiçbir giriş noktası başka bir UI dosyasını import etmiyor; hepsi ortak servis çağırıyor.

### 3) MessageNormalizer / Multimodal Adapter Katmanı
- Tüm girişleri ortak formata dönüştür: text, image, audio, file, tool_result, system/context.
- Her backend için ayrı serializer yaz: Gemini adapter, OpenAI adapter, Anthropic adapter, Ollama/local adapter.
- *Bitti kriteri:* aynı kullanıcı mesajı tüm backend'lere provider-uyumlu biçimde aktarılıyor.

### 4) Dependency Profillerini Ayır
- `requirements.txt` yerine şu profile geç veya `pyproject.toml` + extras kullan:
  - `requirements/base.txt`, `requirements/ui.txt`, `requirements/cloud.txt`, `requirements/whatsapp.txt`, `requirements/dev.txt`
- *Bitti kriteri:* kullanıcı "sadece local", "cloud", "ui", "api" kurulumlarını ayrı yapabiliyor.

### 5) Config Sistemi: Örnek Dosya + Şema Doğrulama (Tamamlandı)
- [x] `config.example.yaml`, `.env.example` oluştur.
- [x] Runtime'da config doğrulaması ekle (Pydantic).
- *Bitti kriteri:* eksik env/config alanı varsa sistem anlaşılır hata veriyor.

### 6) Gradio 6 ve Structured History'yi Tam Sabitle
- Sadece `type="messages"` ile kalmayın; içerik bloklarını da tek standarda çek.
- Text-only ve multimodal history için ortak formatter yazın.
- *Bitti kriteri:* text, image, audio senaryoları için UI smoke test geçiyor.

### 7) Kurulum Smoke Test Matrisi ve CI Validation (Tamamlandı)
- [x] CI'da şu işleri çalıştır: import smoke, CLI smoke, web_ui boot smoke, FastAPI boot smoke, WhatsApp connector import.
- *Bitti kriteri:* Tüm test suite (349+ test) kararlı biçimde, opsiyonel kütüphaneler yokken dahi çalışıyor.

### 8) README / RAPOR / KULLANMA_KILAVUZU Tek Kaynak Disiplini
- Test sayıları, backend listesi, config örnekleri tek yerden türesin.
- Mümkünse otomatik badge üretimi veya docs sync script'i yaz.
- *Bitti kriteri:* aynı bilgi üç farklı dokümanda farklı görünmüyor.

---

## 🟡 P1 — Ölçeklenebilirlik, Güvenilirlik, Kurumsal Sağlamlık
*Amaç: Sistemin "tek makinede demo" sınırını aşıp, kalıcı ve gözlemlenebilir hale gelmesi.*

### 9) API Görev Sistemi: AgentService ile Background Queue (Tamamlandı)
- `api_server.py` içerisinde deep_learning modülü hardcode importlarından kurtarıldı.
- İstekler asenkron in-memory DB'ye (gelecekte Redis/PQ) yatırılıp, doğrudan **AgentService** aracılığıyla işleniyor.
- *Bitti kriteri:* Sunucu otonom olarak background'da AgentService çağırabiliyor ve task status dönüyor.

### 10) `api_server.py` Import ve Modül Yolu Temizliği (Tamamlandı)
- `from deep_learning import quick_train_cnn` çağrısı iptal edildi; iş `AgentService` otonom yeteneklerine devredildi.
- *Bitti kriteri:* CNN endpoint'i bağımlılıklardan arındırıldı, hatasız boot oluyor.

### 11) WhatsApp Katmanını UI'dan Ayır (Tamamlandı)
- `whatsapp_connector.py` artık `web_ui.process_message` yerine doğrudan `services.agent_service.AgentService` katmanını kullanıyor.
- Mesaj geçmişi oturum ID'si (sender_id) ile memory'de (ve diskte) esnekçe tutuluyor.
- *Bitti kriteri:* WhatsApp taşıyıcısı arayüzden (Gradio) koptu. Tam bir mikroservis yapısına evrildi.

### 12) RAG Ingestion Genişletmesi
- Desteklenecek dosyalar: PDF, DOCX, XLSX, PPTX, HTML, Markdown, CSV/TSV.
- Metadata ekleyin: source, page/sheet, section, chunk token count, mime type.
- *Bitti kriteri:* proje raporları ve laboratuvar dökümanları RAG'e alınabiliyor.

### 13) Hybrid Retrieval + Reranking
- Semantic + keyword + metadata filtreleme ve son aşamada reranker.
- *Bitti kriteri:* uzun rapor ve benzer başlıklı dosyalarda retrieval kalitesi gözle görülür artıyor.

### 14) Plugin Güvenliği
- Dinamik Python plugin yükleme için seçenekler: allowlist, imzalı/plugin manifest, subprocess sandbox, Docker/Firecracker izolasyonu.
- *Bitti kriteri:* untrusted plugin doğrudan ana process içinde keyfi kod yürütmüyor.

### 15) Gözlemlenebilirlik (Observability)
- Structured logging, Request/session/task correlation id, Prompt/tool latency, Provider error kodları.
- *Bitti kriteri:* "hangi kullanıcı isteği neden çöktü?" sorusu loglardan takip edilebiliyor.

### 16) Güvenlik Sıkılaştırması
- API auth, Rate limiting, CORS kısıtlaması, Webhook signature doğrulaması, Secret scanning.
- *Bitti kriteri:* public deployment için temel güvenlik checklist'i tamam.

### 17) Hata Modeli ve Kullanıcıya Dönük Hata Mesajları (Tamamlandı)
- [x] Tek tip exception hiyerarşisi: provider error, config error, tool execution error, ingestion error, validation error.
- *Bitti kriteri:* kullanıcı dostu hata + geliştirici dostu log aynı anda sağlanıyor.

---

## 🟢 P2 — Ürünleşme, Geliştirici Deneyimi ve Topluluk
*Amaç: Projeyi sadece çalışan sistem değil, sürdürülebilir açık kaynak ürün haline getirmek.*

### 18) Capability Registry
- Her model/provider için özellik matrisi tut: text, image, audio, file upload, streaming, tool use, context length.
- *Bitti kriteri:* sistem model seçimini capability'ye göre yapıyor; hardcoded tahminler azalıyor.

### 19) Evaluation / Benchmark Harness
- Aynı görev için: yanıt kalitesi, tool call doğruluğu, latency, cost, failure rate.
- *Bitti kriteri:* backend seçimi sezgisel değil ölçülebilir hale geliyor.

### 20) ML Reproducibility ve Experiment Tracking
- Dataset version, random seed, run config, artifact metadata, MLflow/W&B entegrasyonu.
- *Bitti kriteri:* aynı proje çıktısı tekrar üretilebiliyor.

### 21) Packaging ve Sürümleme
- pyproject.toml, console scripts, semantic versioning, changelog, release notes.
- *Bitti kriteri:* `pip install ...` ve sürüm takibi mümkün.

### 22) Dokümantasyon Portalı
- "Quickstart", "Architecture", "Providers", "RAG", "WhatsApp/API", "Troubleshooting".
- *Bitti kriteri:* yeni gelen bir geliştirici 15–20 dakikada sistemi anlayabiliyor.

### 23) Örnek Kullanım Paketleri (Demos)
- Hazır demo akışları: breast cancer classification, EEG/EMG analysis, wastewater quality prediction, medical image classification.
- *Bitti kriteri:* repo, yeteneklerini gösteren tekrar çalıştırılabilir örnekler içeriyor.

### 24) Topluluk ve Katkı Akışı
- CONTRIBUTING.md, issue template, PR template, code owners, roadmap labels.
- *Bitti kriteri:* dış katkı almak kolaylaşıyor.

### 25) Deployment Target’ları
- Docker Compose, Hugging Face Spaces / Gradio hosting, Railway / Render / VPS, self-hosted docs.
- *Bitti kriteri:* en az iki resmi deployment yolu dokümante edilmiş oluyor.

### 26) Kurumsal Özellik Seti (Enterprise)
- Çok kullanıcılı oturumlar, kullanıcı bazlı quota, proje bazlı erişim, audit trail, workspace isolation.
- *Bitti kriteri:* tek kullanıcı ajanından çok kullanıcılı platforma geçiş zemini oluşuyor.

---

> **En Kritik Mimari Karar:** "UI merkezli agent" yapısından, "çekirdek servis merkezli platform" yapısına geçiş. Bunu yaptıktan sonra WhatsApp kırılganlığı azalır, API güvenilirleşir, test yazmak kolaylaşır, provider uyumsuzlukları daha kolay çözülür ve ürünleşme gerçek anlamda başlar.

---

## 🚀 Sonraki Adım Önerileri (Sprint 4)

Testleri, config fallbacks'leri ve backend'leri başarıyla stabilen hale getirdikten ve CI test suite'ini %100 (%100 Pass / 349 test) yeşile çevirdikten sonra odağı yeni "Capability" ve "Use-case" inşasına kaydırmalıyız:

1. **Gözlemlenebilirlik (Observability) (Görev 15):** Ajanın karar alma şemasını, token kullanımını ve execution latency verilerini veritabanına loglayıp `dashboard.py` üzerinden göstermek.
2. **Evaluation / Benchmark Harness (Görev 19):** Ajanın ML analiz başarı oranlarını değerlendirecek ve "hangi model bu göreve uygun" otomatik seçecek Benchmark Harnes aracını entegre etmek.
3. **ML Reproducibility (MLflow) (Görev 20):** Halihazırda bulunan `mlflow_tracker.py` entegrasyonunu CLI argümanlarına bağlayıp model parametrelerini W&B veya MLFlow'a run bazlı pushlamak.
4. **Hazır Demo Akışları (Görev 23):** Kullanıcının `/demo` komutu ile breast-cancer gibi hazır iş akışlarını sıfır prompt ile tetikleyebilmesi.
