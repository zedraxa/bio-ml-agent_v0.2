# Bio-ML Agent: Antigravity Tabanlı Ultra-Ajan Uygulama Yol Haritası

Bu yol haritası, Bio-ML Agent'ı "Antigravity Tabanlı Ultra-Ajan" mimarisine geçirmek için PDF raporundaki detaylı bulgulara dayanarak hazırlanmış 5 fazlı (versiyonlu) bir geliştirme planıdır.
Mevcut görev paketleri (A, B, C, D) temel alınarak üretime alma ve son optimizasyonlar için 5. bir faz (Faz E) eklenmiştir.

## Faz 0: Planlama ve Analiz (Tamamlandı)
- [x] **Görev 0-1:** PDF Raporunun analiz edilmesi ve 5 aşamalı yol haritasının çıkartılması.
- [x] **Görev 0-2:** Kapsamlı geliştirme planının (`implementation_plan.md` ve `task.md`) oluşturulması.


## Faz 1: Güvenlik, Repo Hijyeni ve Risk Kapatma (Versiyon 0.3) (Tamamlandı)
**Odak:** Mevcut güvenlik açıklarını kapatmak, log sızıntılarını önlemek ve temel sanal alan (sandbox) güvenliğini sağlamak.

- [x] **Görev S1-1:** Repo'da commit'li log dosyalarının (api_server.log vb.) temizlenmesi ve CI'a secret scanning eklenmesi.
- [x] **Görev S1-2:** `docker-compose.yml` içindeki `.env` mount riskinin analiz edilmesi ve kapatılması.
- [x] **Görev S1-3:** BASH/PYTHON kullanımı (`shell=True`) için risk analizi ve Bypass vektörlerinin kapatılması.
- [x] **Görev S1-4:** Dinamik Plugin sistemi (`exec_module`) için manifest, imza ve izin listesi (allowlist) tabanlı sertleştirme tasarımı.
- [x] **Görev S1-5:** WhatsApp Web otomasyonu için ToS/Uyum incelemesi, varsayılan kapalı mod ve API alternatifleri.
- [x] **Görev S2-1:** OpenHands yaklaşımıyla minimal Sandbox Runtime v0 Spike'ının oluşturulması (ayrı container/VM).
- [x] **Görev S2-2:** Düz metin (plaintext) risklerine karşı KMS ve izolasyonlu Vault / Kimlik Bilgisi tasarımı.
- [x] **Görev S2-3:** "Otonom API Edinimi" (headless browser / email-client) süreçleri için güvenlik politikası, human-in-the-loop (HITL) kapısı eklenmesi.

## Faz 2: Orkestrasyon ve Kontrol Düzlemi (Versiyon 0.4) (Tamamlandı)
**Odak:** RQ/Redis mimarisinden durable execution için Temporal ve LangGraph'a kademeli geçiş.

- [x] **Görev S3-1:** Mevcut RQ/Redis altyapısını bozmadan Temporal mimari tasarımı ve workflow planlarının çıkarılması.
- [x] **Görev S3-2:** Ajanlar için Cancel / Signal / Kill-Switch (Durdur/Devam Et/İptal) spesifikasyonunun tanımlanması.
- [x] **Görev S3-3:** RQ Bridge Uygulaması: Seçili bir RQ işinin (ör. indeksleme) Temporal'e native bir process olarak taşınması.
- [x] **Görev S3-4:** Temporal Event History üzerinden Replayability (tekrar oynatılabilirlik) test planı tasarlanması.
- [x] **Görev S4-1:** LangGraph Graph Topolojisinin kurgulanması (Plan -> Tool -> Verify -> Artifact döngüsü).
- [x] **Görev S4-2:** Human-in-the-loop (HITL) Middleware yazılması (policy'ye göre onay/red kapıları).
- [x] **Görev S4-3:** Her adımda Antigravity standardına uyumlu (plan, diff, test, screenshot) Artifact Standardizasyonu.
- [x] **Görev S4-4:** Comment-to-Iterate mekanizması (Kullanıcı veya system yorumunun (feedback loop) graph state'ine geri beslenmesi).

## Faz 3: Model Yönlendirme ve Akıllı RAG (Versiyon 0.5) (Tamamlandı)
**Odak:** Çoklu LLM gateway kurulumu, bütçe yönetimi ve belleğin Qdrant ile üretim düzeyine taşınması.

- [x] **Görev S5-1:** LiteLLM Gateway Kurulumu, Virtual key entegrasyonu ve DB ayarı.
- [x] **Görev S5-2:** Routing Politikası kodlaması (Opus 4.6 ana model, Sonnet/Yerel modeller ikincil).
- [x] **Görev S5-3:** Agent veya task bazlı Cost Tracking (Maliyet takibi) ve Request limit mekanizmaları.
- [x] **Görev S5-4:** Anthropic Data Residency (inference_geo) parametrelerinin aktif test edilmesi.
- [x] **Görev S6-1:** Memory (episodic/semantic/procedural vb.) katmanlarının şema ve TTL (redaction) kurallarının tasarlanması.
- [x] **Görev S6-2:** Qdrant kullanarak Metadata filtreli semantic memory v0 adaptörünün yazılması.
- [x] **Görev S6-3:** RAG dokümanları için Provenance (kaynak gösterimi) ve Dedup (aynı veriyi çoklu indekslemeyi önleme) özelliği.
- [x] **Görev S6-4:** BM25 ve Dense Vector (Qdrant) birleşimli Hibrit Arama mekanizmasının test edilmesi.

## Faz 4: Tarayıcı/Vizyon ve Gözlemlenebilirlik (Versiyon 0.6) (Tamamlandı)
**Odak:** Ajanın dış ağlarla olan etkileşimini Playwright üzerinden standartlaştırmak ve Telemetriyi devreye almak.

- [x] **Görev S7-1:** Playwright/Browser-Use tabanlı, login gerektirmeyen temel DOM-first Browser Driver uygulanması.
- [x] **Görev S7-2:** Vision Fallback ve Screenshot alma işlemlerinin doğrulanması (Ajan halüsinasyonlarını engellemek için göz kontrol katmanı).
- [x] **Görev S7-3:** Session ve Cookie yönetimi süreçlerinin tenant (izole profil) tabanlı yönetilmesi.
- [x] **Görev S7-4:** Browser Recording (video/screenshot) sonuçlarının UI'a "Standart Artifact" olarak basılması.
- [x] **Görev S8-1:** OpenTelemetry (Traces/Metrics/Logs) wrapper'ının kurulması (İlk etapta lokaldan dummy log akışı).
- [x] **Görev S8-2:** Prometheus Metric tasarımı (Örn. API hızı, Tool kullanım oranları, error-rate). Low-cardinality PII-free tagler.
- [x] **Görev S8-3:** Governance ve Audit Trails (Denetim İzi) mimarisinin kalıcı loglanması.
- [x] **Görev S8-4:** En riskli 5 tool operasyonunda "Approval Gates" entegrasyonu yapılması.

## Faz 5: Üretime Alma, Sürekli Entegrasyon ve Antigravity Entegrasyonu (Versiyon 1.0) (Tamamlandı)
**Odak:** Rapordaki çıktılara ek olarak sistemin tamamen Antigravity üzerinden kontrol edilebilir şekilde canlıya çıkması.

- [x] **Görev S9-1:** CI/CD Boru Hattının Kurulması: GitHub actions/.gitlab-ci üzerinden container hardening (seccomp vb.), test ve lint otomasyonu.
- [x] **Görev S9-2:** Güvenlik ve Uyumluluk Son Testleri: Ajanın izinsiz dosya okuyup okumadığının otomatik sızma testleri.
- [x] **Görev S9-3:** OpenClaw modelinde tek kullanıcı/tek trust boundary izolasyon testleri.
- [x] **Görev S9-4:** Ajan Kontrol Yüzeyi (Control Plane) Entegrasyonu: Ajanın durumunu, ne üzerinde çalıştığını dışarı aktaran Antigravity UI bağlantısı.
- [x] **Görev S9-5:** Uygulama dökümantasyonlarının güncellenmesi (Readme, Architecture.md vb.).0 sürüm paketinin sunulması.
