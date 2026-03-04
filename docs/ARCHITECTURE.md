# Bio-ML Agent (Ultra Ajan) Mimari Dokümantasyonu

Bu doküman, Bio-ML Agent v1.0'ın "Antigravity Tabanlı Ultra-Ajan" mimarisini teknik detaylarıyla açıklar.

## 🏗️ Genel Bakış
Bio-ML Agent, otonom veri bilimi ve biyoinformatik iş akışlarını yürütmek üzere tasarlanmış, çok katmanlı bir yapay zeka sistemidir.

## 🧱 Ana Katmanlar

### 1. Ultra Agent Core (`agent.py`)
Sistemin beynidir. LLM çıktılarını işler, araçları (PYTHON, BASH, BROWSER vb.) yönetir ve otonom döngüyü kontrol eder.

### 2. Orkestrasyon (`ultra_agent/orchestration/`)
- **LangGraph:** Planla-Uygula-Doğrula (Plan-Execute-Verify) döngüsünü yöneten state machine.
- **Temporal:** Uzun süreli, dayanıklı (durable) iş akışları. Kesintiye uğrayan işlerin kaldığı yerden devam etmesini sağlar.

### 3. Hafıza Katmanı (`ultra_agent/memory/`)
- **Qdrant:** Semantik vektör hafızası.
- **RAG Engine:** Doküman indeksleme ve arama.
- **Provenance:** Her bilginin kaynağını (dosya/URL) takip eden metaveri katmanı.
- **TTL Support:** Bellekteki verilerin otomatik yaşlandırılması ve temizlenmesi.

### 4. Güvenlik ve İzolasyon
- **SandboxRuntime:** Python kodlarının %50 CPU ve kısıtlı bellek ile izole çalıştırılması.
- **Vault:** Kimlik bilgilerinin (API key, şifre) Fernet ile şifrelenmiş olarak saklanması.
- **HITL (Human-in-the-Loop):** Kritik aksiyonlarda (silme, tehlikeli bash komutları) insan onayı gerekliliği.

### 5. Gözlemlenebilirlik (`ultra_agent/observability/`)
- **OpenTelemetry:** Detaylı işlem takibi (traces).
- **Prometheus:** Performans metrikleri (maliyet, hız, hata oranları).
- **Audit Trail:** Değiştirilemez denetim izleri (JSONL formatında).

## 🧬 Çoklu Ajan (Swarm) Yapısı
`swarm/` klasörü altında bulunan uzmanlar:
- **Data Engineer:** Veri temizleme ve hazırlama.
- **ML Expert:** Model seçimi, eğitim ve hiperparametre optimizasyonu.
- **Bioinfo Expert:** Klinik ve biyolojik veri yorumlama.

## 🐳 Dağıtım (Container Hardening)
Docker Compose üzerinde 7 servis çalışır. Worker konteyneri `no-new-privileges` ve `seccomp: default` profili ile sıkılaştırılmıştır.
