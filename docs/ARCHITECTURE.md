# Bio-ML Agent — Mimari Dokümantasyonu (v0.1.0-clean)

Bio-ML Agent, otonom veri bilimi ve biyoinformatik iş akışlarını yürütmek üzere tasarlanmış, **Modüler ve Katmanlı** bir yapay zeka sistemidir.

## 🏗️ Genel Bakış

Sistem, monolitik yapıdan kurtarılarak **Facade Pattern** ve modüler servis mimarisine dönüştürülmüştür. Bu sayede her bileşen (UI, Core, Servis) birbirinden bağımsız olarak geliştirilebilir ve test edilebilir.

## 🧱 Ana Katmanlar

### 1. Core (Çekirdek) — `src/bio_ml_agent/core/`
Sistemin "İcra" katmanıdır.
-   **`agent_core.py`**: Prompt yönetimi ve LLM yönlendirme mantığı.
-   **`tools.py`**: PYTHON, BASH, BROWSER gibi araçların güvenli (sandbox) icra ortamı.
-   **`conversation.py`**: Checkpoint destekli oturum ve mesaj geçmişi yönetimi.

### 2. Services (Servisler) — `src/bio_ml_agent/services/`
Sistemin "Orkestrasyon" katmanıdır.
-   **`agent_service.py`**: UI ve API için tekil giriş noktası (Facade).
-   **`agent/`**: 
    -   `orchestration`: Model seçimi ve rota yönetimi.
    -   `memory_context`: RAG ve bağlam sıkıştırma.
    -   `project_lifecycle`: Otomatik proje klasörü ve metaveri yönetimi.
    -   `execution_policy`: Güvenlik ve onay politikaları.

### 3. Interfaces (Arayüzler) — `src/bio_ml_agent/ui/`
Sistemin "Etkileşim" katmanıdır.
-   **Gradio Web UI**: `web_ui.py` üzerinden başlatılan, modularize edilmiş (chat, session, whatsapp, explorer) web arayüzü.
-   **WhatsApp Gateway**: Uzaktan yönetim için tasarlanmış mesajlaşma arayüzü.

## 🧪 Kalite ve Güvenlik
-   **MockBackend**: Geliştirme sürecinde API maliyetini sıfıra indiren test backend'i.
-   **Auto-Recovery**: `checkpoint.json` mekanizması ile kesintiye uğrayan analizlerin otomatik kurtarılması.
-   **Sert CI**: GitHub Actions üzerinde her değişikliğin lint ve testlerden geçme zorunluluğu.

---
*Son Güncelleme: 20 Mart 2026*
