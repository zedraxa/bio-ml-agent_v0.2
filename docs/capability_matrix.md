# 📑 Bio-ML Agent Capability Matrix (v0.1.0-clean)

Sistemin modüler yapısı ve mevcut yeteneklerinin uygulama katmanlarıyla eşleşmesi aşağıdadır.

## 🏗️ Core Architecture & Services

| Yetenek | Uygulama Modülü | Durum | Açıklama |
| :--- | :--- | :--- | :--- |
| **Orchestration** | `services.agent.orchestration` | ✅ | Mesaj döngüsü, model yönlendirme ve router mantığı. |
| **Memory Management** | `services.agent.memory_context` | ✅ | Bağlam daraltma, briefing ve RAG hazırlığı. |
| **Execution Policy** | `services.agent.execution_policy` | ✅ | Onay mekanizması ve güvenlik kısıtlamaları. |
| **Project Lifecycle** | `services.agent.project_lifecycle` | ✅ | Otomatik proje klasörü, metaveri ve checkpoint yönetimi. |
| **Auto-Recovery** | `services.agent_service.py` | ✅ | Kesinti sonrası son projeden otomatik devam etme. |
| **Unified Facade** | `services.agent_service.py` | ✅ | UI ve dış sistemler için temiz API katmanı. |

## 🧪 Intelligence & Research

| Yetenek | Uygulama Modülü | Durum | Açıklama |
| :--- | :--- | :--- | :--- |
| **Deep Research** | `core/tools.py` (BROWSER) | ✅ | Iterative web araması ve veri toplama. |
| **Mock Simulation** | `llm_backend.py` (MockBackend) | ✅ | API anahtarı olmadan tool döngüsü testi. |
| **Model Routing** | `services.agent.orchestration` | ✅ | Görev zorluğuna göre dinamik model seçimi. |
| **Observability** | `ultra_agent/metrics/` | 🟡 | Performans ve audit log takibi (Geliştirme aşamasında). |

## 📱 User Interfaces

| Yetenek | Uygulama Modülü | Durum | Açıklama |
| :--- | :--- | :--- | :--- |
| **Web UI** | `web_ui.py`, `ui/` | ✅ | Gradio tabanlı ana asistan arayüzü. |
| **Data Explorer** | `ui/data_explorer.py` | ✅ | Workspace içindeki verilerin anlık görselleştirilmesi. |
| **WhatsApp Bridge** | `ui/whatsapp.py` | 🟡 | Uzaktan yönetim altyapısı (Konfigürasyon gerektirir). |

---
*Son Güncelleme: 20 Mart 2026*
