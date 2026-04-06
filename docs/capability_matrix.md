# 📑 Bio-ML Agent Capability Matrix (v0.1.0-clean)

The system's modular structure and how its capabilities map to implementation layers.

## 🏗️ Core Architecture & Services

| Capability | Implementation Module | Status | Description |
| :--- | :--- | :--- | :--- |
| **Orchestration** | `services.agent.orchestration` | ✅ | Message loop, model routing, and router logic. |
| **Memory Management** | `services.agent.memory_context` | ✅ | Context compression, briefing, and RAG preparation. |
| **Execution Policy** | `services.agent.execution_policy` | ✅ | Approval mechanism and security constraints. |
| **Project Lifecycle** | `services.agent.project_lifecycle` | ✅ | Automatic project directory, metadata, and checkpoint management. |
| **Auto-Recovery** | `services.agent_service.py` | ✅ | Automatic resume from the last project after an interruption. |
| **Unified Facade** | `services.agent_service.py` | ✅ | Clean API layer for UI and external systems. |

## 🧪 Intelligence & Research

| Capability | Implementation Module | Status | Description |
| :--- | :--- | :--- | :--- |
| **Deep Research** | `core/tools.py` (BROWSER) | ✅ | Iterative web search and data gathering. |
| **Mock Simulation** | `llm_backend.py` (MockBackend) | ✅ | Tests the tool loop without API key consumption. |
| **Model Routing** | `services.agent.orchestration` | ✅ | Dynamic model selection based on task complexity. |
| **Observability** | `ultra_agent/metrics/` | 🟡 | Performance and audit log tracking (in development). |

## 📱 User Interfaces

| Capability | Implementation Module | Status | Description |
| :--- | :--- | :--- | :--- |
| **Web UI** | `web_ui.py`, `ui/` | ✅ | Gradio-based main assistant interface. |
| **Data Explorer** | `ui/data_explorer.py` | ✅ | Real-time visualization of data in the workspace. |
| **WhatsApp Bridge** | `ui/whatsapp.py` | 🟡 | Remote management infrastructure (requires configuration). |

---

*Last Updated: March 20, 2026*

---

## 🇹🇷 Türkçe (Turkish)

**Yetenek Durumu Göstergesi:** ✅ Tamamlandı | 🟡 Geliştirme Aşamasında

- **Orchestration, Memory, Execution Policy, Project Lifecycle, Auto-Recovery, Facade**: Tüm servis katmanı yetenekleri tamamlandı.
- **Deep Research, Mock Simulation, Model Routing**: Araştırma zekası modülleri tamamlandı.
- **Web UI, Data Explorer**: Kullanıcı arayüzleri tamamlandı. WhatsApp Bridge yapılandırma gerektiriyor.
