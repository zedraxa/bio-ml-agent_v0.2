# Bio-ML Agent — Architecture Documentation (v0.1.0-clean)

Bio-ML Agent is a **modular, layered** AI system designed to execute autonomous data science and bioinformatics workflows.

## 🏗️ Overview

The system was refactored from a monolithic structure to a **Facade Pattern** with a modular service architecture. Each component (UI, Core, Service) can be developed and tested independently.

## 🧱 Main Layers

### 1. Core — `src/bio_ml_agent/core/`
The **execution** layer of the system.
- **`agent_core.py`**: Prompt management and LLM routing logic.
- **`tools.py`**: Secure (sandboxed) execution environment for tools such as PYTHON, BASH, and BROWSER.
- **`conversation.py`**: Checkpoint-backed session and message history management.

### 2. Services — `src/bio_ml_agent/services/`
The **orchestration** layer of the system.
- **`agent_service.py`**: Single entry point (Facade) for the UI and API.
- **`agent/`**:
  - `orchestration`: Model selection and routing management.
  - `memory_context`: RAG and context compression.
  - `project_lifecycle`: Automatic project directory and metadata management.
  - `execution_policy`: Security and approval policies.

### 3. Interfaces — `src/bio_ml_agent/ui/`
The **interaction** layer of the system.
- **Gradio Web UI**: Modularized web interface (chat, session, whatsapp, explorer) launched via `web_ui.py`.
- **WhatsApp Gateway**: Messaging interface designed for remote management.

## 🧪 Quality & Security
- **MockBackend**: Test backend that eliminates API costs during development.
- **Auto-Recovery**: Automatic recovery of interrupted analyses via the `checkpoint.json` mechanism.
- **Strict CI**: Every change must pass lint and tests on GitHub Actions before merge.

---

*Last Updated: March 20, 2026*

---

## 🇹🇷 Türkçe Mimari Özeti (Turkish)

Bio-ML Agent, **Facade Pattern** ve modüler servis mimarisi kullanan katmanlı bir AI sistemidir.

- **Core (`src/bio_ml_agent/core/`)**: Prompt yönetimi, araç icra ortamı ve oturum geçmişi.
- **Services (`src/bio_ml_agent/services/`)**: Orkestrasyon, RAG bağlamı, proje yaşam döngüsü ve güvenlik politikaları.
- **Interfaces (`src/bio_ml_agent/ui/`)**: Gradio Web UI ve WhatsApp Gateway.

Kalite: MockBackend (sıfır API maliyeti), Auto-Recovery (checkpoint.json) ve Katı CI pipeline.
