# 📑 Bio-ML Agent Capability Matrix (v0.1.0-clean)

The system's modular structure and how its capabilities map to implementation layers.

## 🏗️ Core Architecture & Services

| Capability | Implementation Module | Status | Description |
| :--- | :--- | :--- | :--- |
| **Mission Orchestration** | `brain/mission_brain.py` | ✅ | Prompt → MissionPlan + agent DAG + success criteria. |
| **Memory Management** | `services/agent/memory_context.py` | ✅ | Context compression, briefing, and RAG preparation. |
| **Execution Policy** | `services/agent/execution_policy.py` | ✅ | Approval mechanism and security constraints. |
| **Project Lifecycle** | `services/agent/project_lifecycle.py` | ✅ | Automatic project directory, metadata, and checkpoint management. |
| **Auto-Recovery** | `brain/recovery_manager/` + `brain/checkpoint_store/` | ✅ | Automatic resume from the last checkpoint after an interruption. |
| **Unified Facade** | `services/agent_service.py` | ✅ | Clean Facade API layer for Gradio UI, CLI, and REST API. |
| **Artifact Graph** | `brain/artifact_graph.py` | ✅ | Full provenance tracking of agent-produced outputs. |
| **Mission Pack Registry** | `services/mission_pack_registry.py` | ✅ | Research blueprint library (`repo_review_pack`, `lab_report_pack`, `microscopy_pack`). |

## 🧪 Intelligence & Research

| Capability | Implementation Module | Status | Description |
| :--- | :--- | :--- | :--- |
| **Tool Engine** | `core/tools.py` | ✅ | PYTHON, BASH, BROWSER, WEB_SEARCH, CLINICAL_VISION, SWARM, and more. |
| **Deep Research** | `core/tools.py` (DEEP_RESEARCH tag) | ✅ | Iterative web search and data gathering. |
| **Mock / Offline Mode** | `llm_backend.py` (MockBackend) | ✅ | Tests the tool loop without API key consumption. |
| **Model Routing** | `services/agent/orchestration.py` | ✅ | Dynamic model selection based on task type and capability (vision, audio, tool use). |
| **Observability** | `services/observability/` + `ultra_agent/observability/` | ✅ | OpenTelemetry request correlation and JSON latency logging. |
| **RAG / Vector Memory** | `legacy/ultra_agent/rag/` + Qdrant | ✅ | Hybrid semantic + BM25 retrieval over workspace documents. |

## 🤖 Agent Families

| Family | Directory | Tier | Example Agents |
| :--- | :--- | :--- | :--- |
| **Browser** | `agents/browser/` | STABLE | `browser_scout`, `autopilot`, `executor`, `verifier` |
| **Coder** | `agents/coder/` | STABLE | `scientific_python`, `bioinformatics_coder`, `code_architect`, `refactor_repair` |
| **Critic** | `agents/critic/` | STABLE | `consistency_critic`, `adversarial_engine` |
| **Dataset** | `agents/dataset/` | STABLE | `dataset_agent`, `statistical_engine` |
| **Document** | `agents/document/` | STABLE | `document_agent`, `citation_mapper`, `table_extractor` |
| **Biology** | `agents/biology/` | ACTIVE | `histology_expert`, `cell_biology`, `mbg`, `biostatistics`, `wet_lab_protocol` |
| **Microscopy** | `agents/microscopy/` | ACTIVE | `perception_agent`, `identifier_agent`, `morphometrics_agent`, `report_agent` |
| **Academic** | `agents/academic/` | BETA | `abstract_generator`, `lab_report_orchestrator`, `paper_orchestrator`, `reviewer_simulation` |
| **Omics** | `agents/omics/` | EXPERIMENTAL | Genomics and multi-omics pipelines |

## 📱 User Interfaces

| Capability | Implementation Module | Status | Description |
| :--- | :--- | :--- | :--- |
| **Web UI** | `src/bio_ml_agent/web_ui.py` (launched via `run_ui.py`) | ✅ | Gradio-based main assistant interface. |
| **Data Explorer** | `ui/data_explorer.py` | ✅ | Real-time visualization of CSV and Plotly HTML charts in workspace. |
| **XAI Tab** | `web_ui.py` (XAI section) | ✅ | Gallery of SHAP/LIME charts produced by the ML Expert. |
| **WhatsApp Bridge** | `services/whatsapp_connector.py` (Flask + Twilio) | ✅ | Remote management via WhatsApp (requires Twilio credentials). |

## 🔄 Legacy ML Pipeline

| Capability | Implementation Module | Status | Description |
| :--- | :--- | :--- | :--- |
| **Dataset Catalog** | `legacy/dataset_catalog.py` | ✅ | 18 built-in ML/bio datasets. |
| **Model Comparison** | `legacy/ml/model_compare.py` | ✅ | Multi-model training, 5-fold CV, leaderboard. |
| **XAI Engine** | `legacy/ml/xai_engine.py` | ✅ | SHAP and LIME analysis for trained models. |
| **Swarm Orchestrator** | `legacy/swarm/orchestrator.py` | ✅ | 6-agent swarm (DataEngineer, MLExpert, Bioinformatician, Researcher, InSilico, AcademicPublishing). |
| **Active Learning** | `legacy/swarm/active_learning_worker.py` | 🟡 | Background worker for continuous model retraining. |
| **Data Streams** | `legacy/data_streams/` | 🟡 | DB connector + Redis stream consumer. |

---

*Last Updated: April 2026*

---

## 🇹🇷 Türkçe (Turkish)

**Durum:** ✅ Tamamlandı | 🟡 Geliştirme Aşamasında

- **Çekirdek Mimari**: Mission Brain, Recovery, Artifact Graph, Mission Pack Registry — tamamlandı.
- **Ajan Aileleri**: 9 domain ailesi; Browser ve Coder STABLE, Biology ve Microscopy ACTIVE, Academic BETA.
- **Kullanıcı Arayüzleri**: Gradio Web UI, Data Explorer, XAI sekmesi, WhatsApp Bridge.
- **Legacy ML**: Dataset kataloğu (18 veri seti), model karşılaştırma, XAI, 6 ajanlı swarm.
