# Contribution Guide

Bio-ML Agent is an open-source project and welcomes community contributions.

## 🧱 Code Structure

| Module | Path | Description |
| :--- | :--- | :--- |
| CLI entry point | `src/bio_ml_agent/legacy/agent.py` | `bio-ml-agent ui/api/chat/check` subcommands |
| LLM backends | `src/bio_ml_agent/llm_backend.py` | Ollama, OpenAI, Anthropic, Gemini adapters & `ModelCapability` registry |
| Tool engine | `src/bio_ml_agent/core/tools.py` | PYTHON, BASH, BROWSER_AGENT, WEB_SEARCH, and all tool tags |
| Mission Brain | `src/bio_ml_agent/brain/mission_brain.py` | Prompt → MissionPlan, agent DAG, fallback strategies |
| Agent service | `src/bio_ml_agent/services/agent_service.py` | Facade for UI/API interactions |
| Platform routes | `src/bio_ml_agent/routers/platform_routes.py` | REST API endpoints |
| Mission Packs | `src/bio_ml_agent/services/mission_pack_registry.py` | Research blueprints |
| Agent registry | `src/bio_ml_agent/brain/agent_registry.yaml` | Canonical agent metadata |
| Plugin manager | `src/bio_ml_agent/legacy/plugin_manager.py` | Auto-discovered tool plugins |
| Dataset catalog | `src/bio_ml_agent/legacy/dataset_catalog.py` | 18 built-in ML/bio datasets |

## 🛣 Roadmap
Future goals are tracked in [ROADMAP.md](../ROADMAP.md).

## 📣 Communication
Please use GitHub Issues for bug reports and feature requests.

---

## 🇹🇷 Türkçe Katkı Bilgisi (Turkish)

Bio-ML Agent açık kaynaklıdır. Hata bildirimi ve özellik önerileri için GitHub Issues kullanın. Proje hedefleri `ROADMAP.md` dosyasında takip edilebilir.
