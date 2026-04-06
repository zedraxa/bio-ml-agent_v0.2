# Bio-ML Agent Enterprise

Autonomous research platform for bio-informatics, microscopy analysis, and academic publishing.

## 🚀 Quick Start

### 1. Requirements
- Python 3.9+
- Redis (optional, for background jobs)

### 2. Run the API Server
```bash
python run_api.py
```
The API is served at `http://localhost:8001`. Interactive API docs at `/docs`.

Alternatively, use the CLI entry point:
```bash
bio-ml-agent api
```

### 3. Run the Web UI
```bash
python run_ui.py        # or: bio-ml-agent ui
```
Open `http://localhost:7860` in your browser.

### 4. Run Golden Scenario Tests
Verify all primary workflows (Repo Review, Lab Report, Microscopy Swarm):
```bash
PYTHONPATH=src python -m pytest tests/test_golden_scenarios.py -vv
```

## 📂 Project Structure
| Directory | Description |
| :--- | :--- |
| `src/bio_ml_agent/api/` | FastAPI REST API server (`api_server.py`, `gateway_server.py`) |
| `src/bio_ml_agent/routers/` | API route handlers (`platform_routes.py`, `dashboard_routes.py`) |
| `src/bio_ml_agent/brain/` | Mission Brain — planning, execution DAG, recovery, checkpointing |
| `src/bio_ml_agent/services/` | AgentService facade, Mission Pack Registry, Agent Registry |
| `src/bio_ml_agent/agents/` | Specialized research sub-agents (academic, biology, browser, coder, microscopy…) |
| `src/bio_ml_agent/core/` | Tool engine, LLM routing, conversation management |
| `src/bio_ml_agent/models/` | Pydantic models for domain, missions, artifacts |
| `src/bio_ml_agent/static/v2/` | Professional Workspace UI (SPA) |
| `src/bio_ml_agent/legacy/` | Legacy modules (ML pipelines, Swarm, dataset catalog, etc.) |

## 💡 Key Features
- **Mission Packs**: High-level, capability-based research blueprints defined in YAML/JSON.
- **Stateful Recovery**: Mission status is checkpointed after every step; interrupted workflows resume automatically.
- **Artifact Lineage**: Full provenance tracking of data produced by agents.
- **Human-in-the-Loop (HITL)**: Approval gates pause execution before critical actions.

## 🛠 Developer Workflow
1. Define a **Mission Pack** in `src/bio_ml_agent/services/mission_pack_registry.py`.
2. Implement specialized agents under `src/bio_ml_agent/agents/`.
3. Register agents in `src/bio_ml_agent/brain/agent_registry.yaml`, or let them be auto-discovered as `EXPERIMENTAL`.
4. Execute missions via the **MissionOrchestrator** (called from `platform_routes.py`).
5. Verify with **Golden Scenario Tests**: `PYTHONPATH=src pytest tests/test_golden_scenarios.py`.

For a deep-dive into system design, see [ARCHITECTURE.md](ARCHITECTURE.md).  
For contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).  
For the full feature roadmap, see [ROADMAP.md](ROADMAP.md).

---

## 🇹🇷 Türkçe Özet (Turkish Summary)

Bio-ML Agent Enterprise; biyoinformatik, mikroskopi analizi ve akademik yayıncılık için tasarlanmış otonom bir araştırma platformudur.

**Hızlı Başlangıç:**
```bash
python run_api.py      # API sunucusu http://localhost:8001
python run_ui.py       # Web UI http://localhost:7860
bio-ml-agent chat      # Terminal sohbet modu
```

**Temel Özellikler:**
- **Mission Packs**: YAML/JSON tabanlı araştırma şablonları.
- **Durum Kurtarma**: Her adım sonrası checkpoint ile kesintisiz çalışma.
- **Artifact Lineage**: Üretilen verilerin tam köken takibi.
- **Human-in-the-Loop**: Kritik işlemler öncesi onay mekanizması.

Mimari detaylar için [ARCHITECTURE.md](ARCHITECTURE.md), katkı rehberi için [CONTRIBUTING.md](CONTRIBUTING.md) dosyalarına bakın.
