# Bio-ML Agent Enterprise

Autonomous research platform for bio-informatics, microscopy analysis, and academic publishing.

## 🚀 Quick Start

### 1. Requirements
- Python 3.12+
- SQLite
- Redis (optional, for background jobs)

### 2. Run the API Server
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
./.venv/bin/python src/bio_ml_agent/api/api_server.py
```
The API is served at `http://localhost:8001`. The UI is accessible at `/`.

### 3. Run Golden Scenario Tests
Verify all primary workflows (Repo Review, Lab Report, Microscopy Swarm):
```bash
./.venv/bin/python -m pytest tests/test_golden_scenarios.py -vv
```

## 📂 Project Structure
| Directory | Description |
| :--- | :--- |
| `src/bio_ml_agent/api/` | REST API boundary (FastAPI) |
| `src/bio_ml_agent/services/` | Mission Orchestrator & Agent Registry |
| `src/bio_ml_agent/brain/` | Stateful Mission Brain & Recovery Manager |
| `src/bio_ml_agent/db/` | Persistence layer (SQLite / SQLAlchemy) |
| `src/bio_ml_agent/agents/` | Specialized research sub-agents |
| `src/bio_ml_agent/static/v2/` | Professional Workspace UI |
| `src/bio_ml_agent/legacy/` | Experimental and deprecated modules |

## 💡 Key Features
- **Mission Packs**: High-level, capability-based research blueprints defined in YAML/JSON.
- **Stateful Recovery**: Mission status is checkpointed after every step; interrupted workflows resume automatically.
- **Artifact Lineage**: Full provenance tracking of data produced by agents.
- **Human-in-the-Loop (HITL)**: Approval gates pause execution before critical actions.

## 🛠 Developer Workflow
1. Define a **Mission Pack** in `mission_pack_registry.py`.
2. Implement specialized agents under `src/bio_ml_agent/agents/`.
3. Register agents in `agent_registry.yaml`, or let them be auto-discovered as `EXPERIMENTAL`.
4. Execute missions via the **MissionOrchestrator**.
5. Verify with **Golden Scenario Tests**.

For a deep-dive into system design, see [ARCHITECTURE.md](ARCHITECTURE.md).  
For contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).  
For the full feature roadmap, see [ROADMAP.md](ROADMAP.md).

---

## 🇹🇷 Türkçe Özet (Turkish Summary)

Bio-ML Agent Enterprise; biyoinformatik, mikroskopi analizi ve akademik yayıncılık için tasarlanmış otonom bir araştırma platformudur.

**Hızlı Başlangıç:**
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
./.venv/bin/python src/bio_ml_agent/api/api_server.py
```
API `http://localhost:8001` adresinden, kullanıcı arayüzü ise `/` yolundan erişilebilir.

**Temel Özellikler:**
- **Mission Packs**: YAML/JSON tabanlı araştırma şablonları.
- **Durum Kurtarma**: Her adım sonrası checkpoint ile kesintisiz çalışma.
- **Artifact Lineage**: Üretilen verilerin tam köken takibi.
- **Human-in-the-Loop**: Kritik işlemler öncesi onay mekanizması.

Mimari detaylar için [ARCHITECTURE.md](ARCHITECTURE.md), katkı rehberi için [CONTRIBUTING.md](CONTRIBUTING.md) dosyalarına bakın.
