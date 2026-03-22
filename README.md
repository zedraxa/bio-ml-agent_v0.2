# Bio-ML Agent Enterprise

Autonomous research platform for bio-informatics, microscopy analysis and academic publishing.

## 🚀 Quick Start (Verified Core)

### 1. Requirements
- Python 3.12+
- SQLite
- Redis (Optional, for Background Jobs)

### 2. Run API Server (Golden Path)
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
./.venv/bin/python src/bio_ml_agent/api/api_server.py
```
API is served at `http://localhost:8001`. Access the UI at `/`.

### 3. Run Golden Scenario Tests
Verify all primary workflows (Repo Review, Lab Report, Microscopy Swarm) are operational:
```bash
./.venv/bin/python -m pytest tests/test_golden_scenarios.py -vv
```

## 📂 Core Structure
- `src/bio_ml_agent/api/`: REST API boundary.
- `src/bio_ml_agent/services/`: Mission Orchestrator & Agent Registry.
- `src/bio_ml_agent/brain/`: Mission Brain & Recovery Manager (Stateful).
- `src/bio_ml_agent/db/`: Persistence layer.
- `src/bio_ml_agent/agents/`: Specialized research sub-agents.
- `src/bio_ml_agent/static/v2/`: Professional Workspace UI.
- `src/bio_ml_agent/legacy/`: Experimental and deprecated modules.

## 💡 Key Features
- **Mission Packs**: High-level capability-based research blueprints.
- **Stateful Recovery**: Mission status is checkpointed after every step.
- **Artifact Lineage**: Tracking of data provenance across agents.
- **Human-in-the-Loop**: Approval gates for critical actions.

## 🛠 Developer Workflow
1. Define a **Mission Pack** in `mission_pack_registry.py`.
2. Implement specialized agents in `src/bio_ml_agent/agents/`.
3. Register agents in `agent_registry.py` (or let them be auto-discovered as `EXPERIMENTAL`).
4. Execute via the **MissionOrchestrator**.
5. Verify using **Golden Scenario Tests**.

See [ARCHITECTURE.md](ARCHITECTURE.md) for a deep dive into the system design.
 Riverside
