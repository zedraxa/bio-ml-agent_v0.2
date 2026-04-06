# Bio-ML Agent Enterprise

[![CI](https://github.com/zedraxa/bio-ml-agent_v0.2/actions/workflows/test.yml/badge.svg)](https://github.com/zedraxa/bio-ml-agent_v0.2/actions/workflows/test.yml)

Autonomous research platform for bio-informatics, microscopy analysis and academic publishing.

---

## 🚀 Quick Start

### 1. Requirements
- Python 3.12+
- An LLM API key: `GEMINI_API_KEY`, `OPENAI_API_KEY`, or `ANTHROPIC_API_KEY`
- (Optional) Ollama for local models

### 2. Install

```bash
git clone https://github.com/zedraxa/bio-ml-agent_v0.2.git
cd bio-ml-agent_v0.2

pip install -e ".[ui,ml_ops,cloud,viz,xai,test]"
pip install shap lime playwright flask-limiter twilio
playwright install chromium
```

### 3. Configure

```bash
cp config.example.yaml config.yaml
# Set your API key in config.yaml or as an environment variable:
export GEMINI_API_KEY=your_key_here
```

### 4. Run

```bash
export PYTHONPATH=$(pwd)/src
python src/bio_ml_agent/api/api_server.py
```

Open **http://localhost:8001** — the full workspace UI will load automatically.

### 5. Gradio UI (alternative)

```bash
python run_ui.py   # http://localhost:7860
```

---

## 🧪 Running Tests

```bash
# Full suite (459 tests)
PYTHONPATH=src python -m pytest tests/ \
  --ignore=tests/legacy --ignore=tests/benchmarks -q

# Golden scenario tests
PYTHONPATH=src python -m pytest tests/test_golden_scenarios.py -vv

# Smoke tests only
PYTHONPATH=src python -m pytest tests/test_smoke.py -v
```

CI runs automatically on every push via GitHub Actions (`test.yml`).

---

## 💬 Using the UI

1. Open `http://localhost:8001`
2. Navigate to a project → **Operational Log** tab
3. Select a model from the dropdown (gemini-2.0-flash is the default)
4. Type your task and press Enter

The agent streams its response in real-time via SSE. Tool outputs appear as
monospace blocks. When human approval is required, an approval bar appears at
the bottom of the chat.

### REST API

The full Swagger/OpenAPI docs are at `http://localhost:8001/docs`.

Key endpoints:
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/platform/chat/stream` | SSE streaming chat |
| `POST` | `/api/v1/platform/chat/async` | Async chat (callback) |
| `GET`  | `/api/v1/platform/projects` | List projects |
| `WS`   | `/api/v1/platform/ws/events/{session_id}` | Live event stream |

---

## 📂 Core Structure

```
src/bio_ml_agent/
├── api/              REST API boundary (FastAPI)
├── routers/          Platform routes including /chat/stream SSE endpoint
├── services/         Mission Orchestrator & Agent Registry
├── brain/            Mission Brain & Recovery Manager (Stateful)
├── db/               SQLAlchemy 2.0 persistence layer (SQLite)
├── agents/           Specialized research sub-agents
├── static/v2/        Professional Workspace UI (single-page app)
└── legacy/           Experimental and deprecated modules
```

## 💡 Key Features

- **Real-time SSE Streaming** — Chat responses stream token-by-token via `/chat/stream`
- **Mission Packs** — High-level capability-based research blueprints
- **Stateful Recovery** — Checkpointed after every step; resumable from any failure point
- **Artifact Lineage** — Full data provenance tracking across agents
- **Human-in-the-Loop** — Approval gates for critical actions (shown in UI)
- **Multi-model** — Gemini, GPT-4o, Claude, Ollama (local) via dropdown

## 🛠 Developer Workflow

1. Define a **Mission Pack** in `mission_pack_registry.py`
2. Implement specialized agents in `src/bio_ml_agent/agents/`
3. Register agents in `agent_registry.py`
4. Execute via the **MissionOrchestrator**
5. Verify using **Golden Scenario Tests**

See [ARCHITECTURE.md](ARCHITECTURE.md) for a deep dive into the system design.

