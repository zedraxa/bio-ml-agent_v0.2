# Bio-ML Agent Enterprise

[![CI](https://github.com/zedraxa/bio-ml-agent_v0.2/actions/workflows/test.yml/badge.svg)](https://github.com/zedraxa/bio-ml-agent_v0.2/actions/workflows/test.yml)

Autonomous research platform for bio-informatics, microscopy analysis, and academic publishing.

---

## 🚀 Quick Start

### 1. Requirements
- Python 3.12+
- An LLM API key: `GEMINI_API_KEY`, `OPENAI_API_KEY`, or `ANTHROPIC_API_KEY`
- (Optional) Redis for background jobs; Ollama for local models

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

### 4. Run the API Server

```bash
export PYTHONPATH=$(pwd)/src
python src/bio_ml_agent/api/api_server.py
```

Open **http://localhost:8001** — the full workspace UI loads automatically.  
Interactive API docs at `/docs`.

Alternatively, use the CLI entry point:
```bash
bio-ml-agent api
```

### 5. Run the Web UI (Gradio, alternative)

```bash
python run_ui.py        # or: bio-ml-agent ui
```
Open `http://localhost:7860` in your browser.

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

## 💬 Using the Workspace UI

1. Open `http://localhost:8001`
2. Navigate to a project → **Operational Log** tab
3. Select a model from the dropdown (gemini-2.0-flash is the default)
4. Type your task and press Enter

Agent responses stream in real-time via SSE. Tool outputs appear as monospace blocks. When human approval is required, an approval bar appears at the bottom of the chat.

### REST API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/platform/chat/stream` | SSE streaming chat |
| `POST` | `/api/v1/platform/chat/async` | Async chat (callback) |
| `GET`  | `/api/v1/platform/projects` | List projects |
| `WS`   | `/api/v1/platform/ws/events/{session_id}` | Live event stream |

---

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
- **Real-time SSE Streaming** — Chat responses stream token-by-token via `/chat/stream`
- **Mission Packs**: High-level, capability-based research blueprints defined in YAML/JSON.
- **Stateful Recovery**: Mission status is checkpointed after every step; interrupted workflows resume automatically.
- **Artifact Lineage**: Full provenance tracking of data produced by agents.
- **Human-in-the-Loop (HITL)**: Approval gates pause execution before critical actions.
- **Multi-model** — Gemini, GPT-4o, Claude, Ollama (local) via dropdown

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
python src/bio_ml_agent/api/api_server.py   # API + UI → http://localhost:8001
python run_ui.py                             # Gradio UI → http://localhost:7860
bio-ml-agent chat                            # Terminal sohbet modu
```

**Temel Özellikler:**
- **Gerçek Zamanlı SSE**: Sohbet yanıtları `/chat/stream` üzerinden anlık akar.
- **Mission Packs**: YAML/JSON tabanlı araştırma şablonları.
- **Durum Kurtarma**: Her adım sonrası checkpoint ile kesintisiz çalışma.
- **Artifact Lineage**: Üretilen verilerin tam köken takibi.
- **Human-in-the-Loop**: Kritik işlemler öncesi onay mekanizması.

Mimari detaylar için [ARCHITECTURE.md](ARCHITECTURE.md), katkı rehberi için [CONTRIBUTING.md](CONTRIBUTING.md) dosyalarına bakın.


