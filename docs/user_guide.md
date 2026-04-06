# 📖 Bio-ML Agent — User Guide

> **Version:** v0.1.0-clean  
> **API Version:** 6.0.0 (Bio-ML Enterprise API)  
> **Python:** 3.9+  
> **OS:** Linux / macOS

---

## 📋 Table of Contents

1. [Installation](#1--installation)
2. [Quick Start](#2--quick-start)
3. [CLI Entry Points](#3--cli-entry-points)
4. [Web Interface (Gradio)](#4--web-interface-gradio)
5. [Task Dashboard](#5--task-dashboard)
6. [Configuration](#6--configuration)
7. [Agent Commands (Chat Mode)](#7--agent-commands-chat-mode)
8. [Tool System](#8--tool-system)
9. [Creating ML Projects](#9--creating-ml-projects)
10. [Bioengineering Tools](#10--bioengineering-tools)
11. [Plugin System](#11--plugin-system)
12. [Switching LLM Backends](#12--switching-llm-backends)
13. [Conversation History](#13--conversation-history)
14. [Troubleshooting](#14--troubleshooting)
15. [Command Reference](#15--command-reference)
16. [Advanced Features (WhatsApp, Voice, Vision & RAG)](#16--advanced-features-whatsapp-voice-vision--rag)
17. [Multi-Agent Swarm Architecture](#17--multi-agent-swarm-architecture)
18. [Explainable AI (XAI)](#18--explainable-ai-xai)
19. [Continuous Learning & Data Streams](#19--continuous-learning--data-streams)
20. [Docker Compose Deployment](#20--docker-compose-deployment)
21. [Security & Human-in-the-Loop (HITL)](#21--security--human-in-the-loop-hitl)
22. [Durable Workflows (Temporal)](#22--durable-workflows-temporal)

---

## 1. 🔧 Installation

### Prerequisites

- Python 3.9 or higher
- [Ollama](https://ollama.ai/) (for local LLM — optional if using cloud backends)
- pip

### Step-by-Step Installation

```bash
# 1. Navigate to the project directory
cd bio-ml-agent_v0.2/

# 2. Create a virtual environment (skip if already exists)
python3 -m venv venv

# 3. Activate the virtual environment
source venv/bin/activate

# 4. Install all dependencies
pip install -e ".[all]"

# 5. (Optional) Pull an Ollama model for local inference
ollama pull qwen2.5:7b-instruct
```

> **⚠️ Important:** If the project directory has been copied or moved, the venv may break.
> Delete and recreate it:
> ```bash
> rm -rf venv
> python3 -m venv venv
> source venv/bin/activate
> pip install -e ".[all]"
> ```

### Core Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` + `uvicorn` | REST API server |
| `gradio` | Web UI |
| `pydantic` | Config & model validation |
| `sqlalchemy` | ORM (SQLite persistence) |
| `qdrant-client` | Vector memory (RAG) |
| `sentence-transformers` | Embedding for RAG |
| `rank_bm25` | Lexical search |
| `redis` + `rq` | Background task queue |
| `scikit-learn` + `pandas` + `numpy` | ML pipelines |
| `litellm` | Multi-LLM proxy |
| `openai` / `anthropic` / `google-genai` | Cloud LLM backends |
| `shap` + `lime` | Explainable AI (XAI) |
| `twilio` | WhatsApp integration |

---

## 2. 🚀 Quick Start

### Launch the API Server

```bash
python run_api.py
# or: bio-ml-agent api
```
API available at `http://localhost:8001`. Interactive docs at `http://localhost:8001/docs`.

### Launch the Web UI

```bash
python run_ui.py
# or: bio-ml-agent ui
```
Open `http://localhost:7860` in your browser.

### Interactive Terminal Chat

```bash
bio-ml-agent chat
```

When the agent starts you will see:
```
🧠 Bio-ML Agent ready | model=qwen2.5:7b-instruct | workspace=./workspace
📜 Session ID: 20260320_190000_a1b2c3d4
```

---

### Autonomous Account & Email Management
If asked to download gated data or register on a site, the agent uses the `Mail.tm` service via `BROWSER_AGENT` to create a temporary email account, complete registration, read the confirmation email, and store credentials in its internal Vault (`~/.bio-ml-agent/vault.json`) for reuse in future sessions.

---

## 3. 💻 CLI Entry Points

The `bio-ml-agent` command (installed via `pip install -e .`) provides the following subcommands:

| Command | Description |
|---------|-------------|
| `bio-ml-agent ui` | Start the Gradio Web UI |
| `bio-ml-agent api` | Start the FastAPI REST server |
| `bio-ml-agent chat` | Start an interactive terminal chat session |
| `bio-ml-agent check` | Verify installation and configuration |

You can also use the run scripts directly:
```bash
python run_ui.py         # Web UI
python run_api.py        # REST API
python run_whatsapp.py   # WhatsApp connector (Flask)
python run_worker.py     # Background job worker (RQ)
python run_gateway.py    # External API gateway
python start_bio_ml.py   # Unified launcher (all services)
```

---

## 4. 🌐 Web Interface (Gradio)

### Launch

```bash
python run_ui.py
# or: bio-ml-agent ui
```

Open `http://localhost:7860` in your browser.

### Features

- Chat with the agent via the chat box
- **Voice commands** via microphone button (Voice Interface)
- **Multimodal file upload** — images, medical documents for visual analysis (Vision API)
- **Data Explorer tab** — view CSV files and interactive Plotly HTML charts in your workspace
- Change model, timeout, and max_steps from the UI
- Start new sessions, view session list
- **XAI tab** — view SHAP/LIME explainability charts

---

## 5. 📊 Task Dashboard

The status of agent tasks and projects is managed directly within the **Gradio Web UI** using dedicated tabs.

### Launch

```bash
python run_ui.py
```

Open `http://localhost:7860` and click the **Dashboard** tab.

### Dashboard Features

| Feature | Description |
|---------|-------------|
| **Task Management** | View agent tasks in a table |
| **Status Filtering** | Filter by PENDING, IN_PROGRESS, COMPLETED |
| **Actions** | Approve or Reject/Cancel a task |
| **Project Stats** | View summary for the related project |
| **API Key Management** | Set environment variables from the UI |

---

## 6. ⚙️ Configuration

### config.yaml

All settings are managed from `config.yaml` (or `config.example.yaml` as a starting template):

```yaml
agent:
  model: "qwen2.5:7b-instruct"   # Default local LLM
  max_steps: 9999
  timeout: 300

redis:
  host: localhost
  port: 6379
  password: ""
  db: 0

security:
  allow_web_search: true
  api_key: ""           # Empty = security disabled
  webhook_secret: ""    # HMAC-SHA256 for webhooks
  deny_patterns:
    - '\brm\b.*-rf\s+/'
    - '\bshutdown\b'
    - '\breboot\b'

workspace:
  default_project: "scratch_project"
  base_dir: "workspace"

history:
  directory: "conversation_history"
  auto_save_interval: 5
  max_summary_length: 100

logging:
  level: "INFO"
  directory: "logs"
  file_name: "agent.log"
  max_bytes: 5242880
  backup_count: 3
  console_level: "WARNING"

ml:
  test_size: 0.2
  random_state: 42
  cv_folds: 5
  comparison:
    enabled: true
    generate_plots: true
    plot_dpi: 150
    output_formats: [json, csv, markdown]
```

### Configuration Priority

```
CLI arguments > Environment variables > config.yaml > Defaults
```

### Environment Variables

```bash
export AGENT_MODEL="llama3:latest"
export AGENT_TIMEOUT=300
export OLLAMA_HOST="http://localhost:11434"
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AIza..."
export HF_API_TOKEN="hf_..."
```

---

## 7. 📝 Agent Commands (Chat Mode)

### Session Management

| Command | Description |
|---------|-------------|
| `/history` | List saved sessions |
| `/load <session_id>` | Load an existing session |
| `/new` | Start a new session (current is saved) |
| `/save` | Save current session immediately |
| `/delete <session_id>` | Delete a session |
| `/info` | Show current session info |
| `/logs [N]` | Show last N log lines (default: 30) |
| `/rag [keyword]` | Query past conversations in RAG memory |
| `/ragindex` | Manually index workspace files into the vector database |
| `/help` or `/h` | Help menu |
| `exit` or `quit` | Exit (session is saved) |

### Specifying a Project

```
>>> PROJECT: water_quality Build a water quality prediction model
```
If not specified, `scratch_project` is used.

### Enabling Web Search

Web search is disabled by default. Add `ALLOW_WEB_SEARCH` to enable it for a message:
```
>>> ALLOW_WEB_SEARCH research bioengineering datasets from NCBI
```

---

## 8. 🔧 Tool System

The agent detects special XML-style tags in LLM output and executes them:

| Tool Tag | Description |
|----------|-------------|
| `<PYTHON>...</PYTHON>` | Execute Python code (sandboxed) |
| `<BASH>...</BASH>` | Execute Bash commands |
| `<WEB_SEARCH>...</WEB_SEARCH>` | DuckDuckGo search |
| `<WEB_OPEN>...</WEB_OPEN>` | Fetch static text from a URL |
| `<BROWSER_OPEN>...</BROWSER_OPEN>` | Fetch JavaScript-rendered page via Chromium |
| `<BROWSER_ACTION>...</BROWSER_ACTION>` | Manual Playwright commands (click, type, etc.) |
| `<BROWSER_AGENT>...</BROWSER_AGENT>` | LLM-powered autonomous browser sub-agent |
| `<READ_FILE>...</READ_FILE>` | Read a file |
| `<WRITE_FILE>...</WRITE_FILE>` | Write a file |
| `<TODO>...</TODO>` | Append to the TODO list |
| `<CLINICAL_VISION>...</CLINICAL_VISION>` | Medical image analysis |
| `<SWARM>...</SWARM>` | Invoke the multi-agent swarm |
| `<DEEP_RESEARCH>...</DEEP_RESEARCH>` | Deep iterative web research |
| `<INDEX_WORKSPACE>...</INDEX_WORKSPACE>` | Index workspace files into RAG |
| `<BACKGROUND_JOB>...</BACKGROUND_JOB>` | Submit a long-running background task |

### WRITE_FILE Format

```
<WRITE_FILE>
path: project/analysis.py
---
# file content here
import pandas as pd
...
</WRITE_FILE>
```

---

## 9. 📊 Creating ML Projects

### Built-in Dataset Catalog (18 Datasets)

| Dataset | Type | Category |
|---------|------|----------|
| Wisconsin Breast Cancer | Binary Classification | Medical |
| Diabetes Regression | Regression | Medical |
| Iris Flower | Multi-Class Classification | General |
| Wine Recognition | Multi-Class Classification | General |
| Handwritten Digits | Multi-Class Classification | Image |
| Heart Disease (Cleveland) | Binary Classification | Medical |
| Parkinson's Disease | Binary Classification | Medical |
| Indian Liver Patient | Binary Classification | Medical |
| Chronic Kidney Disease | Binary Classification | Medical |
| Water Quality (Potability) | Binary Classification | Environmental |
| Air Quality (UCI) | Regression | Environmental |
| Gene Expression Cancer RNA-Seq | Multi-Class Classification | Genomics |
| EEG Motor Movement/Imagery | Multi-Class Classification | Biosignal |
| Water Treatment Plant | Multi-Class Classification | Environmental |
| Yeast Protein Localization | Multi-Class Classification | Genomics |
| QSAR Biodegradation | Binary Classification | Drug Discovery |
| Chest X-Ray (Pneumonia) | Binary Classification | Medical Imaging |
| EMG Hand Gesture Recognition | Multi-Class Classification | Biosignal |

### Typical ML Workflow

1. User describes the project in natural language
2. Agent finds and loads the dataset from the catalog
3. Project structure created: `data/`, `src/`, `results/`
4. At least **3 models** trained and compared
5. **5-fold cross validation** performed
6. Charts generated: confusion matrix, ROC, feature importance, etc.
7. `report.md` and `README.md` written

All charts are saved as interactive **Plotly HTML** files (viewable in the Data Explorer tab).

### Example

```
>>> PROJECT: cancer Compare at least 5 classifiers on the breast cancer dataset and select the best one.
```

---

## 10. 🧬 Bioengineering Tools

Located in `src/bio_ml_agent/legacy/ml/bioeng_toolkit.py`:

### Protein Analysis

```python
from bio_ml_agent.legacy.ml.bioeng_toolkit import ProteinAnalyzer

pa = ProteinAnalyzer("MKWVTFISLLLLFSSAYS")
print(pa.summary())
print(pa.molecular_weight())
print(pa.amino_acid_composition())
print(pa.isoelectric_point())
```

### Genomic Analysis

```python
from bio_ml_agent.legacy.ml.bioeng_toolkit import GenomicAnalyzer

ga = GenomicAnalyzer("ATGCGATCGATCG")
print(ga.gc_content())
print(ga.reverse_complement())
print(ga.transcribe())
print(ga.translate())
print(ga.find_orfs())
```

---

## 11. 🔌 Plugin System

Add a `.py` file to `src/bio_ml_agent/plugins/` (auto-discovered at startup):

```python
# src/bio_ml_agent/plugins/my_tool.py
from bio_ml_agent.legacy.plugin_manager import ToolPlugin
from pathlib import Path

class MyCustomTool(ToolPlugin):
    @property
    def name(self):
        return "MYTOOL"

    @property
    def description(self):
        return "My custom tool description"

    def execute(self, payload: str, workspace: Path) -> str:
        return f"Output: {payload}"
```

The plugin is registered as `<MYTOOL>...</MYTOOL>`.

> ⚠️ **Security**: Plugins execute Python code in-process. Only load plugins from trusted sources.

---

## 12. 🧠 Switching LLM Backends

### Supported Backends (via `llm_backend.py`)

| Backend | Config Value | Requirements |
|---------|-------------|--------------|
| **Ollama** (default) | e.g. `qwen2.5:7b-instruct` | Ollama server running |
| **OpenAI** | `gpt-4o`, `gpt-4-turbo`, etc. | `OPENAI_API_KEY` env var |
| **Anthropic** | `claude-3-5-sonnet-*`, etc. | `ANTHROPIC_API_KEY` env var |
| **Google Gemini** | `gemini-2.0-flash`, etc. | `GOOGLE_API_KEY` env var |
| **HuggingFace** | `huggingface/...` | `HF_API_TOKEN` env var |

### Switch via config.yaml

```yaml
agent:
  model: "gpt-4o"
```

### Switch via CLI

```bash
bio-ml-agent chat --model gpt-4o
```

You can also change it dynamically from the **Settings** tab in the Web UI.

---

## 13. 📜 Conversation History

### Automatic Saving

Sessions are automatically saved after every message and tool execution. Files are stored in `conversation_history/` as JSON.

### Session Commands (Chat Mode)

```
>>> /history                                 # List all sessions
>>> /load 20260320_190000_a1b2c3d4           # Load a session
>>> /new                                      # Start fresh
>>> /save                                     # Force save now
>>> /delete 20260320_190000_a1b2c3d4          # Delete a session
>>> /info                                     # Current session metadata
```

---

## 14. 🔍 Troubleshooting

### Ollama connection error

```bash
ollama serve                           # Start Ollama server
ollama list                            # Check installed models
ollama pull qwen2.5:7b-instruct       # Download model
```

### Module not found

```bash
source venv/bin/activate
pip install -e ".[all]"
# If directory was moved: rm -rf venv && python3 -m venv venv && ...
```

### Web search blocked

Add `ALLOW_WEB_SEARCH` to your message, or set `security.allow_web_search: true` in `config.yaml`.

### View logs

```
>>> /logs 50          # In chat mode
tail -50 logs/agent.log  # From terminal
```

---

## 15. 📚 Command Reference

### Terminal Commands

| Command | Description |
|---------|-------------|
| `python run_api.py` | Start REST API server (port 8001) |
| `python run_ui.py` | Start Gradio Web UI (port 7860) |
| `python run_whatsapp.py` | Start WhatsApp connector (Flask) |
| `python run_worker.py` | Start RQ background worker |
| `python run_gateway.py` | Start external gateway (port 8000) |
| `python start_bio_ml.py` | Start all services at once |
| `PYTHONPATH=src pytest tests/` | Run test suite |
| `ruff check src/` | Lint check |

### CLI Subcommands

| Command | Description |
|---------|-------------|
| `bio-ml-agent ui` | Gradio Web UI |
| `bio-ml-agent api` | FastAPI REST server |
| `bio-ml-agent chat` | Terminal interactive chat |
| `bio-ml-agent check` | Verify installation |

### Chat Internal Commands

| Command | Description |
|---------|-------------|
| `/history` | List sessions |
| `/load <id>` | Load session |
| `/new` | New session |
| `/save` | Save now |
| `/delete <id>` | Delete session |
| `/info` | Session info |
| `/logs [N]` | View last N log lines |
| `/rag [keyword]` | RAG memory search |
| `/ragindex` | Re-index workspace |
| `/help` | Help |
| `exit` / `quit` | Exit |

---

## 16. 🚀 Advanced Features (WhatsApp, Voice, Vision & RAG)

### 📱 Remote Control via WhatsApp

The WhatsApp connector uses **Twilio** (not `whatsapp-web.js`) and Flask:

```bash
python run_whatsapp.py
```

Configure Twilio credentials in `.env` or `config.yaml`:
```yaml
whatsapp:
  twilio_account_sid: "ACxxxx"
  twilio_auth_token: "xxxx"
  twilio_whatsapp_number: "whatsapp:+14155238886"
```

For local testing, use `start_bio_ml.py` which also launches an ngrok tunnel and prints the webhook URL to configure in your Twilio console.

### 🎙️ Voice Commands & 👁️ Vision (Gradio Web UI)

- **Microphone button**: Voice-to-text input in the Web UI.
- **File upload**: Drag-and-drop images (microscopy, X-ray, etc.) or PDFs. The agent uses the LLM's Vision API for visual analysis.

### 🧠 Long-Term Memory (Qdrant RAG)

- Every conversation turn is indexed to **Qdrant** in the background.
- **Provenance Tracking**: Every stored memory records its source.
- **TTL**: Configurable expiry for sensitive data.
- **Hybrid Search**: RRF fusion of semantic + keyword search.

---

## 17. 🧩 Multi-Agent Swarm Architecture

The `SwarmOrchestrator` (`src/bio_ml_agent/legacy/swarm/orchestrator.py`) coordinates 6 specialist agents:

| Agent Class | Role |
|-------------|------|
| `DataEngineerAgent` | Data loading, cleaning, and preparation |
| `MLExpertAgent` | Model training, hyperparameter tuning, XAI |
| `BioinfoExpertAgent` | Clinical interpretation and biological reasoning |
| `ResearchAgent` | Literature search and web research |
| `InSilicoExpertAgent` | Computational biology and in-silico simulation |
| `AcademicPublishingExpertAgent` | Academic writing and paper generation |

The orchestrator routes incoming intent to the appropriate agent (or runs the full pipeline via `PIPELINE` mode).

### Usage

```python
from bio_ml_agent.legacy.swarm.orchestrator import SwarmOrchestrator
from bio_ml_agent.utils.config import load_config

cfg = load_config()
orchestrator = SwarmOrchestrator(cfg)
for event in orchestrator.process([{"role": "user", "content": "Analyze the diabetes dataset"}]):
    print(event)
```

### Demo Scripts

```bash
PYTHONPATH=src python scripts/demos/swarm_diabetes_demo.py
```

---

## 18. 🔍 Explainable AI (XAI)

After ML training, `xai_engine.py` (`src/bio_ml_agent/legacy/ml/xai_engine.py`) automatically produces SHAP and LIME visualizations.

### Generated Outputs

| Chart | Description |
|-------|-------------|
| SHAP Summary (Bar) | Global feature importance ranking |
| SHAP Summary (Beeswarm) | Per-sample feature contributions |
| SHAP Force Plot | Single-prediction explanation |
| SHAP Dependence Plot | Feature vs. prediction relationship |

### Gradio XAI Tab

In the Web UI, switch to the **🔍 Explainability (XAI)** tab and click "Refresh XAI Charts" to load all generated plots.

---

## 19. 🔄 Continuous Learning & Data Streams

Located in `src/bio_ml_agent/legacy/data_streams/`:

### Database Connection

```python
from bio_ml_agent.legacy.data_streams.db_connector import DBConnector

db = DBConnector("sqlite:///health.db")
new_data = db.get_new_data_since("patients", "created_at", "2026-03-01")
```

### Redis Stream Consumer (Real-time Sensor Data)

```python
from bio_ml_agent.legacy.data_streams.kafka_redis_consumer import StreamConsumer

consumer = StreamConsumer("sensor_stream", "ml_group", "worker_1")
consumer.listen(batch_size=10, callback=retrain_model)
```

### Active Learning Worker

```bash
PYTHONPATH=src python src/bio_ml_agent/legacy/swarm/active_learning_worker.py
```

---

## 20. 🐳 Docker Compose Deployment

The `docker-compose.yml` defines **12 services**:

| Service | Host Port | Description |
|---------|-----------|-------------|
| `redis` | 6380 | Message queue and cache |
| `api` | 8001 | FastAPI REST server |
| `worker` | — | RQ background worker |
| `web_ui` | 7860 | Gradio web interface |
| `mlflow` | 5005 | MLflow Tracking Server |
| `litellm` | 4000 | Multi-LLM Proxy & Router |
| `qdrant` | 6333 / 6334 | Vector memory (RAG) |
| `postgres` | 5432 | PostgreSQL database |
| `minio` | 9000 / 9001 | Object storage (artifacts) |
| `otel_collector` | 4317 / 4318 | OpenTelemetry collector |
| `gateway` | 8000 | External-facing API gateway |
| `notification_worker` | — | Background notification processor |

```bash
docker-compose up -d             # Start all
docker-compose logs -f worker    # Tail worker
docker-compose down              # Stop all
```

---

## 21. 🛡️ Security & Human-in-the-Loop (HITL)

### 🛡️ Sandbox Runtime
Python code executed via the `<PYTHON>` tool runs in a restricted environment with configurable `deny_patterns` (regex blocklist in `config.yaml`). Dangerous commands (e.g., `rm -rf /`, `shutdown`) are blocked before execution.

### 🤝 Human-in-the-Loop (HITL)
Before critical operations, the agent pauses and raises a `ReviewThread` requiring human approval:
- Dangerous Bash commands
- External API registrations or account creation
- High-cost LLM operations

### 🔒 Audit Trail
All critical actions are logged to `audit_logs/` as timestamped, append-only JSONL files. Viewable via:
```
GET /api/v1/observability/audit
```

---

## 22. ⏳ Durable Workflows (Temporal)

For long-running analyses (e.g., large genomic scans), the agent can submit jobs to **Temporal** via `submit_temporal_job()` in `services/agent/orchestration.py`.

### Benefits
- **Durability**: Workflows resume automatically after crash or network interruption.
- **Traceability**: Full event history available in the Temporal UI.
- **Kill-Switch**: Cancel any active workflow at any time.

### Start Temporal locally

```bash
./scripts/start_temporal.sh
```

---

## 🇹🇷 Türkçe Kullanım Özeti (Turkish Summary)

Bu rehberin eski Türkçe sürümü `kullanim_kilavuzu.md` referans olarak saklanmaktadır.

**Hızlı Başlangıç:**
```bash
pip install -e ".[all]"
python run_api.py       # API sunucusu → http://localhost:8001
python run_ui.py        # Web UI → http://localhost:7860
bio-ml-agent chat       # Terminal sohbet modu
```

**Temel Komutlar:** `/history`, `/load`, `/new`, `/save`, `/info`, `/rag`, `/ragindex`, `exit`  
**CLI:** `bio-ml-agent ui | api | chat | check`
