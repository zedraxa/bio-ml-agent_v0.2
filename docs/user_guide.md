# 📖 Bio-ML Agent — User Guide

> **Version:** v0.1.0-clean (Production Ready)  
> **Date:** March 20, 2026  
> **Python:** 3.11+  
> **OS:** Linux / macOS

---

## 📋 Table of Contents

1. [Installation](#1--installation)
2. [Quick Start](#2--quick-start)
3. [Terminal Interface (CLI)](#3--terminal-interface-cli)
4. [Web Interface (Gradio)](#4--web-interface-gradio)
5. [Task Dashboard](#5--task-dashboard)
6. [Configuration](#6--configuration)
7. [Agent Commands](#7--agent-commands)
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

- Python 3.11 or higher
- [Ollama](https://ollama.ai/) (for local LLM)
- pip (Python package manager)

### Step-by-Step Installation

```bash
# 1. Navigate to the project directory
cd bio-ml-agent_v0.2/

# 2. Create a virtual environment (skip if already exists)
python3 -m venv venv

# 3. Activate the virtual environment
source venv/bin/activate

# 4. Install all dependencies (full feature set)
pip install -e ".[all]"

# 5. Pull the Ollama model (Ollama must be running)
ollama pull qwen2.5:7b-instruct
```

> **⚠️ Important:** If the project directory has been copied or moved, the venv may break.
> In that case, delete the old venv and recreate it:
> ```bash
> rm -rf venv
> python3 -m venv venv
> source venv/bin/activate
> pip install -e ".[all]"
> ```

### Dependencies

| Package | Purpose |
|---------|---------|
| `litellm` | Multi-LLM Proxy and Router |
| `qdrant-client` | Agentic Vector Memory |
| `temporalio` | Durable Execution / Workflow |
| `langgraph` | Agent Orchestration (State Control) |
| `playwright` | Autonomous Browser Driver |
| `cryptography` | Vault Encryption |
| `opentelemetry` | Observability (Traces/Metrics) |
| `scikit-learn` | ML models |
| `pandas` | Data processing |
| `numpy==1.26.4` | Numerical computation (pinned version) |
| `matplotlib` | Charts |
| `pytest` | Tests |
| `gradio` | Web interface |

---

## 2. 🚀 Quick Start

### Running in Terminal Mode

```bash
source venv/bin/activate
python3 agent.py
```

When the agent starts, you will see:

```
🧠 Bio-ML Agent ready | model=mock-gpt | workspace=./workspace
📜 Session ID: 20260320_190000_a1b2c3d4
♻️ Session auto-recovered: 20260320_185500_... (if applicable)
```

### Create Your First Project

```
>>> PROJECT: breast_cancer Build a classification model using the breast cancer dataset
```

The agent will automatically:
1. Load the dataset
2. Create the project structure
3. Train and compare models
4. Generate charts
5. Write a report

---

### Autonomous Account & Email Management
If the agent is asked to download data or access a gated resource, it uses the `Mail.tm` infrastructure to **autonomously** create a temporary email account, register on the site (via `BROWSER_AGENT`), read the confirmation email, and save the credentials to its internal Vault (`~/.bio-ml-agent/vault.json`). In subsequent projects, it reuses these credentials automatically.

---

## 3. 💻 Terminal Interface (CLI)

### Launch Options

```bash
# Start with default settings
python3 agent.py

# Use a different model
python3 agent.py --model llama3:latest

# Custom workspace
python3 agent.py --workspace /tmp/my_workspace

# Increase timeout
python3 agent.py --timeout 300

# Debug logging
python3 agent.py --log-level DEBUG

# Load an existing session
python3 agent.py --load-session 20260220_150000_abcd1234

# Custom config file
python3 agent.py --config /path/to/custom_config.yaml
```

### CLI Argument Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `qwen2.5:7b-instruct` | Ollama model name |
| `--workspace` | `workspace` | Workspace directory |
| `--timeout` | `180` | Command timeout (seconds) |
| `--max-steps` | `50` | Maximum tool steps |
| `--history-dir` | `conversation_history` | History save directory |
| `--load-session` | — | Session to load on startup |
| `--log-level` | `INFO` | Log level |
| `--log-dir` | `logs` | Log directory |
| `--config` | `config.yaml` | Config file path |

---

## 4. 🌐 Web Interface (Gradio)

### Launch

```bash
source venv/bin/activate
python3 web_ui.py
```

Open `http://localhost:7860` in your browser.

### Features

- Chat with the agent via the chat box
- **NEW:** Voice commands via microphone button (Voice Interface)
- **NEW:** Upload images and medical documents for visual analysis (Vision API)
- **NEW:** Data Explorer tab — instantly view CSV files and interactive Plotly HTML charts in your workspace
- Change model, timeout, and max_steps settings from the UI
- Start new sessions
- View session list

---

## 5. 📊 Task Dashboard

The status of autonomous agent tasks and projects is managed directly within **Web UI (Gradio)** using dedicated tabs. The dashboard is fully integrated into the Web UI.

### Launch

```bash
source venv/bin/activate
python3 web_ui.py
```

Open `http://localhost:7860` and click on the **Dashboard** tab at the top.

### Dashboard Features

| Feature | Description |
|---------|-------------|
| **Task Management** | View agent tasks in a table |
| **Status Filtering** | Filter by PENDING, IN_PROGRESS, COMPLETED |
| **Actions** | Approve or Reject/Cancel a task |
| **Project Stats** | View summary information for the related project |
| **API Key Management** | Set environment variables from the UI |

---

## 6. ⚙️ Configuration

### config.yaml

All settings are managed from `config.yaml`:

```yaml
# Agent settings
agent:
  model: "qwen2.5:7b-instruct"
  max_steps: 50
  timeout: 180
  language: "en"

# Security
security:
  allow_web_search: true
  deny_patterns:
    - '\brm\b.*-rf\s+/'
    - '\bshutdown\b'
    - '\breboot\b'

# Workspace
workspace:
  default_project: "scratch_project"
  base_dir: "workspace"
  auto_save_web: true

# Conversation history
history:
  directory: "conversation_history"
  auto_save_interval: 5

# Logging
logging:
  level: "INFO"
  directory: "logs"
  file_name: "agent.log"
  max_bytes: 5242880        # 5 MB
  backup_count: 3
  console_level: "WARNING"

# ML settings
ml:
  test_size: 0.2
  random_state: 42
  cv_folds: 5
  default_task: "classification"
  comparison:
    enabled: true
    generate_plots: true
    plot_dpi: 150
    output_formats:
      - json
      - csv
      - markdown
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
export OPENAI_API_KEY="sk-..."       # For OpenAI backend
export ANTHROPIC_API_KEY="sk-..."    # For Anthropic backend
export GOOGLE_API_KEY="..."          # For Google Gemini backend
export HF_API_TOKEN="hf_..."        # For HuggingFace backend
```

---

## 7. 📝 Agent Commands

### Session Management

| Command | Description |
|---------|-------------|
| `/history` | List saved sessions |
| `/load <session_id>` | Load an existing session |
| `/new` | Start a new session (current session is saved) |
| `/save` | Save current session immediately |
| `/delete <session_id>` | Delete a session |
| `/info` | Show current session info |
| `/logs [N]` | Show last N log lines (default: 30) |
| `/help` or `/h` | Help menu |
| `exit` or `quit` | Exit (session is saved) |

### Specifying a Project

Add `PROJECT: project_name` to your message to set a project name:

```
>>> PROJECT: water_quality Build a water quality prediction model
```

If not specified, `scratch_project` is used.

### Web Search

Web search is **disabled by default**. To enable it, add `ALLOW_WEB_SEARCH` to your message:

```
>>> ALLOW_WEB_SEARCH research bioengineering datasets
```

---

## 8. 🔧 Tool System

The agent detects special tags in the LLM output and executes the corresponding tools.

### Built-in Tools

| Tool | Tag | Description |
|------|-----|-------------|
| Python | `<PYTHON>...</PYTHON>` | Execute Python code |
| Bash | `<BASH>...</BASH>` | Execute Bash commands |
| Web Search | `<WEB_SEARCH>...</WEB_SEARCH>` | DuckDuckGo search |
| Web Open | `<WEB_OPEN>...</WEB_OPEN>` | Fetch static text from a URL |
| Browser Open | `<BROWSER_OPEN>...</BROWSER_OPEN>` | Fetch URL content via Chromium (JavaScript-rendered) |
| Browser Action | `<BROWSER_ACTION>...</BROWSER_ACTION>` | Manual Playwright commands (click, type, etc.) |
| Browser Agent | `<BROWSER_AGENT>...</BROWSER_AGENT>` | **NEW:** LLM-powered autonomous browser sub-agent |
| Read File | `<READ_FILE>...</READ_FILE>` | Read a file |
| Write File | `<WRITE_FILE>...</WRITE_FILE>` | Write a file |
| TODO | `<TODO>...</TODO>` | Todo list |

### 🤖 Autonomous Browser Sub-Agent (NEW)
Starting from version 7.0, the agent can invoke an internal **Browser Sub-Agent** for complex research and data-gathering tasks. It operates via Chromium, navigates pages, clicks elements, and types in search boxes. You do not need to use the `<BROWSER_AGENT>` tag manually — the agent invokes it autonomously when needed (e.g., *"research p53 gene mutations on Google"*) and reports the results back to you.

### WRITE_FILE Format

```
<WRITE_FILE>
path: project/file.py
---
file content here...
</WRITE_FILE>
```

---

## 9. 📊 Creating ML Projects

### Supported Datasets

The agent's built-in catalog contains **15+ datasets**:

| Dataset | Type | Category |
|---------|------|----------|
| Breast Cancer | Binary Classification | Medical |
| Wine Quality | Multi-Class Classification | General |
| Diabetes | Regression | Medical |
| Heart Disease | Binary Classification | Medical |
| Parkinson's | Binary Classification | Medical |
| Iris | Multi-Class Classification | General |
| Digits | Multi-Class Classification | General |
| Water Quality | Binary Classification | Environmental |
| Air Quality | Regression | Environmental |
| Wastewater | Multi-Class Classification | Environmental |
| EEG Motor | Multi-Class Classification | Biosignal |
| EMG Hand | Multi-Class Classification | Biosignal |
| Chest X-Ray | Binary Classification | Imaging |
| Biodegradability | Binary Classification | Drug Discovery |
| Liver Disease | Binary Classification | Medical |

### Typical ML Workflow

1. User describes the project in natural language
2. Agent finds and loads the dataset
3. Project structure is created: `data/`, `src/`, `results/`
4. At least **3 models** are trained and compared
5. **5-fold cross validation** is performed
6. Charts are generated: confusion matrix, ROC curve, feature importance, etc.
7. `report.md` and `README.md` are written

### Generated Charts

- Confusion Matrix (normal + normalized)
- ROC Curve
- Feature Importance
- Correlation Matrix (heatmap)
- Learning Curve
- Class Distribution

**Note:** Since v3.5, all charts are saved as interactive **Plotly HTML** files (zoomable in the Data Explorer) rather than static PNGs.

### Example Usage

```
>>> PROJECT: cancer Build a classification model using the breast cancer dataset.
    Compare at least 5 models and select the best one.
```

---

## 10. 🧬 Bioengineering Tools

### Protein Analysis

```python
from bioeng_toolkit import ProteinAnalyzer

pa = ProteinAnalyzer("MKWVTFISLLLLFSSAYS")
print(pa.summary())
print(pa.molecular_weight())
print(pa.amino_acid_composition())
print(pa.hydropathy_profile())
print(pa.isoelectric_point())
print(pa.secondary_structure_tendency())
```

### Genomic Analysis

```python
from bioeng_toolkit import GenomicAnalyzer

ga = GenomicAnalyzer("ATGCGATCGATCG")
print(ga.gc_content())
print(ga.complement())
print(ga.reverse_complement())
print(ga.transcribe())
print(ga.translate())
print(ga.find_orfs())
print(ga.melting_temperature())
```

### Wastewater Analysis

```python
from bioeng_toolkit import WastewaterAnalyzer

ww = WastewaterAnalyzer()
# Analyzes parameters such as pH, BOD, COD, TSS
```

### Drug / Molecule Analysis

```python
from bioeng_toolkit import DrugMolecule

mol = DrugMolecule("CCO")  # Ethanol (SMILES notation)
print(mol.summary())
```

---

## 11. 🔌 Plugin System

### Creating a Plugin

Add a new `.py` file to the `plugins/` directory:

```python
# plugins/my_tool.py
from plugin_manager import ToolPlugin
from pathlib import Path

class MyCustomTool(ToolPlugin):
    @property
    def name(self):
        return "MYTOOL"

    @property
    def description(self):
        return "My custom tool"

    def execute(self, payload: str, workspace: Path) -> str:
        return f"Output: {payload}"
```

The plugin is auto-discovered and registered with the agent as `<MYTOOL>...</MYTOOL>`.

### Available Plugins

| Plugin | File | Description |
|--------|------|-------------|
| Example Plugin | `example_plugin.py` | LISTDIR + SYSINFO tools |

---

## 12. 🧠 Switching LLM Backends

### Supported Backends

| Backend | Setting | Requirements |
|---------|---------|--------------|
| **Ollama** (default) | Local | Ollama server running |
| **OpenAI** | API Key | `OPENAI_API_KEY` env var |
| **Anthropic** | API Key | `ANTHROPIC_API_KEY` env var |
| **Google Gemini** | API Key | `GOOGLE_API_KEY` env var |
| **HuggingFace** | API/Local | `HF_API_TOKEN` env var |

### Switching the Backend

Update the model in `config.yaml`:

```yaml
agent:
  model: "gpt-4"  # Use OpenAI
```

Or via the CLI:

```bash
python3 agent.py --model gpt-4
```

You can also change the model dynamically from the **Settings** tab in the Web UI.

---

## 13. 📜 Conversation History

### Automatic Saving

Sessions are **automatically saved** after every user message, assistant response, and tool execution.

### History Files

Sessions are stored as JSON files in `conversation_history/`:

```
conversation_history/
├── 20260220_150000_a1b2c3d4.json
├── 20260221_033500_e5f6g7h8.json
└── ...
```

### Session Management

```bash
# List history
>>> /history

# Load a session
>>> /load 20260220_150000_a1b2c3d4

# Start a new session
>>> /new

# Save current session
>>> /save

# Delete a session
>>> /delete 20260220_150000_a1b2c3d4

# Session info
>>> /info

# RAG memory search
>>> /rag [keyword]    # Query past conversations
>>> /ragindex         # Manually index workspace files into the vector database
```

---

## 14. 🔍 Troubleshooting

### Common Issues

#### ❌ Ollama connection error

```
❌ LLM connection error (model=qwen2.5:7b-instruct)
```

**Fix:**
```bash
# Check if Ollama is running:
ollama serve

# Check if the model is installed:
ollama list

# Pull the model:
ollama pull qwen2.5:7b-instruct
```

#### ❌ `python` command not found

```bash
# Use python3 or create an alias:
alias python=python3

# Or activate the venv:
source venv/bin/activate
```

#### ❌ Module not found (ImportError / ModuleNotFoundError)

```bash
# Make sure the venv is active:
source venv/bin/activate

# Reinstall dependencies:
pip install -e ".[all]"
```

> **Note:** If the project directory was copied or moved, the venv breaks.
> Recreate it:
> ```bash
> rm -rf venv
> python3 -m venv venv
> source venv/bin/activate
> pip install -e ".[all]"
> ```

#### ❌ Timeout error

```bash
python3 agent.py --timeout 300
```

#### ❌ Web search blocked

Add `ALLOW_WEB_SEARCH` to your message, or set in `config.yaml`:

```yaml
security:
  allow_web_search: true
```

#### ❌ Web UI (Gradio) won't start

```bash
pip install gradio
python3 web_ui.py
# Visit http://localhost:7860
```

### Log Files

Errors are recorded in `logs/agent.log`:

```bash
# View last logs in-agent
>>> /logs 50

# Or directly
tail -50 logs/agent.log
```

---

## 15. 📚 Command Reference

### Terminal Commands

| Command | Description |
|---------|-------------|
| `python3 agent.py` | Start in CLI mode |
| `python3 web_ui.py` | Launch the Gradio web interface (includes Dashboard) |
| `python3 -m pytest tests/` | Run all tests |

### Agent Internal Commands

| Command | Description |
|---------|-------------|
| `/history` | List sessions |
| `/load <id>` | Load a session |
| `/new` | New session |
| `/save` | Save current session |
| `/delete <id>` | Delete a session |
| `/info` | Session info |
| `/logs [N]` | View logs |
| `/help` | Help |
| `exit` / `quit` | Exit |

### Special Keywords

| Keyword | Description |
|---------|-------------|
| `PROJECT: <name>` | Set project name |
| `ALLOW_WEB_SEARCH` | Enable web search for this message |

---

## 16. 🚀 Advanced Features (WhatsApp, Voice, Vision & RAG)

### 📱 Remote Control via WhatsApp

The agent bridges to `whatsapp-web.js` over Node.js, allowing you to create and train ML projects directly from your phone.

**To start:**
```bash
./start_whatsapp_bot.sh
```
1. Scan the QR code with WhatsApp's device-linking feature.
2. Send from your WhatsApp chat: `"Gemini, set up a project using the diabetes dataset, train a Random Forest model, and report the best results here."`
3. When done, the agent sends a Markdown summary back to your phone.

### 🎙️ Voice Commands & 👁️ Vision (Gradio Web UI)

When you launch `web_ui.py`:
- **Microphone Module**: Press the microphone button to give voice commands instead of typing long prompts.
- **Multimodal File Upload**: Drag-and-drop MRI images, cell stains, or analysis papers (PDF/Image). The agent uses the Gemini Vision endpoint to read the visual and provide diagnosis or inference.

### 🧠 Automatic Long-Term Memory (Qdrant RAG)

- After every turn, the conversation is indexed in the background to a **Qdrant** vector database.
- **Provenance Tracking**: The source (file or web page) of every piece of information is always recorded.
- **TTL (Time-to-Live)**: Sensitive data can be configured to expire automatically.
- **Hybrid Search**: Keyword and semantic search are combined for the most accurate context retrieval.

---

## 17. 🧩 Multi-Agent Swarm Architecture

From v6+, Bio-ML Agent runs with **3 specialist sub-agents** coordinated via LangGraph instead of a single monolithic LLM.

### Agent Roles

| Agent | Role |
|-------|------|
| **LangGraph Orchestrator** | Main intelligence directing the Plan → Execute → Verify loop. |
| **Data Engineer** | Specialist in data loading, cleaning, and preparation. |
| **ML Expert** | Specialist in model training, hyperparameters, and XAI. |
| **Bioinfo Expert** | Specialist in clinical interpretation and biological reasoning. |

### Usage

```python
from swarm.orchestrator import SwarmOrchestrator
from utils.config import load_config

cfg = load_config()
orchestrator = SwarmOrchestrator(cfg)
response = orchestrator.process([{"role": "user", "content": "Analyze the diabetes dataset"}])
```

### Demo

```bash
source venv/bin/activate
export GEMINI_API_KEY="YOUR_KEY"
python scripts/demos/swarm_diabetes_demo.py
```

---

## 18. 🔍 Explainable AI (XAI)

After training, the ML Expert automatically runs SHAP and LIME analysis via `xai_engine.py`.

### Generated XAI Outputs

| Chart | Description |
|-------|-------------|
| SHAP Summary Plot (Bar) | Feature importance ranking |
| SHAP Summary Plot (Beeswarm) | Individual effects of each feature |
| SHAP Force Plot | Decision explanation for a single patient |
| SHAP Dependence Plot | Feature-prediction relationship |

### Gradio XAI Tab

In the web interface, switch to the **🔍 Explainability (XAI)** tab:
1. Click "Refresh XAI Charts"
2. View all SHAP/LIME charts produced by the agent in a gallery

---

## 19. 🔄 Continuous Learning & Data Streams

The system can be fed from **live data sources** beyond one-time CSV uploads.

### Database Connection

```python
from data_streams.db_connector import DBConnector

db = DBConnector("postgresql://user:pass@localhost/health_db")
new_data = db.get_new_data_since("patients", "created_at", "2026-03-01")
```

### Redis Streams (Real-time Sensor Data)

```python
from data_streams.kafka_redis_consumer import StreamConsumer

consumer = StreamConsumer("sensor_stream", "ml_group", "worker_1")
consumer.listen(batch_size=10, callback=retrain_model)
```

### Active Learning Worker

```bash
# Start background listener (auto-retrains when new data arrives)
python swarm/active_learning_worker.py
```

### Demo

```bash
python scripts/demos/active_learning_demo.py
```

---

## 20. 🐳 Docker Compose Deployment

The system consists of 7 microservices with hardened security policies:

| Service | Port | Description |
|---------|------|-------------|
| `redis` | 6380 | Message queue and cache (Temporal/RQ) |
| `api` | 8001 | FastAPI REST server + Webhook |
| `worker` | — | Background worker (Temporal / RQ / LangGraph) |
| `web_ui` | 7860 | Gradio web interface |
| `mlflow` | 5005 | MLflow Tracking Server |
| `litellm` | 4000 | Multi-LLM Proxy & Routing Gateway |
| `qdrant` | 6333 | Agentic Vector Memory (RAG) |

### Security & Isolation
- **Non-Root User**: Containers run with restricted permissions under the `agent` user.
- **Seccomp**: Unnecessary system calls are blocked.
- **No-New-Privileges**: Privilege escalation attacks are prevented.

### Running

```bash
# Start all services
docker-compose up -d

# Tail worker logs
docker-compose logs -f worker
```

---

## 21. 🛡️ Security & Human-in-the-Loop (HITL)

### 🛡️ Sandbox Runtime
All Python code executed by the agent runs inside an isolated `SandboxRuntime` with CPU and memory limits. This prevents excessive resource consumption and unauthorized access.

### 🤝 Human-in-the-Loop (HITL)
Before critical operations, the agent pauses and asks for your approval:
- **Dangerous Bash Commands**: File deletion (`rm`), system settings.
- **External API Registrations**: Creating new web accounts or acquiring API keys.
- **High-Cost Operations**: Analyses requiring very high token consumption.

### 🔒 Audit Trail
All critical actions are stored in the `audit_logs/` directory as timestamped JSONL files. These records are immutable and allow you to audit exactly what the system did and why.

---

## 22. ⏳ Durable Workflows (Temporal)

Bio-ML Agent uses **Temporal** to manage long-running analyses (e.g., scanning a large genomic dataset).

### Benefits
- **Durability**: Even if your computer shuts down or the internet disconnects, Temporal resumes the workflow from where it left off.
- **Traceability**: Inspect the workflow event history to see every step the agent took.
- **Kill-Switch**: Stop, cancel, or restart any active analysis at any time.

---

> *This guide was compiled for Bio-ML Agent v1.0 (Ultra-Agent Upgrade) on March 4, 2026.*

---

## 🇹🇷 Türkçe Kullanım Özeti (Turkish Summary)

Bu rehberin Türkçe tam sürümü `kullanim_kilavuzu.md` dosyasında bulunmaktadır.

**Hızlı Kurulum:**
```bash
python3 -m venv venv && source venv/bin/activate
pip install -e ".[all]"
ollama pull qwen2.5:7b-instruct
python3 agent.py  # CLI
python3 web_ui.py # Web UI (http://localhost:7860)
```

**Temel Komutlar:** `/history`, `/load`, `/new`, `/save`, `/info`, `/logs`, `exit`  
**Özel Anahtar Kelimeler:** `PROJECT: <ad>` (proje adı belirle), `ALLOW_WEB_SEARCH` (web aramayı aç)
