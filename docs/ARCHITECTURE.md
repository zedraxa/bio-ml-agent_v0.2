# Bio-ML Agent — Architecture Documentation (v0.1.0-clean)

Bio-ML Agent is a **Mission-Pack based, centrally-orchestrated scientific platform**. It coordinates specialized agents across domains like microscopy, genomics, and academic reporting through a structured mission execution model.

## 🏗️ Overview

The system is built around five architecture principles:

1. **One mission, one truth** — Every research mission has a single, canonical `ProjectState` and `MissionPlan`.
2. **Artifacts are first-class citizens** — Every output is a versioned, traceable entity linked via the `ArtifactGraph`.
3. **Agents must be replaceable** — The system relies on the `UnifiedAgentContract`, not individual model quirks.
4. **Failures are normal; recovery is a feature** — `RecoveryManager` and `CheckpointStore` are core, not afterthoughts.
5. **Integration quality matters as much as model quality** — Governance, observability, and consistency come first.

## 🧱 Core Concepts

### Project
The top-level research container. Tracks multiple **Missions** and maintains a unified history of **Artifacts**.

### Mission
A specific execution unit derived from a **Mission Pack**. Has a lifecycle state (`Running`, `Waiting`, `Completed`, `Failed`), composed of sequential steps.

### Mission Pack
A research blueprint (`JSON/YAML`) defining a capability-based step sequence. Examples: `repo_review_pack`, `lab_report_pack`, `microscopy_pack`.

### Artifact
A versioned output (CSV, PDF, code patch) produced by an agent. Carries **Lineage** (parent inputs) and a review status (`DRAFT`, `REVIEW_NEEDED`, `APPROVED`).

### ReviewThread & Comment
The Human-in-the-Loop (HITL) boundary. Critical artifacts trigger a `ReviewThread`; resolving it resumes the mission.

### ProjectTruthSnapshot
A consolidated view of approved information within a project — the verified "ground truth" layer.

---

## 🧱 System Layers

### API Layer — `src/bio_ml_agent/api/`
FastAPI-based REST surface.
- **`api_server.py`**: Main application, CORS, rate limiting, startup hooks.
- **`gateway_server.py`**: External-facing gateway (`/api/v1/gateway/*`).
- **`src/bio_ml_agent/routers/platform_routes.py`**: Canonical platform routes (projects, missions, artifacts, approvals, auth).
- **`src/bio_ml_agent/routers/dashboard_routes.py`**: Dashboard routes (chat, tasks, stats, config).

### Brain Layer — `src/bio_ml_agent/brain/`
The operating system kernel of Bio-ML Agent.
- **`mission_brain.py`**: Converts prompts to structured `MissionPlan` + agent DAGs + success criteria + fallback strategies.
- **`recovery_manager/`**: Checkpoints mission state after every step; resumes interrupted workflows.
- **`checkpoint_store/`**: Persistent JSON checkpoint store (`data/missions/`).
- **`artifact_graph.py`**: Tracks provenance and relationships between artifacts.
- **`workflow.py`**: `MissionGraphEngine` — executes step DAGs with dependency resolution.

### Service Layer — `src/bio_ml_agent/services/`
The orchestration and lifecycle layer.
- **`agent_service.py`**: Facade for UI/API — wraps the full agent loop (routing → tools → history).
- **`mission_pack_registry.py`**: Stores and serves research blueprints.
- **`agent_registry.py`**: Discovers and instantiates agents; reads `brain/agent_registry.yaml`.
- **`mission/`**: Mission orchestration helpers.
- **`agent/`**: Sub-modules: `orchestration`, `memory_context`, `execution_policy`, `project_lifecycle`.

### Core Layer — `src/bio_ml_agent/core/`
The execution engine for individual agent interactions.
- **`agent_core.py`**: Routes prompts to CHAT / TOOL_LOOP / ML_PIPELINE / SWARM strategies.
- **`tools.py`**: Secure, sandboxed execution of PYTHON, BASH, BROWSER, WEB_SEARCH, and other tool tags.
- **`conversation.py`**: Checkpoint-backed session and message history management.
- **`message_normalizer.py`**: Standard message adapters for all LLM backends.

### Agents Layer — `src/bio_ml_agent/agents/`
The modular "workforce", organized by domain family:

| Family | Directory | Maturity |
| :--- | :--- | :--- |
| Academic | `agents/academic/` | BETA |
| Biology | `agents/biology/` | ACTIVE |
| Browser | `agents/browser/` | STABLE |
| Coder | `agents/coder/` | STABLE |
| Critic | `agents/critic/` | STABLE |
| Dataset | `agents/dataset/` | STABLE |
| Document | `agents/document/` | STABLE |
| Microscopy | `agents/microscopy/` | ACTIVE |
| Omics | `agents/omics/` | EXPERIMENTAL |

### Legacy Layer — `src/bio_ml_agent/legacy/`
Contains the older ML pipeline, Swarm orchestrator, dataset catalog, and other modules that predate the Mission-Pack architecture. Still used by `AgentCore` (ML_PIPELINE and SWARM routes).

---

## 🧪 Quality & Security
- **Auto-Recovery**: `RecoveryManager` + `CheckpointStore` resume interrupted missions automatically.
- **HITL Gates**: `ReviewThread` pauses mission execution pending human approval.
- **Audit Trail**: All critical actions logged to `audit_logs/` as timestamped JSONL.
- **Strict CI**: Every change must pass `ruff` lint + `pytest` on GitHub Actions before merge.

---

## ⚙️ Technical Stack
| Component | Technology |
| :--- | :--- |
| Language | Python 3.9+ |
| API | FastAPI + Uvicorn |
| Validation | Pydantic v2 |
| Database | SQLite (SQLAlchemy) + file-based JSON checkpoints |
| Vector Memory | Qdrant |
| Task Queue | Redis + RQ |
| LLM Routing | LiteLLM (Ollama, OpenAI, Anthropic, Gemini) |
| Observability | OpenTelemetry |

---

*Last Updated: April 2026*

---

## 🇹🇷 Türkçe Mimari Özeti (Turkish)

Bio-ML Agent, **Mission-Pack tabanlı, merkezi olarak orkestre edilen** bir bilimsel platformdur.

- **Beyin Katmanı**: `MissionBrain` kullanıcı isteğini `MissionPlan` + ajan DAG'ına dönüştürür; `RecoveryManager` kesintileri kurtarır.
- **Servis Katmanı**: `AgentService` (facade), `MissionPackRegistry`, `AgentRegistry`.
- **Ajan Ailesi**: academic, biology, browser, coder, critic, dataset, document, microscopy — 9 domain ailesi.
- **Kalite**: HITL onay kapıları, değişmez denetim günlüğü, katı CI pipeline.
