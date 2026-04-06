# REST API Reference

Bio-ML Agent exposes two API surfaces via FastAPI (interactive docs at `/docs`):

| Server | Port | Base Path |
| :--- | :--- | :--- |
| Platform API | 8001 | `/api/v1/platform/` |
| Dashboard API | 8001 | `/` (served by `dashboard_routes.py`) |
| Gateway | 8000 | `/api/v1/gateway/` |

---

## 🔐 Authentication

All requests to `/api/v1/platform/` require an `X-API-Key` header (configured in `config.yaml` under `security.api_key`). If the key is empty, security is disabled.

```http
X-API-Key: your-api-key-here
```

You can also acquire a session token via:

### Login
`POST /api/v1/platform/auth/login`

---

## 📁 Projects

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/projects` | Create a new project |
| `GET` | `/projects` | List all projects |
| `GET` | `/projects/last-active` | Get the most recently active project |
| `GET` | `/projects/{project_id}` | Get project details |
| `GET` | `/projects/{project_id}/missions` | List all missions in a project |
| `GET` | `/projects/{project_id}/artifacts` | List all artifacts in a project |
| `GET` | `/projects/{project_id}/timeline` | Get project timeline events |
| `GET` | `/projects/{project_id}/dashboard` | Get project dashboard summary |
| `GET` | `/projects/{project_id}/memory` | Get project memory items |
| `GET` | `/projects/{project_id}/truth-snapshot` | Get the verified truth snapshot |
| `POST` | `/projects/{project_id}/memory` | Add a project memory item |
| `POST` | `/projects/{project_id}/invite` | Invite a collaborator |

---

## 🚀 Runs & Missions

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/runs` | Start a new mission run |
| `GET` | `/runs/{run_id}` | Get run details |
| `POST` | `/runs/{run_id}/events/emit` | Emit a run event |
| `GET` | `/sessions/{session_id}` | Get a remote session registry |
| `GET` | `/missions/active` | List all active missions |
| `POST` | `/missions/{mission_id}/steps` | Submit a mission step result |
| `GET` | `/missions/{mission_id}/steps` | Get mission steps |
| `POST` | `/missions/{mission_id}/pause` | Pause a running mission |
| `POST` | `/missions/{mission_id}/resume` | Resume a paused mission |
| `POST` | `/missions/{mission_id}/cancel` | Cancel a mission |
| `POST` | `/missions/{mission_id}/intervene` | Submit a human intervention |

---

## ✅ Approvals

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/approvals/action` | Approve or reject a risky action |
| `POST` | `/approvals/browser/takeover` | Approve a browser takeover event |

---

## 📦 Artifacts

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/artifacts/{artifact_id}/manifest` | Get artifact manifest |
| `POST` | `/artifacts/{artifact_id}/approve` | Approve an artifact |
| `POST` | `/artifacts/{artifact_id}/reject` | Reject an artifact |
| `GET` | `/artifacts/{artifact_id}/comments` | List comments on an artifact |
| `POST` | `/artifacts/{artifact_id}/comments` | Add a comment to an artifact |
| `POST` | `/comments/{comment_id}/resolve` | Resolve a comment |
| `GET` | `/projects/{project_id}/review-threads` | List all review threads for a project |
| `POST` | `/artifacts/{artifact_id}/status` | Update artifact review status |
| `GET` | `/artifacts/{artifact_id}/export` | Export an artifact |

---

## 🔔 Notifications & Settings

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/notifications` | List notifications |
| `GET` | `/settings/{settings_id}` | Get settings |
| `POST` | `/settings/{settings_id}` | Update settings |
| `GET` | `/integrations/status` | Check integration status |

---

## 📊 Dashboard API

These endpoints are served directly on the API server for the Gradio dashboard integration:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/agent/chat` | Send a message to the agent |
| `GET` | `/stats` | Get system statistics |
| `GET` | `/tasks` | List dashboard tasks |
| `GET` | `/projects` | List projects (dashboard view) |
| `GET` | `/models` | List available LLM models |
| `GET` | `/datasets` | List available datasets |
| `GET` | `/config` | Get current configuration |
| `GET` | `/config/api-keys` | Get API key status |
| `GET` | `/ollama/models` | List locally available Ollama models |

---

## 🩺 System

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Health check |
| `GET` | `/api/v1/observability/metrics` | Get observability metrics |
| `GET` | `/api/v1/observability/audit` | Get audit log |

---

## 🔒 Security Notes
- API key is set via `security.api_key` in `config.yaml` (or `API_KEY` env var).
- Webhook payloads are verified using HMAC-SHA256 (`security.webhook_secret`).
- Rate limiting is enforced globally via SlowAPI.

---

## 🇹🇷 Türkçe API Özeti (Turkish)

Platform API `http://localhost:8001` adresinde çalışır. Tüm isteklerde `X-API-Key` header'ı gereklidir (boşsa devre dışı).

Ana endpoint grupları:
- `/api/v1/platform/projects/*` — Proje yönetimi
- `/api/v1/platform/missions/*` — Görev (mission) yönetimi
- `/api/v1/platform/artifacts/*` — Artifact yönetimi ve onaylar
- `/api/v1/platform/approvals/*` — HITL onay akışı
- `/health`, `/api/v1/observability/*` — Sistem sağlığı ve gözlemlenebilirlik
