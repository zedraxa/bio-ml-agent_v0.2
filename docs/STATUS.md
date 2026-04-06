# 📦 Project Status & Branch Structure

Bio-ML Agent uses the following branch strategy to manage its development lifecycle.

## 🌿 Main Branches

### `main`
- **Purpose**: Stable, tested, and production-ready code.
- **Status**: Contains the latest features including Mission-Pack orchestration and microscopy agents.
- **Use**: Use this branch for end-users and demos.

### `experimental`
- **Purpose**: The "breakable" playground where new features are first tried.
- **Status**: Merged into `main` when development is complete.
- **Use**: Work on this branch when adding a new tool or agent capability.

## 🛠️ Current Development Status (v0.1.0-clean / API v6.0.0)

### Completed:
- **Architecture**: Mission-Pack based orchestration with MissionBrain, RecoveryManager, ArtifactGraph.
- **API**: FastAPI platform routes (projects, missions, artifacts, approvals, auth, notifications).
- **Agent Families**: 9 domain families — Browser (STABLE), Coder (STABLE), Biology (ACTIVE), Microscopy (ACTIVE), Academic (BETA).
- **QA**: Full test suite with `ruff` linting and `pytest` (290+ passing tests).
- **Core**: Checkpoint-backed automatic recovery (CheckpointStore + RecoveryManager).
- **Advanced Features**:
  - RAG + Qdrant (hybrid semantic + BM25 search).
  - Redis Background Worker (RQ task queue).
  - WhatsApp Gateway (Twilio/Flask).
  - Temporal (durable workflow integration).
  - Multi-agent Swarm (6 specialist agents).
  - OpenTelemetry observability.

---

*Last Updated: April 2026 (v0.1.0-clean)*

---

## 🇹🇷 Türkçe Durum Özeti (Turkish)

**Ana Dallar:**
- `main` — Kararlı, yayına hazır kod.
- `experimental` — Yeni özelliklerin geliştirildiği alan; tamamlanınca `main`'e merge edilir.

**Tamamlananlar:** Mission-Pack mimarisi, 9 ajan ailesi, platform API, RAG+Qdrant, Redis Worker, WhatsApp, Temporal, 6 ajanlı swarm, OpenTelemetry gözlemlenebilirliği.
