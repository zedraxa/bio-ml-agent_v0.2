# 📦 Project Status & Branch Structure

Bio-ML Agent uses the following branch strategy to manage its development lifecycle.

## 🌿 Main Branches

### `main`
- **Purpose**: Stable, tested, and production-ready code.
- **Status**: Contains the latest features (including Deep Research).
- **Use**: Use this branch for end-users and demos.

### `experimental`
- **Purpose**: The "breakable" playground where new features are first tried.
- **Status**: Merged into `main` when development is complete.
- **Use**: Work on this branch when adding a new tool or agent capability.

## 🛠️ Current Development Status (v0.2.0-full)

### Completed (v0.2.0-full):
- **Architecture**: Modular UI and Service layers separated (`src/bio_ml_agent`).
- **QA**: Full test suite (Smoke/Integration) with `ruff` linting and `pytest`.
- **Core**: Checkpoint-backed automatic session recovery (Auto-recovery).
- **Phase 5 (Advanced Features)**:
  - RAG + Qdrant (scientific document search and indexing).
  - Redis Background Worker (queue system for long-running analyses).
  - WhatsApp Gateway (media & report support).
  - Temporal (durable scientific workflows).
  - Multi-agent Swarm (4 specialist agents).

---

*Last Updated: March 20, 2026 (v0.2.0-full)*

---

## 🇹🇷 Türkçe Durum Özeti (Turkish)

**Ana Dallar:**
- `main` — Kararlı, yayına hazır kod.
- `experimental` — Yeni özelliklerin geliştirildiği alan; tamamlanınca `main`'e merge edilir.

**v0.2.0-full Tamamlananlar:** Modüler mimari, CI/test altyapısı, otomatik oturum kurtarma, RAG+Qdrant, Redis Worker, WhatsApp Gateway, Temporal iş akışları, çoklu ajan swarm.
