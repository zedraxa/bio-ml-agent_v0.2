# Bio-ML Agent Architecture

## Overview
Bio-ML Agent is an autonomous research platform designed for complex, multi-stage scientific workflows. It uses a **Mission-Pack** based orchestration model to coordinate specialized agents across domains like microscopy, genomics, and academic reporting.

## Core Concepts

### 1. Project
The top-level container for research. A project tracks multiple **Missions** and maintains a unified history of **Artifacts**.

### 2. Mission
A specific execution unit derived from a **Mission Pack**. A mission has a state (Running, Waiting, Completed, Failed) and is composed of multiple steps.

### 3. Mission Pack
A research blueprint (JSON/YAML) that defines a sequence of capability-based steps. Examples include `repo_review_pack`, `lab_report_pack`, and `microscopy_pack`.

### 4. Artifact
A versioned output produced by an agent (e.g., a CSV, PDF, or code patch). Artifacts carry **Lineage** (tracking parent inputs) and have a review status (`DRAFT`, `REVIEW_NEEDED`, `APPROVED`).

### 5. ReviewThread & Comment
The Human-in-the-Loop (HITL) boundary. Critical artifacts trigger a `ReviewThread` where users can add `Comments`. Resolving or approving these threads allows missions to resume.

### 6. ProjectTruthSnapshot
A consolidated view of verified information within a project. As artifacts are approved, they are promoted to the "Project Truth" layer.

## System Layers

### [API Layer](file:///src/bio_ml_agent/api)
FastAPI-based REST surface. Provides endpoints for project management, mission execution, and artifact retrieval.
- **Router**: [platform_routes.py](file:///src/bio_ml_agent/routers/platform_routes.py)

### [Service Layer](file:///src/bio_ml_agent/services)
The operational core.
- **MissionOrchestrator**: Executes packs, handles step transitions, and pauses at approval gates.
- **AgentRegistry**: Dynamically discovers and instantiates specialized agents.
- **MissionPackRegistry**: Stores standardized research blueprints.

### [Brain Layer](file:///src/bio_ml_agent/brain)
State management and recovery.
- **RecoveryManager**: Checkpoints mission state after every step.
- **Models**: Pydantic-based contracts for execution and telemetry.

### [Persistence Layer](file:///src/bio_ml_agent/db)
Relational storage using SQLite and SQLAlchemy.
- **Models**: [models.py](file:///src/bio_ml_agent/db/models.py)

### [Specialized Agents](file:///src/bio_ml_agent/agents)
The modular "workforce". Categorized by domain (Academic, Biology, Coder, Microscopy, etc.).

---

## Technical Stack
- **Language**: Python 3.12+
- **Database**: SQLite (SQLAlchemy)
- **API**: FastAPI
- **Validation**: Pydantic v2
- **Persistence**: File-based + Relational DB
 Riverside
