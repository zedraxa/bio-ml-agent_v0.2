# Legacy Cleanup Policy

This document defines the lifecycle and removal schedule for components in the `legacy/` directory.

## Current Legacy Components

| Component | Status | Replacement | Targeted Removal |
| :--- | :--- | :--- | :--- |
| `legacy/agent.py` | Deprecated | `src/bio_ml_agent/agent.py` & `ultra_agent/` | Phase 4 (Stabilization) |
| `legacy/memory_manager.py` | Deprecated | `src/bio_ml_agent/core/memory.py` | Phase 4 (Stabilization) |
| `legacy/multi_agent.py` | Experimental | `src/bio_ml_agent/swarm/` | Phase 5 (Advanced) |
| `legacy/rag_engine.py` | Deprecated | `src/bio_ml_agent/core/rag/` | Phase 5 (Advanced) |

## Removal Criteria

1.  **Parity:** The replacement component must implement all features used by current active services.
2.  **Verification:** The replacement must pass all integration tests.
3.  **No References:** No code in `src/bio_ml_agent/` should import from the component.

## Cleanup Status (v0.1.0-clean)

- **Phase 2-4:** ✅ **COMPLETED**. All active logic has been moved to `src/bio_ml_agent/`. 
- **Phase 5:** 🚧 **IN PROGRESS**. Final deletion of redundant files in `legacy/` after ensuring no third-party scripts depend on them.

No new code should be added to the root directory or `legacy/`. All contributions must go into `src/bio_ml_agent/`.
