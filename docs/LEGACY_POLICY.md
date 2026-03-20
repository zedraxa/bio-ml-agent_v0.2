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

## Cleanup Process

- **Phase 2-3:** Ongoing refactoring to move active logic out of legacy.
- **Phase 4:** Official removal of redundant core components.
- **Phase 5:** Final cleanup of all experimental/legacy bits.
