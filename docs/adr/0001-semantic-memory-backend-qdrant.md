# ADR 0001: Semantic Memory Backend Standardization

## Status
Proposed

## Context
The project currently has two parallel vector database implementations for memory and RAG:
1. **ChromaDB**: Used in `memory_manager.py` (legacy interaction memory) and `rag_engine.py` (document RAG).
2. **Qdrant**: Implemented in `ultra_agent/memory/qdrant_store.py` as part of the new ultra-agent architecture, supporting TTL, project scoping, and provenance.

Having two backends increases maintenance overhead and leads to fragmented memory (context exists in one but not the other). The user's vision for "Semantic Memory" requires advanced features like forgetting (TTL), scoping, and provenance, which are already prototyped in the Qdrant implementation.

## Decision
We will standardize **Qdrant** as the single official backend for all "Semantic Memory" operations.
- **ChromaDB** will be moved to a "Legacy/Fallback" status.
- New features must use the unified Qdrant-based memory contract.
- Existing interaction memory in `AgentService` will be migrated to Qdrant.

## Consequences
- **Positive**: Single source of truth for agent memories, structured metadata (tags, projects), and provenance tracking.
- **Positive**: Better scalability and feature set (TTL, advanced filtering).
- **Negative**: Requires adding `qdrant-client` as a core dependency.
- **Negative**: Requires migration effort for existing Chroma data (or starting fresh with the new standard).

## Implementation Path
- **Phase 0**: This ADR.
- **Phase 1**: Dependency (qdrant-client) and Config standardization.
- **Phase 2**: Define `BaseMemoryStore` interface and implement Qdrant provider.
- **Phase 3**: Refactor `AgentService` and `AgentCore` to use the new interface.
- **Phase 4**: Migration/Standardization of Document RAG and legacy cleanup.
