import sys
from bio_ml_agent.legacy.ultra_agent.memory.maintenance import (
    MemoryMerger as _LegacyMemoryMerger,
    synthesize_memories_llm as _leg_synthesize,
)

# Module-level patchable reference (tests patch this via `bio_ml_agent.ultra_agent.memory.maintenance.synthesize_memories_llm`)
synthesize_memories_llm = _leg_synthesize


class MemoryMerger(_LegacyMemoryMerger):
    """
    Thin wrapper that delegates to the legacy MemoryMerger but ensures
    `synthesize_memories_llm` resolves via this shim's namespace so test
    patches targeting ``bio_ml_agent.ultra_agent.memory.maintenance`` work.
    """

    def merge_project_memories(self, project: str, similarity_threshold: float = 0.90) -> int:
        import bio_ml_agent.legacy.ultra_agent.memory.maintenance as _leg_mod

        _shim = sys.modules[__name__]
        _original = _leg_mod.synthesize_memories_llm
        # Redirect the legacy module's reference to whichever version is currently
        # bound in this shim (may be replaced by unittest.mock.patch).
        _leg_mod.synthesize_memories_llm = _shim.synthesize_memories_llm
        try:
            return super().merge_project_memories(project, similarity_threshold)
        finally:
            _leg_mod.synthesize_memories_llm = _original


__all__ = ["MemoryMerger", "synthesize_memories_llm"]
