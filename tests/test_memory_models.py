import unittest
from datetime import datetime
from models.memory import (
    MemoryScope, MemoryLayer, TrustLevel, 
    MemoryTrustScore, MemoryEntry, 
    ConflictStrategy, MemoryConflict, 
    MemoryTenantMeta
)

class TestMemoryModels(unittest.TestCase):
    def test_memory_entry_creation(self):
        trust = MemoryTrustScore(
            level=TrustLevel.OBSERVED,
            score=0.95,
            reason="Direct browser observation"
        )
        entry = MemoryEntry(
            entry_id="mem-001",
            content={"molecule": "C6H12O6", "name": "Glucose"},
            scope=MemoryScope.PROJECT,
            layer=MemoryLayer.BIO,
            trust_score=trust,
            created_by="bio-agent-1",
            created_at=datetime.utcnow().isoformat()
        )
        self.assertEqual(entry.layer, MemoryLayer.BIO)
        self.assertEqual(entry.trust_score.level, TrustLevel.OBSERVED)
        self.assertEqual(entry.content["name"], "Glucose")

    def test_memory_conflict(self):
        trust1 = MemoryTrustScore(level=TrustLevel.INFERRED, score=0.6)
        entry1 = MemoryEntry(
            entry_id="mem-002a",
            content="Temperature is 37C",
            layer=MemoryLayer.BIO,
            trust_score=trust1,
            created_by="agent-a",
            created_at=datetime.utcnow().isoformat()
        )
        
        trust2 = MemoryTrustScore(level=TrustLevel.USER_CONFIRMED, score=1.0)
        entry2 = MemoryEntry(
            entry_id="mem-002b",
            content="Temperature is 38C",
            layer=MemoryLayer.BIO,
            trust_score=trust2,
            created_by="agent-b",
            created_at=datetime.utcnow().isoformat()
        )

        conflict = MemoryConflict(
            conflict_id="con-001",
            key="body_temperature",
            entries=[entry1, entry2],
            strategy=ConflictStrategy.REVIEW_REQUIRED
        )
        self.assertEqual(len(conflict.entries), 2)
        self.assertEqual(conflict.strategy, ConflictStrategy.REVIEW_REQUIRED)

    def test_multitenancy_meta(self):
        meta = MemoryTenantMeta(
            project_id="proj-x",
            tenant_id="user-y",
            access_level="admin"
        )
        self.assertEqual(meta.project_id, "proj-x")
        self.assertEqual(meta.access_level, "admin")

if __name__ == '__main__':
    unittest.main()
