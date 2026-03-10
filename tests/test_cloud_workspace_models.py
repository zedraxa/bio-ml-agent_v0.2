import unittest
from datetime import datetime
from models.cloud_workspace import (
    WorkspaceSnapshot, SandboxType, SandboxStatus,
    SandboxConfig, SyncEventType, SyncTrackRecord,
    SpendGuardrails
)

class TestCloudWorkspaceModels(unittest.TestCase):
    def test_workspace_snapshot(self):
        snapshot = WorkspaceSnapshot(
            snapshot_id="snap-001",
            project_id="proj-123",
            file_indices=["main.py:h1", "utils.py:h2"],
            env_snapshot={"DEBUG": "1"},
            git_commit_hash="abc123def",
            created_at=datetime.utcnow().isoformat(),
            size_mb=45.5
        )
        self.assertEqual(snapshot.snapshot_id, "snap-001")
        self.assertEqual(snapshot.size_mb, 45.5)

    def test_sandbox_config(self):
        config = SandboxConfig(
            sandbox_id="sbx-001",
            sandbox_type=SandboxType.BROWSER,
            image_tag="ultranode/browser-runtime:v2",
            resource_profile="16cpu-64ram",
            status=SandboxStatus.READY,
            active_since=datetime.utcnow().isoformat()
        )
        self.assertEqual(config.sandbox_type, SandboxType.BROWSER)
        self.assertEqual(config.status, SandboxStatus.READY)

    def test_sync_track_record(self):
        record = SyncTrackRecord(
            sync_id="sync-001",
            workspace_id="ws-789",
            event_type=SyncEventType.UPLOAD,
            file_paths=["data.csv", "config.json"],
            checksum_map={"data.csv": "hash1", "config.json": "hash2"},
            timestamp=datetime.utcnow().isoformat(),
            latency_ms=150
        )
        self.assertEqual(record.event_type, SyncEventType.UPLOAD)
        self.assertEqual(record.latency_ms, 150)

    def test_spend_guardrails(self):
        guard = SpendGuardrails(
            project_id="proj-123",
            max_hours=48.0,
            max_gpu_budget_usd=50.0,
            current_spend_usd=5.25
        )
        self.assertEqual(guard.project_id, "proj-123")
        self.assertEqual(guard.max_hours, 48.0)
        self.assertTrue(guard.auto_stop_enabled)

if __name__ == '__main__':
    unittest.main()
