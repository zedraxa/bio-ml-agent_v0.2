import unittest
from datetime import datetime, timezone
from bio_ml_agent.models.cloud_storage import (
    StorageArtifact, SignedAccessURL, ProjectSnapshot,
    DataBranch, OfflineCacheMeta, SyncConflict
)

class TestCloudStorageModels(unittest.TestCase):
    def test_storage_artifact(self):
        artifact = StorageArtifact(
            artifact_id="art-001",
            project_id="proj-abc",
            path="data/dataset.csv",
            version_id="v1.2",
            checksum="sha256:hash123",
            size_bytes=102456,
            content_type="text/csv",
            created_at=datetime.now(timezone.utc).isoformat()
        )
        self.assertEqual(artifact.path, "data/dataset.csv")
        self.assertEqual(artifact.version_id, "v1.2")

    def test_signed_url(self):
        url = SignedAccessURL(
            artifact_id="art-001",
            url="https://s3.amazonaws.com/bucket/file?sig=xxx",
            expires_at="2026-03-10T20:00:00Z"
        )
        self.assertEqual(url.http_method, "GET")
        self.assertTrue(url.url.startswith("https"))

    def test_project_snapshot(self):
        snapshot = ProjectSnapshot(
            snapshot_id="snap-456",
            project_id="proj-abc",
            timestamp=datetime.now(timezone.utc).isoformat(),
            code_version="git:commit123",
            data_manifest_id="art-manifest-01",
            report_ids=["rep-01", "rep-02"]
        )
        self.assertEqual(snapshot.code_version, "git:commit123")
        self.assertIn("rep-01", snapshot.report_ids)

    def test_data_branch(self):
        branch = DataBranch(
            branch_id="br-001",
            project_id="proj-abc",
            name="experimental-feature",
            base_snapshot_id="snap-base",
            head_snapshot_id="snap-456",
            created_by_agent="agent-007"
        )
        self.assertEqual(branch.name, "experimental-feature")
        self.assertFalse(branch.is_merged)

    def test_cache_and_conflict(self):
        meta = OfflineCacheMeta(
            file_path="src/main.py",
            last_synced_checksum="hash-old",
            last_synced_at=datetime.now(timezone.utc).isoformat(),
            local_modification_at=datetime.now(timezone.utc).isoformat(),
            is_dirty=True
        )
        self.assertTrue(meta.is_dirty)

        conflict = SyncConflict(
            conflict_id="conf-01",
            file_path="src/main.py",
            local_checksum="hash-local",
            remote_checksum="hash-remote",
            local_updated_at=datetime.now(timezone.utc).isoformat(),
            remote_updated_at=datetime.now(timezone.utc).isoformat(),
            detected_at=datetime.now(timezone.utc).isoformat(),
            resolution_strategy="use_local"
        )
        self.assertEqual(conflict.resolution_strategy, "use_local")

if __name__ == '__main__':
    unittest.main()
