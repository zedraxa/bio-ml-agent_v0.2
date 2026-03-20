import unittest
from bio_ml_agent.models.remote_storage import (
    CloudArtifactCategory, CloudArtifactItem,
    RemoteFileActionType, RemoteFileBrowserAction,
    SyncDirection, ConflictResolutionStrategy, ProjectSyncSnapshot,
    ProjectAccessRole, SharedProjectAccess
)

class TestRemoteStorageModels(unittest.TestCase):
    def test_cloud_artifact(self):
        item = CloudArtifactItem(
            artifact_id="art-123",
            project_id="proj-456",
            category=CloudArtifactCategory.EXPERIMENT_OUTPUT,
            file_name="results.csv",
            size_bytes=10240,
            storage_path_uri="s3://ai-agent-bucket/proj-456/results.csv",
            created_at="2026-03-10T22:00:00"
        )
        self.assertEqual(item.category, CloudArtifactCategory.EXPERIMENT_OUTPUT)

    def test_remote_file_action(self):
        action = RemoteFileBrowserAction(
            action_id="act-file-1",
            user_id="usr-99",
            action_type=RemoteFileActionType.UPLOAD,
            target_path="data/uploads/",
            device_fingerprint="ios-app",
            supports_drag_and_drop=True,
            timestamp="2026-03-10T22:05:00"
        )
        self.assertTrue(action.supports_drag_and_drop)

    def test_project_sync_snapshot(self):
        snapshot = ProjectSyncSnapshot(
            sync_id="sync-xyz",
            project_id="proj-456",
            user_id="usr-99",
            direction=SyncDirection.LOCAL_TO_CLOUD,
            local_workspace_hash="hash-abc",
            cloud_workspace_hash="hash-def",
            has_conflicts=True,
            conflict_resolution=ConflictResolutionStrategy.MANUAL_MERGE,
            synced_files_count=15,
            timestamp="2026-03-10T22:15:00"
        )
        self.assertEqual(snapshot.conflict_resolution, ConflictResolutionStrategy.MANUAL_MERGE)
        self.assertTrue(snapshot.has_conflicts)

    def test_shared_project_access(self):
        access = SharedProjectAccess(
            access_id="acc-01",
            project_id="proj-456",
            granted_user_id="usr-coworker",
            granted_by_user_id="usr-99",
            role=ProjectAccessRole.REPORT_VIEWER
        )
        self.assertEqual(access.role, ProjectAccessRole.REPORT_VIEWER)

if __name__ == '__main__':
    unittest.main()
