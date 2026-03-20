import unittest
from bio_ml_agent.models.remote_ide import (
    CellEditorRole, CellExecutionState, NotebookCellActivity, LiveNotebookSession,
    PatchApprovalStatus, RemotePatchReviewEvent, StagedCloudDataset
)

class TestRemoteIDEModels(unittest.TestCase):
    def test_notebook_cell_activity(self):
        cell = NotebookCellActivity(
            cell_id="c-def",
            last_edited_by=CellEditorRole.AGENT,
            code_content="import pandas as pd",
            execution_state=CellExecutionState.SUCCESS,
            execution_time_sec=1.5
        )
        self.assertEqual(cell.last_edited_by, CellEditorRole.AGENT)
        self.assertEqual(cell.execution_state, CellExecutionState.SUCCESS)

    def test_live_notebook_session(self):
        session = LiveNotebookSession(
            session_id="n-ses-123",
            workspace_id="ws-99",
            notebook_path="/notebooks/data_analysis.ipynb",
            human_interventions_count=2
        )
        self.assertEqual(session.human_interventions_count, 2)

    def test_remote_patch_review_event(self):
        patch = RemotePatchReviewEvent(
            patch_id="patch-555",
            run_id="run-u3",
            pull_request_or_commit_title="Fix dataframe indexing",
            diff_content_preview="+ data.iloc[0]\n- data[0]",
            files_changed=1,
            status=PatchApprovalStatus.PENDING,
            generated_at="2026-03-10T23:00:00"
        )
        self.assertEqual(patch.status, PatchApprovalStatus.PENDING)
        self.assertEqual(patch.files_changed, 1)

    def test_staged_cloud_dataset(self):
        dataset = StagedCloudDataset(
            dataset_id="ds-tumor-img",
            workspace_id="ws-99",
            dataset_name="Brain_Tumor_MRI_500GB",
            storage_size_gb=500.5,
            is_mounted_on_workspace=True,
            local_proxy_placeholder_uri="cloud://ds-tumor-img/",
            access_format="s3_bucket"
        )
        self.assertTrue(dataset.is_mounted_on_workspace)
        self.assertEqual(dataset.storage_size_gb, 500.5)

if __name__ == '__main__':
    unittest.main()
