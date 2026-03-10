import unittest
from models.ephemeral_workspace import (
    SandboxImageFlavor, NodeState, WorkspaceBudgetAndQuota, EphemeralWorkspaceInfo
)

class TestEphemeralWorkspaceModels(unittest.TestCase):
    def test_budget_and_quota(self):
        quota = WorkspaceBudgetAndQuota(
            quota_id="quota-fast",
            max_duration_seconds=7200,
            idle_timeout_seconds=300,
            max_gpu_count=1,
            max_cost_usd_per_node=5.5
        )
        self.assertEqual(quota.max_gpu_count, 1)
        self.assertEqual(quota.idle_timeout_seconds, 300)

    def test_ephemeral_workspace_info(self):
        workspace = EphemeralWorkspaceInfo(
            workspace_id="ws-uuid-999",
            run_id="run-k99",
            user_id="usr-1",
            image_flavor=SandboxImageFlavor.MACHINE_LEARNING,
            state=NodeState.RUNNING_JOB,
            quota_rules=WorkspaceBudgetAndQuota(quota_id="q1", max_gpu_count=4),
            pulled_local_artifact_refs=["dataset.csv", "main.py"],
            created_at="2026-03-10T22:30:00"
        )
        self.assertEqual(workspace.image_flavor, SandboxImageFlavor.MACHINE_LEARNING)
        self.assertEqual(workspace.quota_rules.max_gpu_count, 4)
        self.assertEqual(len(workspace.pulled_local_artifact_refs), 2)
        self.assertEqual(workspace.state, NodeState.RUNNING_JOB)

if __name__ == '__main__':
    unittest.main()
