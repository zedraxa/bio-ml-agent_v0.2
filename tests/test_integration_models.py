import unittest
from models.integration import (
    ControlPlaneState, StepSource, UnifiedRunGraphNode, UnifiedRunGraph, GlobalIdentity, ArtifactLineageNode, ArtifactLineage, GlobalPolicy
)

class TestIntegrationModels(unittest.TestCase):
    def test_control_plane_state(self):
        state = ControlPlaneState(
            active_runs=10,
            active_browser_sessions=2,
            running_bio_pipelines=1,
            system_health_score=0.99
        )
        self.assertEqual(state.active_runs, 10)

    def test_artifact_lineage(self):
        node = ArtifactLineageNode(
            artifact_id="art-123",
            artifact_type="report",
            produced_by_run_id="run-456",
            derived_from_artifact_ids=["dataset-1", "model-2"]
        )
        lineage = ArtifactLineage(
            root_artifact_id="art-123",
            lineage_nodes=[node]
        )
        self.assertEqual(len(lineage.lineage_nodes[0].derived_from_artifact_ids), 2)

    def test_global_policy(self):
        policy = GlobalPolicy(
            policy_id="pol-001",
            max_budget_usd_per_run=10.0,
            allow_public_publishing=False
        )
        self.assertFalse(policy.allow_public_publishing)

if __name__ == '__main__':
    unittest.main()
