import unittest
from datetime import datetime, timezone
from bio_ml_agent.models.lifecycle import (
    DataStage, DataVersion, ExperimentRun, ModelStatus,
    ModelArtifact, LineageNodeType, LineageNode, ReproBundle
)

class TestLifecycleModels(unittest.TestCase):
    def test_data_versioning(self):
        dv = DataVersion(
            version_id="dv-001",
            dataset_name="genomic_sequences",
            stage=DataStage.CLEANED,
            storage_artifact_id="art-123",
            row_count=5000,
            schema_hash="hash_xy",
            created_at=datetime.now(timezone.utc).isoformat(),
            created_by_run_id="run-789"
        )
        self.assertEqual(dv.stage, DataStage.CLEANED)
        self.assertEqual(dv.row_count, 5000)

    def test_experiment_run(self):
        run = ExperimentRun(
            run_id="run-789",
            experiment_name="dna_sequencing_v1",
            params={"batch_size": 32, "lr": 0.001},
            metrics={"accuracy": 0.95},
            artifact_ids=["art-1", "art-2"],
            start_time=datetime.now(timezone.utc).isoformat(),
            status="completed"
        )
        self.assertEqual(run.experiment_name, "dna_sequencing_v1")
        self.assertEqual(run.metrics["accuracy"], 0.95)

    def test_model_registry(self):
        model = ModelArtifact(
            model_id="mod-001",
            name="sequence_classifier",
            version="1.0.0",
            framework="PyTorch",
            status=ModelStatus.CANDIDATE,
            storage_path="s3://models/seq_v1.pt",
            created_at=datetime.now(timezone.utc).isoformat()
        )
        self.assertEqual(model.framework, "PyTorch")
        self.assertEqual(model.status, ModelStatus.CANDIDATE)

    def test_lineage_and_repro(self):
        node = LineageNode(
            node_id="lin-001",
            type=LineageNodeType.MODEL,
            reference_id="mod-001",
            parents=["run-789", "dv-001"]
        )
        self.assertIn("run-789", node.parents)

        repro = ReproBundle(
            bundle_id="rep-001",
            experiment_run_id="run-789",
            required_data_version_ids=["dv-001"],
            environment_yaml_url="s3://envs/bio_env.yaml",
            entrypoint_script="train.py",
            created_at=datetime.now(timezone.utc).isoformat()
        )
        self.assertEqual(repro.entrypoint_script, "train.py")
        self.assertFalse(repro.is_verified)

if __name__ == '__main__':
    unittest.main()
