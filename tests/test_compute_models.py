import unittest
from datetime import datetime, timezone
from bio_ml_agent.models.compute import (
    WorkloadType, ResourceRequest, ExecutionLocation,
    ComputeNodeStatus, ComputeNode, ExecutionDecision,
    TaskEnvelope, ComputeCheckpoint
)

class TestComputeModels(unittest.TestCase):
    def test_resource_request(self):
        req = ResourceRequest(
            workload_type=WorkloadType.GPU_BOUND,
            min_cpu_cores=8,
            min_ram_gb=32.0,
            requires_gpu=True,
            gpu_type="A100"
        )
        self.assertEqual(req.workload_type, WorkloadType.GPU_BOUND)
        self.assertTrue(req.requires_gpu)

    def test_compute_node_and_decision(self):
        node = ComputeNode(
            node_id="node-cloud-01",
            location=ExecutionLocation.REMOTE,
            address="https://compute.cloud.provider/v1",
            total_cpu_cores=64,
            total_ram_gb=256.0,
            has_gpu=True,
            cost_per_hour=2.5
        )
        self.assertEqual(node.location, ExecutionLocation.REMOTE)
        
        decision = ExecutionDecision(
            task_id="task-train-01",
            selected_node_id="node-cloud-01",
            location=ExecutionLocation.REMOTE,
            reason="Local CPU insufficient for A100 requirement",
            estimated_cost=5.0
        )
        self.assertEqual(decision.selected_node_id, "node-cloud-01")

    def test_task_envelope(self):
        envelope = TaskEnvelope(
            envelope_id="env-001",
            task_id="task-train-01",
            container_image="ultranode/bio-ml-runtime:latest",
            entrypoint=["python", "train_model.py"],
            env_vars={"DATASET_ID": "ds_99"},
            secrets_scope=["BENCHLING_API_KEY"]
        )
        self.assertEqual(envelope.container_image, "ultranode/bio-ml-runtime:latest")
        self.assertIn("BENCHLING_API_KEY", envelope.secrets_scope)

    def test_compute_checkpoint(self):
        cp = ComputeCheckpoint(
            checkpoint_id="chk-001",
            task_id="task-train-01",
            node_id="node-cloud-01",
            timestamp=datetime.now(timezone.utc).isoformat(),
            state_payload_url="s3://checkpoints/task-train-01/chk-001.bin",
            iteration_count=5000
        )
        self.assertEqual(cp.iteration_count, 5000)
        self.assertTrue(cp.is_restorable)

if __name__ == '__main__':
    unittest.main()
