import unittest
from models.cloud_offload import (
    ExecutionTarget, JobClassification, OffloadPolicy,
    RuntimePackage, CheckpointResumeStrategy
)

class TestCloudOffloadModels(unittest.TestCase):
    def test_job_classification(self):
        job = JobClassification(
            classification_id="job-01",
            target=ExecutionTarget.REMOTE_GPU,
            estimated_memory_mb=16384,
            requires_gpu=True,
            reasoning="VRAM ihtiyacı yüksek."
        )
        self.assertEqual(job.target, ExecutionTarget.REMOTE_GPU)
        self.assertTrue(job.requires_gpu)

    def test_offload_policy(self):
        policy = OffloadPolicy(
            policy_id="pol-01",
            max_cost_limit_usd=10.0,
            require_human_approval_for_remote=False
        )
        self.assertFalse(policy.require_human_approval_for_remote)
        self.assertEqual(policy.max_cost_limit_usd, 10.0)

    def test_runtime_package(self):
        package = RuntimePackage(
            package_id="pkg-abc",
            run_id="run-123",
            workspace_snapshot_uri="s3://ai-agent/snapshots/run-123.zip",
            secret_scopes_allowed=["HUGGINGFACE_TOKEN", "WANDB_API_KEY"],
            callback_webhook_url="https://agent-gateway.internal/webhook/run-123",
            env_vars={"NODE_ENV": "production"}
        )
        self.assertEqual(len(package.secret_scopes_allowed), 2)
        self.assertEqual(package.env_vars["NODE_ENV"], "production")

    def test_checkpoint_resume_strategy(self):
        strategy = CheckpointResumeStrategy(
            strategy_id="strat-01",
            run_id="run-123",
            in_progress_lock_id="lock::run-123",
            retry_count=1,
            max_retries=3
        )
        self.assertTrue(strategy.is_resumable)
        self.assertEqual(strategy.in_progress_lock_id, "lock::run-123")

if __name__ == '__main__':
    unittest.main()
