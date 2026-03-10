import unittest
from models.workflow import (
    WorkflowStatus, DurableWorkflow, ChildWorkflow, ErrorType, RetryPolicy, WorkflowSchedule, WorkflowCheckpoint
)

class TestWorkflowModels(unittest.TestCase):
    def test_durable_workflow(self):
        wf = DurableWorkflow(
            workflow_id="wf-001",
            name="Genome Analysis",
            status=WorkflowStatus.RUNNING,
            input_data={"seq": "ATGC"}
        )
        self.assertEqual(wf.status, WorkflowStatus.RUNNING)
        self.assertEqual(wf.input_data["seq"], "ATGC")

    def test_retry_policy(self):
        policy = RetryPolicy(
            policy_id="pol-01",
            max_retries=3,
            retryable_error_types=[ErrorType.TRANSIENT],
            timeout_seconds=30
        )
        self.assertEqual(policy.max_retries, 3)

    def test_workflow_checkpoint(self):
        cp = WorkflowCheckpoint(
            checkpoint_id="cp-01",
            workflow_id="wf-001",
            reason="Human approval needed",
            required_action="approve"
        )
        self.assertFalse(cp.is_resolved)

if __name__ == '__main__':
    unittest.main()
