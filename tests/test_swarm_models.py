import unittest
from datetime import datetime
from models.swarm import (
    AgentRole, AgentCapability,
    SwarmMessageType, SwarmMessage,
    WorkspaceLock, MemoryTag,
    SharedVectorState, DelegationPolicy,
    TaskDelegationContract
)

class TestSwarmModels(unittest.TestCase):
    def test_agent_capabilities(self):
        cap = AgentCapability(
            role=AgentRole.DATA_SCIENTIST,
            tools_available=["train_model", "plot_results"],
            can_delegate=False
        )
        self.assertEqual(cap.role, AgentRole.DATA_SCIENTIST)
        self.assertIn("train_model", cap.tools_available)

    def test_swarm_messaging(self):
        msg = SwarmMessage(
            message_id="msg-001",
            sender_id="orchestrator-1",
            receiver_id="worker-bio-1",
            type=SwarmMessageType.TASK_DELEGATION,
            payload={"task": "sequence_analysis", "target": "human_genome"},
            priority=5,
            timestamp=datetime.utcnow().isoformat()
        )
        self.assertEqual(msg.priority, 5)
        self.assertEqual(msg.type, SwarmMessageType.TASK_DELEGATION)

    def test_shared_memory_and_tags(self):
        memo = SharedVectorState(
            document_id="doc-bio-ref-1",
            content_preview="Analyzing protein folding sequences...",
            tags=[MemoryTag.BIO_LAYER, MemoryTag.ML_LAYER],
            created_by_agent="bio-agent-1"
        )
        self.assertIn(MemoryTag.BIO_LAYER, memo.tags)
        self.assertEqual(memo.created_by_agent, "bio-agent-1")

    def test_task_delegation_flow(self):
        contract = TaskDelegationContract(
            task_id="task-999",
            orchestrator_id="orchestrator-1",
            worker_id="coder-agent-1",
            instructions="Fix the regex in tools.py",
            policy=DelegationPolicy(max_retries=5),
            status="assigned"
        )
        self.assertEqual(contract.policy.max_retries, 5)
        self.assertEqual(contract.worker_id, "coder-agent-1")

if __name__ == '__main__':
    unittest.main()
