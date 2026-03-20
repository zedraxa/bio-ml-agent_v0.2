import unittest
from pathlib import Path
from bio_ml_agent.models.runtime import (
    AgentRun, MessageContract, MessageType,
    ToolContract, SideEffectClass, ApprovalLevel,
    ArtifactContract, ArtifactStatus
)
from bio_ml_agent.core.workspace import LocalWorkspace

class TestRuntimeModels(unittest.TestCase):
    def test_agent_run_model(self):
        run = AgentRun(
            run_id="run-123",
            workspace_id="ws-1",
            project_id="proj-1"
        )
        self.assertEqual(run.run_id, "run-123")
        self.assertEqual(run.cost, 0.0)
        
    def test_message_contract(self):
        msg = MessageContract(
            type=MessageType.PLAN,
            content="Geliştirme planı oluşturuldu."
        )
        self.assertEqual(msg.type, MessageType.PLAN)
        
    def test_local_workspace(self):
        ws = LocalWorkspace("/tmp/test_workspace")
        ws.write_file("test.txt", "merhaba")
        content = ws.read_file("test.txt")
        self.assertEqual(content, "merhaba")
        
    def test_tool_contract(self):
        tool = ToolContract(
            name="terminal_runner",
            description="Run bash commands",
            side_effect=SideEffectClass.DESTRUCTIVE,
            approval_level=ApprovalLevel.REQUIRE_APPROVAL,
            retryability=False,
            idempotent=False
        )
        self.assertEqual(tool.name, "terminal_runner")
        self.assertFalse(tool.idempotent)
        
    def test_artifact_contract(self):
        artifact = ArtifactContract(
            artifact_id="art-555",
            run_id="run-123",
            name="ml_model_weights.pt",
            type="model_weight",
            status=ArtifactStatus.APPROVED
        )
        self.assertEqual(artifact.artifact_id, "art-555")
        self.assertEqual(artifact.status, ArtifactStatus.APPROVED)

if __name__ == '__main__':
    unittest.main()
