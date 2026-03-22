import unittest
import os
import time
import json
from unittest.mock import MagicMock, patch
from bio_ml_agent.services.mission.orchestrator import MissionOrchestrator
from bio_ml_agent.models.mission_pack import MissionPack, MissionPackStep
from bio_ml_agent.models.domain import MissionStatus, AgentRole, TaskType, StepStatus
from bio_ml_agent.brain.models import MissionPlan, MissionStep

class TestMissionRecovery(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()
        self.config.workspace = "/tmp/bio_ml_test_workspace"
        os.makedirs(self.config.workspace, exist_ok=True)
        self.orchestrator = MissionOrchestrator(self.config)

    @patch("bio_ml_agent.services.mission.orchestrator.SessionLocal")
    @patch("bio_ml_agent.services.mission.orchestrator.agent_registry")
    def test_checkpoint_and_resume(self, mock_registry, mock_session):
        # 1. Create a dummy pack
        pack = MissionPack(
            pack_id="test_pack",
            name="Test Pack",
            description="Testing recovery",
            steps=[
                MissionPackStep(step_id="step1", title="Step 1", description="First step", 
                                task_type=TaskType.ANALYZE, assigned_agent=AgentRole.CODING_AGENT),
                MissionPackStep(step_id="step2", title="Step 2", description="Second step", 
                                task_type=TaskType.ANALYZE, assigned_agent=AgentRole.CODING_AGENT,
                                depends_on=["step1"])
            ]
        )

        mission_id = "msn-test-123"
        project_id = "proj-test"
        
        # Mock DB behavior for first run
        mock_db = MagicMock()
        mock_session.return_value.__enter__.return_value = mock_db
        mock_db.query.return_value.filter.return_value.all.return_value = [] # No completed steps initially
        
        # Mock agent
        mock_agent = MagicMock()
        mock_registry.get_agent_instance.return_value = mock_agent
        mock_agent.summarize.return_value = MagicMock(artifacts=[], data={}, message="Done", confidence=0.9, evidence=[])

        # 2. Start mission but simulate "crash" after step 1
        # We simulate a plan
        plan = self.orchestrator._create_plan_from_pack(mission_id, project_id, pack, "Start test")
        
        # Manually run step 1
        self.orchestrator._execute_step(mission_id, pack.steps[0], project_id, plan)
        
        # Verify checkpoint exists
        checkpoint_path = os.path.join(self.orchestrator.recovery.checkpoint_dir, f"{mission_id}_latest.json")
        self.assertTrue(os.path.exists(checkpoint_path), "Checkpoint file should be created after step")
        
        with open(checkpoint_path, "r") as f:
            data = json.load(f)
            self.assertEqual(data["mission_id"], mission_id)
            # StepStatus enum values are used in checkpoint
            self.assertEqual(data["plan_snapshot"]["steps"][0]["status"], "completed")
            self.assertEqual(data["plan_snapshot"]["steps"][1]["status"], "pending")

        # 3. Simulate Resume
        # Mock DB to show step1 is completed
        # Mocking MissionStepDB objects
        class MockStep:
            def __init__(self, step_id, action_type):
                self.step_id = step_id
                self.action_type = action_type
        
        mock_db.query.return_value.filter.return_value.all.return_value = [MockStep(f"{mission_id}-step1", "COMPLETED")]
        
        # Resume the mission
        success = self.orchestrator.resume_pack(mission_id)
        self.assertTrue(success)
        
        # Verify that only step 2 was executed (agent called once during resume)
        # Total calls should be 2 status since we call it once manually and once via resume
        self.assertEqual(mock_agent.perceive.call_count, 2)

if __name__ == "__main__":
    unittest.main()
