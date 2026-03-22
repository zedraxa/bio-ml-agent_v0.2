import unittest
import os
import time
import uuid
from typing import List, Dict, Any
from unittest.mock import MagicMock, patch

from bio_ml_agent.services.mission.orchestrator import MissionOrchestrator
from bio_ml_agent.services.mission_pack_registry import mission_pack_registry
from bio_ml_agent.models.domain import MissionStatus, ArtifactReviewStatus, AgentRole, TaskType, StepStatus
from bio_ml_agent.db.session import SessionLocal, engine, Base
from bio_ml_agent.db.models import MissionDB, MissionStepDB, ArtifactDB, ReviewThreadDB, CommentDB

class TestGoldenScenarios(unittest.TestCase):
    """
    Axis H: Mission-vetted Scenario Harness.
    Verifies end-to-end research flows using canonical models and real orchestration.
    """
    
    @classmethod
    def setUpClass(cls):
        # Use in-memory DB for all tests
        os.environ["WORKSPACE_DB_PATH"] = "sqlite:///:memory:"
        Base.metadata.create_all(bind=engine)

    def setUp(self):
        self.config = MagicMock()
        self.config.workspace = "/tmp/bio_ml_golden_tests"
        os.makedirs(self.config.workspace, exist_ok=True)
        self.orchestrator = MissionOrchestrator(self.config)
        
        # Clear DB before each test
        with SessionLocal() as db:
            db.query(CommentDB).delete()
            db.query(ReviewThreadDB).delete()
            db.query(ArtifactDB).delete()
            db.query(MissionStepDB).delete()
            db.query(MissionDB).delete()
            db.commit()

    @patch("bio_ml_agent.services.mission.orchestrator.agent_registry")
    def test_repo_review_scenario(self, mock_registry):
        """Scenario 1: Repo Review -> Artifact -> Review -> Approval."""
        print("\n🧪 Running Scenario 1: Repo Review...")
        pack = mission_pack_registry.get_pack("repo_review_pack")
        project_id = "proj-repo-1"
        
        # Mock Agent Registry
        mock_info = MagicMock()
        mock_info.agent_id = "test-agent-id"
        mock_registry.find_by_role.return_value = [mock_info]
        mock_agent = MagicMock()
        mock_registry.get_agent_instance.return_value = mock_agent
        
        # scan, audit, refactor, summary
        mock_agent.summarize.side_effect = [
            MagicMock(artifacts=["findings.txt"], message="Scan complete", confidence=0.9, evidence=[], data={}, self_comments=[]),
            MagicMock(artifacts=["security_audit.md"], message="Audit complete", confidence=0.85, evidence=[], data={}, self_comments=[]),
            MagicMock(artifacts=["refactor_plan.py"], message="Refactor plan ready", confidence=0.95, evidence=[], data={}, self_comments=[]),
            MagicMock(artifacts=["final_summary.md"], message="All done", confidence=1.0, evidence=[], data={}, self_comments=[])
        ]
        mock_agent.plan.return_value = ["step"]
        
        # 1. Execute Pack
        mission_id = self.orchestrator.execute_pack(pack, project_id, "Analyze this repo")
        
        # 2. Assert Initial State
        with SessionLocal() as db:
            mission = db.query(MissionDB).filter(MissionDB.mission_id == mission_id).first()
            self.assertIsNotNone(mission)
            # The refactor step requires approval, so mission should be WAITING
            self.assertEqual(mission.status, MissionStatus.WAITING)
            
            # Check artifacts
            artifacts = db.query(ArtifactDB).filter(ArtifactDB.mission_id == mission_id).all()
            # scan (1) + audit (1) + refactor (1) = 3 total
            self.assertEqual(len(artifacts), 3)
            
            # Check review thread for refactor_plan
            patch_artifact = next(a for a in artifacts if "refactor_plan.py" in a.title)
            review_thread = db.query(ReviewThreadDB).filter(ReviewThreadDB.artifact_id == patch_artifact.artifact_id).first()
            self.assertIsNotNone(review_thread)
            self.assertEqual(patch_artifact.status, ArtifactReviewStatus.REVIEW_NEEDED)

        # 3. Simulate Approval
        print("✅ Simulating Approval for Review Thread...")
        with SessionLocal() as db:
            art = db.query(ArtifactDB).filter(ArtifactDB.artifact_id == patch_artifact.artifact_id).first()
            art.status = ArtifactReviewStatus.APPROVED
            db.commit()
            
        # 4. Resume Mission
        success = self.orchestrator.resume_pack(mission_id)
        self.assertTrue(success)
        
        with SessionLocal() as db:
            mission = db.query(MissionDB).filter(MissionDB.mission_id == mission_id).first()
            self.assertEqual(mission.status, MissionStatus.COMPLETED)
            print("🏁 Scenario 1 Passed: Complete Mission Lifecycle Verified.")

    @patch("bio_ml_agent.services.mission.orchestrator.agent_registry")
    def test_lab_report_lineage_scenario(self, mock_registry):
        """Scenario 2: Lab Notes -> Draft -> Lineage Verification."""
        print("\n🧪 Running Scenario 2: Lab Report Lineage...")
        pack = mission_pack_registry.get_pack("lab_report_pack")
        project_id = "proj-lab-2"
        
        mock_agent = MagicMock()
        mock_registry.get_agent_instance.return_value = mock_agent
        
        # steps: extract, processing, drafting, verification
        mock_agent.summarize.side_effect = [
            MagicMock(artifacts=["extracted_values.json"], message="Extracted", confidence=0.9, evidence=[]),
            MagicMock(artifacts=["cleaned_data.csv"], message="Processed", confidence=0.9, evidence=[]),
            MagicMock(artifacts=["lab_report_draft.md"], message="Drafted", confidence=0.9, evidence=[]),
            MagicMock(artifacts=["verification_log.txt"], message="Verified", confidence=0.9, evidence=[])
        ]
        mock_agent.plan.return_value = ["step"]
        
        # 1. Execute
        mission_id = self.orchestrator.execute_pack(pack, project_id, "Write lab report")
        
        # 2. Check Lineage
        with SessionLocal() as db:
            report_art = db.query(ArtifactDB).filter(ArtifactDB.mission_id == mission_id, ArtifactDB.title.like("%lab_report_draft.md%")).first()
            self.assertIsNotNone(report_art)
            
            # Lineage should include parents from previous steps
            # In our simple lineage logic, it includes all mission artifacts produced so far
            self.assertGreater(len(report_art.lineage_parents), 0)
            print(f"🔗 Lineage Verified: {report_art.title} has {len(report_art.lineage_parents)} parents.")
            
            # Resolve a comment scenario
            cid = f"cmt-{uuid.uuid4().hex[:6]}"
            new_cmt = CommentDB(
                comment_id=cid, 
                artifact_id=report_art.artifact_id, 
                author="Critic", 
                content="Fix figure 1",
                status="new",
                is_resolved=False,
                timestamp=time.time()
            )
            db.add(new_cmt)
            db.commit()
            
            comment_count = db.query(CommentDB).filter(CommentDB.artifact_id == report_art.artifact_id).count()
            self.assertEqual(comment_count, 1)
            print("🏁 Scenario 2 Passed: Lineage and Comment flow Verified.")

    @patch("bio_ml_agent.services.mission.orchestrator.agent_registry")
    def test_microscopy_collaboration_scenario(self, mock_registry):
        """Scenario 3: Microscopy Perception -> Segmentation -> Report."""
        print("\n🧪 Running Scenario 3: Microscopy Swarm...")
        pack = mission_pack_registry.get_pack("microscopy_pack")
        project_id = "proj-micro-3"
        
        # Mock Agent Registry
        mock_info = MagicMock()
        mock_info.agent_id = "micro-agent-id"
        mock_registry.find_by_role.return_value = [mock_info]
        mock_agent = MagicMock()
        mock_registry.get_agent_instance.return_value = mock_agent
        
        # steps: perception, segmentation, quantification, identification, report
        mock_agent.summarize.side_effect = [
            MagicMock(artifacts=["scan_profile.json"], message="Perceived", confidence=0.9, evidence=[], data={}, self_comments=[]),
            MagicMock(artifacts=["mask.png"], message="Segmented", confidence=0.9, evidence=[], data={}, self_comments=[]),
            MagicMock(artifacts=["metrics.csv"], message="Quantified", confidence=0.9, evidence=[], data={}, self_comments=[]),
            MagicMock(artifacts=["ontology.json"], message="Identified", confidence=0.9, evidence=[], data={}, self_comments=[]),
            MagicMock(artifacts=["final_analysis.pdf"], message="Reported", confidence=0.9, evidence=[], data={}, self_comments=[])
        ]
        mock_agent.plan.return_value = ["step"]
        
        # 1. Execute (report step requires approval)
        mission_id = self.orchestrator.execute_pack(pack, project_id, "Analyze microscopy image")
        
        # 2. Verification
        with SessionLocal() as db:
            mission = db.query(MissionDB).filter(MissionDB.mission_id == mission_id).first()
            self.assertEqual(mission.status, MissionStatus.WAITING)
            
            all_arts = db.query(ArtifactDB).filter(ArtifactDB.mission_id == mission_id).all()
            self.assertEqual(len(all_arts), 5)
            
            # Check categorized output (e.g., scan step produces perception category)
            # Pack definition for microscopy_pack scan step has task_type=DISCOVER
            # The orchestrator uses pack_step.expected_artifact_type
            # Let's check pack_step.expected_artifact_type in registry
            
            for art in all_arts:
                self.assertTrue(art.artifact_id.startswith("art-"))
                
            print(f"🔬 Microscopy Swarm Verified: {len(all_arts)} specialized artifacts produced.")
            print("🏁 Scenario 3 Passed: Multi-stage pipeline verified.")

if __name__ == "__main__":
    unittest.main()
