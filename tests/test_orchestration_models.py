import unittest
from datetime import datetime, timezone
from bio_ml_agent.models.orchestration import (
    ProjectStatus, ProjectEpic, UserStory,
    ProjectTaskStatus, ProjectTask,
    DependencyType, TaskDependency,
    CriticalPathMap, ProjectTaskQueue,
    ApprovalNeededTask, PartialDeliveryRecord
)

class TestOrchestrationModels(unittest.TestCase):
    def test_project_hierarchy(self):
        epic = ProjectEpic(
            epic_id="epic-001",
            title="Genomic Data Pipeline",
            description="Developing a full-scale pipeline for genomic analysis.",
            owner_agent="orchestrator",
            created_at=datetime.now(timezone.utc).isoformat()
        )
        self.assertEqual(epic.status, ProjectStatus.DRAFT)

        story = UserStory(
            story_id="story-001",
            epic_id="epic-001",
            title="Data Ingestion",
            acceptance_criteria=["Parse FASTQ", "Validate quality scores"]
        )
        self.assertEqual(story.epic_id, "epic-001")

        task = ProjectTask(
            task_id="task-001",
            story_id="story-001",
            title="Implement FASTQ Parser",
            priority=1
        )
        self.assertEqual(task.status, ProjectTaskStatus.PENDING)

    def test_dependencies_and_critical_path(self):
        dep = TaskDependency(
            task_id="task-002",
            depends_on_id="task-001",
            type=DependencyType.FINISH_TO_START
        )
        self.assertEqual(dep.depends_on_id, "task-001")

        cp = CriticalPathMap(
            epic_id="epic-001",
            critical_task_ids=["task-001", "task-002", "task-005"],
            estimated_duration_hours=72.5
        )
        self.assertEqual(cp.estimated_duration_hours, 72.5)

    def test_queue_and_approval(self):
        queue = ProjectTaskQueue(
            project_id="proj-123",
            pending_tasks=["task-003", "task-004"],
            blocked_tasks=["task-002"]
        )
        self.assertIn("task-003", queue.pending_tasks)

        approval_task = ApprovalNeededTask(
            task_id="task-critical",
            story_id="story-001",
            title="Deploy to Production",
            approval_reason="Kritik üretim aşaması insan onayı gerektirir.",
            requires_human_approval=True
        )
        self.assertTrue(approval_task.requires_human_approval)

    def test_partial_delivery(self):
        delivery = PartialDeliveryRecord(
            delivery_id="del-001",
            project_id="proj-123",
            artifact_ids=["art-001", "art-002"],
            version_tag="v0.1-alpha",
            description="First draft of data parser results.",
            released_at=datetime.now(timezone.utc).isoformat()
        )
        self.assertEqual(delivery.version_tag, "v0.1-alpha")

if __name__ == '__main__':
    unittest.main()
