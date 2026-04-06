import logging
import uuid
from typing import List, Dict, Any, Optional
from .models import MissionPlan, MissionStep, TaskType, AgentRole, StepStatus, Comment, IntentType

logger = logging.getLogger("bio_ml_agent.brain.refinement")

class RefinementPlanner:
    """
    R7-2: Feedback-to-Action Intelligence.
    Adapts mission plans by injecting refinement steps based on human commentary.
    """
    def __init__(self):
        pass

    def plan_refinement_step(self, plan: MissionPlan, refinement_task: Dict[str, Any]) -> MissionStep:
        """
        Generates a new MissionStep to address the feedback.
        """
        step_id = f"refine_{uuid.uuid4().hex[:4]}"

        new_step = MissionStep(
            step_id=step_id,
            title=f"Refinement: {refinement_task['intent'].value.capitalize()} for {refinement_task['target_id']}",
            description=f"Action: {refinement_task['instruction']}",
            task_type=TaskType.ANALYZE if refinement_task['intent'] == IntentType.REVISE else TaskType.SYNTHESIZE,
            assigned_agent=refinement_task['suggested_role'],
            status=StepStatus.PENDING,
            metadata={
                "source_comment_id": refinement_task['comment_id'],
                "is_refinement": True
            }
        )

        # Inject into plan steps
        plan.steps.append(new_step)

        logger.info(f"[RefinementPlanner] Planned new step {step_id} based on comment {refinement_task['comment_id']}")
        return new_step

    def link_refinement_to_graph(self, plan: MissionPlan, new_step_id: str, target_id: Optional[str] = None):
        """
        Ensures the new refinement step is correctly positioned in the execution DAG.
        Typically, refinement happens after the original step that produced the target artifact.
        """
        new_step = next((s for s in plan.steps if s.step_id == new_step_id), None)
        if not new_step:
            return

        # If there's a specific target, make the new step depend on it.
        # Otherwise, make it depend on the currently last completed step or just the last step before it.
        if target_id and any(s.step_id == target_id for s in plan.steps):
            if target_id not in new_step.depends_on:
                new_step.depends_on.append(target_id)
        else:
            idx = plan.steps.index(new_step)
            if idx > 0:
                prev_id = plan.steps[idx - 1].step_id
                if prev_id not in new_step.depends_on:
                    new_step.depends_on.append(prev_id)

        # Update the AgentGraph using the Graph Engine
        try:
            from .graph_engine import MissionGraphEngine
            engine = MissionGraphEngine()
            plan.agent_graph = engine.build_graph(plan.mission_id, plan.steps)
            logger.info(f"[RefinementPlanner] AgentGraph successfully rebuilt with refinement step {new_step_id}")
        except Exception as e:
            logger.error(f"[RefinementPlanner] Failed to rebuild AgentGraph: {e}", exc_info=True)

    def execute_feedback_loop(self, plan: MissionPlan, interpreter_result: Dict[str, Any]):
        """Runs the full loop of planning and linking a refinement."""
        new_step = self.plan_refinement_step(plan, interpreter_result)
        target_id = interpreter_result.get("target_id")
        self.link_refinement_to_graph(plan, new_step.step_id, target_id)
        return new_step

    def create_bulk_revision_plan(self, mission_id: str, comments: List[Comment]) -> Dict[str, Any]:
        """
        B3: Revision Planner.
        Groups and prioritizes multiple comments into a coherent revision strategy.
        """
        log_msg = f"[RefinementPlanner] Creating bulk revision plan for {len(comments)} comments."
        logger.info(log_msg)

        # simulated B3 logic:
        # 1. Group by target_id
        # 2. Sort by severity
        # 3. Identify overlaps

        revision_plan = {
            "plan_id": f"rev_{uuid.uuid4().hex[:6]}",
            "priority_order": [c.comment_id for c in sorted(comments, key=lambda x: x.severity, reverse=True)],
            "conflicts": [], # Placeholder for actual conflict detection
            "summary": "Coherent revision plan generated to address multiple observer feedbacks."
        }
        return revision_plan
