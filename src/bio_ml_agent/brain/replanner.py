import logging
from typing import Optional, List, Dict, Any
from .models import (
    MissionPlan,
    MissionStep,
    StepStatus,
    ReplanResult,
    AgentRole,
    AgentRetryStrategy,
    RiskLevel,
    TaskType
)

logger = logging.getLogger(__name__)

# ─── Retry Policies ───────────────────────────────────────────────────────────

AGENT_RETRY_POLICIES = {
    AgentRole.BROWSER_AGENT: {"strategy": AgentRetryStrategy.LIMITED_RETRY, "max_retries": 3},
    AgentRole.WRITING_AGENT: {"strategy": AgentRetryStrategy.LIMITED_RETRY, "max_retries": 2},
    AgentRole.CODING_AGENT: {"strategy": AgentRetryStrategy.SELF_DEBUG, "max_retries": 5},
    AgentRole.MICROSCOPY_AGENT: {"strategy": AgentRetryStrategy.HUMAN_REVIEW, "max_retries": 0},
    AgentRole.RESEARCHER: {"strategy": AgentRetryStrategy.LIMITED_RETRY, "max_retries": 2},
}

class Replanner:
    """
    Part VI: Replanner Module.
    
    Handles failure recovery and dynamic mission adjustment 
    using Strategy 1-5 (Substitution, Upgrading, Simplifying, Bypassing, Escalation).
    """

    def replan(self, plan: MissionPlan, failed_step_id: str, error_context: str, confidence: Optional[float] = None) -> ReplanResult:
        """Executes the recovery lifecycle for a failed step."""
        failed_step = next((s for s in plan.steps if s.step_id == failed_step_id), None)
        if not failed_step:
            raise ValueError(f"Step {failed_step_id} not found in mission {plan.mission_id}")

        strategy_used = "none"
        strategy_detail = "No recovery strategy applied."
        recovery_succeeded = False

        # 1. Update Telemetry
        if plan.telemetry:
            plan.telemetry.failure_points.append(failed_step_id)
            plan.telemetry.replan_count += 1

        # 2. Local Retry Logic (G3)
        policy_data = AGENT_RETRY_POLICIES.get(failed_step.assigned_agent, {"strategy": AgentRetryStrategy.LIMITED_RETRY, "max_retries": 2})
        retry_count = failed_step.metadata.get("retry_count", 0)

        if retry_count < policy_data["max_retries"]:
            failed_step.metadata["retry_count"] = retry_count + 1
            failed_step.status = StepStatus.RETRYING
            strategy_used = "retry"
            strategy_detail = f"Auto-retry {failed_step.metadata['retry_count']}/{policy_data['max_retries']}"
            recovery_succeeded = True

        # 3. Structural Strategies (A4)
        if not recovery_succeeded:
            # Strategy: Substitute Agent
            substitute = self._find_substitute_agent(failed_step.assigned_agent)
            if substitute and not failed_step.metadata.get("substituted"):
                failed_step.metadata["substituted"] = True
                failed_step.assigned_agent = substitute
                failed_step.status = StepStatus.RETRYING
                strategy_used = "substitute_agent"
                strategy_detail = f"Swapped agent to {substitute}"
                recovery_succeeded = True

            # Strategy: Simplify Task
            elif not failed_step.metadata.get("simplified"):
                failed_step.metadata["simplified"] = True
                failed_step.description = f"[Simplified] {failed_step.description} (Reduced scope)"
                failed_step.status = StepStatus.RETRYING
                strategy_used = "simplify_task"
                strategy_detail = "Reduced task complexity and retrying."
                recovery_succeeded = True

            # Strategy: User Escalation
            else:
                failed_step.status = StepStatus.AAWAITING_APPROVAL
                failed_step.requires_approval = True
                strategy_used = "user_escalation"
                strategy_detail = "All automated strategies exhausted. Awaiting human input."
                recovery_succeeded = False

        result = ReplanResult(
            mission_id=plan.mission_id,
            failed_step_id=failed_step_id,
            strategy_used=strategy_used,
            strategy_detail=strategy_detail,
            recovery_succeeded=recovery_succeeded,
            replan_count=len([s for s in plan.steps if s.status == StepStatus.RETRYING]),
            error_context=error_context,
            confidence=confidence
        )

        return result

    def _find_substitute_agent(self, role: AgentRole) -> Optional[AgentRole]:
        """Finds a viable alternative for a failed agent role."""
        substitutes = {
            AgentRole.MICROSCOPY_AGENT: AgentRole.IN_SILICO_EXPERT,
            AgentRole.ML_EXPERT: AgentRole.CODING_AGENT,
            AgentRole.RESEARCHER: AgentRole.ACADEMIC_EXPERT
        }
        return substitutes.get(role)
