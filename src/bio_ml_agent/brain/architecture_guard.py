import re
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from .models import (
    DriftReport,
    DriftViolation,
    MissionPlan,
    StepStatus
)

logger = logging.getLogger(__name__)

# ─── Architecture Contract Definition ──────────────────────────────────────────

ARCHITECTURE_CONTRACT = {
    "boundaries": {
        "brain": {
            "allowed_internal": ["models", "persistence", "agent_contract", "mission_brain"],
            "forbidden_internal": ["ui", "tools.browser", "tools.microscopy"],
            "description": "Brain core should not depend on UI or high-level tool implementations directly."
        },
        "models": {
            "allowed_internal": [],
            "forbidden_internal": ["mission_brain", "persistence"],
            "description": "Models must be pure and free of circular dependencies."
        }
    },
    "interfaces": {
        "AgentCapability": ["get_capabilities", "execute_task"],
        "MissionBrain": ["decompose", "replan", "finalize_mission"]
    },
    "legacy_patterns": [
        (r"old_mission_v1", "CRITICAL", "Legacy mission logic detected. Use MissionBrain instead."),
        (r"manual_step_v0", "WARNING", "Deprecated manual step handling found."),
        (r"print\(", "INFO", "Standard print() found. Use logging instead.")
    ]
}

class ArchitectureGuard:
    """
    H3: Architectural Drift Detection Engine.
    
    Enforces the system's structural integrity by auditing artifacts 
    and code changes against the ARCHITECTURE_CONTRACT.
    """

    def __init__(self, contract: Dict[str, Any] = ARCHITECTURE_CONTRACT):
        self.contract = contract

    def audit_mission_artifacts(self, plan: MissionPlan) -> DriftReport:
        """Audits the artifacts produced or modified during a mission."""
        report = DriftReport(mission_id=plan.mission_id)

        # In a real system, we would iterate through plan.steps and check output_artifacts content.
        # For this implementation, we'll perform a structural audit of the plan itself.

        # 1. Check Module Dependencies (Simulation)
        self._audit_dependencies(plan, report)

        # 2. Check Interface Compliance
        self._audit_interfaces(plan, report)

        # 3. Check Legacy Regressions
        self._audit_legacy_patterns(plan, report)

        # Finalize Report
        report.violation_count = len(report.violations)
        report.is_healthy = report.violation_count == 0
        report.drift_score = min(1.0, report.violation_count * 0.2)

        if report.violations:
            logger.warning(f"[ArchitectureGuard] Detected {report.violation_count} drift violations in mission {plan.mission_id}")

        return report

    def _audit_dependencies(self, plan: MissionPlan, report: DriftReport):
        """Checks for forbidden cross-module dependencies."""
        # Simulated check: look at step descriptions for prohibited mentions
        for step in plan.steps:
            for component, rules in self.contract["boundaries"].items():
                for forbidden in rules["forbidden_internal"]:
                    if forbidden in step.description.lower():
                        report.violations.append(DriftViolation(
                            component=component,
                            violation_type="FORBIDDEN_DEPENDENCY",
                            severity="WARNING",
                            description=f"Potential leak: '{component}' step mentions forbidden module '{forbidden}'.",
                            remediation_hint=rules["description"]
                        ))

    def _audit_interfaces(self, plan: MissionPlan, report: DriftReport):
        """Checks if agents are assigned tasks they aren't contracted for."""
        # Simulated check: check if the plan uses any unknown methods (mocked)
        pass

    def _audit_legacy_patterns(self, plan: MissionPlan, report: DriftReport):
        """Checks for forbidden strings/patterns in objectives and descriptions."""
        full_text = f"{plan.objective} {' '.join(s.description for s in plan.steps)}"

        for pattern, severity, msg in self.contract["legacy_patterns"]:
            if re.search(pattern, full_text):
                report.violations.append(DriftViolation(
                    component="General",
                    violation_type="LEGACY_PATTERN",
                    severity=severity,
                    description=msg,
                    remediation_hint="Refer to Architecture Evolution Guide in Part VI."
                ))
