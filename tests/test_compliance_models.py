import unittest
from datetime import datetime, timezone
from bio_ml_agent.models.compliance import (
    AccessRole, PermissionSet, PolicyMode, PolicyTemplate,
    ConsentRecord, ViolationSeverity, ComplianceViolation,
    DegradationState
)

class TestComplianceModels(unittest.TestCase):
    def test_rbac_and_permissions(self):
        permissions = PermissionSet(
            can_read=True,
            can_write=False,
            can_execute_shell=False
        )
        self.assertTrue(permissions.can_read)
        self.assertFalse(permissions.can_write)
        
        role = AccessRole.VIEWER
        self.assertEqual(role, "viewer")

    def test_policy_template(self):
        template = PolicyTemplate(
            mode=PolicyMode.STUDENT,
            enforce_hitl=True,
            max_tools_per_turn=2,
            allowed_file_extensions=[".md", ".txt"]
        )
        self.assertEqual(template.mode, PolicyMode.STUDENT)
        self.assertEqual(template.max_tools_per_turn, 2)
        self.assertIn(".md", template.allowed_file_extensions)

    def test_consent_record(self):
        consent = ConsentRecord(
            consent_id="con-001",
            user_id="user-456",
            action_type="github_push",
            granted_at=datetime.now(timezone.utc).isoformat(),
            digital_signature="sig_abc123"
        )
        self.assertEqual(consent.user_id, "user-456")
        self.assertEqual(consent.digital_signature, "sig_abc123")

    def test_violation_and_degradation(self):
        violation = ComplianceViolation(
            violated_rule_id="rule-shell-01",
            policy_mode=PolicyMode.REGULATED_DATA,
            agent_id="agent-007",
            detected_at=datetime.now(timezone.utc).isoformat(),
            severity=ViolationSeverity.BLOCKING,
            description="Attempted shell execution in regulated mode.",
            remediation_action="Switch to Read-Only"
        )
        self.assertEqual(violation.severity, ViolationSeverity.BLOCKING)

        degradation = DegradationState(
            is_degraded=True,
            original_intent="Modify production database",
            degration_reason="Insufficient permissions",
            active_limitations=["Read-Only Mode", "No Shell Access"],
            suggested_fix="Ask admin for EDITOR role"
        )
        self.assertTrue(degradation.is_degraded)
        self.assertIn("Read-Only Mode", degradation.active_limitations)

if __name__ == '__main__':
    unittest.main()
