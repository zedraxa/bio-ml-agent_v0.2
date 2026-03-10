import unittest
from datetime import datetime
from models.security import (
    RiskLevel, ActionType, ActionApprovalPolicy,
    HITLResponseType, HITLResponse,
    SecurityReviewRequest
)

class TestSecurityModels(unittest.TestCase):
    def test_approval_policy(self):
        policy = ActionApprovalPolicy(
            action_type=ActionType.DESTRUCTIVE_DELETE,
            risk_level=RiskLevel.CRITICAL,
            requires_approval=True
        )
        self.assertEqual(policy.action_type, ActionType.DESTRUCTIVE_DELETE)
        self.assertTrue(policy.requires_approval)

    def test_hitl_response(self):
        response = HITLResponse(
            response_type=HITLResponseType.EDIT,
            modified_action={"command": "rm -rf /tmp/safe_dir"},
            reason="User corrected the path",
            responder_id="user-123",
            responded_at=datetime.utcnow().isoformat()
        )
        self.assertEqual(response.response_type, HITLResponseType.EDIT)
        self.assertEqual(response.responder_id, "user-123")

    def test_security_review_request(self):
        request = SecurityReviewRequest(
            request_id="rev-001",
            action_type=ActionType.SHELL_EXECUTION,
            proposed_action={"command": "pip install some-pkg"},
            intent_explanation="Installing dependencies for the genomic pipeline.",
            expected_impact="Adds new libraries to the environment.",
            risk_assessment="Low risk if pkg is trusted.",
            rollback_supported=True,
            rollback_plan="pip uninstall some-pkg"
        )
        self.assertEqual(request.action_type, ActionType.SHELL_EXECUTION)
        self.assertTrue(request.rollback_supported)

if __name__ == '__main__':
    unittest.main()
