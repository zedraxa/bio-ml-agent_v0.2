import unittest
from models.auth import (
    BrowserProfile, LoginFlowType, LoginState,
    ChallengeType, ChallengeState,
    HandoffStatus, HumanHandoffRequest,
    ClearanceLevel, CredentialPolicy
)

class TestAuthModels(unittest.TestCase):
    def test_browser_profile(self):
        profile = BrowserProfile(
            profile_id="prod-profile-user-1",
            cookie_vault_path="/secure/vault/user1_cookies.json"
        )
        self.assertEqual(profile.profile_id, "prod-profile-user-1")

    def test_challenge_and_handoff(self):
        challenge = ChallengeState(
            type=ChallengeType.CAPTCHA,
            is_blocking=True
        )
        self.assertEqual(challenge.type, ChallengeType.CAPTCHA)
        self.assertTrue(challenge.is_blocking)
        
        handoff = HumanHandoffRequest(
            handoff_id="hand-777",
            run_id="run-123",
            reason="Lütfen telefonunuza gelen SMS'i girin.",
            challenge=challenge,
            status=HandoffStatus.PENDING_USER
        )
        self.assertEqual(handoff.status, HandoffStatus.PENDING_USER)

    def test_credential_policy(self):
        policy = CredentialPolicy(
            policy_id="policy-001",
            credential_id="cred-kaggle-api",
            allowed_domains=["*.kaggle.com", "kaggle.com"],
            clearance_level=ClearanceLevel.MEDIUM,
            max_usage_count=5
        )
        self.assertIn("kaggle.com", policy.allowed_domains)
        self.assertEqual(policy.clearance_level, ClearanceLevel.MEDIUM)

if __name__ == '__main__':
    unittest.main()
