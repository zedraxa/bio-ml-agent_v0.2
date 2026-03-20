import unittest
from datetime import datetime, timezone
from bio_ml_agent.models.vault import (
    SecretType, AccessScope, SecretIdentifier,
    SecretLease, RedactionType, RedactionRule,
    RedactedArtifact, SecretAuditLog
)

class TestVaultModels(unittest.TestCase):
    def test_secret_identifier_and_scope(self):
        scope = AccessScope(
            allowed_domains=["*.benchling.com"],
            allowed_tools=["browser_agent"],
            requires_approval=True
        )
        secret = SecretIdentifier(
            secret_id="sec-001",
            key_name="BENCHLING_API_KEY",
            secret_type=SecretType.API_KEY,
            scope=scope,
            created_at=datetime.now(timezone.utc).isoformat()
        )
        self.assertEqual(secret.key_name, "BENCHLING_API_KEY")
        self.assertIn("browser_agent", secret.scope.allowed_tools)
        self.assertTrue(secret.scope.requires_approval)

    def test_secret_lease(self):
        lease = SecretLease(
            lease_id="lse-001",
            secret_id="sec-001",
            token_value="sk_live_123456789",
            issued_at=datetime.now(timezone.utc).isoformat(),
            expires_at="2026-03-10T19:54:32Z",
            ttl_seconds=3600
        )
        self.assertEqual(lease.ttl_seconds, 3600)
        self.assertTrue(lease.auto_revoke)

    def test_redaction_rule(self):
        rule = RedactionRule(
            rule_id="rud-001",
            pattern=r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
            type=RedactionType.REPLACE,
            replacement="[EMAIL_REDACTED]",
            description="Mask email addresses"
        )
        self.assertEqual(rule.type, RedactionType.REPLACE)
        self.assertEqual(rule.replacement, "[EMAIL_REDACTED]")

    def test_audit_log(self):
        log = SecretAuditLog(
            audit_id="aud-001",
            secret_id="sec-001",
            agent_id="agent-x",
            task_id="task-y",
            timestamp=datetime.now(timezone.utc).isoformat(),
            action="read",
            status="success"
        )
        self.assertEqual(log.action, "read")
        self.assertEqual(log.status, "success")

if __name__ == '__main__':
    unittest.main()
