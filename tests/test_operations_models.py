import unittest
from bio_ml_agent.models.operations import (
    CostEntry, QuotaLimit, FinOpsRecommendation, IncidentSeverity, IncidentPlaybook
)

class TestOperationsModels(unittest.TestCase):
    def test_cost_entry(self):
        entry = CostEntry(
            entry_id="e-001",
            timestamp="2026-03-10T19:30:00",
            user_id="u-123",
            project_id="p-456",
            resource_type="gpt-4",
            cost_amount=0.15
        )
        self.assertEqual(entry.cost_amount, 0.15)

    def test_quota_limit(self):
        quota = QuotaLimit(
            quota_id="q-01",
            tenant_id="t-01",
            daily_token_limit=1000000,
            monthly_gpu_limit_hours=100.0,
            max_concurrent_browsers=5,
            storage_quota_gb=50.0
        )
        self.assertEqual(quota.daily_token_limit, 1000000)

    def test_incident_playbook(self):
        pb = IncidentPlaybook(
            playbook_id="pb-01",
            component="qdrant",
            trigger_condition="down",
            severity=IncidentSeverity.P0,
            remediation_steps=["restart pod", "check logs"]
        )
        self.assertEqual(pb.severity, IncidentSeverity.P0)

if __name__ == '__main__':
    unittest.main()
