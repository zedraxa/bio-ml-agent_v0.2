import unittest
from models.infrastructure import (
    DeploymentStatus, HelmRelease, InferenceService, NodeClass, ScalingPolicy, TenantIsolation
)

class TestInfrastructureModels(unittest.TestCase):
    def test_helm_release(self):
        rel = HelmRelease(
            release_id="rel-01",
            name="bio-api",
            namespace="prod",
            version="1.2.3",
            status=DeploymentStatus.DEPLOYED,
            last_updated="2026-03-10T19:00:00"
        )
        self.assertEqual(rel.status, DeploymentStatus.DEPLOYED)

    def test_inference_service(self):
        svc = InferenceService(
            service_id="svc-01",
            model_name="dnabert-2",
            model_version="v1",
            endpoint_url="http://inference.bio",
            min_replicas=2,
            max_replicas=20
        )
        self.assertEqual(svc.max_replicas, 20)

    def test_tenant_isolation(self):
        tenant = TenantIsolation(
            tenant_id="t-01",
            namespace="tenant-a",
            quota_id="q-01",
            secret_vault_path="/secrets/t-a",
            storage_root="/data/t-a"
        )
        self.assertTrue(tenant.network_policy_enabled)

if __name__ == '__main__':
    unittest.main()
