import unittest
from bio_ml_agent.models.plugin import (
    PluginCategory, PluginManifest,
    SandboxClass, PluginSignature,
    PluginState, PluginLifecycleState,
    PluginTelemetry
)

class TestPluginModels(unittest.TestCase):
    def test_plugin_manifest(self):
        manifest = PluginManifest(
            plugin_id="plugin-pubmed-adapter",
            name="PubMed Official Adapters",
            version="2.1.0",
            category=PluginCategory.SCIENTIFIC_DB,
            required_permissions=["network.read"]
        )
        self.assertEqual(manifest.category, PluginCategory.SCIENTIFIC_DB)
        self.assertEqual(manifest.version, "2.1.0")

    def test_plugin_signature_and_lifecycle(self):
        sig = PluginSignature(
            publisher_id="pub-benchling",
            signature_hash="sha256-abcxyz",
            trust_score=0.99,
            sandbox_class=SandboxClass.SEMI_TRUSTED
        )
        self.assertEqual(sig.trust_score, 0.99)
        self.assertEqual(sig.sandbox_class, SandboxClass.SEMI_TRUSTED)

        lifecycle = PluginLifecycleState(
            plugin_id="plugin-benchling-connector",
            current_state=PluginState.ENABLED,
            installed_at="2026-03-01T15:00:00Z"
        )
        self.assertEqual(lifecycle.current_state, PluginState.ENABLED)

    def test_plugin_telemetry(self):
        metrics = PluginTelemetry(
            plugin_id="plugin-huggingface",
            invocation_count=50,
            error_rate=0.01,
            avg_latency_ms=120.5,
            token_impact=450
        )
        self.assertTrue(metrics.error_rate < 0.05)
        self.assertEqual(metrics.token_impact, 450)

if __name__ == '__main__':
    unittest.main()
