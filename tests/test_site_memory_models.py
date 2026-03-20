import unittest
from bio_ml_agent.models.site_memory import (
    KnownSelectors, SiteProfile,
    ExtractionFormat, ExtractionConfig,
    ExtractionResult, SelectorMemory,
    SiteInteractionHistory, SitePolicyHint
)

class TestSiteMemoryModels(unittest.TestCase):
    def test_site_profile_registry(self):
        selectors = KnownSelectors(
            login_button="button.login-btn",
            pagination_next="a.next-page"
        )
        profile = SiteProfile(
            domain="example.com",
            known_selectors=selectors,
            login_route="/account/login"
        )
        self.assertEqual(profile.domain, "example.com")
        self.assertEqual(profile.known_selectors.login_button, "button.login-btn")

    def test_extraction_config(self):
        config = ExtractionConfig(
            target_selector="table.data-grid",
            desired_format=ExtractionFormat.CSV,
            paginate_if_possible=True
        )
        self.assertEqual(config.desired_format, ExtractionFormat.CSV)
        self.assertTrue(config.paginate_if_possible)

    def test_browser_memory(self):
        memory = SelectorMemory(
            selector_str="#buy-now",
            success_count=8,
            fail_count=2
        )
        self.assertEqual(memory.reliability_score, 0.8) # 8 / 10

    def test_site_policy_hints(self):
        hint = SitePolicyHint(
            domain="api.example.com",
            slow_mode_enabled=True,
            delay_between_clicks_ms=2000,
            banned_paths=["/admin", "/robots.txt"]
        )
        self.assertTrue(hint.slow_mode_enabled)
        self.assertEqual(hint.delay_between_clicks_ms, 2000)

if __name__ == '__main__':
    unittest.main()
