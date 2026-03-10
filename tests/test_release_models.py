import unittest
from models.release import (
    ReleaseType, ReleaseManifest, StarterTemplate, PresetConfig, SecurityScanReport
)

class TestReleaseModels(unittest.TestCase):
    def test_release_manifest(self):
        manifest = ReleaseManifest(
            version_tag="v1.0.0",
            release_type=ReleaseType.MAJOR,
            deprecated_apis_removed=True,
            technical_debt_score=0.5,
            release_notes=["First public stable release!"]
        )
        self.assertTrue(manifest.is_public_roadmap_v1)

    def test_starter_template(self):
        template = StarterTemplate(
            template_id="tpl-01",
            name="RNA-Seq Pipeline",
            description="End to end RNA-Seq template",
            preloaded_data_refs=[],
            default_agent_roles=["Bioinformatics Analyst", "Report Writer"]
        )
        self.assertEqual(len(template.default_agent_roles), 2)

    def test_security_scan_report(self):
        report = SecurityScanReport(
            scan_id="sec-001",
            target_version="v1.0.0",
            critical_vulnerabilities_count=0,
            high_vulnerabilities_count=0,
            all_licenses_compliant=True,
            scan_passed=True
        )
        self.assertTrue(report.scan_passed)

if __name__ == '__main__':
    unittest.main()
