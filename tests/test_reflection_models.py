import unittest
from models.reflection import (
    FailureCategory, FailureRecord, RefinementRule, ReflectionReport
)

class TestReflectionModels(unittest.TestCase):
    def test_failure_record(self):
        rec = FailureRecord(
            failure_id="f-001",
            run_id="run-123",
            category=FailureCategory.PLANNING_FAILURE,
            error_message="Plan was circular",
            root_cause_analysis="Logic loop in planner"
        )
        self.assertEqual(rec.category, FailureCategory.PLANNING_FAILURE)

    def test_refinement_rule(self):
        rule = RefinementRule(
            rule_id="rule-01",
            scope="global",
            pattern="Always use X instead of Y",
            correction="Updated system prompt with X",
            source_failure_ids=["f-001"],
            confidence=0.9
        )
        self.assertTrue(rule.is_active)
        self.assertEqual(rule.confidence, 0.9)

    def test_reflection_report(self):
        report = ReflectionReport(
            report_id="rep-001",
            period_start="2026-03-01",
            period_end="2026-03-07",
            top_failure_patterns=[{"pattern": "API timeout", "count": 5}],
            new_rules_derived=2,
            quality_trend={"accuracy": 0.85},
            recommendations=["Increase timeout"]
        )
        self.assertEqual(report.new_rules_derived, 2)
        self.assertIn("accuracy", report.quality_trend)

if __name__ == '__main__':
    unittest.main()
