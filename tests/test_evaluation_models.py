import unittest
from bio_ml_agent.models.evaluation import (
    EvalScenario, BenchmarkSuite, EvalMetricResult, RegressionGateConfig
)

class TestEvaluationModels(unittest.TestCase):
    def test_eval_scenario(self):
        scenario = EvalScenario(
            scenario_id="sc-001",
            name="Test Task",
            difficulty="easy",
            domain="coding",
            input_data={"task": "write hello world"}
        )
        self.assertEqual(scenario.difficulty, "easy")

    def test_benchmark_suite(self):
        suite = BenchmarkSuite(
            suite_id="suite-01",
            name="Smoke Tests",
            scenarios=[],
            version="1.0"
        )
        self.assertEqual(suite.version, "1.0")

    def test_eval_metric_result(self):
        res = EvalMetricResult(
            run_id="run-123",
            scenario_id="sc-001",
            task_completion=0.9,
            groundedness=1.0,
            citation_faithfulness=1.0,
            cost_efficiency=0.8,
            latency_ms=2000.0
        )
        self.assertEqual(res.task_completion, 0.9)
        self.assertEqual(res.groundedness, 1.0)

    def test_regression_gate(self):
        gate = RegressionGateConfig(
            gate_id="gate-01",
            metric_name="task_completion",
            threshold=0.8,
            comparison_operator="ge",
            source_benchmark="suite-01"
        )
        self.assertEqual(gate.threshold, 0.8)

if __name__ == '__main__':
    unittest.main()
