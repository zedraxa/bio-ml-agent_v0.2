import unittest
from bio_ml_agent.models.scenarios import (
    ScenarioDifficulty, EndToEndScenario, DemoFlow, ChaosAction, ChaosTestConfig, RecoveryValidation
)

class TestScenariosModels(unittest.TestCase):
    def test_end_to_end_scenario(self):
        scenario = EndToEndScenario(
            scenario_id="sc-01",
            title="DICOM Segmentation",
            difficulty=ScenarioDifficulty.PRODUCTION_GRADE,
            required_services=["gpu_node"],
            entry_prompt="Run baseline segmentation on dataset A",
            expected_final_artifact_type="scientific_report"
        )
        self.assertEqual(scenario.difficulty, ScenarioDifficulty.PRODUCTION_GRADE)

    def test_chaos_test_config(self):
        chaos = ChaosTestConfig(
            test_id="ch-01",
            base_scenario_id="sc-01",
            injected_chaos=ChaosAction.NODE_RESET,
            injection_delay_seconds=120,
            expected_system_behavior="Resume from last checkpoint"
        )
        self.assertEqual(chaos.injected_chaos, ChaosAction.NODE_RESET)

    def test_recovery_validation(self):
        recovery = RecoveryValidation(
            validation_id="val-01",
            chaos_test_id="ch-01",
            state_resumed_successfully=True,
            recovery_time_ms=5000
        )
        self.assertTrue(recovery.state_resumed_successfully)
        self.assertFalse(recovery.duplicate_billing_detected)

if __name__ == '__main__':
    unittest.main()
