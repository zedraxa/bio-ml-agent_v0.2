import unittest
from datetime import datetime
from models.medical_segmentation import (
    ModelFramework, BioMedicalModelConfig, SegmentationTarget, SegmentationTask,
    InferenceStrategy, PostProcessingAction, InferenceConfig,
    UncertaintyMethod, SegmentationReport
)

class TestMedicalSegmentationModels(unittest.TestCase):
    def test_model_config(self):
        config = BioMedicalModelConfig(
            model_id="model-nnunet-01",
            name="nnU-Net Liver Segmentation",
            framework=ModelFramework.NNUNET,
            version="2.0",
            architecture="3d_fullres",
            input_shape=[1, 128, 128, 128],
            output_classes=2
        )
        self.assertEqual(config.framework, ModelFramework.NNUNET)
        self.assertEqual(config.output_classes, 2)

    def test_segmentation_task(self):
        task = SegmentationTask(
            task_id="task-liver-01",
            name="Liver Segmentation Task",
            target=SegmentationTarget.ORGAN,
            target_name="Liver",
            base_model_id="model-nnunet-01",
            required_modality=["CT"]
        )
        self.assertEqual(task.target, SegmentationTarget.ORGAN)
        self.assertIn("CT", task.required_modality)

    def test_inference_config(self):
        config = InferenceConfig(
            config_id="inf-01",
            strategy=InferenceStrategy.SLIDING_WINDOW,
            overlap=0.5,
            tile_size=[96, 96, 96],
            post_processing=[PostProcessingAction.KEEP_LARGEST_COMPONENT, PostProcessingAction.FILL_HOLES]
        )
        self.assertEqual(config.overlap, 0.5)
        self.assertEqual(len(config.post_processing), 2)

    def test_segmentation_report(self):
        report = SegmentationReport(
            report_id="rep-001",
            artifact_id="img-123",
            task_id="task-liver-01",
            model_id="model-nnunet-01",
            dice_score=0.94,
            uncertainty_method=UncertaintyMethod.MONTE_CARLO_DROPOUT,
            mean_uncertainty=0.05,
            failure_risk=0.02,
            findings=["Successful segmentation with high confidence"],
            timestamp=datetime.utcnow().isoformat()
        )
        self.assertEqual(report.dice_score, 0.94)
        self.assertEqual(report.uncertainty_method, UncertaintyMethod.MONTE_CARLO_DROPOUT)

if __name__ == '__main__':
    unittest.main()
