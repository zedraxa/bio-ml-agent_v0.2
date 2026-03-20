import unittest
from datetime import datetime, timezone
from bio_ml_agent.models.medical_imaging import (
    TransformType, ImageTransform, ImageTransformStack, ROIConfig,
    DatasetCard, AnnotationStatus, AnnotationSession, ActiveLearningSample
)

class TestMedicalImagingModels(unittest.TestCase):
    def test_transform_stack(self):
        t1 = ImageTransform(type=TransformType.RESAMPLE, params={"spacing": [1.0, 1.0, 1.0]})
        t2 = ImageTransform(type=TransformType.NORMALIZE, params={"std": 1.0, "mean": 0.0})
        
        stack = ImageTransformStack(
            stack_id="stack-001",
            name="standard_mri_prep",
            transforms=[t1, t2],
            version="1.0.0",
            created_at=datetime.now(timezone.utc).isoformat()
        )
        self.assertEqual(len(stack.transforms), 2)
        self.assertEqual(stack.transforms[0].type, TransformType.RESAMPLE)

    def test_roi_config(self):
        roi = ROIConfig(
            roi_id="roi-001",
            artifact_id="img-123",
            center=[100.0, 150.0, 50.0],
            size=[50.0, 50.0, 20.0],
            label="tumor",
            confidence=0.85
        )
        self.assertEqual(roi.label, "tumor")
        self.assertEqual(roi.center[0], 100.0)

    def test_dataset_card(self):
        card = DatasetCard(
            dataset_id="ds-brain-01",
            name="Brain MRI Dataset",
            modalities=["MRI", "CT"],
            voxel_spacing_stats={"mean": [1.0, 1.0, 1.0]},
            class_distribution={"normal": 50, "tumor": 25},
            institution_split={"Hospital A": 0.6, "Clinic B": 0.4},
            total_samples=75,
            created_at=datetime.now(timezone.utc).isoformat()
        )
        self.assertEqual(card.total_samples, 75)
        self.assertIn("MRI", card.modalities)

    def test_annotation_and_active_learning(self):
        session = AnnotationSession(
            session_id="sess-001",
            project_id="proj-abc",
            status=AnnotationStatus.IN_PROGRESS,
            samples_count=100,
            created_at=datetime.now(timezone.utc).isoformat()
        )
        self.assertEqual(session.status, AnnotationStatus.IN_PROGRESS)

        sample = ActiveLearningSample(
            sample_id="als-001",
            artifact_id="img-123",
            uncertainty_score=0.92,
            reason="High boundary uncertainty",
            priority=5
        )
        self.assertEqual(sample.priority, 5)
        self.assertFalse(sample.is_processed)

if __name__ == '__main__':
    unittest.main()
