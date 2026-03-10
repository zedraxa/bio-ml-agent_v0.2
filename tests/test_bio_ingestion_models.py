import unittest
from datetime import datetime
from models.bio_ingestion import (
    BioFormat, BioSequenceMeta, Modality, MedicalImageMeta,
    QCCheckType, QCCheckResult, DeIdAction, DeIdPolicy, AnonymizationAudit
)

class TestBioIngestionModels(unittest.TestCase):
    def test_bio_sequence_meta(self):
        meta = BioSequenceMeta(
            artifact_id="bio-001",
            format=BioFormat.FASTA,
            organism="Homo sapiens",
            sequence_type="DNA",
            read_count=1000000,
            avg_read_length=150.5,
            reference_genome="GRCh38"
        )
        self.assertEqual(meta.format, BioFormat.FASTA)
        self.assertEqual(meta.organism, "Homo sapiens")

    def test_medical_image_meta(self):
        meta = MedicalImageMeta(
            artifact_id="img-001",
            modality=Modality.MRI,
            study_instance_uid="1.2.3.4.5",
            body_part="BRAIN",
            pixel_spacing=[0.5, 0.5],
            slice_thickness=1.0,
            dimensions=[512, 512, 160]
        )
        self.assertEqual(meta.modality, Modality.MRI)
        self.assertEqual(meta.dimensions[0], 512)

    def test_qc_check_result(self):
        result = QCCheckResult(
            check_id="qc-001",
            artifact_id="bio-001",
            check_type=QCCheckType.CORRUPTION,
            is_passed=True,
            score=0.99,
            findings=["No corruption detected"],
            timestamp=datetime.utcnow().isoformat()
        )
        self.assertTrue(result.is_passed)
        self.assertEqual(result.score, 0.99)

    def test_de_id_and_audit(self):
        policy = DeIdPolicy(
            policy_id="pol-001",
            name="Standard HIPAA de-id",
            tag_actions={"PatientName": DeIdAction.REMOVE, "PatientID": DeIdAction.MASK}
        )
        self.assertEqual(policy.tag_actions["PatientName"], DeIdAction.REMOVE)

        audit = AnonymizationAudit(
            audit_id="aud-001",
            original_artifact_id="img-001",
            anonymized_artifact_id="img-001-anon",
            policy_id="pol-001",
            agent_id="agent-01",
            timestamp=datetime.utcnow().isoformat()
        )
        self.assertEqual(audit.agent_id, "agent-01")

if __name__ == '__main__':
    unittest.main()
