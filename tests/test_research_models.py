import unittest
from datetime import datetime, timezone
from bio_ml_agent.models.research import (
    SourcePlatform, ScientificSource, SourceType, SourceQualityScore,
    EvidenceType, EvidenceNode, ResearchClaim, ContradictionReport
)

class TestResearchModels(unittest.TestCase):
    def test_scientific_source(self):
        source = ScientificSource(
            source_id="src-arxiv-01",
            title="Attention Is All You Need",
            authors=["Vaswani et al."],
            platform=SourcePlatform.ARXIV,
            doi="10.48550/arXiv.1706.03762",
            is_peer_reviewed=True
        )
        self.assertEqual(source.platform, SourcePlatform.ARXIV)
        self.assertTrue(source.is_peer_reviewed)

    def test_source_quality_score(self):
        score = SourceQualityScore(
            source_id="src-arxiv-01",
            source_type=SourceType.PEER_REVIEWED,
            overall_score=0.95,
            factors={"citation_impact": 0.9, "journal_rank": 0.98},
            review_status="verified"
        )
        self.assertEqual(score.overall_score, 0.95)
        self.assertEqual(score.source_type, SourceType.PEER_REVIEWED)

    def test_evidence_and_claim(self):
        node = EvidenceNode(
            source_id="src-arxiv-01",
            evidence_type=EvidenceType.SUPPORTING,
            snippet="Transformer architectures outperform RNNs in translation tasks.",
            confidence=0.98
        )
        claim = ResearchClaim(
            claim_id="claim-transformer-01",
            claim_text="Transformers are state-of-the-art for NLP.",
            evidence_nodes=[node],
            consensus_score=0.95,
            status="validated"
        )
        self.assertEqual(len(claim.evidence_nodes), 1)
        self.assertEqual(claim.evidence_nodes[0].evidence_type, EvidenceType.SUPPORTING)

    def test_contradiction_report(self):
        report = ContradictionReport(
            report_id="err-001",
            claim_id="claim-transformer-01",
            contradicting_source_ids=["src-old-01", "src-new-02"],
            details="Source A says X while Source B says Y",
            severity="high",
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        self.assertEqual(report.severity, "high")
        self.assertEqual(len(report.contradicting_source_ids), 2)

if __name__ == '__main__':
    unittest.main()
