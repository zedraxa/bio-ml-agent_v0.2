import unittest
from datetime import datetime
from models.scientific_reporting import (
    ReportSection, ScientificReport, CitationType, CitationRecord,
    VisualType, TechnicalFigure, ScientificTable, ReproAppendix
)

class TestScientificReportingModels(unittest.TestCase):
    def test_scientific_report(self):
        methods = ReportSection(title="Methods", content="We used a Transformer model.")
        report = ScientificReport(
            report_id="rep-001",
            title="A New AI Era",
            authors=["Antigravity"],
            abstract="This is a breakthrough.",
            sections=[methods],
            creation_timestamp=datetime.utcnow().isoformat()
        )
        self.assertEqual(report.sections[0].title, "Methods")
        self.assertEqual(len(report.authors), 1)

    def test_citation_record(self):
        citation = CitationRecord(
            citation_id="cit-001",
            source_id="src-123",
            text_segment="AI is powerful.",
            citation_type=CitationType.PARAPHRASE
        )
        self.assertTrue(citation.is_supported)
        self.assertEqual(citation.citation_type, CitationType.PARAPHRASE)

    def test_technical_figure(self):
        fig = TechnicalFigure(
            figure_id="fig-001",
            title="ROC Curve",
            visual_type=VisualType.ROC_CURVE,
            artifact_path="/path/to/roc.png",
            caption="Accuracy plot"
        )
        self.assertEqual(fig.visual_type, VisualType.ROC_CURVE)

    def test_repro_appendix(self):
        appendix = ReproAppendix(
            appendix_id="app-001",
            experiment_id="exp-123",
            repro_bundle_id="bun-123",
            environment_info={"os": "linux"},
            package_versions={"pydantic": "2.0"},
            random_seed=42,
            hardware_specs={"gpu": "A100"}
        )
        self.assertEqual(appendix.random_seed, 42)
        self.assertEqual(appendix.environment_info["os"], "linux")

if __name__ == '__main__':
    unittest.main()
