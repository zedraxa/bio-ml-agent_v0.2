from .lab_report_orchestrator import LabReportOrchestratorAgent
from .section_writer_agent import SectionWriterAgent
from .observation_report_agent import ObservationToReportAgent
from .calculation_agent import CalculationIntegrationAgent
from .lab_report_critic import LabReportCriticAgent

from .paper_orchestrator import PaperDraftOrchestrator
from .abstract_agent import ScientificAbstractAgent
from .intro_methods_agents import IntroductionBuilderAgent, MethodsFormalizerAgent
from .results_discussion_agents import ResultsNarrationAgent, DiscussionInterpretationAgent
from .review_article_agent import ReviewArticleAgent

from .literature_matrix_agent import LiteratureMatrixAgent
from .citation_context_agent import CitationContextAgent
from .contradiction_synthesizer import ContradictionSynthesizerAgent
from .gap_finder_agent import GapFinderAgent

from .figure_caption_agent import FigureCaptionAgent
from .table_narration_agent import TableNarrationAgent
from .statistical_narration_agent import StatisticalNarrationAgent
from .microscopy_narration_agent import MicroscopyResultNarrationAgent
from .multimodal_integrator import MultiModalIntegratorAgent

from .style_paraphrase_agent import AcademicStyleAgent
from .structure_compliance_agent import StructureComplianceAgent
from .integrity_guardrail_agent import IntegrityGuardrailAgent
from .reviewer_simulation_agent import ReviewerSimulationAgent

from .presentation_script_agent import PresentationScriptAgent
from .poster_content_agent import PosterContentAgent
from .thesis_section_agent import ThesisSectionAgent
from .cover_letter_agent import CoverLetterAgent

__all__ = [
    "LabReportOrchestratorAgent",
    "SectionWriterAgent",
    "ObservationToReportAgent",
    "CalculationIntegrationAgent",
    "LabReportCriticAgent",
    "PaperDraftOrchestrator",
    "ScientificAbstractAgent",
    "IntroductionBuilderAgent",
    "MethodsFormalizerAgent",
    "ResultsNarrationAgent",
    "DiscussionInterpretationAgent",
    "ReviewArticleAgent",
    "LiteratureMatrixAgent",
    "CitationContextAgent",
    "ContradictionSynthesizerAgent",
    "GapFinderAgent",
    "FigureCaptionAgent",
    "TableNarrationAgent",
    "StatisticalNarrationAgent",
    "MicroscopyResultNarrationAgent",
    "MultiModalIntegratorAgent",
    "AcademicStyleAgent",
    "StructureComplianceAgent",
    "IntegrityGuardrailAgent",
    "ReviewerSimulationAgent",
    "PresentationScriptAgent",
    "PosterContentAgent",
    "ThesisSectionAgent",
    "CoverLetterAgent"
]
