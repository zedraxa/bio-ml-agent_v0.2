import logging
from typing import Dict, List, Optional
from bio_ml_agent.models.mission_pack import MissionPack, MissionPackStep
from bio_ml_agent.models.workspace_ux import AgentRole, TaskType, RiskLevel

logger = logging.getLogger(__name__)

class MissionPackRegistry:
    """Registry for professional Mission Packs (Capability-based scenarios)."""
    
    def __init__(self):
        self._packs: Dict[str, MissionPack] = {}
        self._initialize_default_packs()

    def register(self, pack: MissionPack):
        self._packs[pack.pack_id] = pack
        logger.info(f"[MissionPack] Registered: {pack.pack_id}")

    def get_pack(self, pack_id: str) -> Optional[MissionPack]:
        return self._packs.get(pack_id)

    def list_packs(self) -> List[MissionPack]:
        return list(self._packs.values())

    def _initialize_default_packs(self):
        # 1. repo_review_pack
        self.register(MissionPack(
            pack_id="repo_review_pack",
            name="Codebase Audit & Refactor",
            description="End-to-end repository analysis, security audit, and refactoring proposal.",
            capabilities=["CODE_ANALYSIS", "SECURITY_AUDIT", "REFACTORING"],
            steps=[
                MissionPackStep(
                    step_id="scan", title="Static Scan", description="Finding architectural smells",
                    task_type=TaskType.DISCOVER, assigned_agent=AgentRole.CODING_AGENT,
                    preferred_agent_id="code-architect"
                ),
                MissionPackStep(
                    step_id="audit", title="Security Audit", description="Adversarial check",
                    task_type=TaskType.CRITIQUE, assigned_agent=AgentRole.CRITIC,
                    depends_on=["scan"], preferred_agent_id="critic-agent"
                ),
                MissionPackStep(
                    step_id="refactor", title="Refactor Plan", description="Designing the new structure",
                    task_type=TaskType.SYNTHESIZE, assigned_agent=AgentRole.CODING_AGENT,
                    depends_on=["audit"], preferred_agent_id="refactor-repair",
                    requires_approval=True
                ),
                MissionPackStep(
                    step_id="summary", title="Final Brief", description="Executive summary",
                    task_type=TaskType.WRITE, assigned_agent=AgentRole.WRITING_AGENT,
                    depends_on=["refactor"]
                )
            ],
            final_outputs=["audit_report.md", "refactor_proposal.py"]
        ))

        # 2. lab_report_pack
        self.register(MissionPack(
            pack_id="lab_report_pack",
            name="Lab Report Synthesis",
            description="Transforms messy researcher notes into professional lab reports.",
            capabilities=["DOCUMENT_ANALYSIS", "DATA_SYNTHESIS", "ACADEMIC_WRITING"],
            steps=[
                MissionPackStep(
                    step_id="extract", title="Note Extraction", description="Pulling numbers from logs",
                    task_type=TaskType.ANALYZE, assigned_agent=AgentRole.ACADEMIC_EXPERT,
                    preferred_agent_id="document-agent"
                ),
                MissionPackStep(
                    step_id="processing", title="Data Processing", description="Cleaning and encoding",
                    task_type=TaskType.ANALYZE, assigned_agent=AgentRole.BIOINFORMATICIAN,
                    depends_on=["extract"], preferred_agent_id="bio-info"
                ),
                MissionPackStep(
                    step_id="drafting", title="Initial Draft", description="First report draft",
                    task_type=TaskType.WRITE, assigned_agent=AgentRole.WRITING_AGENT,
                    depends_on=["processing"], preferred_agent_id="academic-paper"
                ),
                MissionPackStep(
                    step_id="verification", title="Fact Check", description="Verifying data vs notes",
                    task_type=TaskType.VERIFY, assigned_agent=AgentRole.CRITIC,
                    depends_on=["drafting"], preferred_agent_id="critic-agent",
                    requires_approval=True
                )
            ],
            final_outputs=["lab_report.pdf", "data_summary.csv"]
        ))

        # 3. microscopy_pack
        self.register(MissionPack(
            pack_id="microscopy_pack",
            name="Advanced Microscopy Analysis",
            description="Full automated pipeline from image scanning to biological ontology classification.",
            capabilities=["IMAGE_SEGMENTATION", "MORPHOMETRICS", "BIO_ONTOLOGY"],
            steps=[
                MissionPackStep(
                    step_id="perception", title="Initial Scan", description="Image profiling",
                    task_type=TaskType.DISCOVER, assigned_agent=AgentRole.MICROSCOPY_AGENT,
                    preferred_agent_id="percept-agent"
                ),
                MissionPackStep(
                    step_id="segmentation", title="Denoising & Seg", description="Mask generation",
                    task_type=TaskType.ANALYZE, assigned_agent=AgentRole.MICROSCOPY_AGENT,
                    depends_on=["perception"], preferred_agent_id="segment-agent"
                ),
                MissionPackStep(
                    step_id="quantification", title="Morphometrics", description="Geometric Metrics",
                    task_type=TaskType.ANALYZE, assigned_agent=AgentRole.MICROSCOPY_AGENT,
                    depends_on=["segmentation"], preferred_agent_id="morpho-agent"
                ),
                MissionPackStep(
                    step_id="identification", title="Ontology ID", description="Bio-Ontology Mapping",
                    task_type=TaskType.ANALYZE, assigned_agent=AgentRole.MICROSCOPY_AGENT,
                    depends_on=["quantification"], preferred_agent_id="ident-agent"
                ),
                MissionPackStep(
                    step_id="report", title="Final Analysis", description="Generating summary",
                    task_type=TaskType.WRITE, assigned_agent=AgentRole.WRITING_AGENT,
                    depends_on=["identification"], requires_approval=True
                )
            ],
            final_outputs=["analysis_results.json", "segmented_overlay.png"]
        ))

# Singleton
mission_pack_registry = MissionPackRegistry()
