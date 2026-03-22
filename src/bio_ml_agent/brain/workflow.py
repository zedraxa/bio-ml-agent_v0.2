import hashlib
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from .models import MissionPlan, MissionStep, TaskType, AgentRole, RiskLevel, StepStatus, AgentGraph, SuccessCriteria, FallbackStrategy, ProjectState

class StepTemplate(BaseModel):
    """The blueprint for a single step in a larger workflow."""
    title: str
    description: str
    task_type: TaskType
    assigned_agent: AgentRole
    depends_on: List[int] = Field(default_factory=list, description="Indices of steps in the template this depends on")
    is_parallel: bool = Field(default=False)
    requires_approval: bool = Field(default=False)
    risk_level: RiskLevel = Field(default=RiskLevel.LOW)

class MissionWorkflowTemplate(BaseModel):
    """A reusable graph of mission steps (D1)."""
    template_id: str
    name: str
    description: str
    steps: List[StepTemplate] = Field(default_factory=list)
    version: str = Field(default="1.0.0")

class MissionGraphEngine:
    """
    Orchestrator for managing and instantiating MissionWorkflowTemplates.
    Satisfies Axis D1: Mission Graph Engine.
    """
    
    def __init__(self):
        self._registry: Dict[str, MissionWorkflowTemplate] = {}

    def register_template(self, template: MissionWorkflowTemplate):
        self._registry[template.template_id] = template

    def get_template(self, template_id: str) -> Optional[MissionWorkflowTemplate]:
        return self._registry.get(template_id)

    def instantiate(self, template_id: str, mission_id: str, project_id: str, user_prompt: str) -> MissionPlan:
        """Creates a concrete MissionPlan from a template."""
        template = self.get_template(template_id)
        if not template:
            raise ValueError(f"Template {template_id} not found.")

        steps = []
        for i, st in enumerate(template.steps):
            step_id = f"{mission_id}_step_{i+1:03d}"
            
            # Resolve dependency IDs from indices
            depends_on_ids = [f"{mission_id}_step_{idx+1:03d}" for idx in st.depends_on]
            
            step = MissionStep(
                step_id=step_id,
                title=st.title,
                description=st.description,
                task_type=st.task_type,
                assigned_agent=st.assigned_agent,
                depends_on=depends_on_ids,
                is_parallel=st.is_parallel,
                requires_approval=st.requires_approval,
                risk_level=st.risk_level
            )
            steps.append(step)

        return MissionPlan(
            mission_id=mission_id,
            project_id=project_id,
            user_prompt=user_prompt,
            title=template.name,
            objective=template.description,
            steps=steps,
            agent_graph=AgentGraph(mission_id=mission_id),
            success_criteria=SuccessCriteria(mission_id=mission_id),
            fallback_strategy=FallbackStrategy(mission_id=mission_id)
        )

    def find_resume_point(self, plan: MissionPlan) -> Optional[int]:
        """Identifies the index of the first step that needs execution."""
        from .models import StepStatus
        for i, step in enumerate(plan.steps):
            if step.status not in [StepStatus.COMPLETED, StepStatus.SKIPPED]:
                return i
        return None

    def checkpoint(self, plan: MissionPlan):
        """Saves current plan state to the global store."""
        from .persistence import MISSION_STORE
        MISSION_STORE.save(plan)

    def generate_fingerprint(self, step: MissionStep) -> str:
        """Generates a stable hash for a step's action."""
        # Normalize inputs for stable hashing
        inputs = sorted(step.input_artifacts)
        payload = f"{step.task_type}:{step.assigned_agent}:{step.description}:{inputs}"
        return hashlib.sha256(payload.encode()).hexdigest()

    def check_idempotency(self, step: MissionStep, project: ProjectState) -> Optional[str]:
        """
        Checks if an identical task was already completed in the project.
        Returns the ID of the existing output artifact if found.
        """
        # This is a simplified check for the demo. 
        # In a real system, we'd query an Artifact Registry.
        # For now, we simulate by checking if the fingerprint is in 'approved_artifact_ids' 
        # (assuming we tagged them with fingerprints)
        if step.fingerprint and step.fingerprint in project.approved_artifact_ids:
            return f"artifact_reused_{step.fingerprint[:8]}"
        return None

# Pre-defined professional workflows
WORKFLOW_REGISTRY = MissionGraphEngine()

# D2 Scenario: Microscopy Report
WORKFLOW_REGISTRY.register_template(MissionWorkflowTemplate(
    template_id="microscopy_report",
    name="Microscopy Analysis & Report",
    description="Full lifecycle from raw imagery to interpreted report.",
    steps=[
        StepTemplate(title="Profiling", description="Initial scan", task_type=TaskType.DISCOVER, assigned_agent=AgentRole.BROWSER_AGENT),
        StepTemplate(title="Identification", description="Biological structure ID", task_type=TaskType.ANALYZE, assigned_agent=AgentRole.MICROSCOPY_AGENT, depends_on=[0]),
        StepTemplate(title="Segmentation", description="Mask generation", task_type=TaskType.ANALYZE, assigned_agent=AgentRole.MICROSCOPY_AGENT, depends_on=[1]),
        StepTemplate(title="Morphometrics", description="Quantitative data", task_type=TaskType.ANALYZE, assigned_agent=AgentRole.BIOINFORMATICIAN, depends_on=[2]),
        StepTemplate(title="Narration", description="Narrative synthesis", task_type=TaskType.WRITE, assigned_agent=AgentRole.WRITING_AGENT, depends_on=[3]),
        StepTemplate(title="Critique", description="Adversarial check", task_type=TaskType.CRITIQUE, assigned_agent=AgentRole.CRITIC, depends_on=[4]),
        StepTemplate(title="Export", description="Final PDF delivery", task_type=TaskType.EXPORT, assigned_agent=AgentRole.WRITING_AGENT, depends_on=[5]),
    ]
))

# D2 Scenario: Lab Report from Notes
WORKFLOW_REGISTRY.register_template(MissionWorkflowTemplate(
    template_id="lab_report_from_notes",
    name="Lab Report from Raw Notes",
    description="Synthesizes structured reports from messy researcher notes and data files.",
    steps=[
        StepTemplate(title="Note Cleaning", description="Standardizing researcher notes", task_type=TaskType.DISCOVER, assigned_agent=AgentRole.RESEARCHER),
        StepTemplate(title="Data Extraction", description="Pulling numbers from logs", task_type=TaskType.ANALYZE, assigned_agent=AgentRole.BIOINFORMATICIAN, depends_on=[0]),
        StepTemplate(title="Drafting", description="First report draft", task_type=TaskType.WRITE, assigned_agent=AgentRole.WRITING_AGENT, depends_on=[1]),
        StepTemplate(title="Fact Check", description="Verifying data vs notes", task_type=TaskType.VERIFY, assigned_agent=AgentRole.CRITIC, depends_on=[2]),
        StepTemplate(title="Final Polish", description="Adding citations & abstract", task_type=TaskType.WRITE, assigned_agent=AgentRole.ACADEMIC_EXPERT, depends_on=[3]),
    ]
))

# D2 Scenario: Literature to Review
WORKFLOW_REGISTRY.register_template(MissionWorkflowTemplate(
    template_id="literature_to_review",
    name="Literature Review Matrix",
    description="Scales research from keywords to a comparative knowledge matrix.",
    steps=[
        StepTemplate(title="Global Search", description="PubMed and ArXiv crawling", task_type=TaskType.DISCOVER, assigned_agent=AgentRole.BROWSER_AGENT),
        StepTemplate(title="Paper Screening", description="Selection based on abstract", task_type=TaskType.VERIFY, assigned_agent=AgentRole.RESEARCHER, depends_on=[0]),
        StepTemplate(title="Extraction", description="Extracting key findings", task_type=TaskType.ANALYZE, assigned_agent=AgentRole.ACADEMIC_EXPERT, depends_on=[1]),
        StepTemplate(title="Matrix Synthesis", description="Comparing methodologies", task_type=TaskType.SYNTHESIZE, assigned_agent=AgentRole.ACADEMIC_EXPERT, depends_on=[2]),
        StepTemplate(title="Review Writing", description="Drafting the narrative review", task_type=TaskType.WRITE, assigned_agent=AgentRole.WRITING_AGENT, depends_on=[3]),
    ]
))

# D2 Scenario: Dataset to Model
WORKFLOW_REGISTRY.register_template(MissionWorkflowTemplate(
    template_id="dataset_to_model",
    name="Dataset to Trained Model",
    description="Automates data profiling, preprocessing, and model training.",
    steps=[
        StepTemplate(title="Data Profiling", description="Exploratory Data Analysis", task_type=TaskType.DISCOVER, assigned_agent=AgentRole.DATA_ENGINEER),
        StepTemplate(title="Preprocessing", description="Cleaning and encoding", task_type=TaskType.ANALYZE, assigned_agent=AgentRole.ML_EXPERT, depends_on=[0]),
        StepTemplate(title="Baseline Training", description="First model iteration", task_type=TaskType.ANALYZE, assigned_agent=AgentRole.ML_EXPERT, depends_on=[1]),
        StepTemplate(title="Evaluation", description="Performance metrics", task_type=TaskType.VERIFY, assigned_agent=AgentRole.ML_EXPERT, depends_on=[2]),
        StepTemplate(title="Documentation", description="Model card generation", task_type=TaskType.WRITE, assigned_agent=AgentRole.CODING_AGENT, depends_on=[3]),
    ]
))

# D2 Scenario: Repo Refactor Review
WORKFLOW_REGISTRY.register_template(MissionWorkflowTemplate(
    template_id="repo_refactor_review",
    name="Codebase Refactor & Audit",
    description="End-to-end code improvement from analysis to test coverage.",
    steps=[
        StepTemplate(title="Static Analysis", description="Finding architectural smells", task_type=TaskType.DISCOVER, assigned_agent=AgentRole.CODING_AGENT),
        StepTemplate(title="Refactor Plan", description="Designing the new structure", task_type=TaskType.SYNTHESIZE, assigned_agent=AgentRole.PLANNER, depends_on=[0]),
        StepTemplate(title="Implementation", description="Executing the changes", task_type=TaskType.ANALYZE, assigned_agent=AgentRole.CODING_AGENT, depends_on=[1]),
        StepTemplate(title="Test Generation", description="Updating unit tests", task_type=TaskType.WRITE, assigned_agent=AgentRole.CODING_AGENT, depends_on=[2]),
        StepTemplate(title="Adversarial Audit", description="Security and perf check", task_type=TaskType.CRITIQUE, assigned_agent=AgentRole.CRITIC, depends_on=[3]),
    ]
))

# D2 Scenario: Sequence to Structure Brief
WORKFLOW_REGISTRY.register_template(MissionWorkflowTemplate(
    template_id="sequence_to_structure_brief",
    name="Sequence Analysis & Structural Insight",
    description="Goes from genomic/proteomic sequence to structural briefing.",
    steps=[
        StepTemplate(title="Sequence Profiling", description="Homology and conservation", task_type=TaskType.ANALYZE, assigned_agent=AgentRole.BIOINFORMATICIAN),
        StepTemplate(title="Structure Prediction", description="AlphaFold-tier prediction", task_type=TaskType.ANALYZE, assigned_agent=AgentRole.IN_SILICO_EXPERT, depends_on=[0]),
        StepTemplate(title="Mapping", description="Active site identification", task_type=TaskType.SYNTHESIZE, assigned_agent=AgentRole.IN_SILICO_EXPERT, depends_on=[1]),
        StepTemplate(title="Technical Briefing", description="Writing the structural notes", task_type=TaskType.WRITE, assigned_agent=AgentRole.WRITING_AGENT, depends_on=[2]),
    ]
))

# D2 Scenario: Experiment to Presentation
WORKFLOW_REGISTRY.register_template(MissionWorkflowTemplate(
    template_id="experiment_to_presentation",
    name="Experiment Data to Presentation Slidework",
    description="Prepares conference-ready slides from raw experiment logs.",
    steps=[
        StepTemplate(title="Key Findings", description="Extracting the 'wow' factors", task_type=TaskType.SYNTHESIZE, assigned_agent=AgentRole.RESEARCHER),
        StepTemplate(title="Visual Design", description="Defining chart requirements", task_type=TaskType.SYNTHESIZE, assigned_agent=AgentRole.WRITING_AGENT, depends_on=[0]),
        StepTemplate(title="Slide Scripting", description="Narration for each slide", task_type=TaskType.WRITE, assigned_agent=AgentRole.WRITING_AGENT, depends_on=[1]),
        StepTemplate(title="Review & Polish", description="Visual and flow audit", task_type=TaskType.CRITIQUE, assigned_agent=AgentRole.RESEARCHER, depends_on=[2]),
    ]
))

# D3 Scenario: Cross-Domain Fusion (Drug Discovery Proof)
WORKFLOW_REGISTRY.register_template(MissionWorkflowTemplate(
    template_id="cross_domain_drug_discovery",
    name="Full Drug Discovery Proof (Sequence to Brief)",
    description="Complex fusion of bio-structural analysis, pharmacology, and executive communication.",
    steps=[
        StepTemplate(title="Sequence Profiling", description="Genomic/Homology context", task_type=TaskType.ANALYZE, assigned_agent=AgentRole.BIOINFORMATICIAN),
        StepTemplate(title="Structure Prediction", description="AlphaFold or similar 3D modeling", task_type=TaskType.ANALYZE, assigned_agent=AgentRole.IN_SILICO_EXPERT, depends_on=[0]),
        StepTemplate(title="Target Suitability", description="Pharmacological suitability review", task_type=TaskType.SYNTHESIZE, assigned_agent=AgentRole.RESEARCHER, depends_on=[1]),
        StepTemplate(title="Technical Brief", description="Drafting the detailed MS report", task_type=TaskType.WRITE, assigned_agent=AgentRole.WRITING_AGENT, depends_on=[2]),
        StepTemplate(title="Executive Summary", description="Slide deck content for leadership", task_type=TaskType.WRITE, assigned_agent=AgentRole.WRITING_AGENT, depends_on=[3]),
        StepTemplate(title="Critique & Compliance", description="Adversarial and ethics review", task_type=TaskType.CRITIQUE, assigned_agent=AgentRole.CRITIC, depends_on=[3, 4]),
    ]
))

# F6 Scenario: Mitosis Phase Identification
WORKFLOW_REGISTRY.register_template(MissionWorkflowTemplate(
    template_id="mitosis_id_workflow",
    name="Mitosis Phase Identification",
    description="Automated cell cycle analysis from microscopy imagery.",
    steps=[
        StepTemplate(title="Image Acquisition", description="Fetching raw microscopy slides", task_type=TaskType.DISCOVER, assigned_agent=AgentRole.BROWSER_AGENT),
        StepTemplate(title="Segmentation", description="UNet-based cell segmentation", task_type=TaskType.ANALYZE, assigned_agent=AgentRole.MICROSCOPY_AGENT, depends_on=[0]),
        StepTemplate(title="Phase Classification", description="Classifying prophase, metaphase, etc.", task_type=TaskType.ANALYZE, assigned_agent=AgentRole.MICROSCOPY_AGENT, depends_on=[1]),
        StepTemplate(title="Cycle Summary", description="Generating mitotic index and counts", task_type=TaskType.WRITE, assigned_agent=AgentRole.BIOINFORMATICIAN, depends_on=[2]),
    ]
))

# F6 Scenario: Sequence to Target Assessment
WORKFLOW_REGISTRY.register_template(MissionWorkflowTemplate(
    template_id="target_assessment_workflow",
    name="Sequence to Target Assessment",
    description="Genetic sequence analysis for drug target suitability.",
    steps=[
        StepTemplate(title="Sequence Fetch", description="Retrieving FASTA from UniProt", task_type=TaskType.DISCOVER, assigned_agent=AgentRole.BROWSER_AGENT),
        StepTemplate(title="Structural Prediction", description="Folding and pocket analysis", task_type=TaskType.ANALYZE, assigned_agent=AgentRole.IN_SILICO_EXPERT, depends_on=[0]),
        StepTemplate(title="Druggability Cross-ref", description="Querying ChEMBL/TargetDB", task_type=TaskType.ANALYZE, assigned_agent=AgentRole.RESEARCHER, depends_on=[1]),
        StepTemplate(title="Suitability Report", description="Final assessment and tiering", task_type=TaskType.WRITE, assigned_agent=AgentRole.ACADEMIC_EXPERT, depends_on=[2]),
    ]
))

# F6 Scenario: Proposal Draft Workflow
WORKFLOW_REGISTRY.register_template(MissionWorkflowTemplate(
    template_id="proposal_draft_workflow",
    name="Scientific Proposal Drafting",
    description="From literature synthesis to a grant-ready proposal.",
    steps=[
        StepTemplate(title="Literature Review", description="Gathering foundational papers", task_type=TaskType.DISCOVER, assigned_agent=AgentRole.BROWSER_AGENT),
        StepTemplate(title="Gap Analysis", description="Identifying research novelties", task_type=TaskType.SYNTHESIZE, assigned_agent=AgentRole.ACADEMIC_EXPERT, depends_on=[0]),
        StepTemplate(title="Methodology Design", description="Drafting the experimental plan", task_type=TaskType.WRITE, assigned_agent=AgentRole.RESEARCHER, depends_on=[1]),
        StepTemplate(title="Full Proposal", description="Synthesizing the final draft", task_type=TaskType.WRITE, assigned_agent=AgentRole.WRITING_AGENT, depends_on=[2]),
    ]
))
