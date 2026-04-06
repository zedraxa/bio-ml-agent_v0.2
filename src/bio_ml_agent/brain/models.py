# bio_ml_agent/brain/models.py
"""
Part VI - Axis A: Mission Brain Data Models

Pydantic models for the structured outputs of the Mission Brain:
  - MissionPlan (mission_plan.json)
  - AgentGraph (agent_graph.json)
  - SuccessCriteria (success_criteria.json)
  - FallbackStrategy (fallback_strategy.json)
"""

from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import uuid
from datetime import datetime


from bio_ml_agent.models.domain import (
    MissionPriority,
    ResourceIntensity,
    StepStatus,
    RiskLevel,
    AgentRole,
    TaskType,
    ArtifactType,
    ArtifactReviewStatus,
    CommentStatus,
    ReviewMode,
    ReviewerRole,
    IntentType,
    Severity,
    TargetType,
    ArtifactStateTransition,
    Comment,
    MissionStatus,
    Confidence
)

class AgentRetryStrategy(str, Enum):
    """G3: Specialized retry strategies by agent type."""
    LIMITED_RETRY = "limited_retry"
    REGENERATE = "regenerate"
    SELF_DEBUG = "self_debug"
    QUEUE_RESUME = "queue_resume"
    HUMAN_REVIEW = "human_review"

# Enums moved to domain.py


# ─── Core Data structures ─────────────────────────────────────────────────────

# Structures moved to domain.py or handled via import

class AgentResult(BaseModel):
    """The outcome of an agent's execution for a specific step."""
    agent_id: str
    step_id: str
    status: StepStatus
    output_artifacts: List[str] = Field(default_factory=list)
    logs: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None
    retry_count: int = 0
    confidence: Confidence
    evidence: List[Any] = Field(default_factory=list) # Changed from Evidence to Any as Evidence is defined later
    message: str = ""
    # R7-D1/D2: Agent-generated self-reflection and inline critique
    self_comments: List[Comment] = Field(default_factory=list)

class WhatsAppCardType(str, Enum):
    """E2: Types of structured WhatsApp messages."""
    MISSION_STARTED = "mission_started"
    MISSION_COMPLETED = "mission_completed"
    APPROVAL_NEEDED = "approval_needed"
    ERROR_ALERT = "error_alert"
    STATUS_UPDATE = "status_update"
    SUMMARY_REPORT = "summary_report"
    REVIEW_BUNDLE = "review_bundle"
    MEDIA_RECEIVED = "media_received"
    VOICE_NOTE_PROCESSED = "voice_note_processed"
    PROJECT_BRIEFING = "project_briefing"
    CONTEXT_SWITCH = "context_switch"
    INFO_RESPONSE = "info_response"
    CONVERSATIONAL_SUMMARY = "conversational_summary"

class WhatsAppMissionCard(BaseModel):
    """
    R7-E2: Structured format for WhatsApp interactions.
    Simulates interactive "cards" via strict text formatting.
    """
    card_type: WhatsAppCardType
    title: str = Field(..., description="Card header/title")
    body: str = Field(..., description="Main content of the card")
    action_buttons: List[str] = Field(default_factory=list, description="List of possible actions (e.g., ['Oayla', 'Reddet', 'Revize İste'])")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Hidden context, like mission_id or step_id")
    
    def render_to_text(self) -> str:
        """Converts the structured card into a formatted WhatsApp text message."""
        icons = {
            WhatsAppCardType.MISSION_STARTED: "🚀",
            WhatsAppCardType.MISSION_COMPLETED: "✅",
            WhatsAppCardType.APPROVAL_NEEDED: "⏳",
            WhatsAppCardType.ERROR_ALERT: "🚨",
            WhatsAppCardType.STATUS_UPDATE: "🔄",
            WhatsAppCardType.SUMMARY_REPORT: "📊",
            WhatsAppCardType.REVIEW_BUNDLE: "📝"
        }
        icon = icons.get(self.card_type, "ℹ️")
        
        lines = [f"{icon} *{self.title}*"]
        if self.body:
            lines.append("")
            lines.append(self.body)
            
        if self.action_buttons:
            lines.append("")
            lines.append("⚡ *Aksiyonlar:*")
            for i, btn in enumerate(self.action_buttons, 1):
                lines.append(f"[{i}] {btn}")
            lines.append("\n_(Yanıtlamak için numarayı veya aksiyonu yazın)_")
            
        return "\n".join(lines)

class Point(BaseModel):
    x: float
    y: float

class Box(BaseModel):
    x: float
    y: float
    width: float
    height: float

class TextRange(BaseModel):
    start_index: int
    end_index: int
    context_snippet: Optional[str] = None

class CodeAnchor(BaseModel):
    file_path: str
    line_number: int
    column: Optional[int] = None

class DataAnchor(BaseModel):
    row_id: str
    column_name: Optional[str] = None

class PatchAnchor(BaseModel):
    patch_id: str
    diff_hunk: str
    line_offset: int = 0

class Annotation(BaseModel):
    """R7-1/A2: A high-fidelity inline anchor for feedback."""
    annotation_id: str
    comment_id: Optional[str] = None
    tag: str = Field(..., description="e.g., 'nuclei', 'roi', 'typo'")
    
    # Typed Coordinates (A2)
    point: Optional[Point] = None
    box: Optional[Box] = None
    text_range: Optional[TextRange] = None
    code_anchor: Optional[CodeAnchor] = None
    data_anchor: Optional[DataAnchor] = None
    patch_anchor: Optional[PatchAnchor] = None
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ArtifactRecord(BaseModel):
    """A single artifact produced by the agent."""
    artifact_id: str = Field(..., description="Unique artifact identifier")
    artifact_type: ArtifactType
    title: str = Field(default="")
    description: str = Field(default="")
    file_path: Optional[str] = Field(None)
    data: Optional[Any] = Field(None)
    producer_agent_id: str
    producer_agent_role: str
    mission_id: str
    project_id: str
    source_input_ids: List[str] = Field(default_factory=list)
    parent_artifact_id: Optional[str] = Field(None)
    version: str = Field(default="1.0.0")
    status: ArtifactReviewStatus = Field(default=ArtifactReviewStatus.DRAFT)
    status_history: List[ArtifactStateTransition] = Field(default_factory=list)
    is_outdated: bool = Field(default=False)
    confidence: float = Field(default=0.5)
    comments: List[Comment] = Field(default_factory=list)
    annotations: List[Annotation] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryRecord(BaseModel):
    """A single memory entry to persist in long-term storage."""
    memory_id: str
    category: str = "general"
    content: str
    importance: float = 0.5
    tags: List[str] = Field(default_factory=list)
    source_step_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Evidence(BaseModel):
    """Proof or source material justifying an agent's conclusion."""
    source: str
    content_snippet: str
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ─── Mission Step ─────────────────────────────────────────────────────────────

class MissionStep(BaseModel):
    """A single atomic step within a mission plan."""
    step_id: str = Field(..., description="Unique step identifier, e.g. 'step_001'")
    title: str = Field(default="", description="Human-readable step title")
    description: str = Field(default="", description="Detailed description of what this step does")
    task_type: TaskType = Field(default=TaskType.ANALYZE, description="Universal task category from the Mission Decomposer (A3)")
    assigned_agent: AgentRole = Field(default=AgentRole.RESEARCHER, description="Which agent is responsible")
    depends_on: List[str] = Field(default_factory=list, description="List of step_ids this step depends on")
    is_parallel: bool = Field(default=False, description="Can this run in parallel with siblings?")
    requires_approval: bool = Field(default=False, description="Does this step need human sign-off before proceeding?")
    risk_level: RiskLevel = Field(default=RiskLevel.LOW, description="Risk assessment for this step")
    estimated_duration_seconds: int = Field(default=60, description="Estimated wall-clock time")
    status: StepStatus = Field(default=StepStatus.PENDING)
    input_artifacts: List[str] = Field(default_factory=list, description="Artifact IDs this step consumes")
    output_artifacts: List[str] = Field(default_factory=list, description="Artifact IDs this step produces")
    comments: List[Comment] = Field(default_factory=list)
    
    # E3: Idempotency
    fingerprint: Optional[str] = Field(None, description="Unique hash of task + inputs to prevent duplicate execution")
    
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # G4: Resource Scheduling
    resource_intensity: ResourceIntensity = Field(default=ResourceIntensity.LIGHT)
    estimated_tokens: Optional[int] = None


# ─── Agent Graph ──────────────────────────────────────────────────────────────

class AgentNode(BaseModel):
    """A node in the agent execution graph."""
    agent_id: str = Field(..., description="Unique agent instance identifier")
    role: AgentRole
    model_tier: int = Field(default=2, ge=1, le=3, description="LLM tier: 1=fast, 2=balanced, 3=powerful")
    capabilities: List[str] = Field(default_factory=list, description="Required capabilities for this node")
    upstream: List[str] = Field(default_factory=list, description="Agent IDs that feed into this node")
    downstream: List[str] = Field(default_factory=list, description="Agent IDs that consume this node's output")


class AgentGraph(BaseModel):
    """The directed acyclic graph of agents for a mission."""
    mission_id: str
    nodes: List[AgentNode] = Field(default_factory=list)
    execution_order: List[List[str]] = Field(
        default_factory=list,
        description="Topologically sorted layers: [[parallel_group_1], [parallel_group_2], ...]"
    )
    total_estimated_seconds: int = Field(default=0)


# ─── Success Criteria ─────────────────────────────────────────────────────────

class SuccessCriterion(BaseModel):
    """A single measurable success criterion."""
    criterion_id: str
    description: str
    metric: Optional[str] = Field(None, description="Quantitative metric name, e.g. 'accuracy', 'confidence_score'")
    threshold: Optional[float] = Field(None, description="Minimum acceptable value for the metric")
    is_mandatory: bool = Field(default=True)
    evaluation_agent: AgentRole = Field(default=AgentRole.CRITIC)


class SuccessCriteria(BaseModel):
    """Collection of success criteria for a mission."""
    mission_id: str
    criteria: List[SuccessCriterion] = Field(default_factory=list)
    min_mandatory_pass_rate: float = Field(default=1.0, description="Fraction of mandatory criteria that must pass")


# ─── Fallback Strategy ────────────────────────────────────────────────────────

class FallbackAction(BaseModel):
    """A single fallback action when a step fails."""
    trigger_step_id: str = Field(..., description="The step that triggers this fallback if it fails")
    action: str = Field(..., description="What to do: 'retry', 'skip', 'substitute_agent', 'escalate', 'abort'")
    substitute_agent: Optional[AgentRole] = Field(None, description="Agent to use if action is 'substitute_agent'")
    max_retries: int = Field(default=2)
    escalation_message: Optional[str] = Field(None, description="Message to surface to user if escalated")


class FallbackStrategy(BaseModel):
    """Complete fallback strategy for a mission."""
    mission_id: str
    actions: List[FallbackAction] = Field(default_factory=list)
    global_timeout_seconds: int = Field(default=3600, description="Maximum total mission runtime")
    abort_on_critical_failure: bool = Field(default=True)


class AgentRetryPolicy(BaseModel):
    """G3: Policy defining how a specific agent type handles failure."""
    strategy: AgentRetryStrategy
    max_retries: int = Field(default=2)
    alternative_strategy_description: Optional[str] = None
    debug_depth: int = Field(default=1, description="For self_debug: how deep to go in the stack")


# ─── Mission Plan (Top-level) ─────────────────────────────────────────────────

class MissionPlan(BaseModel):
    """
    The master output of the Mission Brain.
    
    Encapsulates the full execution blueprint:
      - mission_plan.json  → steps + dependencies
      - agent_graph.json   → agent DAG with execution order
      - success_criteria.json → measurable quality gates
      - fallback_strategy.json → recovery playbook
    """
    mission_id: str = Field(..., description="Unique mission identifier")
    project_id: str = Field(default="", description="Parent project this mission belongs to")
    pack_id: Optional[str] = Field(None, description="The Mission Pack ID this plan was derived from")
    user_prompt: str = Field(default="", description="Original user request")
    title: str = Field(default="", description="Generated mission title")
    objective: str = Field(default="", description="Distilled mission objective")
    
    # The 4 structured outputs
    steps: List[MissionStep] = Field(default_factory=list)
    agent_graph: Optional[AgentGraph] = Field(default=None)
    success_criteria: Optional[SuccessCriteria] = Field(default=None)
    fallback_strategy: Optional[FallbackStrategy] = Field(default=None)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    replanned_count: int = Field(default=0, description="How many times the Brain has replanned this mission")
    approval_gates: List[str] = Field(default_factory=list, description="Step IDs that require human approval")
    risk_summary: str = Field(default="", description="Human-readable risk assessment")
    
    # G4: Resource Scheduling
    priority: MissionPriority = Field(default=MissionPriority.NORMAL)
    total_resource_score: int = Field(default=0, description="Aggregated resource cost for scheduling")
    is_paused: bool = Field(default=False)
    
    # H1: Telemetry
    telemetry: Optional["MissionTelemetry"] = None
    
    # H2: Quality Scorecards
    scorecard: Optional["QualityScorecard"] = None
    
    # H3: Architectural Drift Detection
    drift_report: Optional["DriftReport"] = None
    
    # Sync & Versioning
    version: int = Field(default=1)
    last_synced_at: Optional[datetime] = Field(None)
    origin_device_id: Optional[str] = Field(None)
    
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class MissionTelemetry(BaseModel):
    """H1: Unified Mission Telemetry for monitoring and governance."""
    mission_id: str
    start_time: float = Field(default_factory=lambda: datetime.now().timestamp())
    end_time: Optional[float] = None
    total_duration_seconds: float = 0.0
    
    # Cost & Usage
    total_tokens_used: int = 0
    estimated_cost_usd: float = 0.0
    
    # Activity Metrics
    agents_involved: List[AgentRole] = Field(default_factory=list)
    artifacts_produced_count: int = 0
    review_cycles: int = 0
    approval_count: int = 0
    
    # Success/Failure Tracking
    failure_points: List[str] = Field(default_factory=list, description="Step IDs that failed/replanned")
    replan_count: int = 0
    status: MissionStatus = Field(default=MissionStatus.PENDING)
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ─── H2: Quality Scorecards ──────────────────────────────────────────────────

class QualityScorecard(BaseModel):
    """H2: Automated assessment of mission quality and health."""
    mission_id: str
    
    # Quantitative Scores (0.0 to 1.0)
    completeness: float = 0.0
    evidence_sufficiency: float = 0.0
    confidence_profile: float = 0.0
    review_burden: float = 0.0
    artifact_health: float = 0.0
    
    # Summary
    overall_quality_score: float = 0.0
    critical_gaps: List[str] = Field(default_factory=list)
    key_strengths: List[str] = Field(default_factory=list)
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ─── H3: Architectural Drift Detection ───────────────────────────────────────

class DriftViolation(BaseModel):
    """H3: Specific instance of architectural drift."""
    component: str
    violation_type: str # e.g., "INTERFACE_MISMATCH", "FORBIDDEN_DEPENDENCY", "LEGACY_PATTERN"
    severity: str # "INFO", "WARNING", "CRITICAL"
    description: str
    file_path: Optional[str] = None
    remediation_hint: Optional[str] = None

class DriftReport(BaseModel):
    """H3: Report summarizing detected architectural drift."""
    mission_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Summary
    is_healthy: bool = True
    violation_count: int = 0
    drift_score: float = 0.0 # 0.0 to 1.0 (1.0 = heavy drift)
    
    # Detailed Findings
    violations: List[DriftViolation] = Field(default_factory=list)
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ─── H4: Controlled Feature Flags ──────────────────────────────────────────

class FeatureStatus(str, Enum):
    """H4: Possible states for a controlled feature."""
    DISABLED = "disabled"
    EXPERIMENTAL = "experimental"
    BETA = "beta"
    STABLE = "stable"

class FeatureFlag(BaseModel):
    """H4: Control record for a system feature or agent."""
    name: str
    status: FeatureStatus = FeatureStatus.DISABLED
    enabled: bool = False
    
    # Targeting
    min_agent_version: Optional[str] = None
    allowed_roles: List[str] = Field(default_factory=list)
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ─── A4: Replan Result ────────────────────────────────────────────────────────

class ReplanResult(BaseModel):
    """
    Structured output from the Mission Replanner (A4).
    
    Documents exactly which recovery strategy was chosen and why,
    providing full traceability for the Decision Journal.
    """
    mission_id: str
    failed_step_id: str
    strategy_used: str = Field(..., description="One of: substitute_agent, model_switch, simplify_task, alternate_path, user_escalation")
    strategy_detail: str = Field(..., description="Human-readable explanation of the recovery action")
    recovery_succeeded: bool = Field(..., description="True if automatic recovery was applied; False if human input needed")
    replan_count: int = Field(default=1, description="Total number of replans for this mission so far")
    error_context: str = Field(default="", description="Original error that triggered replanning")
    confidence: Optional[float] = Field(None, description="Confidence score at time of failure, if applicable")


# ─── E1: Unified Project State ────────────────────────────────────────────────

class ProjectState(BaseModel):
    """
    The Single Source of Truth for a research project.
    
    Bridges missions, artifacts, and memory into a persistent state.
    """
    project_id: str = Field(..., description="Unique project identifier")
    name: str = Field(..., description="Human-readable project name")
    
    # Execution State
    active_mission_id: Optional[str] = Field(None, description="ID of the currently running mission")
    
    # Reliability & Consistency
    artifacts: List[ArtifactRecord] = Field(default_factory=list, description="All artifacts produced in this project")
    approved_artifact_ids: List[str] = Field(default_factory=list, description="IDs of artifacts that passed review")
    pending_approval_step_ids: List[str] = Field(default_factory=list, description="Steps awaiting human sign-off")
    
    # Intelligence State
    critical_findings: List[str] = Field(default_factory=list, description="High-impact discoveries")
    open_blockers: List[str] = Field(default_factory=list, description="Things preventing progress")
    memory_snapshot_ref: Optional[str] = Field(None, description="Reference to the latest project memory point")
    
    # History
    mission_history: List[str] = Field(default_factory=list, description="List of all missions attempted in this project")
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    # Sync & Versioning
    version: int = Field(default=1)
    last_synced_at: Optional[datetime] = Field(None)
    origin_device_id: Optional[str] = Field(None)
    
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# ─── G1: Mission Checkpointing ────────────────────────────────────────────────

class MissionSnapshot(BaseModel):
    """
    Axis G1: Mission Checkpoint / Snapshot.
    
    A point-in-time capture of the entire project context for recovery.
    """
    snapshot_id: str = Field(..., description="Unique snapshot identifier")
    mission_id: str
    project_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # The 'Source of Truth' states
    plan_snapshot: MissionPlan
    project_snapshot: ProjectState
    
    # Traceability
    last_completed_step_id: Optional[str] = Field(None)
    memory_delta: Dict[str, Any] = Field(default_factory=dict, description="New findings/facts since last checkpoint")
    
    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
