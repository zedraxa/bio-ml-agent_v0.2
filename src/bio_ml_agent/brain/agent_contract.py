# bio_ml_agent/brain/agent_contract.py
"""
Part VI - Axis B: Unified Agent Contract & Lifecycle

B1: 8-Phase Lifecycle + B2: Agent Capability Declaration

Every agent in the Bio-ML Agent ecosystem — whether it is a BrowserAgent,
MicroscopyAgent, WritingAgent, or any future specialist — MUST speak the
same language. This module defines that language.

The 8-Phase Agent Lifecycle:
    1. perceive     — Observe inputs, environment, context
    2. plan         — Decompose the goal into sub-steps
    3. act          — Execute the primary action
    4. verify       — Validate the output quality
    5. summarize    — Produce a human-readable summary
    6. emit_artifacts — Declare produced files/data
    7. emit_memory  — Write to long-term memory
    8. emit_events  — Broadcast lifecycle events to the MessageBus

By conforming to this contract, any agent can be:
  - Orchestrated by MissionBrain
  - Monitored by the Live Run system
  - Reviewed by CriticAgent
  - Logged by the Audit system
  - Substituted by the Replanner
"""

import time
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from .models import (
    AgentRole, ArtifactType, ArtifactReviewStatus, ArtifactStateTransition, 
    ArtifactRecord, MemoryRecord, Evidence, AgentRetryPolicy, AgentRetryStrategy
)

logger = logging.getLogger("bio_ml_agent.brain.contract")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Lifecycle Phase Enums
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LifecyclePhase(str, Enum):
    """The 8 phases every agent must pass through."""
    PERCEIVE = "perceive"
    PLAN = "plan"
    ACT = "act"
    VERIFY = "verify"
    SUMMARIZE = "summarize"
    EMIT_ARTIFACTS = "emit_artifacts"
    EMIT_MEMORY = "emit_memory"
    EMIT_EVENTS = "emit_events"


class ConfidenceLevel(str, Enum):
    """Standardized confidence scoring across all agents."""
    VERY_LOW = "very_low"      # < 0.3
    LOW = "low"                # 0.3 - 0.5
    MEDIUM = "medium"          # 0.5 - 0.7
    HIGH = "high"              # 0.7 - 0.9
    VERY_HIGH = "very_high"    # > 0.9


# VALID_ARTIFACT_TRANSITIONS moved to where it's needed or maintained here
VALID_ARTIFACT_TRANSITIONS = {
    ArtifactReviewStatus.DRAFT: [ArtifactReviewStatus.REVIEW_NEEDED, ArtifactReviewStatus.OUTDATED, ArtifactReviewStatus.CONFLICT],
    ArtifactReviewStatus.REVIEW_NEEDED: [ArtifactReviewStatus.APPROVED, ArtifactReviewStatus.REJECTED, ArtifactReviewStatus.DRAFT, ArtifactReviewStatus.OUTDATED, ArtifactReviewStatus.CONFLICT],
    ArtifactReviewStatus.APPROVED: [ArtifactReviewStatus.FINAL, ArtifactReviewStatus.OUTDATED, ArtifactReviewStatus.CONFLICT],
    ArtifactReviewStatus.OUTDATED: [ArtifactReviewStatus.DRAFT, ArtifactReviewStatus.CONFLICT],
    ArtifactReviewStatus.CONFLICT: [ArtifactReviewStatus.DRAFT], # E4: Resolution path
    ArtifactReviewStatus.FINAL: [ArtifactReviewStatus.EXPORTED, ArtifactReviewStatus.OUTDATED, ArtifactReviewStatus.CONFLICT],
    ArtifactReviewStatus.EXPORTED: [ArtifactReviewStatus.OUTDATED, ArtifactReviewStatus.CONFLICT],
}


class EventType(str, Enum):
    """Types of lifecycle events emitted to the MessageBus."""
    PHASE_STARTED = "phase_started"
    PHASE_COMPLETED = "phase_completed"
    PHASE_FAILED = "phase_failed"
    CONFIDENCE_DROP = "confidence_drop"
    APPROVAL_NEEDED = "approval_needed"
    ARTIFACT_PRODUCED = "artifact_produced"
    MEMORY_WRITTEN = "memory_written"
    AGENT_SUBSTITUTED = "agent_substituted"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# B3: Agent State Model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AgentState(str, Enum):
    """B3: Canonical agent execution states for UI and orchestration."""
    QUEUED = "queued"                       # Waiting in the mission queue
    RUNNING = "running"                     # Actively executing a lifecycle phase
    WAITING_INPUT = "waiting_input"         # Blocked on upstream data/artifact
    AWAITING_APPROVAL = "waiting_approval"   # Paused until human approves
    BLOCKED = "blocked"                     # Cannot proceed (dependency / error)
    COMPLETED = "completed"                 # Finished successfully
    FAILED = "failed"                       # Terminated with error
    SUPERSEDED = "superseded"               # Replaced by another agent (via Replanner)


# Valid state transitions enforced by the orchestrator
VALID_STATE_TRANSITIONS: Dict[str, List[str]] = {
    AgentState.QUEUED:            [AgentState.RUNNING, AgentState.BLOCKED, AgentState.SUPERSEDED],
    AgentState.RUNNING:           [AgentState.COMPLETED, AgentState.FAILED, AgentState.WAITING_INPUT, AgentState.AWAITING_APPROVAL, AgentState.BLOCKED],
    AgentState.WAITING_INPUT:     [AgentState.RUNNING, AgentState.BLOCKED, AgentState.FAILED],
    AgentState.AWAITING_APPROVAL:  [AgentState.RUNNING, AgentState.BLOCKED, AgentState.FAILED, AgentState.SUPERSEDED],
    AgentState.BLOCKED:           [AgentState.RUNNING, AgentState.FAILED, AgentState.SUPERSEDED],
    AgentState.COMPLETED:         [AgentState.SUPERSEDED],  # Only superseded can follow completed
    AgentState.FAILED:            [AgentState.QUEUED, AgentState.SUPERSEDED],  # Retry or replace
    AgentState.SUPERSEDED:        [],  # Terminal state
}


class AgentStateTransition(BaseModel):
    """Records a single state change in the agent lifecycle."""
    from_state: AgentState
    to_state: AgentState
    timestamp: float = Field(default_factory=time.time)
    reason: str = Field(default="", description="Why this transition occurred")
    phase: Optional[LifecyclePhase] = Field(None, description="Which lifecycle phase triggered the transition")


class AgentStateRecord(BaseModel):
    """
    B3: Complete state history for one agent execution.
    
    Provides:
      - Current state for UI rendering
      - Full transition history for Audit Journal
      - Duration tracking per state
    """
    agent_name: str
    current_state: AgentState = Field(default=AgentState.QUEUED)
    transitions: List[AgentStateTransition] = Field(default_factory=list)
    
    def transition_to(self, new_state: AgentState, reason: str = "",
                      phase: Optional[LifecyclePhase] = None) -> bool:
        """Attempt a state transition. Returns True if valid, False if rejected."""
        valid_targets = VALID_STATE_TRANSITIONS.get(self.current_state, [])
        if new_state not in valid_targets:
            logger.warning(
                f"[B3:StateModel] Invalid transition: {self.current_state} → {new_state} "
                f"(valid: {valid_targets}) for agent '{self.agent_name}'"
            )
            return False
        
        self.transitions.append(AgentStateTransition(
            from_state=self.current_state,
            to_state=new_state,
            reason=reason,
            phase=phase,
        ))
        self.current_state = new_state
        return True
    
    @property
    def is_terminal(self) -> bool:
        """Whether the agent has reached a terminal state."""
        return self.current_state in (AgentState.COMPLETED, AgentState.FAILED, AgentState.SUPERSEDED)
    
    @property
    def is_actionable(self) -> bool:
        """Whether the agent needs intervention (approval or input)."""
        return self.current_state in (AgentState.WAITING_INPUT, AgentState.AWAITING_APPROVAL)
    
    @property
    def transition_count(self) -> int:
        return len(self.transitions)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Lifecycle Phase Result Models
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PerceptionResult(BaseModel):
    """Output of the PERCEIVE phase — what the agent observed."""
    observed_inputs: List[str] = Field(default_factory=list, description="Input artifact IDs or data references observed")
    context_summary: str = Field(default="", description="Brief summary of the environmental context")
    detected_constraints: List[str] = Field(default_factory=list, description="Constraints or limitations detected")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    raw_perception: Dict[str, Any] = Field(default_factory=dict, description="Agent-specific perception data")


class PlanResult(BaseModel):
    """Output of the PLAN phase — the agent's execution strategy."""
    goal: str = Field(..., description="The high-level goal being pursued")
    sub_steps: List[str] = Field(default_factory=list, description="Ordered list of planned sub-steps")
    estimated_duration_seconds: int = Field(default=60)
    risk_assessment: str = Field(default="low", description="Agent's self-assessment of risk")
    requires_tools: List[str] = Field(default_factory=list, description="Tools/APIs the agent will need")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ActionResult(BaseModel):
    """Output of the ACT phase — the primary execution result."""
    success: bool = Field(default=False)
    output_data: Any = Field(default=None, description="Primary output of the action")
    steps_completed: int = Field(default=0)
    steps_total: int = Field(default=0)
    duration_seconds: float = Field(default=0.0)
    errors: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list, description="Record of tools invoked")


class VerificationResult(BaseModel):
    """Output of the VERIFY phase — quality validation."""
    passed: bool = Field(default=False)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    issues_found: List[str] = Field(default_factory=list)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    reviewer_notes: str = Field(default="")
    needs_human_review: bool = Field(default=False)


class SummaryResult(BaseModel):
    """Output of the SUMMARIZE phase — human-readable report."""
    title: str = Field(default="")
    summary_text: str = Field(default="")
    key_findings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_gathered: List["Evidence"] = Field(default_factory=list, description="Evidence collected during execution")


# Models moved to models.py


class AgentEvent(BaseModel):
    """A lifecycle event emitted to the MessageBus."""
    event_type: EventType
    agent_name: str
    phase: LifecyclePhase
    mission_id: Optional[str] = None
    step_id: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)
    data: Dict[str, Any] = Field(default_factory=dict)
    message: str = Field(default="")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# B2: Agent Capability Declaration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class InputSpec(BaseModel):
    """Declares one type of input the agent can consume."""
    name: str = Field(..., description="Input name, e.g. 'image_path', 'dataset_csv', 'protein_sequence'")
    data_type: str = Field(default="any", description="Expected type: file_path, text, json, dataframe, image, sequence, etc.")
    required: bool = Field(default=True)
    description: str = Field(default="")
    example: Optional[str] = Field(None, description="Example value for documentation")


class OutputSpec(BaseModel):
    """Declares one type of output the agent produces."""
    name: str = Field(..., description="Output name, e.g. 'segmentation_mask', 'report_draft', 'model_metrics'")
    data_type: str = Field(default="any", description="Produced type: file_path, json, text, image, dataframe, etc.")
    artifact_type: Optional[ArtifactType] = Field(None, description="If this output becomes an artifact, which type")
    description: str = Field(default="")


class ApprovalCondition(BaseModel):
    """
    Declares when this agent requires human approval.
    
    The MissionBrain reads these conditions to automatically
    insert approval gates into the mission plan.
    """
    condition: str = Field(..., description="Human-readable description of when approval is needed")
    trigger: str = Field(..., description="Programmatic trigger: 'confidence_below', 'destructive_action', 'external_publish', 'cost_above', 'always'")
    threshold: Optional[float] = Field(None, description="Threshold value for numeric triggers, e.g. confidence < 0.7")
    severity: str = Field(default="medium", description="How critical: low, medium, high, critical")


class AgentCapabilityCard(BaseModel):
    """
    B2: Agent Capability Declaration — The Agent's Self-Portrait.
    
    Every agent declares:
      - What it does (domain, task types)
      - What inputs it needs
      - What outputs it produces  
      - Which artifact types it creates
      - Which confidence levels it reports
      - Under what conditions it needs human approval
      - Its model requirements and resource profile
    
    This card is used by:
      - MissionBrain: to select the right agent for a task
      - Replanner: to find viable substitutes
      - UI: to display agent profiles in the cockpit
      - Audit: to validate that agents stayed within scope
    """
    # Identity
    agent_name: str = Field(..., description="Unique agent name")
    agent_role: str = Field(..., description="AgentRole enum value")
    version: str = Field(default="1.0.0")
    description: str = Field(default="", description="One-paragraph description of what this agent does")
    domain: str = Field(default="general", description="Primary domain: microscopy, coding, writing, structural_biology, etc.")
    
    # Capabilities
    task_types: List[str] = Field(default_factory=list, description="TaskType values this agent can handle: discover, analyze, verify, etc.")
    capabilities: List[str] = Field(default_factory=list, description="Fine-grained capability tags: image_segmentation, cell_counting, etc.")
    
    # I/O Contract
    inputs: List[InputSpec] = Field(default_factory=list, description="What inputs this agent consumes")
    outputs: List[OutputSpec] = Field(default_factory=list, description="What outputs this agent produces")
    artifact_types_produced: List[ArtifactType] = Field(default_factory=list, description="Artifact types this agent can create")
    
    # Confidence & Quality
    confidence_range: str = Field(default="0.0-1.0", description="Typical confidence range for this agent")
    min_acceptable_confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Below this, the agent's output should be flagged")
    
    # Approval Conditions
    approval_conditions: List[ApprovalCondition] = Field(default_factory=list, description="When does this agent need human sign-off")
    
    # Resource Profile
    model_tier: int = Field(default=2, ge=1, le=3, description="LLM tier required: 1=fast, 2=balanced, 3=powerful")
    estimated_duration_seconds: int = Field(default=60, description="Typical execution time")
    requires_gpu: bool = Field(default=False)
    requires_network: bool = Field(default=False)
    
    # Substitution
    can_substitute_for: List[str] = Field(default_factory=list, description="AgentRole values this agent can stand in for")
    can_be_substituted_by: List[str] = Field(default_factory=list, description="AgentRole values that can replace this agent")
    
    # G3: Retry Policy
    retry_policy: AgentRetryPolicy = Field(
        default_factory=lambda: AgentRetryPolicy(strategy=AgentRetryStrategy.LIMITED_RETRY)
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# B4: Agent Result Contract
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class UnifiedAgentResult(BaseModel):
    """
    B4: Unified Agent Result Contract
    
    No matter what an agent does internally across its 8 phases,
    it MUST yield this exact structure to the next agent in the pipeline.
    This guarantees that outputs can be safely chained.
    """
    agent_name: str
    success: bool
    
    summary: str = Field(..., description="Human-readable summary of what was accomplished")
    artifacts: List[ArtifactRecord] = Field(default_factory=list, description="New files, figures, datasets created")
    evidence: List["Evidence"] = Field(default_factory=list, description="Proof points supporting the summary")
    
    confidence: float = Field(..., description="Overall confidence (0.0 to 1.0) in the result")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal issues, caveats, or low-confidence flags")
    
    next_recommendation: Optional[str] = Field(None, description="Agent's suggestion for what should happen next")
    memory_candidates: List[MemoryRecord] = Field(default_factory=list, description="Important learnings to persist globally")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Aggregate Lifecycle Output
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AgentLifecycleOutput(BaseModel):
    """
    The complete output of one agent execution cycle.
    
    This is what the MissionBrain receives back after an agent runs.
    It aggregates all 8 lifecycle phases into a single structured response.
    """
    agent_name: str
    mission_id: Optional[str] = None
    step_id: Optional[str] = None
    
    # Phase results
    perception: PerceptionResult = Field(default_factory=PerceptionResult)
    plan: PlanResult = Field(default=None)
    action: ActionResult = Field(default_factory=ActionResult)
    verification: VerificationResult = Field(default_factory=VerificationResult)
    summary: SummaryResult = Field(default_factory=SummaryResult)
    
    # Emissions
    artifacts: List[ArtifactRecord] = Field(default_factory=list)
    memories: List[MemoryRecord] = Field(default_factory=list)
    events: List[AgentEvent] = Field(default_factory=list)
    
    # Meta
    total_duration_seconds: float = Field(default=0.0)
    overall_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    overall_success: bool = Field(default=False)
    current_phase: LifecyclePhase = Field(default=LifecyclePhase.PERCEIVE)
    error: Optional[str] = Field(None)
    
    # B3: State tracking
    state_record: AgentStateRecord = Field(default=None, description="Full state history for UI and audit")
    
    def to_unified_result(self) -> 'UnifiedAgentResult':
        """
        B4: Compiles all 8 lifecycle phases into the strict Agent Result Contract.
        This provides a standardized, chained-ready object for downstream modules.
        """
        warnings = []
        if self.verification and not self.verification.passed:
            warnings.extend(self.verification.issues_found)
        if self.overall_confidence < 0.7:
            warnings.append(f"Low overall confidence: {self.overall_confidence:.2f}")
        if self.error:
            warnings.append(f"Lifecycle error encountered: {self.error}")
        
        # Determine recommendation
        rec = None
        if self.summary and self.summary.recommendations:
            rec = " ".join(self.summary.recommendations)
        elif not self.overall_success:
            rec = "Replan or escalate to human."
            
        evidence = getattr(self.summary, 'evidence_gathered', []) if self.summary else []
            
        return UnifiedAgentResult(
            agent_name=self.agent_name,
            success=self.overall_success,
            summary=self.summary.summary_text if self.summary else (
                self.error or "Execution failed before summary could be generated."
            ),
            artifacts=self.artifacts or [],
            evidence=evidence,
            confidence=self.overall_confidence,
            warnings=warnings,
            next_recommendation=rec,
            memory_candidates=self.memories or [],
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# The Unified Agent Contract (Abstract Base Class)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class UnifiedAgentContract(ABC):
    """
    THE universal contract for every agent in Bio-ML Agent.
    
    Any module that wants to participate in MissionBrain orchestration
    MUST subclass this ABC and implement all 8 lifecycle methods.
    
    The lifecycle is executed sequentially by the orchestrator:
    
        ┌──────────┐   ┌──────┐   ┌─────┐   ┌────────┐   ┌───────────┐
        │ PERCEIVE │──▶│ PLAN │──▶│ ACT │──▶│ VERIFY │──▶│ SUMMARIZE │
        └──────────┘   └──────┘   └─────┘   └────────┘   └─────┬─────┘
                                                                │
                    ┌───────────────────────────────────────────┘
                    ▼
        ┌────────────────┐   ┌─────────────┐   ┌─────────────┐
        │ EMIT_ARTIFACTS │──▶│ EMIT_MEMORY │──▶│ EMIT_EVENTS │
        └────────────────┘   └─────────────┘   └─────────────┘
    
    Each phase has a typed input and output, enabling:
      - Uniform orchestration by MissionBrain
      - Live monitoring by the UI
      - Decision logging for the Audit Journal
      - Agent substitution by the Replanner
    """
    
    # ─── Identity ─────────────────────────────────────────────────────────
    
    @property
    @abstractmethod
    def agent_name(self) -> str:
        """Unique human-readable name for this agent."""
        ...
    
    @property
    @abstractmethod
    def agent_role(self) -> str:
        """The AgentRole enum value this agent fulfills."""
        ...
    
    @property
    def agent_version(self) -> str:
        """Semantic version of this agent implementation."""
        return "1.0.0"
    
    @property
    def capabilities(self) -> List[str]:
        """List of capability tags this agent provides."""
        return []
    
    # ─── B2: Capability Declaration ───────────────────────────────────────
    
    @property
    @abstractmethod
    def capability_card(self) -> AgentCapabilityCard:
        """
        B2: The agent's self-declaration.
        
        Every agent MUST return a complete AgentCapabilityCard that
        describes what it does, what it needs, what it produces,
        and when it needs human approval.
        
        The MissionBrain reads this card to:
          - Select the right agent for each mission step
          - Find substitutes when an agent fails
          - Insert approval gates where needed
          - Display agent profiles in the UI
        """
        ...
    
    # ─── Lifecycle Phase 1: PERCEIVE ──────────────────────────────────────
    
    @abstractmethod
    def perceive(self, context: Dict[str, Any]) -> PerceptionResult:
        """
        Phase 1: Observe the environment and inputs.
        
        The agent examines the context provided by the orchestrator:
        input data, previous step results, project state, etc.
        
        Args:
            context: Dict containing input_data, previous_results,
                     project_metadata, and mission_context.
        
        Returns:
            PerceptionResult with observed inputs and environmental summary.
        """
        ...
    
    # ─── Lifecycle Phase 2: PLAN ──────────────────────────────────────────
    
    @abstractmethod
    def plan(self, goal: str, perception: PerceptionResult) -> PlanResult:
        """
        Phase 2: Decompose the goal into an execution strategy.
        
        Based on what was perceived, the agent decides HOW to accomplish
        the goal — which sub-steps to take and in what order.
        
        Args:
            goal: The high-level objective from the MissionStep.
            perception: The output of Phase 1.
        
        Returns:
            PlanResult with ordered sub-steps and risk assessment.
        """
        ...
    
    # ─── Lifecycle Phase 3: ACT ───────────────────────────────────────────
    
    @abstractmethod
    def act(self, plan: PlanResult) -> ActionResult:
        """
        Phase 3: Execute the plan.
        
        This is where the actual work happens — calling LLMs, running
        models, processing data, navigating browsers, etc.
        
        Args:
            plan: The output of Phase 2.
        
        Returns:
            ActionResult with success/failure, output data, and tool call logs.
        """
        ...
    
    # ─── Lifecycle Phase 4: VERIFY ────────────────────────────────────────
    
    @abstractmethod
    def verify(self, action: ActionResult) -> VerificationResult:
        """
        Phase 4: Validate the quality of the action's output.
        
        Self-check: did the output meet expectations? Are there anomalies?
        Should a human review this?
        
        Args:
            action: The output of Phase 3.
        
        Returns:
            VerificationResult with pass/fail, quality score, and issues.
        """
        ...
    
    # ─── Lifecycle Phase 5: SUMMARIZE ─────────────────────────────────────
    
    @abstractmethod
    def summarize(self, action: ActionResult, verification: VerificationResult) -> SummaryResult:
        """
        Phase 5: Produce a human-readable summary of the work.
        
        This is what the user and downstream agents will read.
        
        Args:
            action: The output of Phase 3.
            verification: The output of Phase 4.
        
        Returns:
            SummaryResult with title, narrative, findings, and recommendations.
        """
        ...
    
    # ─── Lifecycle Phase 6: EMIT_ARTIFACTS ────────────────────────────────
    
    @abstractmethod
    def emit_artifacts(self, action: ActionResult) -> List[ArtifactRecord]:
        """
        Phase 6: Declare all artifacts produced during execution.
        
        Every file, dataset, model, or visualization created by the agent
        must be registered here for the Artifact Hub.
        
        Args:
            action: The output of Phase 3.
        
        Returns:
            List of ArtifactRecords with type, path, and metadata.
        """
        ...
    
    # ─── Lifecycle Phase 7: EMIT_MEMORY ───────────────────────────────────
    
    @abstractmethod
    def emit_memory(self, action: ActionResult, summary: SummaryResult) -> List[MemoryRecord]:
        """
        Phase 7: Write findings to long-term project memory.
        
        Key decisions, findings, and learnings are persisted so that
        future agents and missions can recall them.
        
        Args:
            action: The output of Phase 3.
            summary: The output of Phase 5.
        
        Returns:
            List of MemoryRecords to persist in the project's memory store.
        """
        ...
    
    # ─── Lifecycle Phase 8: EMIT_EVENTS ───────────────────────────────────
    
    @abstractmethod
    def emit_events(self, output: 'AgentLifecycleOutput') -> List[AgentEvent]:
        """
        Phase 8: Broadcast lifecycle events to the system MessageBus.
        
        This enables real-time monitoring, audit logging, and
        reactive triggers (e.g., auto-start downstream agents).
        
        Args:
            output: The aggregate lifecycle output so far.
        
        Returns:
            List of AgentEvents to publish on the MessageBus.
        """
        ...
    
    # ─── Orchestrator Entry Point ─────────────────────────────────────────
    
    def execute(self, context: Dict[str, Any], goal: str,
                mission_id: Optional[str] = None,
                step_id: Optional[str] = None) -> AgentLifecycleOutput:
        """
        Run the full 8-phase lifecycle.
        
        This method is called by the MissionBrain orchestrator.
        Subclasses should NOT override this — override the individual
        phase methods instead.
        """
        start_time = time.time()
        state = AgentStateRecord(agent_name=self.agent_name)
        output = AgentLifecycleOutput(
            agent_name=self.agent_name,
            mission_id=mission_id,
            step_id=step_id,
            state_record=state,
        )
        
        # Transition: QUEUED → RUNNING
        state.transition_to(AgentState.RUNNING, reason="Lifecycle started", phase=LifecyclePhase.PERCEIVE)
        
        try:
            # Phase 1: PERCEIVE
            output.current_phase = LifecyclePhase.PERCEIVE
            logger.info(f"[{self.agent_name}] Phase 1: PERCEIVE")
            output.perception = self.perceive(context)
            
            # Phase 2: PLAN
            output.current_phase = LifecyclePhase.PLAN
            logger.info(f"[{self.agent_name}] Phase 2: PLAN")
            output.plan = self.plan(goal, output.perception)
            
            # Phase 3: ACT
            output.current_phase = LifecyclePhase.ACT
            logger.info(f"[{self.agent_name}] Phase 3: ACT")
            output.action = self.act(output.plan)
            
            # Phase 4: VERIFY
            output.current_phase = LifecyclePhase.VERIFY
            logger.info(f"[{self.agent_name}] Phase 4: VERIFY")
            output.verification = self.verify(output.action)
            
            # Check if human review needed → state transition
            if output.verification.needs_human_review:
                state.transition_to(
                    AgentState.AWAITING_APPROVAL,
                    reason="Verification flagged for human review",
                    phase=LifecyclePhase.VERIFY,
                )
                # In real execution, orchestrator would pause here.
                # For now, continue and mark the transition back.
                state.transition_to(
                    AgentState.RUNNING,
                    reason="Continuing after review flag (demo mode)",
                    phase=LifecyclePhase.SUMMARIZE,
                )
            
            # Phase 5: SUMMARIZE
            output.current_phase = LifecyclePhase.SUMMARIZE
            logger.info(f"[{self.agent_name}] Phase 5: SUMMARIZE")
            output.summary = self.summarize(output.action, output.verification)
            
            # Phase 6: EMIT_ARTIFACTS
            output.current_phase = LifecyclePhase.EMIT_ARTIFACTS
            logger.info(f"[{self.agent_name}] Phase 6: EMIT_ARTIFACTS")
            output.artifacts = self.emit_artifacts(output.action)
            
            # Phase 7: EMIT_MEMORY
            output.current_phase = LifecyclePhase.EMIT_MEMORY
            logger.info(f"[{self.agent_name}] Phase 7: EMIT_MEMORY")
            output.memories = self.emit_memory(output.action, output.summary)
            
            # Phase 8: EMIT_EVENTS
            output.current_phase = LifecyclePhase.EMIT_EVENTS
            logger.info(f"[{self.agent_name}] Phase 8: EMIT_EVENTS")
            output.events = self.emit_events(output)
            
            # Aggregate
            output.overall_success = output.action.success and output.verification.passed
            output.overall_confidence = min(
                output.action.confidence,
                output.verification.confidence,
            )
            
            # Transition: RUNNING → COMPLETED
            state.transition_to(
                AgentState.COMPLETED,
                reason=f"All 8 phases done. confidence={output.overall_confidence:.2f}",
                phase=LifecyclePhase.EMIT_EVENTS,
            )
            
        except Exception as e:
            output.error = f"[{output.current_phase.value}] {type(e).__name__}: {str(e)}"
            output.overall_success = False
            output.overall_confidence = 0.0
            # Transition: RUNNING → FAILED
            state.transition_to(
                AgentState.FAILED,
                reason=f"Exception at {output.current_phase.value}: {str(e)[:100]}",
                phase=output.current_phase,
            )
            logger.error(f"[{self.agent_name}] Lifecycle failed at {output.current_phase.value}: {e}")
        
        output.total_duration_seconds = time.time() - start_time
        
        logger.info(
            f"[{self.agent_name}] Lifecycle complete: "
            f"success={output.overall_success}, "
            f"confidence={output.overall_confidence:.2f}, "
            f"duration={output.total_duration_seconds:.1f}s, "
            f"artifacts={len(output.artifacts)}, "
            f"memories={len(output.memories)}"
        )
        
        return output
