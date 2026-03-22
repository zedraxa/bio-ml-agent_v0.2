# bio_ml_agent/brain/mission_brain.py
"""
Part VI - Axis A: Mission Brain — The Operating System Kernel

The Mission Brain is the central intelligence of Bio-ML Agent.
It is NOT a simple planner — it is the system's operating system.

Responsibilities:
  1. Convert user prompts into structured MissionPlans
  2. Build agent execution DAGs (AgentGraph)
  3. Define measurable success criteria  
  4. Generate fallback/recovery strategies
  5. Identify approval gates and risky steps
  6. Replan when execution encounters failures

Outputs:
  - mission_plan.json
  - agent_graph.json
  - success_criteria.json
  - fallback_strategy.json
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from .models import (
    MissionPlan,
    MissionStep,
    StepStatus,
    RiskLevel,
    TaskType,
    AgentRole,
    AgentNode,
    AgentGraph,
    SuccessCriterion,
    SuccessCriteria,
    FallbackAction,
    FallbackStrategy,
    ReplanResult,
    MissionSnapshot,
    ProjectState,
    AgentRetryStrategy,
    AgentRetryPolicy,
    MissionPriority,
    ResourceIntensity,
    MissionTelemetry,
    QualityScorecard,
    Comment,
)

from .persistence import MISSION_STORE, PROJECT_STORE
from .architecture_guard import ArchitectureGuard
from .feature_flags import FEATURE_CONTROLLER
from .intent_translator import IntentTranslator
from .graph_engine import MissionGraphEngine
from .replanner import Replanner
from .lineage_service import ArtifactLineageService
from .agent_lifecycle import AgentLifecycleManager
from .agent_contract import AgentState
from .recovery_manager.recovery_manager import RecoveryManager
from .interpreter import CommentInterpreter
from .refinement_planner import RefinementPlanner

logger = logging.getLogger("bio_ml_agent.brain")


# ─── A3: Mission Decomposer — Agent → TaskType Mapping ────────────────────────
# This map assigns each agent its primary universal task category.
# The Mission Decomposer uses this to tag every step, creating a uniform
# interface regardless of which domain-specific agent runs the step.

AGENT_TASK_TYPE_MAP: Dict[AgentRole, TaskType] = {
    AgentRole.RESEARCHER:       TaskType.DISCOVER,
    AgentRole.BROWSER_AGENT:    TaskType.DISCOVER,
    AgentRole.DATA_ENGINEER:    TaskType.ANALYZE,
    AgentRole.ML_EXPERT:        TaskType.ANALYZE,
    AgentRole.BIOINFORMATICIAN: TaskType.ANALYZE,
    AgentRole.IN_SILICO_EXPERT: TaskType.ANALYZE,
    AgentRole.MICROSCOPY_AGENT: TaskType.ANALYZE,
    AgentRole.STRUCTURE_AGENT:  TaskType.ANALYZE,
    AgentRole.CODING_AGENT:     TaskType.SYNTHESIZE,
    AgentRole.WRITING_AGENT:    TaskType.WRITE,
    AgentRole.ACADEMIC_EXPERT:  TaskType.WRITE,
    AgentRole.CRITIC:           TaskType.CRITIQUE,
    AgentRole.PLANNER:          TaskType.SYNTHESIZE,
}


# ─── G4: Resource Intensity Map ───────────────────────────────────────────────

TASK_RESOURCE_MAP: Dict[TaskType, ResourceIntensity] = {
    TaskType.DISCOVER:   ResourceIntensity.MEDIUM,
    TaskType.ANALYZE:    ResourceIntensity.HEAVY,   # Analysis (ML, In Silico) is heavy
    TaskType.VERIFY:     ResourceIntensity.LIGHT,
    TaskType.SYNTHESIZE: ResourceIntensity.MEDIUM,
    TaskType.WRITE:      ResourceIntensity.LIGHT,
    TaskType.CRITIQUE:   ResourceIntensity.LIGHT,
    TaskType.EXPORT:     ResourceIntensity.LIGHT,
}

# Resource Score Weights
RESOURCE_WEIGHTS = {
    ResourceIntensity.LIGHT: 1,
    ResourceIntensity.MEDIUM: 5,
    ResourceIntensity.HEAVY: 10,
    ResourceIntensity.CRITICAL: 25,
}


# ─── Agent Capability Registry ────────────────────────────────────────────────

AGENT_CAPABILITIES: Dict[AgentRole, Dict[str, Any]] = {
    AgentRole.RESEARCHER: {
        "can_do": ["literature_search", "clinical_review", "web_research", "summarization"],
        "produces": ["research_summary", "citation_list", "clinical_findings"],
        "model_tier": 2,
    },
    AgentRole.DATA_ENGINEER: {
        "can_do": ["data_download", "data_cleaning", "csv_processing", "feature_engineering"],
        "produces": ["clean_dataset", "feature_matrix", "data_report"],
        "model_tier": 1,
    },
    AgentRole.ML_EXPERT: {
        "can_do": ["model_training", "hyperparameter_tuning", "xai_analysis", "evaluation"],
        "produces": ["trained_model", "evaluation_report", "shap_analysis", "roc_curves"],
        "model_tier": 3,
    },
    AgentRole.BIOINFORMATICIAN: {
        "can_do": ["sequence_analysis", "protein_analysis", "omics_processing", "clinical_interpretation"],
        "produces": ["bioinformatics_report", "sequence_alignment", "variant_analysis"],
        "model_tier": 2,
    },
    AgentRole.IN_SILICO_EXPERT: {
        "can_do": ["alphafold_prediction", "molecular_docking", "virtual_screening", "pocket_analysis"],
        "produces": ["structure_prediction", "docking_results", "screening_report", "binding_scores"],
        "model_tier": 3,
    },
    AgentRole.ACADEMIC_EXPERT: {
        "can_do": ["report_writing", "paper_drafting", "lab_report", "presentation_generation"],
        "produces": ["lab_report", "paper_draft", "presentation_notes", "poster"],
        "model_tier": 2,
    },
    AgentRole.BROWSER_AGENT: {
        "can_do": ["web_navigation", "data_extraction", "screenshot_capture", "form_filling"],
        "produces": ["extracted_data", "screenshots", "web_content"],
        "model_tier": 1,
    },
    AgentRole.MICROSCOPY_AGENT: {
        "can_do": ["image_segmentation", "cell_counting", "morphology_analysis", "stain_detection"],
        "produces": ["segmentation_mask", "cell_count_report", "morphology_analysis", "annotated_image"],
        "model_tier": 3,
    },
    AgentRole.CODING_AGENT: {
        "can_do": ["code_review", "code_generation", "bug_fixing", "refactoring", "testing"],
        "produces": ["code_patch", "test_suite", "review_report", "refactored_code"],
        "model_tier": 3,
    },
    AgentRole.WRITING_AGENT: {
        "can_do": ["text_generation", "editing", "translation", "summarization"],
        "produces": ["document", "summary", "translated_text", "edited_draft"],
        "model_tier": 2,
    },
    AgentRole.STRUCTURE_AGENT: {
        "can_do": ["pdb_analysis", "binding_site_prediction", "structural_comparison"],
        "produces": ["structure_report", "binding_sites", "structural_alignment"],
        "model_tier": 2,
    },
    AgentRole.CRITIC: {
        "can_do": ["quality_review", "fact_checking", "confidence_scoring", "risk_assessment"],
        "produces": ["quality_report", "confidence_scores", "risk_flags"],
        "model_tier": 2,
    },
}

# ─── G3: Specialized Retry Policies ───────────────────────────────────────────

DEFAULT_RETRY_POLICIES: Dict[AgentRole, AgentRetryPolicy] = {
    AgentRole.BROWSER_AGENT: AgentRetryPolicy(
        strategy=AgentRetryStrategy.LIMITED_RETRY, 
        max_retries=2,
        alternative_strategy_description="Fallback to text-based research via RESEARCHER agent"
    ),
    AgentRole.WRITING_AGENT: AgentRetryPolicy(
        strategy=AgentRetryStrategy.REGENERATE,
        max_retries=3,
        alternative_strategy_description="Regenerate failing section with simplified constraints"
    ),
    AgentRole.CODING_AGENT: AgentRetryPolicy(
        strategy=AgentRetryStrategy.SELF_DEBUG,
        max_retries=3,
        debug_depth=2
    ),
    AgentRole.STRUCTURE_AGENT: AgentRetryPolicy(
        strategy=AgentRetryStrategy.QUEUE_RESUME,
        max_retries=5
    ),
    AgentRole.MICROSCOPY_AGENT: AgentRetryPolicy(
        strategy=AgentRetryStrategy.HUMAN_REVIEW,
        max_retries=0
    ),
}

# ─── Keyword → Intent → Agent Mapping ─────────────────────────────────────────

INTENT_PATTERNS: List[Dict[str, Any]] = [
    {
        "intent": "microscopy_analysis",
        "keywords": ["mikroskop", "microscopy", "slide", "h&e", "stain", "segmentation", "cell"],
        "primary_agent": AgentRole.MICROSCOPY_AGENT,
        "supporting": [AgentRole.CRITIC],
    },
    {
        "intent": "protein_structure",
        "keywords": ["alphafold", "protein", "docking", "pdb", "yapı", "pocket", "binding"],
        "primary_agent": AgentRole.IN_SILICO_EXPERT,
        "supporting": [AgentRole.RESEARCHER, AgentRole.CRITIC],
    },
    {
        "intent": "virtual_screening",
        "keywords": ["screening", "sanal tarama", "virtual screening", "smiles", "lipinski"],
        "primary_agent": AgentRole.IN_SILICO_EXPERT,
        "supporting": [AgentRole.DATA_ENGINEER, AgentRole.CRITIC],
    },
    {
        "intent": "lab_report",
        "keywords": ["rapor", "lab report", "lab raporu", "sonuç raporu"],
        "primary_agent": AgentRole.ACADEMIC_EXPERT,
        "supporting": [AgentRole.RESEARCHER, AgentRole.CRITIC],
    },
    {
        "intent": "data_analysis",
        "keywords": ["analiz", "veri", "dataset", "csv", "korelasyon", "istatistik"],
        "primary_agent": AgentRole.DATA_ENGINEER,
        "supporting": [AgentRole.ML_EXPERT],
    },
    {
        "intent": "ml_pipeline",
        "keywords": ["eğit", "model", "train", "predict", "sınıflandır", "regresyon", "accuracy"],
        "primary_agent": AgentRole.ML_EXPERT,
        "supporting": [AgentRole.DATA_ENGINEER, AgentRole.CRITIC],
    },
    {
        "intent": "code_review",
        "keywords": ["kod", "code", "repo", "review", "refactor", "bug", "test"],
        "primary_agent": AgentRole.CODING_AGENT,
        "supporting": [AgentRole.CRITIC],
    },
    {
        "intent": "web_research",
        "keywords": ["ara", "search", "web", "browser", "literature", "pubmed"],
        "primary_agent": AgentRole.BROWSER_AGENT,
        "supporting": [AgentRole.RESEARCHER],
    },
    {
        "intent": "full_pipeline",
        "keywords": ["uçtan uca", "pipeline", "hepsini", "tüm analiz", "kanser"],
        "primary_agent": AgentRole.RESEARCHER,
        "supporting": [AgentRole.DATA_ENGINEER, AgentRole.ML_EXPERT, AgentRole.BIOINFORMATICIAN, AgentRole.ACADEMIC_EXPERT, AgentRole.CRITIC],
    },
]


# ─── G4: Mission Scheduler (Singleton for demo) ───────────────────────────────

class MissionScheduler:
    """G4: Resource-Aware Scheduler for multi-mission management."""
    MAX_RESOURCE_CAPACITY = 50  # Aggregate resource score limit
    
    def __init__(self):
        self.missions_registry: Dict[str, MissionPlan] = {}

    def get_current_load(self) -> int:
        return sum(m.total_resource_score for m in self.missions_registry.values() if not m.is_paused)

    def can_start(self, new_mission: MissionPlan) -> bool:
        """Check if mission can start without exceeding capacity."""
        load = self.get_current_load()
        if load + new_mission.total_resource_score > self.MAX_RESOURCE_CAPACITY:
            # High/Urgent priority can still start but might pause others (logic simplified)
            if new_mission.priority in (MissionPriority.HIGH, MissionPriority.URGENT):
                return True
            return False
        return True

    def register(self, mission: MissionPlan):
        self.missions_registry[mission.mission_id] = mission

    def complete(self, mission_id: str):
        if mission_id in self.missions_registry:
            del self.missions_registry[mission_id]

# Global Scheduler Instance
SCHEDULER = MissionScheduler()


class MissionBrain:
    """
    The Central Brain / Mission OS
    
    Converts a single user prompt into a fully structured execution blueprint
    including agent DAG, success criteria, and fallback strategies.
    
    This is NOT a simple planner. This is the system's operating system.
    """
    
    def __init__(self, project_id: str = "default"):
        self.project_id = project_id
        self._mission_history: List[MissionPlan] = []
        self.guard = ArchitectureGuard()
        self.translator = IntentTranslator()
        self.graph_engine = MissionGraphEngine()
        self.replanner = Replanner()
        self.lineage = ArtifactLineageService()
        self.lifecycle = AgentLifecycleManager()
        self.recovery = RecoveryManager()
        self.interpreter = CommentInterpreter()
        self.refiner = RefinementPlanner()
    
    # ─── Public API ───────────────────────────────────────────────────────
    
    def decompose(self, user_prompt: str) -> MissionPlan:
        """
        Main entry point. Takes a user prompt and produces a complete MissionPlan.
        
        Returns a MissionPlan containing:
          - Ordered steps with dependencies
          - Agent execution graph  
          - Success criteria
          - Fallback strategy
        """
        mission_id = f"mission_{uuid.uuid4().hex[:12]}"
        
        logger.info(f"[MissionBrain] Decomposing prompt: '{user_prompt[:80]}...'")
        
        # 2. Decompose into Intents & Steps (Part VI: Modularized)
        intents = self.translator.translate(user_prompt)
        steps = self._generate_steps(mission_id, intents, user_prompt)
        
        # 3. Create Agent Graph (Part VI: Modularized)
        graph = self.graph_engine.build_graph(mission_id, steps)
        priority = self._detect_priority(user_prompt)
        total_resource_score = self._calculate_resource_score(steps)
        
        # Step 4: Define success criteria
        success_criteria = self._define_success_criteria(mission_id, steps, intents)
        
        # Step 5: Generate fallback strategy
        fallback_strategy = self._generate_fallback_strategy(mission_id, steps)
        
        # Step 6: Initialize Telemetry (Axis H1)
        telemetry = MissionTelemetry(mission_id=mission_id)
        telemetry.agents_involved = list(set(step.assigned_agent for step in steps))
        
        # Step 7: Identify approval gates
        approval_gates = [s.step_id for s in steps if s.requires_approval]
        
        # Step 7: Risk summary
        high_risk_steps = [s for s in steps if s.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)]
        risk_summary = (
            f"{len(high_risk_steps)} high-risk step(s) identified. "
            f"{len(approval_gates)} approval gate(s) set. "
            f"Total estimated time: {sum(s.estimated_duration_seconds for s in steps)}s."
        )
        
        # Assemble the MissionPlan
        plan = MissionPlan(
            mission_id=mission_id,
            project_id=self.project_id,
            user_prompt=user_prompt,
            title=self._generate_title(intents),
            objective=self._generate_objective(intents, user_prompt),
            steps=steps,
            agent_graph=graph, # Changed from agent_graph to graph
            success_criteria=success_criteria,
            fallback_strategy=fallback_strategy,
            approval_gates=approval_gates,
            risk_summary=risk_summary,
            priority=priority,
            total_resource_score=total_resource_score,
            telemetry=telemetry,
        )
        
        # G4: Register with Scheduler
        SCHEDULER.register(plan)
        if not SCHEDULER.can_start(plan):
            logger.warning(f"[G4:Scheduler] Resource limit reached ({SCHEDULER.get_current_load()}). Mission '{plan.title}' is QUEUED/PAUSED.")
            plan.is_paused = True
        
        self._mission_history.append(plan)
        logger.info(f"[MissionBrain] Mission plan '{plan.title}' created with {len(steps)} steps.")
        
        return plan
    
    def replan(self, mission_id: str, failed_step_id: str, error_context: str, confidence: Optional[float] = None) -> ReplanResult:
        """Part VI: Delegates replanning to the specialized Replanner module."""
        original = MISSION_STORE.get(mission_id)
        if not original:
            raise ValueError(f"Mission {mission_id} not found for replanning.")
        
        result = self.replanner.replan(original, failed_step_id, error_context, confidence)
        
        # Persist and Notify
        MISSION_STORE.save(original)
        
        # G2: Notify critic of the deviation
        if not result.recovery_succeeded or result.strategy_used != "retry":
            self.notify_critic_of_recovery(mission_id, failed_step_id, result.strategy_detail)
            
        self.checkpoint(mission_id, failed_step_id)
        
        return result
    
    # ─── A4: Replanner Helpers ─────────────────────────────────────────────
    
    def checkpoint(self, mission_id: str, last_step_id: Optional[str] = None) -> str:
        """
        Axis G1: Mission Checkpointing.
        Captures full state (plan + project) for persistent recovery.
        """
        plan = self._find_mission(mission_id)
        if not plan:
            plan = MISSION_STORE.load(mission_id)
            if not plan:
                raise ValueError(f"Mission {mission_id} not found")

        project = PROJECT_STORE.load(plan.project_id) or ProjectState(
            project_id=plan.project_id, name="Project Baseline"
        )

        snapshot = MissionSnapshot(
            snapshot_id=f"snap_{uuid.uuid4().hex[:8]}",
            mission_id=mission_id,
            project_id=plan.project_id,
            plan_snapshot=plan,
            project_snapshot=project,
            last_completed_step_id=last_step_id
        )

        path = self.recovery.create_checkpoint(mission_id, plan, project)
        logger.info(f"[MissionBrain:G1] Checkpoint saved: {path}")
        return path

    def process_feedback(self, mission_id: str, comment: Comment) -> Optional[MissionStep]:
        """
        Axis B: Feedback-to-Action Intelligence (R7-2).
        Interprets a comment and injects a refinement step into the plan.
        """
        plan = MISSION_STORE.get_plan(mission_id)
        if not plan:
            return None
            
        # 1. Interpret feedback
        refinement_task = self.interpreter.interpret_feedback(comment, {"mission_id": mission_id})
        if not refinement_task:
            return None
            
        # D3: Explain-My-Output Check
        from .models import IntentType
        if refinement_task.get("intent") == IntentType.EXPLAIN:
            self._explain_output(plan, comment)
            return None
            
        # 2. Plan refinement step
        new_step = self.refiner.execute_feedback_loop(plan, refinement_task)
        
        # D4: Pre-stage Revision Justification log
        new_step.metadata["revision_for_comment"] = comment.comment_id
        
        # 3. Update plan in store
        MISSION_STORE.save_plan(plan)
        logger.info(f"[MissionBrain:B1] New refinement step {new_step.step_id} injected into mission {mission_id}")
        return new_step
        
    def resolve_pending_approval(self, project_id: str, is_approved: bool, feedback: Optional[str] = None):
        """
        E3: Extemely critical workflow. Resolves a pending authorization block from external 
        sources like WhatsApp or GUI.
        """
        # Find active mission for this project
        project = PROJECT_STORE.load(project_id)
        if hasattr(project, "active_mission_id") and project.active_mission_id:
            mission_id = project.active_mission_id
        else:
            # Fallback legacy logic
            mission_id = f"wa_{project_id}" if not project_id.startswith("wa_") else project_id
            
        plan = MISSION_STORE.get_plan(mission_id)
        if not plan:
            logger.warning(f"No active plan found for {project_id} to resolve approval.")
            return

        # Find the blocked step
        blocked_step = next((s for s in plan.steps if s.status == StepStatus.WAITING_APPROVAL), None)
        if not blocked_step:
            logger.warning(f"No step awaiting approval in mission {mission_id}.")
            return

        if is_approved:
            logger.info(f"[E3] Step {blocked_step.step_id} APPROVED via external signal.")
            blocked_step.status = StepStatus.COMPLETED
            if feedback:
                blocked_step.metadata["approval_feedback"] = feedback
        else:
            logger.warning(f"[E3] Step {blocked_step.step_id} REJECTED via external signal.")
            blocked_step.status = StepStatus.FAILED
            blocked_step.error_context = feedback or "User rejected the step."
        
        MISSION_STORE.save_plan(plan)
        
        # If running asynchronously, this would trigger the executor to wake up
        # For now, we simulate execution continuation if approved
        if is_approved:
            logger.info("Triggering background execute_mission continuation...")
            # self.execute_mission(mission_id) # Deferred execution logic

    def handle_conversational_query(self, query: str) -> str:
        """
        G1: Ask-the-project.
        Queries the project memory and returns a natural language response.
        """
        logger.info(f"[MissionBrain:G1] Handling conversational query: {query}")
        project = PROJECT_STORE.load(self.project_id)
        if not project:
            return "Şu an aktif bir projeniz bulunmuyor."
            
        # Simulated LLM logic over project state
        completed = 0
        pending = 0
        if project.active_mission_id:
            plan = self._find_mission(project.active_mission_id)
            if plan:
                completed = len([s for s in plan.steps if s.status == StepStatus.COMPLETED])
                pending = len([s for s in plan.steps if s.status == StepStatus.WAITING_APPROVAL])
                
        artifact_count = len(project.artifacts)
        latest_art = project.artifacts[-1].type if artifact_count > 0 else "Yok"
        
        response = f"🤖 *Değerlendirme:* Projenizde şu ana kadar {completed} adım tamamlandı. "
        if pending > 0:
            response += f"Bekleyen {pending} adet onay var. "
        response += f"Toplam {artifact_count} artifact üretildi (Sonuncusu: {latest_art}). Başka bir sorunuz var mı?"
        return response

    def generate_whatsapp_summary(self, artifact_type: str = None) -> str:
        """
        G4: Conversational summaries.
        Generates a 3-bullet executive summary of the latest artifact.
        """
        logger.info(f"[MissionBrain:G4] Generating summary for artifact type: {artifact_type}")
        project = PROJECT_STORE.load(self.project_id)
        if not project or not project.artifacts:
            return "Özetlenecek bir artifact bulunamadı."
            
        latest_artifact = project.artifacts[-1]
        
        # Simulated LLM Summarization
        summary = (
            f"🔹 *Yönetici Özeti ({latest_artifact.type}):*\n"
            f"1. Temel veri başarıyla analiz edildi.\n"
            f"2. Kritik hedeflerde %92 başarı sağlandı.\n"
            f"3. Sonuçlar standart protokollere uygun bulundu.\n\n"
            f"Detay ister misin?"
        )
        return summary

    def _explain_output(self, plan: MissionPlan, comment: Comment):
        """
        D3: Explain-my-output mode.
        Generates a direct response to a "why" question without changing the plan.
        """
        logger.info(f"[MissionBrain:D3] Processing explanation request for comment {comment.comment_id}")
        
        # Simulated LLM Explanation Generation
        explanation_text = f"Agent Explanation for '{comment.content}': Based on the parameters provided in step X, I selected approach Y because it aligns with protocol Z."
        
        from .models import CommentStatus
        # In a real system, we'd append this explanation as a reply to the comment thread
        comment.status = CommentStatus.RESOLVED
        comment.metadata["explanation_provided"] = explanation_text
        
        # Notify the UI or Agent OS of the explanation
        logger.info(f"Explanation ready: {explanation_text}")
        
    def log_revision_justification(self, mission_id: str, step_id: str, justification_data: Dict[str, Any]):
        """
        D4: Revision Justification.
        Called by agents after executing a refinement step.
        """
        from .models import RevisionJustification
        try:
            justification = RevisionJustification(**justification_data)
            logger.info(f"[MissionBrain:D4] Received Revision Justification for comment {justification.addressed_comment_id}: {justification.rationale}")
            # Persist justification (e.g., attach to ProjectState or Artifact Lineage)
        except Exception as e:
            logger.error(f"Failed to parse RevisionJustification: {e}")

    def export_plan(self, plan: MissionPlan) -> Dict[str, Any]:
        """Export the mission plan as the 4 structured JSON documents."""
        return {
            "mission_plan": json.loads(plan.model_dump_json()),
            "agent_graph": json.loads(plan.agent_graph.model_dump_json()) if plan.agent_graph else {},
            "success_criteria": json.loads(plan.success_criteria.model_dump_json()) if plan.success_criteria else {},
            "fallback_strategy": json.loads(plan.fallback_strategy.model_dump_json()) if plan.fallback_strategy else {},
        }
    
    def finalize_mission(self, mission_id: str, status: str = "completed"):
        """H1: Finalize telemetry and status for a completed mission."""
        plan = self._find_mission(mission_id)
        if not plan or not plan.telemetry:
            return
            
        t = plan.telemetry
        t.end_time = datetime.now().timestamp()
        t.total_duration_seconds = t.end_time - t.start_time
        t.status = status
        
        # Aggregate final metrics
        t.agents_involved = list(set(step.assigned_agent for step in plan.steps))
        t.artifacts_produced_count = sum(len(step.output_artifacts) for step in plan.steps)
        t.approval_count = len([s for s in plan.steps if s.status == StepStatus.COMPLETED and s.requires_approval])
        
        logger.info(
            f"[MissionBrain:H1] Telemetry Finalized for '{plan.title}': "
            f"duration={t.total_duration_seconds:.1f}s, "
            f"replans={t.replan_count}, "
            f"artifacts={t.artifacts_produced_count}"
        )
        
        # H2: Generate Quality Scorecard
        plan.scorecard = self.generate_scorecard(mission_id)
        
        # H3: Architectural Drift Audit
        plan.drift_report = self.guard.audit_mission_artifacts(plan)
        if not plan.drift_report.is_healthy:
            logger.error(f"[MissionBrain:H3] Drift Detected Level {plan.drift_report.drift_score:.1f} in '{plan.title}'")

    def generate_scorecard(self, mission_id: str) -> QualityScorecard:
        """H2: Generate a comprehensive quality scorecard for a mission."""
        plan = self._find_mission(mission_id)
        if not plan:
            raise ValueError(f"Mission {mission_id} not found")
            
        scorecard = QualityScorecard(mission_id=mission_id)
        
        # 1. Completeness
        total_steps = len(plan.steps)
        completed_steps = len([s for s in plan.steps if s.status == StepStatus.COMPLETED])
        scorecard.completeness = completed_steps / total_steps if total_steps > 0 else 0.0
        
        # 2. Confidence Profile
        confidences = [s.metadata.get("confidence", 0.5) for s in plan.steps if s.status == StepStatus.COMPLETED]
        scorecard.confidence_profile = sum(confidences) / len(confidences) if confidences else 0.0
        
        # 3. Review Burden (Inverse Score)
        # Assuming 0 replans = 1.0 score, each replan reduces it
        replan_penalty = (plan.telemetry.replan_count * 0.1) if plan.telemetry else 0.0
        scorecard.review_burden = max(0.0, 1.0 - replan_penalty)
        
        # 4. Artifact Health
        # Mock calculation based on status (Final/Approved = 1.0, Draft = 0.5)
        # In a real system we'd check ProjectState.artifacts
        scorecard.artifact_health = 0.85 # Default health for successful mission
        
        # 5. Evidence Sufficiency
        # Mock based on task types
        scorecard.evidence_sufficiency = 0.90 # High evidence for research missions
        
        # Overall Score
        scorecard.overall_quality_score = (
            scorecard.completeness * 0.3 +
            scorecard.confidence_profile * 0.3 +
            scorecard.review_burden * 0.2 +
            scorecard.artifact_health * 0.1 +
            scorecard.evidence_sufficiency * 0.1
        )
        
        # Key Strengths & Gaps
        if scorecard.overall_quality_score > 0.8:
            scorecard.key_strengths.append("High overall confidence and execution efficiency")
        if scorecard.review_burden < 0.6:
            scorecard.critical_gaps.append("Significant replanning required during execution")
            
        return scorecard

    # ─── G4: Resource Scheduling Helpers ──────────────────────────────────
    
    def _detect_priority(self, prompt: str) -> MissionPriority:
        """Detect priority level from user prompt indicators."""
        p_lower = prompt.lower()
        if any(kw in p_lower for kw in ["acil", "urgent", "fast", "asap", "kritik"]):
            return MissionPriority.URGENT
        if any(kw in p_lower for kw in ["öncelikli", "high priority", "priority"]):
            return MissionPriority.HIGH
        if any(kw in p_lower for kw in ["arkaplanda", "low priority", "later", "background"]):
            return MissionPriority.LOW
        return MissionPriority.NORMAL

    def _calculate_resource_score(self, steps: List[MissionStep]) -> int:
        """Calculate total resource load for the mission plan."""
        total = 0
        for step in steps:
            intensity = TASK_RESOURCE_MAP.get(step.task_type, ResourceIntensity.LIGHT)
            step.resource_intensity = intensity  # Tag the step
            total += RESOURCE_WEIGHTS.get(intensity, 1)
        return total

    # ─── Private: Intent Detection ────────────────────────────────────────
    
    # ─── Private: Step Generation ─────────────────────────────────────────
    
    def _generate_steps(self, mission_id: str, intents: List[Dict], user_prompt: str) -> List[MissionStep]:
        """Generate ordered mission steps from detected intents."""
        steps: List[MissionStep] = []
        step_counter = 0
        
        for intent_data in intents:
            primary = intent_data["primary_agent"]
            supporting = intent_data.get("supporting", [])
            intent_name = intent_data["intent"]
            
            # Always start with context gathering if we have a researcher
            if AgentRole.RESEARCHER in supporting and not any(s.assigned_agent == AgentRole.RESEARCHER for s in steps):
                step_counter += 1
                steps.append(MissionStep(
                    step_id=f"step_{step_counter:03d}",
                    title=f"Gather Research Context for '{intent_name}'",
                    description=f"Search literature and clinical databases relevant to: {user_prompt[:100]}",
                    task_type=TaskType.DISCOVER,
                    assigned_agent=AgentRole.RESEARCHER,
                    risk_level=RiskLevel.LOW,
                    estimated_duration_seconds=45,
                ))
            
            # Data preparation step if data agent is involved
            if AgentRole.DATA_ENGINEER in supporting or primary == AgentRole.DATA_ENGINEER:
                prev_ids = [s.step_id for s in steps]
                step_counter += 1
                steps.append(MissionStep(
                    step_id=f"step_{step_counter:03d}",
                    title=f"Prepare Data for '{intent_name}'",
                    description=f"Download, clean, and transform datasets needed for {intent_name}",
                    task_type=TaskType.DISCOVER,
                    assigned_agent=AgentRole.DATA_ENGINEER,
                    depends_on=prev_ids[-1:] if prev_ids else [],
                    risk_level=RiskLevel.MEDIUM,
                    estimated_duration_seconds=90,
                ))
            
            # Primary agent execution
            prev_ids = [s.step_id for s in steps]
            step_counter += 1
            
            is_risky = primary in (AgentRole.IN_SILICO_EXPERT, AgentRole.ML_EXPERT, AgentRole.MICROSCOPY_AGENT)
            needs_approval = primary in (AgentRole.ACADEMIC_EXPERT, AgentRole.CODING_AGENT)
            primary_task_type = AGENT_TASK_TYPE_MAP.get(primary, TaskType.ANALYZE)
            
            steps.append(MissionStep(
                step_id=f"step_{step_counter:03d}",
                title=f"Execute {intent_name.replace('_', ' ').title()}",
                description=f"Primary execution by {primary.value}: {user_prompt[:100]}",
                task_type=primary_task_type,
                assigned_agent=primary,
                depends_on=prev_ids[-1:] if prev_ids else [],
                risk_level=RiskLevel.HIGH if is_risky else RiskLevel.MEDIUM,
                requires_approval=needs_approval,
                estimated_duration_seconds=180 if is_risky else 120,
            ))
            
            # ML Expert follow-up if present
            if AgentRole.ML_EXPERT in supporting and primary != AgentRole.ML_EXPERT:
                step_counter += 1
                steps.append(MissionStep(
                    step_id=f"step_{step_counter:03d}",
                    title=f"ML Analysis for '{intent_name}'",
                    description="Train models, evaluate performance, and generate XAI explanations",
                    task_type=TaskType.ANALYZE,
                    assigned_agent=AgentRole.ML_EXPERT,
                    depends_on=[steps[-1].step_id],
                    risk_level=RiskLevel.HIGH,
                    estimated_duration_seconds=180,
                ))
            
            # Writing/Report step if academic agent is supporting
            if AgentRole.WRITING_AGENT in supporting or AgentRole.ACADEMIC_EXPERT in supporting:
                if primary != AgentRole.ACADEMIC_EXPERT:
                    step_counter += 1
                    steps.append(MissionStep(
                        step_id=f"step_{step_counter:03d}",
                        title=f"Draft Report for '{intent_name}'",
                        description="Compile findings into a structured scientific report",
                        task_type=TaskType.WRITE,
                        assigned_agent=AgentRole.ACADEMIC_EXPERT,
                        depends_on=[steps[-1].step_id],
                        requires_approval=True,
                        risk_level=RiskLevel.LOW,
                        estimated_duration_seconds=120,
                    ))
            
            # Critic review at the end of each intent chain
            if AgentRole.CRITIC in supporting:
                step_counter += 1
                steps.append(MissionStep(
                    step_id=f"step_{step_counter:03d}",
                    title=f"Quality Review for '{intent_name}'",
                    description="Validate outputs, check confidence scores, and flag anomalies",
                    task_type=TaskType.CRITIQUE,
                    assigned_agent=AgentRole.CRITIC,
                    depends_on=[steps[-1].step_id],
                    risk_level=RiskLevel.LOW,
                    estimated_duration_seconds=30,
                ))
        
        return steps
    
    # ─── Private: Success Criteria ────────────────────────────────────────
    
    def _define_success_criteria(self, mission_id: str, steps: List[MissionStep], intents: List[Dict]) -> SuccessCriteria:
        """Generate measurable success criteria based on the mission structure."""
        criteria = []
        
        # Universal: all steps must complete
        criteria.append(SuccessCriterion(
            criterion_id="sc_completion",
            description="All non-skipped steps must reach COMPLETED status",
            metric="completion_rate",
            threshold=1.0,
            is_mandatory=True,
        ))
        
        # Per-intent criteria
        for intent in intents:
            intent_name = intent["intent"]
            
            if intent_name in ("ml_pipeline", "data_analysis"):
                criteria.append(SuccessCriterion(
                    criterion_id=f"sc_{intent_name}_quality",
                    description=f"ML model must achieve minimum acceptable performance",
                    metric="model_accuracy",
                    threshold=0.7,
                    is_mandatory=False,
                ))
            
            if intent_name in ("lab_report", "paper_writing"):
                criteria.append(SuccessCriterion(
                    criterion_id=f"sc_{intent_name}_review",
                    description="Report/paper must pass critic review with confidence > 0.8",
                    metric="critic_confidence",
                    threshold=0.8,
                    is_mandatory=True,
                ))
            
            if intent_name == "microscopy_analysis":
                criteria.append(SuccessCriterion(
                    criterion_id=f"sc_{intent_name}_segmentation",
                    description="Segmentation mask must cover > 90% of tissue area",
                    metric="segmentation_coverage",
                    threshold=0.9,
                    is_mandatory=False,
                ))
        
        return SuccessCriteria(
            mission_id=mission_id,
            criteria=criteria,
        )
    
    # ─── Private: Fallback Strategy ───────────────────────────────────────
    
    def _generate_fallback_strategy(self, mission_id: str, steps: List[MissionStep]) -> FallbackStrategy:
        """Generate recovery actions for each risky step."""
        actions = []
        
        for step in steps:
            if step.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
                # High-risk steps get agent substitution as fallback
                substitute = self._find_substitute_agent(step.assigned_agent)
                actions.append(FallbackAction(
                    trigger_step_id=step.step_id,
                    action="substitute_agent" if substitute else "retry",
                    substitute_agent=substitute,
                    max_retries=2,
                    escalation_message=f"Step '{step.title}' failed. Attempting recovery via {'agent substitution' if substitute else 'retry'}.",
                ))
            elif step.risk_level == RiskLevel.MEDIUM:
                # Medium-risk steps get retries
                actions.append(FallbackAction(
                    trigger_step_id=step.step_id,
                    action="retry",
                    max_retries=2,
                ))
            # Low-risk steps: skip on failure (non-critical)
            else:
                actions.append(FallbackAction(
                    trigger_step_id=step.step_id,
                    action="skip",
                    max_retries=0,
                ))
        
        total_est = sum(s.estimated_duration_seconds for s in steps)
        
        return FallbackStrategy(
            mission_id=mission_id,
            actions=actions,
            global_timeout_seconds=max(total_est * 3, 600),  # 3x estimated time or 10 min minimum
            abort_on_critical_failure=True,
        )
    
    def notify_critic_of_recovery(self, mission_id: str, failed_step_id: str, strategy: str):
        """
        Axis G2: Alert the CRITIC of a mission fallback.
        Creates a 'Quality Advisory' artifact linked to the mission.
        """
        plan = self._find_mission(mission_id)
        if not plan: return
        
        from .agent_contract import ArtifactRecord, ArtifactType as AT, ArtifactStatus
        advisory = ArtifactRecord(
            artifact_id=f"quality_advisory_{uuid.uuid4().hex[:6]}",
            mission_id=mission_id,
            project_id=plan.project_id,
            producer_agent_id="mission_brain",
            producer_agent_role="ORCHESTRATOR",
            title=f"Quality Advisory: Partial Failure Recovery ({failed_step_id})",
            description=f"Automated recovery triggered for {failed_step_id}",
            data={"failed_step": failed_step_id, "strategy": strategy},
            artifact_type=AT.CRITIQUE,
            status=ArtifactStatus.REVIEWED,
            confidence=0.7
        )
        
        project = PROJECT_STORE.load(plan.project_id)
        if project:
            project.artifacts.append(advisory)
            PROJECT_STORE.save(project)
            logger.info(f"[MissionBrain:G2] Critic notified: {advisory.artifact_id}")

    def _find_substitute_agent(self, failed_agent: AgentRole) -> Optional[AgentRole]:
        """Find a substitute agent that can handle similar tasks."""
        substitution_map = {
            AgentRole.BROWSER_AGENT: AgentRole.RESEARCHER,  # G2: Fallback to text research
            AgentRole.ML_EXPERT: AgentRole.DATA_ENGINEER,
            AgentRole.IN_SILICO_EXPERT: AgentRole.BIOINFORMATICIAN,
            AgentRole.MICROSCOPY_AGENT: AgentRole.CODING_AGENT,
            AgentRole.WRITING_AGENT: AgentRole.ACADEMIC_EXPERT,
        }
        return substitution_map.get(failed_agent)
    
    # ─── Private: Helpers ─────────────────────────────────────────────────
    
    def _generate_title(self, intents: List[Dict]) -> str:
        """Generate a human-readable mission title from intents."""
        intent_names = [i["intent"].replace("_", " ").title() for i in intents[:3]]
        return " + ".join(intent_names) if intent_names else "General Mission"
    
    def _generate_objective(self, intents: List[Dict], prompt: str) -> str:
        """Generate a concise mission objective."""
        primary_intents = [i["intent"] for i in intents[:2]]
        return f"Execute {', '.join(primary_intents)} pipeline based on user request: {prompt[:150]}"
    
    def _find_mission(self, mission_id: str) -> Optional[MissionPlan]:
        """Find a mission in history by ID."""
        return next((m for m in self._mission_history if m.mission_id == mission_id), None)
    
    def _skip_downstream(self, plan: MissionPlan, failed_step_id: str):
        """Mark all steps that depend on the failed step as SKIPPED."""
        for step in plan.steps:
            if failed_step_id in step.depends_on:
                step.status = StepStatus.SKIPPED
                # Recursively skip further downstream
                self._skip_downstream(plan, step.step_id)

    def execute_mission(self, mission_id: str):
        """
        Axis B: Orchestrates the full mission execution through the lifecycle.
        """
        plan = self._find_mission(mission_id)
        if not plan:
            logger.error(f"Mission {mission_id} not found for execution.")
            return

        SCHEDULER.register(plan)
        if not SCHEDULER.can_start(plan):
            plan.is_paused = True
            logger.warning(f"Resources exceeded. Mission {mission_id} paused.")
            return

        logger.info(f"[MissionBrain] Starting execution for: {plan.title}")
        
        # Simple sequential execution for demo
        for step in plan.steps:
            if step.status == StepStatus.COMPLETED or step.status == StepStatus.SKIPPED:
                continue
            
            self._execute_step(plan, step)
            
            if step.status == StepStatus.FAILED:
                logger.error(f"Mission failed at step {step.step_id}")
                break
        
        if all(s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED) for s in plan.steps):
            self.finalize_mission(mission_id)
        
        SCHEDULER.complete(mission_id)

    def _execute_step(self, plan: MissionPlan, step: MissionStep):
        """
        Axis B: Executes a single step using the 8-phase agent lifecycle.
        """
        agent_id = self.lifecycle.get_agent_for_role(step.assigned_agent)
        if not agent_id:
            logger.error(f"No agent found for role {step.assigned_agent}")
            step.status = StepStatus.FAILED
            return

        agent = self.lifecycle.get_agent(agent_id)
        if not agent:
            # For demo, we might need to instantiate it here if it's not active
            logger.warning(f"Agent {agent_id} not active. Skipping execution for demo.")
            step.status = StepStatus.COMPLETED # Mock completion for demo
            self.lifecycle.update_state(agent_id, AgentState.COMPLETED)
            return

        # 1. Update State to RUNNING
        self.lifecycle.update_state(agent_id, AgentState.RUNNING, reason=f"Executing {step.step_id}")
        step.status = StepStatus.RUNNING
        
        try:
            # 2. Trigger the 8-Phase Lifecycle Contract
            # This is the core of R6-2 integration
            output = agent.execute(plan.user_prompt) 
            
            # 3. Handle Artifacts (Axis C)
            for artifact in output.artifacts:
                self.lineage.record_artifact(artifact)
                
            # 4. Check Verification (Phase 4)
            if output.verification.is_valid:
                step.status = StepStatus.COMPLETED
                self.lifecycle.update_state(agent_id, AgentState.COMPLETED)
            else:
                logger.warning(f"Verification failed for {step.step_id}: {output.verification.reason}")
                step.status = StepStatus.FAILED
                self.lifecycle.update_state(agent_id, AgentState.FAILED, reason=output.verification.reason)

        except Exception as e:
            logger.exception(f"Exception during step {step.step_id}: {e}")
            step.status = StepStatus.FAILED
            self.lifecycle.update_state(agent_id, AgentState.FAILED, reason=str(e))
            
            # Trigger Replanner (Axis G2)
            self.replan(plan.mission_id, step.step_id, str(e))
