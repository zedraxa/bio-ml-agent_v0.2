import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from .models import ProjectState, MissionPlan
from .agent_contract import ArtifactRecord, ArtifactReviewStatus, ArtifactType
from .artifact_graph import ArtifactGraph

logger = logging.getLogger("bio_ml_agent")

class ConsistencyIssue(BaseModel):
    issue_type: str # 'lineage_break', 'outdated_mismatch', 'finality_violation', 'memory_conflict'
    severity: str # 'low', 'medium', 'high', 'critical'
    description: str
    artifact_id: Optional[str] = None
    affected_nodes: List[str] = Field(default_factory=list)

class ConsistencyReport(BaseModel):
    project_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    is_consistent: bool
    issues: List[ConsistencyIssue] = Field(default_factory=list)
    summary: str

class ConsistencyChecker:
    """
    Axis E5: Consistency Checker.
    Audits the project state for integrity, staleness, and memory alignment.
    """
    
    def __init__(self, project_id: str):
        self.project_id = project_id

    def audit(self, project: ProjectState, graph: ArtifactGraph) -> ConsistencyReport:
        """Runs a full suite of consistency checks."""
        issues = []
        
        # 1. Lineage Integrity & Outdated Propagation
        lineage_issues = self._check_lineage_and_staleness(graph)
        issues.extend(lineage_issues)
        
        # 2. Finality Uniqueness
        finality_issues = self._check_finality_uniqueness(graph)
        issues.extend(finality_issues)
        
        # 3. Memory Alignment
        memory_issues = self._check_memory_alignment(project)
        issues.extend(memory_issues)
        
        # 4. State Consistency
        state_issues = self._check_state_consistency(project, graph)
        issues.extend(state_issues)
        
        # 5. [F3] Handoff Integrity (Mission Level)
        # Note: Requires a MissionPlan to check against
        # We'll pass it optionally or as part of a separate call for now.
        
        is_consistent = len([i for i in issues if i.severity in ["high", "critical"]]) == 0
        
        summary = f"Audit complete. Found {len(issues)} issues."
        if not is_consistent:
            summary += " CRITICAL INCONSISTENCIES DETECTED."
            
        return ConsistencyReport(
            project_id=self.project_id,
            is_consistent=is_consistent,
            issues=issues,
            summary=summary
        )

    def _check_lineage_and_staleness(self, graph: ArtifactGraph) -> List[ConsistencyIssue]:
        issues = []
        for node_id, node in graph.nodes.items():
            # Check if sources exist
            for src_id in node.record.source_input_ids:
                if src_id not in graph.nodes:
                    issues.append(ConsistencyIssue(
                        issue_type="lineage_break",
                        severity="high",
                        description=f"Artifact {node_id} references missing source {src_id}",
                        artifact_id=node_id
                    ))
                else:
                    # check outdated propagation
                    src_node = graph.nodes[src_id]
                    if src_node.record.is_outdated and not node.record.is_outdated:
                        issues.append(ConsistencyIssue(
                            issue_type="outdated_mismatch",
                            severity="medium",
                            description=f"Artifact {node_id} is not marked outdated even though source {src_id} is.",
                            artifact_id=node_id,
                            affected_nodes=[src_id]
                        ))
        return issues

    def _check_finality_uniqueness(self, graph: ArtifactGraph) -> List[ConsistencyIssue]:
        issues = []
        # Group by type and mission
        final_counts: Dict[str, List[str]] = {} # "mission_id:type" -> [artifact_id]
        
        for node_id, node in graph.nodes.items():
            if node.record.status == ArtifactReviewStatus.FINAL:
                key = f"{node.record.mission_id}:{node.record.artifact_type.value}"
                if key not in final_counts:
                    final_counts[key] = []
                final_counts[key].append(node_id)
        
        for key, ids in final_counts.items():
            if len(ids) > 1:
                issues.append(ConsistencyIssue(
                    issue_type="finality_violation",
                    severity="high",
                    description=f"Multiple FINAL artifacts found for {key}: {ids}",
                    affected_nodes=ids
                ))
        return issues

    def _check_memory_alignment(self, project: ProjectState) -> List[ConsistencyIssue]:
        issues = []
        # Simple check: Critical findings in ProjectState should also be reflected in the findings list
        # This is a placeholder for actual Memory Hub integration
        if project.critical_findings and not project.memory_snapshot_ref:
             issues.append(ConsistencyIssue(
                issue_type="memory_conflict",
                severity="low",
                description="Project has critical findings but no linked memory snapshot reference."
            ))
        return issues

    def _check_state_consistency(self, project: ProjectState, graph: ArtifactGraph) -> List[ConsistencyIssue]:
        issues = []
        # check if all approved_artifact_ids actually exist in the graph
        for art_id in project.approved_artifact_ids:
            if art_id not in graph.nodes:
                issues.append(ConsistencyIssue(
                    issue_type="state_fragmentation",
                    severity="medium",
                    description=f"Project marks {art_id} as approved, but it is missing from the artifact graph.",
                    artifact_id=art_id
                ))
            elif graph.nodes[art_id].record.status not in [ArtifactReviewStatus.APPROVED, ArtifactReviewStatus.FINAL, ArtifactReviewStatus.EXPORTED]:
                issues.append(ConsistencyIssue(
                    issue_type="state_mismatch",
                    severity="medium",
                    description=f"Project marks {art_id} as approved, but its graph status is {graph.nodes[art_id].record.status}",
                    artifact_id=art_id
                ))
        return issues

    def audit_handoffs(self, plan: MissionPlan, graph: ArtifactGraph) -> List[ConsistencyIssue]:
        """
        Axis F3: Multi-Agent Integration (Handoff) Audit.
        Ensures that dependencies are satisfied by actual artifact flow.
        """
        issues = []
        step_map = {s.step_id: s for s in plan.steps}
        
        for step in plan.steps:
            for dep_id in step.depends_on:
                if dep_id not in step_map:
                    issues.append(ConsistencyIssue(
                        issue_type="handoff_logic_error",
                        severity="high",
                        description=f"Step {step.step_id} depends on non-existent step {dep_id}",
                    ))
                    continue
                
                parent = step_map[dep_id]
                
                # Check if parent produced something this step consumes
                # Step.input_artifacts should contains at least one of parent.output_artifacts
                shared_artifacts = set(parent.output_artifacts).intersection(set(step.input_artifacts))
                
                if not shared_artifacts and parent.output_artifacts:
                    issues.append(ConsistencyIssue(
                        issue_type="handoff_gap",
                        severity="medium",
                        description=f"Handoff Gap: Step {step.step_id} ({step.assigned_agent.value}) depends on {dep_id} ({parent.assigned_agent.value}) but does not consume its output artifacts.",
                        affected_nodes=[step.step_id, dep_id]
                    ))
        
        return issues
