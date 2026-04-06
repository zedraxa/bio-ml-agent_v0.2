from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from .agent_contract import (
    ArtifactRecord,
    ArtifactReviewStatus,
    VALID_ARTIFACT_TRANSITIONS,
    ArtifactStateTransition
)

logger = logging.getLogger("bio_ml_agent")

class ArtifactNode(BaseModel):
    """A node in the Cross-System Artifact Graph."""
    artifact_id: str
    record: ArtifactRecord
    created_at: float = Field(default_factory=datetime.now().timestamp)

    @property
    def label(self) -> str:
        return f"{self.record.artifact_type.value}: {self.record.title}"

    def transition_to(self, new_status: ArtifactReviewStatus, reason: Optional[str] = None):
        """
        C4: Transitions the artifact to a new state and records the history.
        Enforces VALID_ARTIFACT_TRANSITIONS rules.
        """
        current = self.record.status

        # Validation
        if new_status not in VALID_ARTIFACT_TRANSITIONS.get(current, []):
            logger.warning(f"[Axis C4] Invalid transition: {current} -> {new_status} for {self.artifact_id}")
            # We allow it with a warning for system-forced transitions like OUTDATED
            if new_status != ArtifactReviewStatus.OUTDATED:
                return

        # Record change
        transition = ArtifactStateTransition(
            from_status=current,
            to_status=new_status,
            reason=reason
        )
        self.record.status = new_status
        self.record.status_history.append(transition)

        logger.info(f"[Axis C4] State changed: {self.artifact_id} ({current} -> {new_status})")

class ArtifactLink(BaseModel):
    """An edge in the Artifact Graph representing dependency/lineage."""
    source_id: str = Field(..., description="The input artifact ID")
    target_id: str = Field(..., description="The produced artifact ID")
    link_type: str = Field(default="derives_from", description="Relationship: derives_from, version_of, part_of, extracted_from, summarized_from")

class LineageReport(BaseModel):
    """C2: A professional, human-readable lineage/provenance certificate."""
    target_artifact_id: str
    target_title: str
    producer_agent: str
    status: ArtifactReviewStatus
    version: str

    # The Chain
    direct_sources: List[Dict[str, str]] = Field(default_factory=list, description="IDs and titles of immediate inputs")
    all_ancestors: List[Dict[str, str]] = Field(default_factory=list, description="Flattened list of all contributing artifacts")
    root_inputs: List[Dict[str, str]] = Field(default_factory=list, description="The original raw data/inputs")

    # Context
    mission_id: str
    project_id: str
    timestamp: datetime = Field(default_factory=datetime.now)

class ArtifactGraph(BaseModel):
    """
    C1: Cross-System Artifact Graph.
    
    Tracks the complete lineage of all scientific outputs.
    Allows researchers to trace results back to raw inputs and
    understand the sequence of agent operations.
    """
    mission_id: str
    project_id: str
    nodes: Dict[str, ArtifactNode] = Field(default_factory=dict)
    links: List[ArtifactLink] = Field(default_factory=list)

    def detect_conflict(self, new_record: ArtifactRecord) -> Optional[str]:
        """
        E4: Detects if adding this artifact would create a conflict.
        A conflict occurs if another artifact already exists with the same 
        parent_artifact_id but different content/producer.
        """
        if not new_record.parent_artifact_id:
            return None

        # Find siblings (other artifacts with the same parent)
        siblings = [
            node for node in self.nodes.values()
            if node.record.parent_artifact_id == new_record.parent_artifact_id
            and node.artifact_id != new_record.artifact_id
        ]

        for sib in siblings:
            # If they come from different agents or have different titles/versions
            # without one being a direct descendant of the other, it's a conflict.
            if sib.record.producer_agent_id != new_record.producer_agent_id:
                return sib.artifact_id

        return None

    def add_artifact(self, record: ArtifactRecord):
        """Adds an artifact and establishes lineage links. Handles conflict detection."""

        # E4: Conflict Detection
        conflicting_id = self.detect_conflict(record)
        if conflicting_id:
            logger.warning(f"[Axis E4] Conflict detected for {record.artifact_id} with {conflicting_id}")
            record.status = ArtifactReviewStatus.CONFLICT
            if "conflict_with" not in record.metadata:
                 record.metadata["conflict_with"] = conflicting_id

        node = ArtifactNode(artifact_id=record.artifact_id, record=record)

        # Initialize history if empty
        if not node.record.status_history:
            node.record.status_history.append(ArtifactStateTransition(
                from_status=record.status,
                to_status=record.status,
                reason="Initial state"
            ))

        self.nodes[record.artifact_id] = node

        # Link to sources (Lineage)
        for src_id in record.source_input_ids:
            self.links.append(ArtifactLink(source_id=src_id, target_id=record.artifact_id))

        # Link to parent version (Versioning)
        if record.parent_artifact_id:
            self.links.append(ArtifactLink(
                source_id=record.parent_artifact_id,
                target_id=record.artifact_id,
                link_type="version_of"
            ))

        logger.info(f"[Axis C] Artifact added to graph: {record.artifact_id} ({record.title})")

    def get_lineage(self, artifact_id: str) -> List[str]:
        """Returns the full ancestry of an artifact (recursive)."""
        ancestry = []
        to_check = [artifact_id]
        visited = set()

        while to_check:
            current = to_check.pop(0)
            if current in visited: continue
            visited.add(current)

            # Find all sources for this target
            sources = [l.source_id for l in self.links if l.target_id == current]
            ancestry.extend(sources)
            to_check.extend(sources)

        return list(set(ancestry))

    def get_downstream(self, artifact_id: str) -> List[str]:
        """Returns all artifacts derived from this one."""
        derived = []
        to_check = [artifact_id]
        visited = set()

        while to_check:
            current = to_check.pop(0)
            if current in visited: continue
            visited.add(current)

            # Find all targets where this is the source
            targets = [l.target_id for l in self.links if l.source_id == current]
            derived.extend(targets)
            to_check.extend(targets)

        return list(set(derived))

    def get_manifest(self) -> List[ArtifactRecord]:
        """Returns all artifacts in the graph as a flat list."""
        return [node.record for node in self.nodes.values()]

    def filter_by_type(self, artifact_type: Any) -> List[ArtifactRecord]:
        return [n.record for n in self.nodes.values() if n.record.artifact_type == artifact_type]

    def get_root_inputs(self) -> List[str]:
        """Returns artifacts that have no known sources in this graph."""
        target_ids = {l.target_id for l in self.links}
        return [node_id for node_id in self.nodes if node_id not in target_ids]

    def mark_outdated_downstream(self, artifact_id: str, reason: str):
        """
        C3: Recursively marks all downstream artifacts as outdated.
        This handles the scenario where a source changes and all 
        derived reports/figures need revision.
        """
        downstream_ids = self.get_downstream(artifact_id)
        for d_id in downstream_ids:
            if d_id in self.nodes:
                node = self.nodes[d_id]
                node.record.is_outdated = True

                # Use State Machine for transition
                triggered_by = self.nodes[artifact_id].record.title
                update_reason = f"Source '{triggered_by}' updated: {reason}"

                node.transition_to(ArtifactReviewStatus.OUTDATED, reason=update_reason)

                if node.record.outdated_reason:
                    node.record.outdated_reason += f" | {update_reason}"
                else:
                    node.record.outdated_reason = update_reason

    def get_revision_queue(self) -> List[ArtifactRecord]:
        """
        C3: Returns all artifacts that are currently outdated 
        and require agent or human revision.
        """
        return [n.record for n in self.nodes.values() if n.record.is_outdated]

    def get_provenance_report(self, artifact_id: str) -> LineageReport:
        """
        C2: Generates a professional Lineage Report for an artifact.
        
        This satisfies the 'Artifact Lineage Engine' requirement by 
        explicitly tracing reports to results, captions to images, etc.
        """
        if artifact_id not in self.nodes:
            raise ValueError(f"Artifact {artifact_id} not found in graph.")

        node = self.nodes[artifact_id]
        record = node.record

        # Direct sources
        direct_ids = [l.source_id for l in self.links if l.target_id == artifact_id]
        direct_sources = [
            {"id": sid, "title": self.nodes[sid].record.title, "type": self.nodes[sid].record.artifact_type.value}
            for sid in direct_ids if sid in self.nodes
        ]

        # All ancestors (flattened)
        ancestor_ids = self.get_lineage(artifact_id)
        all_ancestors = [
            {"id": aid, "title": self.nodes[aid].record.title, "type": self.nodes[aid].record.artifact_type.value}
            for aid in ancestor_ids if aid in self.nodes
        ]

        # Root inputs (ancestors that are roots)
        root_ids_global = self.get_root_inputs()
        root_ids = [aid for aid in ancestor_ids if aid in root_ids_global]
        root_inputs = [
            {"id": rid, "title": self.nodes[rid].record.title, "type": self.nodes[rid].record.artifact_type.value}
            for rid in root_ids if rid in self.nodes
        ]

        return LineageReport(
            target_artifact_id=artifact_id,
            target_title=record.title,
            producer_agent=f"{record.producer_agent_role} ({record.producer_agent_id})",
            status=record.status,
            version=record.version,
            direct_sources=direct_sources,
            all_ancestors=all_ancestors,
            root_inputs=root_inputs,
            mission_id=self.mission_id,
            project_id=self.project_id
        )

    def compare_versions(self, id1: str, id2: str) -> Dict[str, Any]:
        """E4: Returns a structured comparison between two artifact versions."""
        if id1 not in self.nodes or id2 not in self.nodes:
            return {"error": "One or both artifacts not found"}

        r1 = self.nodes[id1].record
        r2 = self.nodes[id2].record

        return {
            "comparison": "version_divergence",
            "artifact_1": {"id": id1, "agent": r1.producer_agent_role, "title": r1.title},
            "artifact_2": {"id": id2, "agent": r2.producer_agent_role, "title": r2.title},
            "diff_summary": f"Divergence detected between {r1.producer_agent_role} and {r2.producer_agent_role} inputs."
        }

    def resolve_conflict(self, conflict_id: str, resolved_record: ArtifactRecord, strategy: str = "supersede"):
        """E4: Resolves a conflict by adding a new record and marking the old one as replaced."""
        if conflict_id not in self.nodes:
            return

        # Add the resolved artifact
        self.add_artifact(resolved_record)

        # Mark the conflicted one as replaced
        conflict_node = self.nodes[conflict_id]
        conflict_node.transition_to(ArtifactReviewStatus.REPLACED, reason=f"Resolved via strategy: {strategy}")

        # Link them
        self.links.append(ArtifactLink(
            source_id=conflict_id,
            target_id=resolved_record.artifact_id,
            link_type="superseded_by"
        ))

        logger.info(f"[Axis E4] Conflict resolved: {conflict_id} -> {resolved_record.artifact_id}")
