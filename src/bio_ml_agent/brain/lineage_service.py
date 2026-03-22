import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from .models import ArtifactRecord, ProjectState

logger = logging.getLogger(__name__)

class ArtifactLineageService:
    """
    Part VI: Artifact Lineage Service.
    
    Tracks the provenance and evolution of scientific artifacts 
    across multiple missions.
    """

    def track_derivation(self, project_state: ProjectState, artifact_id: str) -> List[ArtifactRecord]:
        """Returns the full lineage of an artifact, tracing back to its sources."""
        lineage = []
        current_id = artifact_id
        
        # Build map for fast lookup
        artifact_map = {a.artifact_id: a for a in project_state.artifacts}
        
        visited = set()
        while current_id in artifact_map and current_id not in visited:
            visited.add(current_id)
            art = artifact_map[current_id]
            lineage.append(art)
            
            # Trace back to the first source
            if art.lineage.sources:
                current_id = art.lineage.sources[0]
            else:
                break
                
        return lineage

    def get_version_history(self, project_state: ProjectState, original_artifact_id: str) -> List[ArtifactRecord]:
        """Returns all versions/evolutions of a specific artifact."""
        # In this simplistic model, we look for artifacts derived from this one
        history = [a for a in project_state.artifacts if original_artifact_id in a.lineage.sources]
        
        # Sort by timestamp
        history.sort(key=lambda x: x.created_at)
        return history

    def audit_artifact_health(self, artifact: ArtifactRecord) -> Dict[str, Any]:
        """Performs a consistency check on a single artifact."""
        return {
            "artifact_id": artifact.artifact_id,
            "has_sources": len(artifact.lineage.sources) > 0,
            "has_confidence": artifact.lineage.confidence_score > 0,
            "is_stable": artifact.status == "final",
            "health_score": 0.9 if artifact.status == "final" else 0.5
        }
