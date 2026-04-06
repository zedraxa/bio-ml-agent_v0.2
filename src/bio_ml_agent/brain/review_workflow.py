import logging
import uuid
from typing import List, Dict, Any, Optional
from .models import ReviewBundle, ReviewMode, ReviewerRole, Comment, CommentStatus

logger = logging.getLogger("bio_ml_agent.brain.review")

class ReviewWorkflowManager:
    """
    R7-C: Multi-Layer Review Workflows.
    Manages complex review cycles, version comparisons, and approval gates.
    """
    def __init__(self, comment_manager: Any):
        self.comment_manager = comment_manager

    def create_bundle(self, artifact_id: str, mode: ReviewMode) -> ReviewBundle:
        """C3: Review Bundles. Groups all comments for an artifact into a package."""
        relevant_comments = [c for c in self.comment_manager.comments.values() if c.target_id == artifact_id]

        bundle_id = f"bundle_{uuid.uuid4().hex[:6]}"
        blocking = sum(1 for c in relevant_comments if c.severity == "critical")

        bundle = ReviewBundle(
            bundle_id=bundle_id,
            artifact_id=artifact_id,
            mode=mode,
            comments=[c.comment_id for c in relevant_comments],
            summary=f"Review summary for artifact {artifact_id} in {mode} mode. Found {len(relevant_comments)} comments.",
            blocking_count=blocking,
            is_ready_for_approval=(blocking == 0)
        )

        logger.info(f"[ReviewWorkflow:C3] Created bundle {bundle_id} for {artifact_id}")
        return bundle

    def initiate_comparison(self, version_a_id: str, version_b_id: str) -> Dict[str, Any]:
        """C4: Compare-and-review flows. Prepares a comparison context."""
        logger.info(f"[ReviewWorkflow:C4] Initiating comparison between {version_a_id} and {version_b_id}")
        return {
            "comparison_id": f"comp_{uuid.uuid4().hex[:6]}",
            "v_old": version_a_id,
            "v_new": version_b_id,
            "diff_summary": "Simulated structural diff for scientific reviewing."
        }
