import logging
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime
from .models import (
    Comment, CommentStatus, Annotation, ProjectState, 
    IntentType, Severity, TargetType, Point, Box, TextRange, CodeAnchor, DataAnchor,
    ReviewerRole,
)

logger = logging.getLogger("bio_ml_agent.brain.comments")

class CommentManager:
    """
    R7-1: Commenting & Annotation System.
    Manages human and agent feedback linked to artifacts and mission steps.
    """
    def __init__(self, project_state_or_id=None, iteration_id: str = ""):
        if isinstance(project_state_or_id, ProjectState):
            self.project = project_state_or_id
        elif project_state_or_id is not None:
            # Called with (project_id, iteration_id) strings
            pid = str(project_state_or_id)
            self.project = ProjectState(project_id=pid, name=pid)
        else:
            self.project = ProjectState(project_id="default", name="default")
        self._iteration_id = iteration_id
        self._comment_store: Dict[str, "Comment"] = {}

    def add_comment(
        self,
        content: str,
        target_id: str = "",
        target_type: "TargetType | None" = None,
        author: str = "unknown",
        intent: IntentType = IntentType.GENERAL,
        severity: Severity = Severity.INFO,
        requested_action: Optional[str] = None,
        is_actionable: bool = True,
        parent_comment_id: Optional[str] = None,
        # Simplified API (used by tests and newer callers)
        target_uri: Optional[str] = None,
        role: Optional[object] = None,
    ) -> "Comment":
        """Adds a new comment to the project and links it to the target."""
        # Normalise args from simplified API
        if target_uri and not target_id:
            target_id = target_uri
        if target_type is None:
            target_type = TargetType.TEXT_RANGE
        if role is not None and author == "unknown":
            author = str(role.value) if hasattr(role, "value") else str(role)

        u_hex = uuid.uuid4().hex
        comment_id = f"cmt_{u_hex[:8]}"
        comment = Comment(
            comment_id=comment_id,
            author=author,
            content=content,
            target_id=target_id,
            target_type=target_type,
            intent=intent,
            severity=severity,
            requested_action=requested_action,
            is_actionable=is_actionable,
            parent_comment_id=parent_comment_id,
            role=role if role is not None else ReviewerRole.USER,
        )

        # Link to artifact if target is an artifact
        if target_type == TargetType.ARTIFACT:
            for art in self.project.artifacts:
                if art.artifact_id == target_id:
                    art.comments.append(comment)
                    break

        logger.info(f"[Comments:A1] New comment by {author} on {target_type} {target_id}")
        self._comment_store[comment_id] = comment
        return comment
        
    def add_inline_comment(
        self,
        target_id: str,
        target_type: TargetType,
        content: str,
        author: str,
        anchor_data: Dict[str, Any],
        intent: IntentType = IntentType.REVISE,
        severity: Severity = Severity.MEDIUM
    ) -> Comment:
        """Adds a comment with a specific inline anchor (A2)."""
        comment = self.add_comment(
            target_id=target_id,
            target_type=target_type,
            content=content,
            author=author,
            intent=intent,
            severity=severity
        )
        
        # Create corresponding annotation based on anchor_data
        annotation_id = f"ann_{uuid.uuid4().hex[:8]}"
        annotation = Annotation(
            annotation_id=annotation_id,
            comment_id=comment.comment_id,
            tag="inline_anchor"
        )
        
        # Map anchor_data to typed coordinates
        if target_type == TargetType.IMAGE_REGION:
            annotation.box = Box(**anchor_data)
        elif target_type in [TargetType.PARAGRAPH, TargetType.SECTION]:
            annotation.text_range = TextRange(**anchor_data)
        elif target_type == TargetType.CODE_LINE:
            annotation.code_anchor = CodeAnchor(**anchor_data)
        elif target_type == TargetType.TABLE_ROW:
            annotation.data_anchor = DataAnchor(**anchor_data)
            
        # Attach to target artifact
        for art in self.project.artifacts:
            if art.artifact_id == target_id:
                art.annotations.append(annotation)
                break
                
        return comment

    def add_annotation(
        self,
        comment_id: str,
        tag: str,
        coordinates: Dict[str, Any]
    ) -> Annotation:
        """Adds a visual/spatial annotation linked to a comment."""
        annotation = Annotation(
            annotation_id=f"ann_{uuid.uuid4().hex[:8]}",
            comment_id=comment_id,
            tag=tag,
            coordinates=coordinates
        )
        
        # In a real system, we'd search the artifact that contains the comment
        # and attach the annotation there.
        logger.info(f"[Comments:A1] New annotation {annotation.annotation_id} added to comment {comment_id}")
        return annotation

    def resolve_comment(self, comment_id: str, resolution_note: Optional[str] = None):
        """Marks a comment as resolved."""
        comment = self.get_comment(comment_id)
        if comment:
            comment.status = CommentStatus.RESOLVED
            if resolution_note:
                comment.resolution_note = resolution_note
        logger.info(f"[Comments:A1] Comment {comment_id} marked as RESOLVED")

    def get_comment(self, comment_id: str) -> Optional["Comment"]:
        """Retrieves a specific comment by ID from internal store."""
        return self._comment_store.get(comment_id)

    def get_active_threads(self) -> List["Comment"]:
        """Returns all non-resolved comments (active review threads)."""
        return [c for c in self._comment_store.values() if c.status != CommentStatus.RESOLVED]

    def prepare_review_bundle(self) -> Dict[str, Any]:
        """Returns a summary dict for the current review session."""
        comments = list(self._comment_store.values())
        summary_parts = [c.content for c in comments]
        return {
            "total_comments": len(comments),
            "active_comments": len([c for c in comments if c.status != CommentStatus.RESOLVED]),
            "resolved_comments": len([c for c in comments if c.status == CommentStatus.RESOLVED]),
            "summary": " | ".join(summary_parts),
            "comments": [{"id": c.comment_id, "content": c.content, "author": c.author, "status": c.status} for c in comments],
        }

    def export_state(self) -> Dict[str, Any]:
        """Exports the current comment state for persistence or analysis."""
        all_comments = list(self._comment_store.values())
        threads = self.get_active_threads()
        return {
            "project_id": self.project.project_id,
            "active_threads": [{"id": c.comment_id, "author": c.author, "content": c.content} for c in threads],
            "total_threads": len(self._comment_store),
            "active_count": len(threads),
            "comments": [{"id": c.comment_id, "content": c.content, "author": c.author, "status": c.status} for c in all_comments],
        }

    def get_comments_for_target(self, target_id: str) -> List["Comment"]:
        """
        Retrieves all comments for a specific target by searching across
        artifacts, steps, and project-level records.
        """
        all_comments = []
        
        # 1. Search in artifacts
        for art in self.project.artifacts:
            if art.artifact_id == target_id:
                all_comments.extend(art.comments)
            # Also check internal artifact comments (e.g. on sections)
            for c in art.comments:
                if c.target_id == target_id:
                     if c not in all_comments:
                         all_comments.append(c)
                         
        # 2. Search in project-level findings or general project comments
        # (Implementation would expand as we add more containers)
        
        return all_comments
