"""
bio_ml_agent.db.models — SQLAlchemy ORM models.

All tables use SQLite-compatible column types.  The schema is created
automatically via ``Base.metadata.create_all(bind=engine)``.
"""
import time
import uuid
from typing import Any, Optional

from sqlalchemy import (
    Boolean,
    Column,
    Float,
    Integer,
    JSON,
    String,
    Text,
)

from bio_ml_agent.db.session import Base


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


class ProjectDB(Base):
    __tablename__ = "projects"

    project_id = Column(String, primary_key=True, default=_new_id)
    name = Column(String, nullable=False, default="")
    description = Column(Text, default="")
    status = Column(String, default="active")


class MissionDB(Base):
    __tablename__ = "missions"

    mission_id = Column(String, primary_key=True, default=_new_id)
    project_id = Column(String, nullable=False, default="")
    name = Column(String, default="")
    title = Column(String, default="")
    objective = Column(Text, default="")
    status = Column(String, default="pending")
    created_at = Column(Float, default=time.time)
    updated_at = Column(Float, default=time.time)


class ArtifactDB(Base):
    __tablename__ = "artifacts"

    artifact_id = Column(String, primary_key=True, default=_new_id)
    mission_id = Column(String, nullable=False, default="")
    project_id = Column(String, default="")
    name = Column(String, default="")
    path = Column(String, default="")
    title = Column(String, default="")
    category = Column(String, default="")
    file_type = Column(String, default="")
    status = Column(String, default="draft")
    created_by = Column(String, default="")
    lineage_parents = Column(JSON, default=list)
    content_uri = Column(String, default="")
    created_at = Column(Float, default=time.time)
    updated_at = Column(Float, default=time.time)


class TimelineEventDB(Base):
    __tablename__ = "timeline_events"

    event_id = Column(String, primary_key=True, default=_new_id)
    project_id = Column(String, default="")
    event_type = Column(String, default="")
    description = Column(Text, default="")


class ProjectMemoryDB(Base):
    __tablename__ = "project_memories"

    memory_id = Column(String, primary_key=True, default=_new_id)
    project_id = Column(String, default="")
    content = Column(Text, default="")


class MissionStepDB(Base):
    __tablename__ = "mission_steps"

    step_id = Column(String, primary_key=True, default=_new_id)
    mission_id = Column(String, nullable=False, default="")
    name = Column(String, default="")
    status = Column(String, default="pending")
    action_type = Column(String, default="")
    timestamp = Column(Float, default=time.time)
    agent_name = Column(String, default="")
    agent_role = Column(String, default="")
    content = Column(Text, default="")
    thought = Column(Text, default="")
    confidence = Column(Float, default=0.0)
    metadata_json = Column(JSON, default=dict)


class NotificationDB(Base):
    __tablename__ = "notifications"

    notification_id = Column(String, primary_key=True, default=_new_id)
    title = Column(String, default="")
    message = Column(Text, default="")
    read = Column(Boolean, default=False)


class CommentDB(Base):
    __tablename__ = "comments"

    comment_id = Column(String, primary_key=True, default=_new_id)
    artifact_id = Column(String, default="")
    author = Column(String, default="")
    content = Column(Text, default="")
    status = Column(String, default="new")
    is_resolved = Column(Boolean, default=False)
    timestamp = Column(Float, default=time.time)


class SettingsDB(Base):
    __tablename__ = "settings"

    settings_id = Column(String, primary_key=True, default=_new_id)
    config = Column(JSON, default=dict)


class ReviewThreadDB(Base):
    __tablename__ = "review_threads"

    thread_id = Column(String, primary_key=True, default=_new_id)
    artifact_id = Column(String, default="")
    status = Column(String, default="open")
    created_at = Column(Float, default=time.time)
    updated_at = Column(Float, default=time.time)


class ProjectTruthSnapshotDB(Base):
    __tablename__ = "project_truth_snapshots"

    snapshot_id = Column(String, primary_key=True, default=_new_id)
    project_id = Column(String, default="")
    data = Column(JSON, default=dict)
