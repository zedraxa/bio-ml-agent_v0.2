"""
bio_ml_agent.db.session — SQLAlchemy session factory.

Uses SQLite by default (URL from env ``WORKSPACE_DB_PATH``).
In-memory SQLite is used for testing when the URL is set to
``sqlite:///:memory:``.
"""
import os
import logging
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase

log = logging.getLogger("bio_ml_agent")

_DB_URL = os.environ.get("WORKSPACE_DB_PATH", "sqlite:///bio_ml_agent.db")

# SQLite-specific: enable WAL mode and foreign keys for file-backed DBs
connect_args = {}
if _DB_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(_DB_URL, connect_args=connect_args)


class Base(DeclarativeBase):
    pass


SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
