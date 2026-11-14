"""
app/database/db.py
SQLAlchemy engine, session factory, declarative base, and dependency helpers.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
_connect_args = {}
if settings.database_url.startswith("sqlite"):
    # Enable WAL mode and foreign-key enforcement for SQLite
    _connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.database_url,
    connect_args=_connect_args,
    echo=settings.debug,       # logs SQL when DEBUG=True
    pool_pre_ping=True,        # validate connections before use
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragmas(dbapi_conn, _connection_record):
    """Apply SQLite-specific PRAGMAs on every new connection."""
    if settings.database_url.startswith("sqlite"):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


# ---------------------------------------------------------------------------
# Declarative base (shared by all ORM models)
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    """Project-wide SQLAlchemy declarative base."""


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------
def get_db() -> Generator[Session, None, None]:
    """
    Yield a SQLAlchemy session for use as a FastAPI dependency.

    Usage::

        @router.get("/claims")
        def list_claims(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Context-manager variant (for scripts / background tasks)
# ---------------------------------------------------------------------------
@contextmanager
def db_session() -> Generator[Session, None, None]:
    """
    Context-manager wrapper around SessionLocal.

    Usage::

        with db_session() as db:
            db.add(some_object)
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------
def create_all_tables() -> None:
    """
    Create all tables defined via the ORM in the target database.
    Safe to call on every startup — existing tables are left untouched.
    """
    # Import models so their classes are registered on Base.metadata
    import app.database.models  # noqa: F401

    logger.info("Creating database tables (if not exist) on: %s", settings.database_url)
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ready.")


def health_check() -> bool:
    """Return True if the database is reachable."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.error("Database health check failed: %s", exc)
        return False
