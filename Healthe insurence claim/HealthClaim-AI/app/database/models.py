"""
app/database/models.py
SQLAlchemy ORM models for HealthClaim AI.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime
from enum import Enum as PyEnum
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.db import Base


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class UserRole(str, PyEnum):
    admin = "admin"
    adjuster = "adjuster"
    auditor = "auditor"


class ClaimStatus(str, PyEnum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    UNDER_REVIEW = "UNDER_REVIEW"
    FRAUD_SUSPECTED = "FRAUD_SUSPECTED"


# ---------------------------------------------------------------------------
# User
# ---------------------------------------------------------------------------
class User(Base):
    """Platform user (adjuster / auditor / admin)."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_uuid
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole, name="user_role"), nullable=False, default=UserRole.adjuster
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)

    # Relationships
    audit_logs: Mapped[list["AuditLog"]] = relationship(
        "AuditLog", back_populates="user", foreign_keys="AuditLog.actor"
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email} role={self.role}>"


# ---------------------------------------------------------------------------
# Claim
# ---------------------------------------------------------------------------
class Claim(Base):
    """An insurance claim submitted for processing."""

    __tablename__ = "claims"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    claim_number: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True
    )

    # Patient info
    patient_name: Mapped[str] = mapped_column(String(255), nullable=False)
    patient_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    patient_dob: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # Provider info
    provider_name: Mapped[str] = mapped_column(String(255), nullable=False)
    provider_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    # Claim details
    claim_date: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)
    diagnosis_codes: Mapped[list[Any]] = mapped_column(JSON, nullable=False, default=list)
    procedure_codes: Mapped[list[Any]] = mapped_column(JSON, nullable=False, default=list)
    amount_claimed: Mapped[float] = mapped_column(Float, nullable=False)
    amount_approved: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Status & AI outputs
    status: Mapped[ClaimStatus] = mapped_column(
        Enum(ClaimStatus, name="claim_status"),
        nullable=False,
        default=ClaimStatus.PENDING,
        index=True,
    )
    fraud_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    fraud_flags: Mapped[list[Any]] = mapped_column(JSON, nullable=False, default=list)
    ai_decision: Mapped[str | None] = mapped_column(Text, nullable=True)
    ai_reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=_now, onupdate=_now
    )

    # Relationships
    documents: Mapped[list["Document"]] = relationship(
        "Document", back_populates="claim", cascade="all, delete-orphan"
    )
    audit_logs: Mapped[list["AuditLog"]] = relationship(
        "AuditLog", back_populates="claim"
    )

    def __repr__(self) -> str:
        return f"<Claim {self.claim_number} status={self.status}>"


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------
class Document(Base):
    """A supporting document attached to a claim."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    claim_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("claims.id", ondelete="CASCADE"), nullable=False, index=True
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_type: Mapped[str] = mapped_column(String(64), nullable=False)   # e.g. "pdf", "jpg"
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)  # bytes
    extracted_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)

    # Relationships
    claim: Mapped["Claim"] = relationship("Claim", back_populates="documents")

    def __repr__(self) -> str:
        return f"<Document {self.filename} claim={self.claim_id}>"


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------
class Policy(Base):
    """An insurance policy held by a patient / member."""

    __tablename__ = "policies"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    policy_number: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True
    )
    holder_name: Mapped[str] = mapped_column(String(255), nullable=False)
    holder_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    coverage_type: Mapped[str] = mapped_column(
        String(64), nullable=False, default="Basic"
    )  # Comprehensive / Basic / Premium

    # Financial
    deductible: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    max_coverage: Mapped[float] = mapped_column(Float, nullable=False)
    copay_percent: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Coverage details
    covered_procedures: Mapped[list[Any]] = mapped_column(JSON, nullable=False, default=list)
    excluded_conditions: Mapped[list[Any]] = mapped_column(JSON, nullable=False, default=list)

    # Validity
    effective_date: Mapped[date] = mapped_column(Date, nullable=False)
    expiry_date: Mapped[date] = mapped_column(Date, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    def __repr__(self) -> str:
        return f"<Policy {self.policy_number} holder={self.holder_name}>"


# ---------------------------------------------------------------------------
# AuditLog
# ---------------------------------------------------------------------------
class AuditLog(Base):
    """Immutable audit trail for every action taken on a claim."""

    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    claim_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("claims.id", ondelete="SET NULL"), nullable=True, index=True
    )
    action: Mapped[str] = mapped_column(String(128), nullable=False)
    # actor stores the user email/id; also FK to users.id for join convenience
    actor: Mapped[str] = mapped_column(
        String(255), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    details: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)

    # Relationships
    claim: Mapped["Claim | None"] = relationship("Claim", back_populates="audit_logs")
    user: Mapped["User | None"] = relationship(
        "User", back_populates="audit_logs", foreign_keys=[actor]
    )

    def __repr__(self) -> str:
        return f"<AuditLog action={self.action} claim={self.claim_id} actor={self.actor}>"
