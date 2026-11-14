"""
app/schemas/claim_schema.py
Pydantic v2 request / response schemas for the HealthClaim AI API.
"""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations (mirror ORM enums for serialisation)
# ---------------------------------------------------------------------------
class ClaimStatusEnum(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    UNDER_REVIEW = "UNDER_REVIEW"
    FRAUD_SUSPECTED = "FRAUD_SUSPECTED"


class UserRoleEnum(str, Enum):
    admin = "admin"
    adjuster = "adjuster"
    auditor = "auditor"


# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------
class _BaseSchema(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,          # allow ORM-model → schema conversion
        populate_by_name=True,
        str_strip_whitespace=True,
    )


# ---------------------------------------------------------------------------
# Claim schemas
# ---------------------------------------------------------------------------
class ClaimCreate(_BaseSchema):
    """Fields required to submit a new insurance claim."""

    patient_name: str = Field(..., min_length=2, max_length=255, description="Full name of the patient")
    patient_id: str = Field(..., description="Patient's insurance member ID")
    patient_dob: Optional[str] = Field(None, description="Patient date of birth (YYYY-MM-DD)")
    provider_name: str = Field(..., description="Name of the healthcare provider / facility")
    provider_id: str = Field(..., description="Provider NPI or facility code")
    claim_date: Optional[datetime] = Field(None, description="Date the claim is for; defaults to now")
    diagnosis_codes: list[str] = Field(default_factory=list, description="ICD-10 diagnosis codes")
    procedure_codes: list[str] = Field(default_factory=list, description="CPT procedure codes")
    amount_claimed: float = Field(..., gt=0, description="Total amount claimed in USD")
    clinical_notes: Optional[str] = Field(None, max_length=10000)

    @field_validator("amount_claimed")
    @classmethod
    def _positive_amount(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("amount_claimed must be positive")
        return round(v, 2)

    @field_validator("patient_dob")
    @classmethod
    def _valid_dob(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("patient_dob must be in YYYY-MM-DD format")
        return v


class ClaimUpdate(_BaseSchema):
    """Partial update payload for an existing claim."""

    status: Optional[ClaimStatusEnum] = None
    amount_approved: Optional[float] = Field(None, ge=0)
    ai_decision: Optional[str] = None
    ai_reasoning: Optional[str] = None
    fraud_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    fraud_flags: Optional[list[str]] = None


class ClaimResponse(_BaseSchema):
    """Full claim representation returned by the API."""

    id: str
    claim_number: str
    patient_name: str
    patient_id: str
    patient_dob: Optional[str]
    provider_name: str
    provider_id: str
    claim_date: datetime
    diagnosis_codes: list[str]
    procedure_codes: list[str]
    amount_claimed: float
    amount_approved: Optional[float]
    status: ClaimStatusEnum
    fraud_score: Optional[float]
    fraud_flags: list[Any]
    ai_decision: Optional[str]
    ai_reasoning: Optional[str]
    created_at: datetime
    updated_at: datetime


class ClaimListResponse(_BaseSchema):
    """Paginated list of claims."""

    total: int
    page: int
    page_size: int
    items: list[ClaimResponse]


# ---------------------------------------------------------------------------
# Document schemas
# ---------------------------------------------------------------------------
class DocumentUpload(_BaseSchema):
    """Metadata returned after a document is uploaded."""

    id: str
    claim_id: str
    filename: str
    file_type: str
    file_size: int
    is_verified: bool
    created_at: datetime


class DocumentResponse(DocumentUpload):
    """Full document record including extracted text."""

    extracted_text: Optional[str] = None
    file_path: str


# ---------------------------------------------------------------------------
# Policy schemas
# ---------------------------------------------------------------------------
class PolicyCheckRequest(_BaseSchema):
    """Request to check whether a claim is covered by a policy."""

    policy_number: str = Field(..., description="Policy to validate against")
    patient_id: str = Field(..., description="Member ID on the policy")
    diagnosis_codes: list[str] = Field(default_factory=list)
    procedure_codes: list[str] = Field(default_factory=list)
    amount_claimed: float = Field(..., gt=0)


class PolicyCheckResponse(_BaseSchema):
    """Result of a policy coverage check."""

    policy_number: str
    is_covered: bool
    coverage_type: str
    amount_claimable: float = Field(description="Maximum payable after deductible and copay")
    deductible_applied: float
    copay_applied: float
    uncovered_procedures: list[str] = Field(default_factory=list)
    excluded_conditions_matched: list[str] = Field(default_factory=list)
    notes: str = ""


# ---------------------------------------------------------------------------
# Fraud analysis schemas
# ---------------------------------------------------------------------------
class FraudFlag(_BaseSchema):
    """A single fraud indicator."""

    flag: str = Field(description="Short code identifying the flag")
    description: str = Field(description="Human-readable explanation")
    severity: str = Field(default="medium", description="low / medium / high")


class FraudAnalysisResult(_BaseSchema):
    """Output of the fraud-detection agent for a claim."""

    claim_id: str
    fraud_score: float = Field(ge=0.0, le=1.0, description="Probability of fraud [0, 1]")
    is_fraud_suspected: bool
    flags: list[FraudFlag] = Field(default_factory=list)
    model_version: str = "1.0.0"
    analysed_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Agent decision schemas
# ---------------------------------------------------------------------------
class AgentDecision(_BaseSchema):
    """A structured decision produced by an AI agent."""

    agent_name: str = Field(description="Name of the agent that produced this decision")
    decision: str = Field(description="APPROVE / REJECT / ESCALATE / REVIEW")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the decision")
    reasoning: str = Field(description="Chain-of-thought or rule-based reasoning text")
    recommendations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# End-to-end claim processing schemas
# ---------------------------------------------------------------------------
class ClaimProcessingRequest(_BaseSchema):
    """Request to trigger the full AI agentic pipeline for a claim."""

    claim_id: str = Field(description="UUID of the claim to process")
    force_reprocess: bool = Field(
        default=False,
        description="Re-run even if the claim has already been processed",
    )
    agents_to_run: list[str] = Field(
        default_factory=lambda: ["document", "policy", "fraud", "decision"],
        description="Subset of agents to execute",
    )


class ClaimProcessingResponse(_BaseSchema):
    """Final output of the agentic claim-processing pipeline."""

    claim_id: str
    claim_number: str
    final_status: ClaimStatusEnum
    amount_approved: Optional[float]
    fraud_analysis: Optional[FraudAnalysisResult]
    policy_check: Optional[PolicyCheckResponse]
    agent_decisions: list[AgentDecision] = Field(default_factory=list)
    processing_time_ms: int = Field(description="Wall-clock time for the pipeline in milliseconds")
    processed_at: datetime = Field(default_factory=datetime.utcnow)
