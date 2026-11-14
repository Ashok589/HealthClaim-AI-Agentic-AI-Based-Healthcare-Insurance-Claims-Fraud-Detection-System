"""
Claim Intake Agent
==================
Receives raw claim submission data, validates it using Pydantic,
normalizes/standardizes the data format, extracts key entities from
free-text descriptions, and returns a structured ClaimIntakeResult.
"""

import json
import re
from datetime import datetime
from typing import Any, Optional

import google.generativeai as genai
from pydantic import BaseModel, Field, field_validator

from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Pydantic result models
# ---------------------------------------------------------------------------

class PatientInfo(BaseModel):
    """Normalized patient information."""
    patient_name: str
    patient_id: str
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    contact: Optional[str] = None


class ProviderInfo(BaseModel):
    """Normalized provider information."""
    provider_name: str
    provider_id: str
    npi: Optional[str] = None
    specialty: Optional[str] = None
    address: Optional[str] = None


class ClaimIntakeResult(BaseModel):
    """Structured result returned by ClaimIntakeAgent.process_claim()."""
    success: bool
    claim_id: Optional[str] = None
    patient_info: Optional[PatientInfo] = None
    provider_info: Optional[ProviderInfo] = None
    claim_date: Optional[str] = None
    diagnosis_codes: list[str] = Field(default_factory=list)
    procedure_codes: list[str] = Field(default_factory=list)
    amount_claimed: Optional[float] = None
    policy_id: Optional[str] = None
    clinical_notes: Optional[str] = None
    validation_errors: list[str] = Field(default_factory=list)
    normalized_data: dict[str, Any] = Field(default_factory=dict)
    extracted_entities: dict[str, Any] = Field(default_factory=dict)
    processing_timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )


# ---------------------------------------------------------------------------
# Internal raw-claim validator
# ---------------------------------------------------------------------------

class _RawClaim(BaseModel):
    """Minimal Pydantic model to validate required raw fields."""
    patient_name: str = Field(..., min_length=2)
    patient_id: str = Field(..., min_length=1)
    provider_name: str = Field(..., min_length=2)
    provider_id: str = Field(..., min_length=1)
    amount_claimed: float = Field(..., gt=0)
    policy_id: str = Field(..., min_length=1)
    diagnosis_codes: list[str] = Field(default_factory=list)
    procedure_codes: list[str] = Field(default_factory=list)

    @field_validator("diagnosis_codes", "procedure_codes", mode="before")
    @classmethod
    def coerce_to_list(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [c.strip() for c in v.split(",") if c.strip()]
        return v or []

    @field_validator("amount_claimed", mode="before")
    @classmethod
    def coerce_amount(cls, v: Any) -> float:
        if isinstance(v, str):
            v = re.sub(r"[^\d.]", "", v)
        return float(v)


# ---------------------------------------------------------------------------
# Main agent class
# ---------------------------------------------------------------------------

class ClaimIntakeAgent:
    """
    AI agent that ingests raw claim data, validates it, normalises it, and
    uses Google Gemini to extract structured entities from free-text fields.
    """

    _SYSTEM_PROMPT = (
        "You are a medical insurance claim data extraction specialist. "
        "Given raw claim information (possibly messy or free-text), extract "
        "and return a valid JSON object with keys: patient_name, patient_id, "
        "date_of_birth, gender, provider_name, provider_id, npi, specialty, "
        "claim_date, diagnosis_codes (array), procedure_codes (array), "
        "amount_claimed (number), clinical_notes, extracted_entities (dict "
        "of any other relevant medical facts found). "
        "Return ONLY the JSON object, no extra text."
    )

    def __init__(self) -> None:
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self._model = genai.GenerativeModel(settings.LLM_MODEL)
        logger.info("ClaimIntakeAgent initialised with model=%s", settings.LLM_MODEL)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_claim(self, raw_data: dict) -> ClaimIntakeResult:
        """
        Validate, normalize, and enrich a raw claim payload.

        Parameters
        ----------
        raw_data : dict
            Arbitrary dict of claim fields submitted by the caller.

        Returns
        -------
        ClaimIntakeResult
            Structured result containing validated/enriched claim data or
            a list of validation errors.
        """
        logger.info("Processing claim for patient=%s", raw_data.get("patient_name", "?"))
        validation_errors: list[str] = []

        # 1. Validate required fields
        try:
            validated = _RawClaim(**raw_data)
        except Exception as exc:
            errors = self._extract_pydantic_errors(exc)
            logger.warning("Claim validation failed: %s", errors)
            return ClaimIntakeResult(success=False, validation_errors=errors)

        # 2. Normalize codes to uppercase stripped strings
        diagnosis_codes = [c.upper().strip() for c in validated.diagnosis_codes]
        procedure_codes = [c.upper().strip() for c in validated.procedure_codes]

        # 3. Use Gemini to extract/enrich from any free-text clinical notes
        extracted_entities: dict[str, Any] = {}
        clinical_notes = raw_data.get("clinical_notes") or raw_data.get("notes", "")
        if clinical_notes:
            try:
                extracted_entities = self._extract_entities_with_gemini(
                    raw_data, clinical_notes
                )
                # Merge extracted codes if any
                if "diagnosis_codes" in extracted_entities:
                    extras = [
                        c.upper().strip()
                        for c in extracted_entities.pop("diagnosis_codes", [])
                        if c.upper().strip() not in diagnosis_codes
                    ]
                    diagnosis_codes.extend(extras)
                if "procedure_codes" in extracted_entities:
                    extras = [
                        c.upper().strip()
                        for c in extracted_entities.pop("procedure_codes", [])
                        if c.upper().strip() not in procedure_codes
                    ]
                    procedure_codes.extend(extras)
            except Exception as exc:
                logger.warning("Gemini extraction failed (non-fatal): %s", exc)
                validation_errors.append(f"AI entity extraction warning: {exc}")

        # 4. Build normalized data snapshot
        normalized_data = {
            "patient_name": validated.patient_name.title().strip(),
            "patient_id": validated.patient_id.strip(),
            "provider_name": validated.provider_name.title().strip(),
            "provider_id": validated.provider_id.strip(),
            "amount_claimed": round(validated.amount_claimed, 2),
            "policy_id": validated.policy_id.strip(),
            "diagnosis_codes": diagnosis_codes,
            "procedure_codes": procedure_codes,
            "claim_date": raw_data.get("claim_date") or datetime.utcnow().strftime("%Y-%m-%d"),
        }

        return ClaimIntakeResult(
            success=True,
            patient_info=PatientInfo(
                patient_name=normalized_data["patient_name"],
                patient_id=normalized_data["patient_id"],
                date_of_birth=raw_data.get("date_of_birth"),
                gender=raw_data.get("gender"),
                contact=raw_data.get("contact"),
            ),
            provider_info=ProviderInfo(
                provider_name=normalized_data["provider_name"],
                provider_id=normalized_data["provider_id"],
                npi=raw_data.get("npi"),
                specialty=raw_data.get("specialty") or extracted_entities.get("specialty"),
            ),
            claim_date=normalized_data["claim_date"],
            diagnosis_codes=diagnosis_codes,
            procedure_codes=procedure_codes,
            amount_claimed=normalized_data["amount_claimed"],
            policy_id=normalized_data["policy_id"],
            clinical_notes=clinical_notes or None,
            validation_errors=validation_errors,
            normalized_data=normalized_data,
            extracted_entities=extracted_entities,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_entities_with_gemini(
        self, raw_data: dict, clinical_notes: str
    ) -> dict[str, Any]:
        """Call Gemini to extract structured entities from free-text."""
        prompt = (
            f"{self._SYSTEM_PROMPT}\n\n"
            f"Raw claim data:\n{json.dumps(raw_data, indent=2)}\n\n"
            f"Clinical notes:\n{clinical_notes}"
        )
        response = self._model.generate_content(prompt)
        text = response.text.strip()
        # Strip markdown fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return json.loads(text)

    @staticmethod
    def _extract_pydantic_errors(exc: Exception) -> list[str]:
        """Convert a Pydantic ValidationError into a list of human-readable strings."""
        try:
            return [f"{'.'.join(str(l) for l in e['loc'])}: {e['msg']}" for e in exc.errors()]
        except Exception:
            return [str(exc)]
