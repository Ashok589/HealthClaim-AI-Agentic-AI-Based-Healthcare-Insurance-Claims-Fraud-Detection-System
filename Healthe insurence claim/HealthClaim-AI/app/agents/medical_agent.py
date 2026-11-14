"""
Medical Agent
=============
Validates the medical necessity of claimed procedures based on diagnosis
codes (ICD-10) and procedure codes (CPT).  Uses Google Gemini to reason
about clinical appropriateness and flags unusual combinations.
"""

import json
import re
from typing import Optional

import google.generativeai as genai
from pydantic import BaseModel, Field

from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

class MedicalValidationResult(BaseModel):
    """Structured result returned by MedicalAgent.validate_medical_necessity()."""
    is_medically_necessary: bool
    icd10_valid: bool
    cpt_valid: bool
    red_flags: list[str] = Field(default_factory=list)
    recommendations: str = ""
    code_analysis: dict[str, str] = Field(default_factory=dict)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class MedicalAgent:
    """
    AI agent that validates medical necessity and clinical appropriateness
    of insurance claims using ICD-10 and CPT code analysis via Gemini.
    """

    # Known ICD-10 format: letter + 2+ alphanumerics (e.g., J06.9, M54.5)
    _ICD10_PATTERN = re.compile(r"^[A-Z][0-9A-Z]{1,6}(\.[0-9A-Z]{1,4})?$", re.IGNORECASE)
    # CPT codes are 5-digit numeric (or with suffix for Category II/III)
    _CPT_PATTERN = re.compile(r"^\d{5}[A-Z0-9]?$")

    _SYSTEM_PROMPT = (
        "You are a clinical coding and medical necessity expert with deep knowledge "
        "of ICD-10 diagnosis codes and CPT procedure codes. "
        "Evaluate the following medical claim for:\n"
        "1. Whether ICD-10 codes are valid and accurately describe conditions.\n"
        "2. Whether CPT codes are appropriate and necessary for the given diagnoses.\n"
        "3. Any unusual or suspicious procedure-diagnosis combinations.\n"
        "4. Overall medical necessity.\n\n"
        "Respond ONLY with a JSON object using these exact keys:\n"
        "  is_medically_necessary (bool),\n"
        "  icd10_valid (bool),\n"
        "  cpt_valid (bool),\n"
        "  red_flags (array of strings describing any concerns),\n"
        "  code_analysis (dict mapping each code to a brief description/assessment),\n"
        "  recommendations (string with overall clinical assessment).\n"
        "Return ONLY the JSON, no markdown fences or extra commentary."
    )

    def __init__(self) -> None:
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self._model = genai.GenerativeModel(settings.LLM_MODEL)
        logger.info("MedicalAgent initialised with model=%s", settings.LLM_MODEL)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_medical_necessity(
        self,
        diagnosis_codes: list[str],
        procedure_codes: list[str],
        clinical_notes: str = "",
    ) -> MedicalValidationResult:
        """
        Validate medical necessity and code appropriateness.

        Parameters
        ----------
        diagnosis_codes : list[str]
            ICD-10 diagnosis codes from the claim.
        procedure_codes : list[str]
            CPT procedure codes from the claim.
        clinical_notes : str
            Free-text clinical notes (optional).

        Returns
        -------
        MedicalValidationResult
        """
        logger.info(
            "Validating medical necessity: %d ICD-10 codes, %d CPT codes",
            len(diagnosis_codes),
            len(procedure_codes),
        )

        # Basic format validation first (fast, no API call)
        format_flags: list[str] = []
        icd10_format_ok = self._validate_icd10_format(diagnosis_codes, format_flags)
        cpt_format_ok = self._validate_cpt_format(procedure_codes, format_flags)

        if not diagnosis_codes and not procedure_codes:
            return MedicalValidationResult(
                is_medically_necessary=False,
                icd10_valid=False,
                cpt_valid=False,
                red_flags=["No diagnosis or procedure codes provided"],
                recommendations="Claim cannot be evaluated without codes.",
            )

        # AI-based clinical assessment
        try:
            ai_result = self._ai_validate(diagnosis_codes, procedure_codes, clinical_notes)
        except Exception as exc:
            logger.error("MedicalAgent AI validation error: %s", exc, exc_info=True)
            return MedicalValidationResult(
                is_medically_necessary=False,
                icd10_valid=icd10_format_ok,
                cpt_valid=cpt_format_ok,
                red_flags=format_flags + [f"AI analysis unavailable: {exc}"],
                recommendations="Manual clinical review required.",
                error=str(exc),
            )

        # Merge format flags with AI red flags
        all_flags = format_flags + ai_result.get("red_flags", [])

        return MedicalValidationResult(
            is_medically_necessary=ai_result.get("is_medically_necessary", False),
            icd10_valid=ai_result.get("icd10_valid", icd10_format_ok),
            cpt_valid=ai_result.get("cpt_valid", cpt_format_ok),
            red_flags=all_flags,
            code_analysis=ai_result.get("code_analysis", {}),
            recommendations=ai_result.get("recommendations", ""),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ai_validate(
        self,
        diagnosis_codes: list[str],
        procedure_codes: list[str],
        clinical_notes: str,
    ) -> dict:
        """Call Gemini to perform clinical code analysis."""
        payload = {
            "diagnosis_codes": diagnosis_codes,
            "procedure_codes": procedure_codes,
            "clinical_notes": clinical_notes[:3000] if clinical_notes else "",
        }
        prompt = f"{self._SYSTEM_PROMPT}\n\nClaim data:\n{json.dumps(payload, indent=2)}"
        response = self._model.generate_content(prompt)
        text = response.text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return json.loads(text)

    def _validate_icd10_format(self, codes: list[str], flags: list[str]) -> bool:
        """Check ICD-10 code format; append flag messages for bad codes."""
        all_ok = True
        for code in codes:
            if not self._ICD10_PATTERN.match(code.strip()):
                flags.append(f"Possibly invalid ICD-10 format: '{code}'")
                all_ok = False
        return all_ok

    def _validate_cpt_format(self, codes: list[str], flags: list[str]) -> bool:
        """Check CPT code format; append flag messages for bad codes."""
        all_ok = True
        for code in codes:
            if not self._CPT_PATTERN.match(code.strip()):
                flags.append(f"Possibly invalid CPT format: '{code}'")
                all_ok = False
        return all_ok
