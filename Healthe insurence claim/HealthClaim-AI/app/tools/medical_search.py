"""
Medical Code Lookup Tool for validating ICD-10 and CPT codes.
Uses a local JSON database and Gemini LLM for validation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import google.generativeai as genai

from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Default path to the medical codes JSON database
_DEFAULT_CODES_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "data",
    "medical_documents",
    "sample_medical_codes.json",
)


class MedicalSearchTool:
    """Tool for validating and searching ICD-10/CPT medical codes."""

    def __init__(self, codes_json_path: str | None = None) -> None:
        """
        Initialize MedicalSearchTool.

        Args:
            codes_json_path: Path to the medical codes JSON file.
                             Falls back to the default data path.
        """
        self._codes_path = Path(codes_json_path or _DEFAULT_CODES_PATH).resolve()
        self._codes_db: dict[str, Any] | None = None
        self._model: genai.GenerativeModel | None = None
        logger.info("MedicalSearchTool initialized with codes_path='%s'", self._codes_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_codes_db(self) -> dict[str, Any]:
        """Load (or return cached) medical codes database."""
        if self._codes_db is None:
            if not self._codes_path.exists():
                logger.warning(
                    "Medical codes file not found at '%s'. Using empty DB.", self._codes_path
                )
                self._codes_db = {"icd10_codes": {}, "cpt_codes": {}}
            else:
                with open(self._codes_path, "r", encoding="utf-8") as fh:
                    self._codes_db = json.load(fh)
                logger.info("Loaded medical codes database from '%s'", self._codes_path)
        return self._codes_db

    def _get_gemini_model(self) -> genai.GenerativeModel:
        """Lazy-load Gemini model."""
        if self._model is None:
            genai.configure(api_key=settings.google_api_key)
            self._model = genai.GenerativeModel(settings.llm_model)
        return self._model

    def _ask_gemini(self, prompt: str) -> str:
        """Call Gemini and return text response, with error handling."""
        try:
            model = self._get_gemini_model()
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as exc:
            logger.error("Gemini API error: %s", exc, exc_info=True)
            return f"[Gemini unavailable: {exc}]"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_icd10_code(self, code: str) -> dict[str, Any]:
        """
        Validate an ICD-10 diagnosis code and return its description.

        Args:
            code: ICD-10 code string (e.g. 'J18.9').

        Returns:
            Dict with keys: valid (bool), code, description, category, notes.
        """
        code = code.strip().upper()
        db = self._load_codes_db()
        icd10_db: dict[str, Any] = db.get("icd10_codes", {})

        if code in icd10_db:
            entry = icd10_db[code]
            return {
                "valid": True,
                "code": code,
                "description": entry.get("description", ""),
                "category": entry.get("category", ""),
                "notes": entry.get("notes", ""),
                "source": "local_db",
            }

        # Fallback: ask Gemini
        logger.info("ICD-10 code '%s' not in local DB; querying Gemini", code)
        prompt = (
            f"You are a medical coding expert. Is '{code}' a valid ICD-10 diagnosis code? "
            "If yes, provide the official description and category. "
            "Respond in JSON with keys: valid (bool), description (str), category (str), notes (str)."
        )
        gemini_text = self._ask_gemini(prompt)
        try:
            # Extract JSON from response
            start = gemini_text.find("{")
            end = gemini_text.rfind("}") + 1
            parsed = json.loads(gemini_text[start:end]) if start >= 0 else {}
            parsed["code"] = code
            parsed["source"] = "gemini"
            return parsed
        except (json.JSONDecodeError, ValueError):
            return {
                "valid": False,
                "code": code,
                "description": gemini_text,
                "category": "",
                "notes": "Could not parse structured response",
                "source": "gemini",
            }

    def validate_cpt_code(self, code: str) -> dict[str, Any]:
        """
        Validate a CPT procedure code and return its description.

        Args:
            code: CPT code string (e.g. '99213').

        Returns:
            Dict with keys: valid (bool), code, description, category, notes.
        """
        code = code.strip().upper()
        db = self._load_codes_db()
        cpt_db: dict[str, Any] = db.get("cpt_codes", {})

        if code in cpt_db:
            entry = cpt_db[code]
            return {
                "valid": True,
                "code": code,
                "description": entry.get("description", ""),
                "category": entry.get("category", ""),
                "notes": entry.get("notes", ""),
                "source": "local_db",
            }

        # Fallback: ask Gemini
        logger.info("CPT code '%s' not in local DB; querying Gemini", code)
        prompt = (
            f"You are a medical coding expert. Is '{code}' a valid CPT procedure code? "
            "If yes, provide the official description and category. "
            "Respond in JSON with keys: valid (bool), description (str), category (str), notes (str)."
        )
        gemini_text = self._ask_gemini(prompt)
        try:
            start = gemini_text.find("{")
            end = gemini_text.rfind("}") + 1
            parsed = json.loads(gemini_text[start:end]) if start >= 0 else {}
            parsed["code"] = code
            parsed["source"] = "gemini"
            return parsed
        except (json.JSONDecodeError, ValueError):
            return {
                "valid": False,
                "code": code,
                "description": gemini_text,
                "category": "",
                "notes": "Could not parse structured response",
                "source": "gemini",
            }

    def get_related_procedures(self, diagnosis_code: str) -> list[str]:
        """
        Get a list of typical CPT procedure codes for a given ICD-10 diagnosis.

        Args:
            diagnosis_code: ICD-10 code (e.g. 'J18.9').

        Returns:
            List of typical CPT codes associated with this diagnosis.
        """
        diagnosis_code = diagnosis_code.strip().upper()
        db = self._load_codes_db()

        # Check local database for pre-defined mappings
        mappings: dict[str, list[str]] = db.get("diagnosis_procedure_mappings", {})
        if diagnosis_code in mappings:
            logger.info(
                "Found local mappings for diagnosis '%s': %s",
                diagnosis_code,
                mappings[diagnosis_code],
            )
            return mappings[diagnosis_code]

        # Fallback: ask Gemini
        logger.info(
            "No local mappings for diagnosis '%s'; querying Gemini", diagnosis_code
        )
        prompt = (
            f"Given ICD-10 diagnosis code '{diagnosis_code}', list up to 5 typical CPT procedure codes "
            "that would be performed for this condition. "
            "Respond with a JSON array of CPT code strings only, e.g. [\"99213\", \"71046\"]."
        )
        gemini_text = self._ask_gemini(prompt)
        try:
            start = gemini_text.find("[")
            end = gemini_text.rfind("]") + 1
            codes = json.loads(gemini_text[start:end]) if start >= 0 else []
            return [str(c).strip() for c in codes]
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                "Could not parse Gemini response for diagnosis '%s'", diagnosis_code
            )
            return []

    def check_code_compatibility(
        self, icd10_codes: list[str], cpt_codes: list[str]
    ) -> dict[str, Any]:
        """
        Check whether the given ICD-10 and CPT codes are medically compatible.

        Args:
            icd10_codes: List of ICD-10 diagnosis codes.
            cpt_codes: List of CPT procedure codes.

        Returns:
            Dict with keys: compatible (bool), confidence (float),
            explanation (str), warnings (list[str]), flags (list[str]).
        """
        if not icd10_codes or not cpt_codes:
            return {
                "compatible": False,
                "confidence": 0.0,
                "explanation": "Cannot check compatibility with empty code lists",
                "warnings": [],
                "flags": ["MISSING_CODES"],
            }

        # Validate all codes first
        icd_results = [self.validate_icd10_code(c) for c in icd10_codes]
        cpt_results = [self.validate_cpt_code(c) for c in cpt_codes]

        invalid_icd = [r["code"] for r in icd_results if not r.get("valid")]
        invalid_cpt = [r["code"] for r in cpt_results if not r.get("valid")]
        warnings: list[str] = []
        flags: list[str] = []

        if invalid_icd:
            warnings.append(f"Invalid ICD-10 codes: {', '.join(invalid_icd)}")
            flags.append("INVALID_DIAGNOSIS_CODES")
        if invalid_cpt:
            warnings.append(f"Invalid CPT codes: {', '.join(invalid_cpt)}")
            flags.append("INVALID_PROCEDURE_CODES")

        # Build descriptions for Gemini
        icd_desc = "; ".join(
            f"{r['code']} ({r.get('description', 'unknown')})" for r in icd_results
        )
        cpt_desc = "; ".join(
            f"{r['code']} ({r.get('description', 'unknown')})" for r in cpt_results
        )

        logger.info("Checking code compatibility via Gemini")
        prompt = (
            "You are a medical coding compliance expert. "
            f"Diagnoses: {icd_desc}. "
            f"Procedures: {cpt_desc}. "
            "Are these codes medically compatible and do the procedures make sense for the diagnoses? "
            "Respond in JSON with keys: compatible (bool), confidence (0-1 float), "
            "explanation (str), additional_warnings (list of str), additional_flags (list of str)."
        )
        gemini_text = self._ask_gemini(prompt)
        try:
            start = gemini_text.find("{")
            end = gemini_text.rfind("}") + 1
            parsed = json.loads(gemini_text[start:end]) if start >= 0 else {}
            warnings.extend(parsed.get("additional_warnings", []))
            flags.extend(parsed.get("additional_flags", []))
            return {
                "compatible": parsed.get("compatible", True),
                "confidence": float(parsed.get("confidence", 0.5)),
                "explanation": parsed.get("explanation", gemini_text),
                "warnings": warnings,
                "flags": flags,
            }
        except (json.JSONDecodeError, ValueError):
            logger.warning("Could not parse Gemini compatibility response")
            return {
                "compatible": len(invalid_icd) == 0 and len(invalid_cpt) == 0,
                "confidence": 0.4,
                "explanation": gemini_text,
                "warnings": warnings,
                "flags": flags,
            }
