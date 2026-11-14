"""
Document Agent
==============
Accepts uploaded documents (PDF, images, plain text), extracts raw text,
uses Google Gemini to parse structured medical/claim fields, and optionally
verifies document authenticity via consistency checks.
"""

import io
import json
import os
import re
from pathlib import Path
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

class DocumentExtractionResult(BaseModel):
    """Structured result returned by DocumentAgent."""
    success: bool
    file_path: str = ""
    file_type: str = ""
    raw_text: str = ""
    patient_name: Optional[str] = None
    patient_id: Optional[str] = None
    provider_name: Optional[str] = None
    provider_id: Optional[str] = None
    claim_date: Optional[str] = None
    service_dates: list[str] = Field(default_factory=list)
    diagnosis_codes: list[str] = Field(default_factory=list)
    procedure_codes: list[str] = Field(default_factory=list)
    amount_billed: Optional[float] = None
    amount_paid: Optional[float] = None
    is_authentic: bool = False
    confidence_scores: dict[str, float] = Field(default_factory=dict)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class DocumentAgent:
    """
    AI agent that extracts text from uploaded medical documents (PDF/image/text)
    and uses Gemini to parse structured claim fields with confidence scores.
    """

    _PARSE_PROMPT = (
        "You are a medical document parsing expert. "
        "Given the following extracted text from a medical/insurance document, "
        "return a JSON object with these exact keys:\n"
        "  patient_name, patient_id, provider_name, provider_id, claim_date,\n"
        "  service_dates (array of strings), diagnosis_codes (array),\n"
        "  procedure_codes (array), amount_billed (number or null),\n"
        "  amount_paid (number or null),\n"
        "  confidence_scores (dict mapping each key above to 0.0–1.0).\n"
        "Return ONLY the JSON, no markdown fences or commentary."
    )

    def __init__(self) -> None:
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self._model = genai.GenerativeModel(settings.LLM_MODEL)
        logger.info("DocumentAgent initialised with model=%s", settings.LLM_MODEL)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_text(self, file_path: str) -> str:
        """
        Extract raw text from a file.  Dispatches to the appropriate
        backend (pdfplumber, pytesseract, or plain read) based on extension.

        Parameters
        ----------
        file_path : str
            Absolute or relative path to the file.

        Returns
        -------
        str
            Extracted plain text.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        logger.info("Extracting text from %s (type=%s)", path.name, ext)

        if ext == ".pdf":
            return self._extract_pdf(path)
        elif ext in {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"}:
            return self._extract_image(path)
        else:
            return path.read_text(encoding="utf-8", errors="replace")

    def parse_document(self, extracted_text: str) -> DocumentExtractionResult:
        """
        Use Gemini to parse structured medical claim fields from extracted text.

        Parameters
        ----------
        extracted_text : str
            Raw text obtained from extract_text().

        Returns
        -------
        DocumentExtractionResult
        """
        if not extracted_text or not extracted_text.strip():
            return DocumentExtractionResult(
                success=False, error="Empty extracted text – nothing to parse."
            )

        prompt = f"{self._PARSE_PROMPT}\n\nDocument text:\n{extracted_text[:6000]}"
        try:
            response = self._model.generate_content(prompt)
            parsed = self._parse_json_response(response.text)
        except Exception as exc:
            logger.error("Gemini parsing error: %s", exc)
            return DocumentExtractionResult(success=False, error=str(exc))

        return DocumentExtractionResult(
            success=True,
            raw_text=extracted_text,
            patient_name=parsed.get("patient_name"),
            patient_id=parsed.get("patient_id"),
            provider_name=parsed.get("provider_name"),
            provider_id=parsed.get("provider_id"),
            claim_date=parsed.get("claim_date"),
            service_dates=parsed.get("service_dates") or [],
            diagnosis_codes=[c.upper().strip() for c in (parsed.get("diagnosis_codes") or [])],
            procedure_codes=[c.upper().strip() for c in (parsed.get("procedure_codes") or [])],
            amount_billed=self._safe_float(parsed.get("amount_billed")),
            amount_paid=self._safe_float(parsed.get("amount_paid")),
            confidence_scores=parsed.get("confidence_scores") or {},
        )

    def verify_document_authenticity(self, doc_data: DocumentExtractionResult) -> bool:
        """
        Perform heuristic authenticity checks on extracted document data.

        Returns True when no obvious inconsistencies are detected.

        Parameters
        ----------
        doc_data : DocumentExtractionResult
            Parsed document result to verify.
        """
        issues = []

        if not doc_data.patient_name:
            issues.append("Missing patient name")
        if not doc_data.provider_name:
            issues.append("Missing provider name")
        if doc_data.amount_billed is not None and doc_data.amount_billed <= 0:
            issues.append("Non-positive billed amount")
        if doc_data.amount_paid is not None and doc_data.amount_billed is not None:
            if doc_data.amount_paid > doc_data.amount_billed * 1.05:
                issues.append("Amount paid exceeds billed amount by >5%")

        # Confidence threshold check
        low_conf_fields = [
            k for k, v in doc_data.confidence_scores.items() if v < 0.5
        ]
        if len(low_conf_fields) > 3:
            issues.append(f"Low confidence in fields: {low_conf_fields}")

        if issues:
            logger.warning("Authenticity issues found: %s", issues)
            return False

        logger.info("Document passed authenticity checks")
        return True

    def process_file(self, file_path: str) -> DocumentExtractionResult:
        """
        End-to-end helper: extract text, parse fields, run authenticity check.
        """
        try:
            text = self.extract_text(file_path)
            result = self.parse_document(text)
            result.file_path = file_path
            result.file_type = Path(file_path).suffix.lower().lstrip(".")
            if result.success:
                result.is_authentic = self.verify_document_authenticity(result)
            return result
        except Exception as exc:
            logger.error("process_file failed for %s: %s", file_path, exc)
            return DocumentExtractionResult(
                success=False, file_path=file_path, error=str(exc)
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_pdf(self, path: Path) -> str:
        """Extract text from PDF using pdfplumber (preferred) or PyPDF2."""
        try:
            import pdfplumber  # type: ignore
            with pdfplumber.open(str(path)) as pdf:
                parts = [page.extract_text() or "" for page in pdf.pages]
            return "\n".join(parts)
        except ImportError:
            pass

        try:
            import PyPDF2  # type: ignore
            reader = PyPDF2.PdfReader(str(path))
            return "\n".join(
                page.extract_text() or "" for page in reader.pages
            )
        except ImportError:
            raise RuntimeError("Install pdfplumber or PyPDF2 to process PDFs.")

    def _extract_image(self, path: Path) -> str:
        """Extract text from an image using pytesseract OCR."""
        try:
            from PIL import Image  # type: ignore
            import pytesseract  # type: ignore
            img = Image.open(str(path))
            return pytesseract.image_to_string(img)
        except ImportError:
            raise RuntimeError(
                "Install Pillow and pytesseract (and Tesseract-OCR system package) "
                "to process images."
            )

    @staticmethod
    def _parse_json_response(text: str) -> dict:
        """Strip markdown fences and parse JSON from Gemini response."""
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Attempt to extract JSON object via regex
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                return json.loads(m.group())
            return {}

    @staticmethod
    def _safe_float(value) -> Optional[float]:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None
