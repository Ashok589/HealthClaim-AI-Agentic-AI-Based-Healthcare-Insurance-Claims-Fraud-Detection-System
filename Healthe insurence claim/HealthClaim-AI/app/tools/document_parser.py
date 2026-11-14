"""
Document parsing tool supporting PDF, images, and plain text.
Extracts structured data using regex and Gemini LLM.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any

import google.generativeai as genai

from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Known image MIME types
_IMAGE_MIME_TYPES = {
    "image/png", "image/jpeg", "image/jpg", "image/tiff",
    "image/bmp", "image/gif", "image/webp",
}


class DocumentParser:
    """
    Multi-format document parser for insurance claim processing.

    Supports PDF, image (via OCR), and plain text files.
    Uses regex and Gemini LLM for structured data extraction.
    """

    def __init__(self) -> None:
        """Initialize DocumentParser with lazy-loaded Gemini model."""
        self._model: genai.GenerativeModel | None = None
        logger.info("DocumentParser initialized")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_gemini_model(self) -> genai.GenerativeModel:
        """Lazy-load Gemini model."""
        if self._model is None:
            genai.configure(api_key=settings.google_api_key)
            self._model = genai.GenerativeModel(settings.llm_model)
        return self._model

    def _ask_gemini(self, prompt: str) -> str:
        """Call Gemini and return text response with error handling."""
        try:
            model = self._get_gemini_model()
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as exc:
            logger.error("Gemini API error: %s", exc, exc_info=True)
            return f"[Gemini unavailable: {exc}]"

    # ------------------------------------------------------------------
    # File type detection
    # ------------------------------------------------------------------

    def detect_file_type(self, file_path: str) -> str:
        """
        Detect the type of a file based on extension and MIME type.

        Args:
            file_path: Path to the file.

        Returns:
            One of: 'pdf', 'image', 'text', 'unknown'.
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext == ".pdf":
            return "pdf"
        if ext in {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif", ".webp"}:
            return "image"
        if ext in {".txt", ".md", ".csv", ".json", ".xml", ".html"}:
            return "text"

        # Try MIME type
        mime_type, _ = mimetypes.guess_type(str(path))
        if mime_type:
            if mime_type == "application/pdf":
                return "pdf"
            if mime_type in _IMAGE_MIME_TYPES:
                return "image"
            if mime_type.startswith("text/"):
                return "text"

        logger.warning("Could not determine file type for '%s'", file_path)
        return "unknown"

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------

    def parse_pdf(self, file_path: str) -> str:
        """
        Extract text from a PDF file using pdfplumber.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Extracted text as a single string.
        """
        try:
            import pdfplumber  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "pdfplumber is not installed. Run: pip install pdfplumber"
            ) from exc

        path = Path(file_path)
        if not path.exists():
            logger.error("PDF file not found: '%s'", file_path)
            return ""

        try:
            logger.info("Parsing PDF: '%s'", file_path)
            text_parts: list[str] = []
            with pdfplumber.open(str(path)) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(f"[Page {page_num}]\n{page_text}")
            full_text = "\n\n".join(text_parts)
            logger.info("Extracted %d characters from PDF '%s'", len(full_text), file_path)
            return full_text
        except Exception as exc:
            logger.error("Error parsing PDF '%s': %s", file_path, exc, exc_info=True)
            return ""

    def parse_image(self, file_path: str) -> str:
        """
        Extract text from an image using pytesseract OCR.

        Args:
            file_path: Path to the image file.

        Returns:
            OCR-extracted text as a string.
        """
        try:
            from PIL import Image  # noqa: PLC0415
            import pytesseract  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "Pillow and pytesseract are required. "
                "Run: pip install Pillow pytesseract"
            ) from exc

        path = Path(file_path)
        if not path.exists():
            logger.error("Image file not found: '%s'", file_path)
            return ""

        try:
            logger.info("Running OCR on image: '%s'", file_path)
            img = Image.open(str(path))
            text = pytesseract.image_to_string(img, lang="eng")
            logger.info(
                "OCR extracted %d characters from '%s'", len(text), file_path
            )
            return text.strip()
        except Exception as exc:
            logger.error("Error parsing image '%s': %s", file_path, exc, exc_info=True)
            return ""

    def parse_text(self, file_path: str) -> str:
        """
        Read text from a plain text file.

        Args:
            file_path: Path to the text file.

        Returns:
            File contents as a string.
        """
        path = Path(file_path)
        if not path.exists():
            logger.error("Text file not found: '%s'", file_path)
            return ""

        try:
            logger.info("Reading text file: '%s'", file_path)
            with open(str(path), "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read()
            logger.info("Read %d characters from '%s'", len(content), file_path)
            return content
        except Exception as exc:
            logger.error("Error reading text file '%s': %s", file_path, exc, exc_info=True)
            return ""

    # ------------------------------------------------------------------
    # Structured data extraction
    # ------------------------------------------------------------------

    def extract_structured_data(
        self, text: str, doc_type: str = "claim"
    ) -> dict[str, Any]:
        """
        Extract structured fields from raw text using regex and Gemini.

        Args:
            text: Raw text from a parsed document.
            doc_type: Type of document ('claim', 'policy', 'medical_report', etc.).

        Returns:
            Dict of extracted fields (varies by doc_type).
        """
        if not text or not text.strip():
            return {}

        # Regex extraction for common fields
        extracted: dict[str, Any] = {}

        # Date patterns
        date_pattern = re.compile(
            r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b"
        )
        dates = date_pattern.findall(text)
        if dates:
            extracted["dates_found"] = dates[:5]

        # Dollar amounts
        amount_pattern = re.compile(r"\$\s*([\d,]+(?:\.\d{2})?)")
        amounts = [a.replace(",", "") for a in amount_pattern.findall(text)]
        if amounts:
            extracted["amounts_found"] = [float(a) for a in amounts[:10]]

        # ICD-10 codes (format: letter + digits + optional .digits)
        icd_pattern = re.compile(r"\b([A-Z]\d{2}(?:\.\d{1,4})?)\b")
        icd_codes = list(set(icd_pattern.findall(text)))
        if icd_codes:
            extracted["icd10_codes"] = icd_codes[:10]

        # CPT codes (5-digit numeric)
        cpt_pattern = re.compile(r"\b(\d{5})\b")
        cpt_codes = list(set(cpt_pattern.findall(text)))
        if cpt_codes:
            extracted["cpt_codes"] = cpt_codes[:10]

        # Policy number patterns
        policy_pattern = re.compile(
            r"[Pp]olicy\s*[Nn]o?\.?\s*[:#]?\s*([A-Z0-9-]{6,20})"
        )
        policy_match = policy_pattern.search(text)
        if policy_match:
            extracted["policy_number"] = policy_match.group(1)

        # Claim number
        claim_pattern = re.compile(
            r"[Cc]laim\s*[Nn]o?\.?\s*[:#]?\s*([A-Z0-9-]{6,20})"
        )
        claim_match = claim_pattern.search(text)
        if claim_match:
            extracted["claim_number"] = claim_match.group(1)

        # Patient/provider names (heuristic)
        name_pattern = re.compile(
            r"(?:[Pp]atient|[Ii]nsured)\s*[Nn]ame\s*[:#]?\s*([A-Z][a-z]+ [A-Z][a-z]+)"
        )
        name_match = name_pattern.search(text)
        if name_match:
            extracted["patient_name"] = name_match.group(1)

        # Now use Gemini to fill gaps and validate
        logger.info("Using Gemini to extract structured fields from %s document", doc_type)
        prompt = (
            f"You are an expert at extracting information from {doc_type} documents. "
            f"Extract the following fields from this text and return valid JSON:\n\n"
            f"Text (first 3000 chars):\n{text[:3000]}\n\n"
            "Required fields: patient_name, patient_id, provider_name, provider_id, "
            "claim_date, diagnosis_codes (array), procedure_codes (array), "
            "amount_claimed (number), policy_number, claim_number. "
            "Use null for missing fields. Return only the JSON object."
        )
        gemini_text = self._ask_gemini(prompt)
        try:
            start = gemini_text.find("{")
            end = gemini_text.rfind("}") + 1
            gemini_data = json.loads(gemini_text[start:end]) if start >= 0 else {}
            # Merge: Gemini fills missing regex fields
            for key, value in gemini_data.items():
                if key not in extracted and value is not None:
                    extracted[key] = value
        except (json.JSONDecodeError, ValueError):
            logger.warning("Could not parse Gemini structured extraction response")
            extracted["gemini_raw"] = gemini_text[:500]

        logger.info("Structured extraction complete, %d fields found", len(extracted))
        return extracted

    # ------------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------------

    def calculate_document_hash(self, file_path: str) -> str:
        """
        Calculate SHA-256 hash of a file for deduplication.

        Args:
            file_path: Path to the file.

        Returns:
            Hex string SHA-256 digest, or empty string on error.
        """
        path = Path(file_path)
        if not path.exists():
            logger.error("File not found for hashing: '%s'", file_path)
            return ""

        try:
            sha256 = hashlib.sha256()
            with open(str(path), "rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    sha256.update(chunk)
            digest = sha256.hexdigest()
            logger.debug("SHA-256 for '%s': %s", file_path, digest)
            return digest
        except Exception as exc:
            logger.error(
                "Error hashing file '%s': %s", file_path, exc, exc_info=True
            )
            return ""

    def parse_document(self, file_path: str) -> str:
        """
        Auto-detect file type and parse accordingly.

        Args:
            file_path: Path to the document.

        Returns:
            Extracted text string.
        """
        file_type = self.detect_file_type(file_path)
        if file_type == "pdf":
            return self.parse_pdf(file_path)
        elif file_type == "image":
            return self.parse_image(file_path)
        elif file_type == "text":
            return self.parse_text(file_path)
        else:
            logger.warning("Unsupported file type for '%s'", file_path)
            return ""
