"""
RAG data ingestion pipeline for policies, medical codes, and claim history.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from app.rag.vectorstore import VectorStore
from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class RAGIngestionPipeline:
    """
    Handles ingestion of documents into ChromaDB collections for RAG retrieval.

    Supports policies, medical codes, and historical claims.
    """

    def __init__(self) -> None:
        """Initialize the ingestion pipeline with vectorstore connections."""
        self._policy_store: VectorStore | None = None
        self._medical_store: VectorStore | None = None
        self._claims_store: VectorStore | None = None
        logger.info("RAGIngestionPipeline initialized")

    def _get_policy_store(self) -> VectorStore:
        if self._policy_store is None:
            self._policy_store = VectorStore(
                persist_dir=settings.chroma_persist_dir,
                collection_name="policies",
            )
        return self._policy_store

    def _get_medical_store(self) -> VectorStore:
        if self._medical_store is None:
            self._medical_store = VectorStore(
                persist_dir=settings.chroma_persist_dir,
                collection_name="medical_codes",
            )
        return self._medical_store

    def _get_claims_store(self) -> VectorStore:
        if self._claims_store is None:
            self._claims_store = VectorStore(
                persist_dir=settings.chroma_persist_dir,
                collection_name="claim_history",
            )
        return self._claims_store

    # ------------------------------------------------------------------
    # Text utilities
    # ------------------------------------------------------------------

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> list[str]:
        """
        Split text into overlapping chunks for indexing.

        Tries to split on sentence boundaries first, then falls back
        to character-level chunking.

        Args:
            text: Input text to chunk.
            chunk_size: Target chunk size in characters.
            overlap: Number of overlapping characters between chunks.

        Returns:
            List of text chunks.
        """
        if not text or not text.strip():
            return []

        text = text.strip()

        # Split on sentence boundaries
        import re  # noqa: PLC0415
        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks: list[str] = []
        current_chunk: list[str] = []
        current_len = 0

        for sentence in sentences:
            sentence_len = len(sentence)
            if current_len + sentence_len > chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                # Overlap: keep last few sentences
                overlap_text = chunk_text[-overlap:] if overlap else ""
                current_chunk = [overlap_text] if overlap_text else []
                current_len = len(overlap_text)

            current_chunk.append(sentence)
            current_len += sentence_len + 1  # +1 for space

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        logger.debug("Text chunked into %d chunks (chunk_size=%d)", len(chunks), chunk_size)
        return [c for c in chunks if c.strip()]

    def process_json_document(self, data: dict[str, Any]) -> str:
        """
        Convert a JSON object to a searchable text representation.

        Args:
            data: JSON/dict object to convert.

        Returns:
            Human-readable text representation.
        """

        def flatten(obj: Any, prefix: str = "") -> list[str]:
            parts: list[str] = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
                    parts.extend(flatten(v, key))
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    parts.extend(flatten(item, f"{prefix}[{i}]"))
            else:
                parts.append(f"{prefix}: {obj}")
            return parts

        lines = flatten(data)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Ingestion methods
    # ------------------------------------------------------------------

    def ingest_policy_documents(self, directory: str) -> int:
        """
        Load and index all JSON policy documents from a directory.

        Args:
            directory: Path to directory containing policy JSON files.

        Returns:
            Number of documents successfully indexed.
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.warning("Policy documents directory not found: '%s'", directory)
            return 0

        store = self._get_policy_store()
        count = 0
        json_files = list(dir_path.glob("*.json"))
        logger.info("Found %d JSON files in '%s'", len(json_files), directory)

        for json_file in json_files:
            try:
                with open(str(json_file), "r", encoding="utf-8") as fh:
                    data = json.load(fh)

                # Handle list or single dict
                policies = data if isinstance(data, list) else [data]

                texts: list[str] = []
                metadatas: list[dict[str, Any]] = []
                ids: list[str] = []

                for idx, policy in enumerate(policies):
                    policy_text = self.process_json_document(policy)
                    policy_id = str(
                        policy.get("id", policy.get("policy_number", f"{json_file.stem}_{idx}"))
                    )

                    # Chunk large policy documents
                    chunks = self.chunk_text(policy_text, chunk_size=800, overlap=100)
                    for chunk_idx, chunk in enumerate(chunks):
                        texts.append(chunk)
                        metadatas.append(
                            {
                                "policy_id": policy_id,
                                "policy_number": str(policy.get("policy_number", "")),
                                "holder_name": str(policy.get("holder_name", "")),
                                "coverage_type": str(policy.get("coverage_type", "")),
                                "deductible": float(policy.get("deductible", 0)),
                                "max_coverage": float(policy.get("max_coverage", 0)),
                                "copay_percent": float(policy.get("copay_percent", 0)),
                                "covered_procedures": json.dumps(
                                    policy.get("covered_procedures", [])
                                ),
                                "excluded_conditions": json.dumps(
                                    policy.get("excluded_conditions", [])
                                ),
                                "effective_date": str(policy.get("effective_date", "")),
                                "expiry_date": str(policy.get("expiry_date", "")),
                                "is_active": bool(policy.get("is_active", True)),
                                "source_file": json_file.name,
                                "chunk_index": chunk_idx,
                            }
                        )
                        ids.append(f"{policy_id}_chunk_{chunk_idx}")

                if texts:
                    store.add_documents(texts=texts, metadatas=metadatas, ids=ids)
                    count += len(policies)
                    logger.info(
                        "Indexed %d policies from '%s'", len(policies), json_file.name
                    )
            except Exception as exc:
                logger.error(
                    "Error ingesting policy file '%s': %s", json_file, exc, exc_info=True
                )

        logger.info("Policy ingestion complete: %d policies indexed", count)
        return count

    def ingest_medical_codes(self, json_path: str) -> int:
        """
        Load and index the medical code database (ICD-10 and CPT codes).

        Args:
            json_path: Path to the medical codes JSON file.

        Returns:
            Number of codes indexed.
        """
        path = Path(json_path)
        if not path.exists():
            logger.warning("Medical codes file not found: '%s'", json_path)
            return 0

        try:
            with open(str(path), "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            logger.error("Error reading medical codes file: %s", exc, exc_info=True)
            return 0

        store = self._get_medical_store()
        texts: list[str] = []
        metadatas: list[dict[str, Any]] = []
        ids: list[str] = []

        icd10_codes: dict[str, Any] = data.get("icd10_codes", {})
        cpt_codes: dict[str, Any] = data.get("cpt_codes", {})

        for code, entry in icd10_codes.items():
            text = (
                f"ICD-10 Code: {code}\n"
                f"Description: {entry.get('description', '')}\n"
                f"Category: {entry.get('category', '')}\n"
                f"Notes: {entry.get('notes', '')}"
            )
            texts.append(text)
            metadatas.append(
                {
                    "code": code,
                    "code_type": "ICD-10",
                    "description": str(entry.get("description", "")),
                    "category": str(entry.get("category", "")),
                }
            )
            ids.append(f"icd10_{code}")

        for code, entry in cpt_codes.items():
            text = (
                f"CPT Code: {code}\n"
                f"Description: {entry.get('description', '')}\n"
                f"Category: {entry.get('category', '')}\n"
                f"Notes: {entry.get('notes', '')}"
            )
            texts.append(text)
            metadatas.append(
                {
                    "code": code,
                    "code_type": "CPT",
                    "description": str(entry.get("description", "")),
                    "category": str(entry.get("category", "")),
                }
            )
            ids.append(f"cpt_{code}")

        if texts:
            store.add_documents(texts=texts, metadatas=metadatas, ids=ids)

        total = len(texts)
        logger.info("Medical codes ingestion complete: %d codes indexed", total)
        return total

    def ingest_claim_history(self, claims: list[dict[str, Any]]) -> int:
        """
        Index historical claims for similarity search.

        Args:
            claims: List of historical claim dicts.

        Returns:
            Number of claims indexed.
        """
        if not claims:
            logger.warning("ingest_claim_history called with empty claims list")
            return 0

        store = self._get_claims_store()
        texts: list[str] = []
        metadatas: list[dict[str, Any]] = []
        ids: list[str] = []

        for claim in claims:
            claim_text = self.process_json_document(claim)
            claim_id = str(claim.get("id", claim.get("claim_number", f"claim_{len(texts)}")))
            texts.append(claim_text)
            metadatas.append(
                {
                    "claim_id": claim_id,
                    "claim_number": str(claim.get("claim_number", "")),
                    "status": str(claim.get("status", "")),
                    "amount_claimed": float(claim.get("amount_claimed", 0)),
                    "is_fraud": bool(claim.get("is_fraud", False)),
                    "fraud_score": float(claim.get("fraud_score", 0.0)),
                    "provider_id": str(claim.get("provider_id", "")),
                }
            )
            ids.append(f"claim_{claim_id}")

        store.add_documents(texts=texts, metadatas=metadatas, ids=ids)
        logger.info("Claim history ingestion complete: %d claims indexed", len(texts))
        return len(texts)
