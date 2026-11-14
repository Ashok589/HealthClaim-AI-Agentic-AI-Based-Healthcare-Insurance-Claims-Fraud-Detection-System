"""
High-level RAG retriever interface for the HealthClaim AI system.
Provides context retrieval for policies, medical validation, and similar claims.
"""

from __future__ import annotations

from typing import Any

from app.rag.vectorstore import VectorStore
from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class RAGRetriever:
    """
    High-level retrieval interface that wraps VectorStore operations and
    formats results into LLM-ready context strings.
    """

    def __init__(self) -> None:
        """Initialize RAGRetriever with lazy-loaded vector stores."""
        self._policy_store: VectorStore | None = None
        self._medical_store: VectorStore | None = None
        self._claims_store: VectorStore | None = None
        logger.info("RAGRetriever initialized")

    # ------------------------------------------------------------------
    # Private store accessors
    # ------------------------------------------------------------------

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
    # Public API
    # ------------------------------------------------------------------

    def retrieve_policy_context(
        self, query: str, policy_id: str | None = None
    ) -> str:
        """
        Retrieve relevant policy context for a given query.

        Args:
            query: Natural language query about coverage, benefits, etc.
            policy_id: Optional policy ID to restrict search to a specific policy.

        Returns:
            Formatted context string ready for use in LLM prompts.
        """
        if not query or not query.strip():
            logger.warning("retrieve_policy_context called with empty query")
            return "No policy context available."

        try:
            store = self._get_policy_store()

            if policy_id:
                # Build a targeted query
                augmented_query = f"{query} policy_id:{policy_id}"
            else:
                augmented_query = query

            docs = store.similarity_search(query=augmented_query, top_k=5)

            # Filter by policy_id if specified
            if policy_id:
                docs = [
                    d for d in docs
                    if d.get("metadata", {}).get("policy_id") == policy_id
                ]

            if not docs:
                return f"No policy documents found matching query: '{query}'"

            return self.format_context(docs, section_title="Policy Context")
        except Exception as exc:
            logger.error(
                "Error retrieving policy context: %s", exc, exc_info=True
            )
            return f"Error retrieving policy context: {exc}"

    def retrieve_medical_context(
        self,
        diagnosis_codes: list[str],
        procedure_codes: list[str],
    ) -> str:
        """
        Retrieve medical validation context for given diagnosis and procedure codes.

        Args:
            diagnosis_codes: List of ICD-10 diagnosis codes.
            procedure_codes: List of CPT procedure codes.

        Returns:
            Formatted context string with descriptions for each code.
        """
        if not diagnosis_codes and not procedure_codes:
            logger.warning(
                "retrieve_medical_context called with no codes"
            )
            return "No medical codes provided for context retrieval."

        try:
            store = self._get_medical_store()
            all_docs: list[dict[str, Any]] = []

            # Retrieve context for each code
            all_codes = list(diagnosis_codes) + list(procedure_codes)
            for code in all_codes:
                query = f"medical code {code} diagnosis procedure description"
                docs = store.similarity_search(query=query, top_k=2)
                all_docs.extend(docs)

            # Deduplicate by id
            seen_ids: set[str] = set()
            unique_docs: list[dict[str, Any]] = []
            for doc in all_docs:
                doc_id = doc.get("id", "")
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    unique_docs.append(doc)

            if not unique_docs:
                return (
                    f"No medical context found for diagnosis codes: "
                    f"{diagnosis_codes} or procedure codes: {procedure_codes}"
                )

            return self.format_context(unique_docs, section_title="Medical Code Context")
        except Exception as exc:
            logger.error(
                "Error retrieving medical context: %s", exc, exc_info=True
            )
            return f"Error retrieving medical context: {exc}"

    def retrieve_similar_claims(
        self, claim_features: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Find historical claims similar to the given claim features.

        Args:
            claim_features: Dict with claim data (amounts, codes, provider, etc.).

        Returns:
            List of similar claim dicts with similarity scores.
        """
        if not claim_features:
            logger.warning("retrieve_similar_claims called with empty features")
            return []

        try:
            store = self._get_claims_store()

            # Build a text query from claim features
            query_parts: list[str] = []
            if "diagnosis_codes" in claim_features:
                query_parts.append(
                    f"diagnosis codes {' '.join(claim_features['diagnosis_codes'])}"
                )
            if "procedure_codes" in claim_features:
                query_parts.append(
                    f"procedure codes {' '.join(claim_features['procedure_codes'])}"
                )
            if "provider_id" in claim_features:
                query_parts.append(f"provider {claim_features['provider_id']}")
            if "amount_claimed" in claim_features:
                query_parts.append(f"amount claimed {claim_features['amount_claimed']}")

            query = " ".join(query_parts) if query_parts else "insurance claim"
            docs = store.similarity_search(query=query, top_k=5)
            logger.info(
                "Found %d similar historical claims", len(docs)
            )
            return docs
        except Exception as exc:
            logger.error(
                "Error retrieving similar claims: %s", exc, exc_info=True
            )
            return []

    def format_context(
        self,
        documents: list[dict[str, Any]],
        section_title: str = "Retrieved Context",
    ) -> str:
        """
        Format a list of retrieved documents into a readable LLM context string.

        Args:
            documents: List of dicts with 'text', 'score', and 'metadata' keys.
            section_title: Optional title for the context section.

        Returns:
            Formatted multi-line string.
        """
        if not documents:
            return f"[{section_title}]\nNo relevant documents found."

        lines: list[str] = [f"[{section_title}]", f"({len(documents)} relevant documents found)", ""]

        for idx, doc in enumerate(documents, start=1):
            text = doc.get("text", "").strip()
            score = doc.get("score", 0.0)
            metadata = doc.get("metadata", {})

            lines.append(f"--- Document {idx} (relevance: {score:.3f}) ---")

            # Include key metadata fields
            meta_highlights: list[str] = []
            for field in [
                "policy_number", "coverage_type", "code", "code_type",
                "claim_number", "status", "policy_id"
            ]:
                if field in metadata and metadata[field]:
                    meta_highlights.append(f"{field}: {metadata[field]}")
            if meta_highlights:
                lines.append("Metadata: " + " | ".join(meta_highlights))

            lines.append(text[:800])  # Truncate very long chunks
            lines.append("")

        return "\n".join(lines)
