"""
Policy Search Tool for semantic search over insurance policies.
Uses ChromaDB vector store for semantic retrieval.
"""

from __future__ import annotations

import json
from typing import Any

from app.rag.vectorstore import VectorStore
from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class PolicySearchTool:
    """Tool for searching insurance policies using semantic search and exact lookup."""

    def __init__(self) -> None:
        """Initialize the PolicySearchTool with a ChromaDB vector store."""
        self._vectorstore: VectorStore | None = None
        logger.info("PolicySearchTool initialized")

    @property
    def vectorstore(self) -> VectorStore:
        """Lazy-load the vector store."""
        if self._vectorstore is None:
            self._vectorstore = VectorStore(
                persist_dir=settings.chroma_persist_dir,
                collection_name="policies",
            )
        return self._vectorstore

    def search_policies(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """
        Perform semantic search over indexed insurance policy documents.

        Args:
            query: Natural language search query.
            top_k: Number of top results to return (default 3).

        Returns:
            List of dicts with policy details, coverage info, and relevance scores.
        """
        if not query or not query.strip():
            logger.warning("search_policies called with empty query")
            return []

        try:
            logger.info("Searching policies with query='%s', top_k=%d", query, top_k)
            results = self.vectorstore.similarity_search(query=query, top_k=top_k)
            enriched: list[dict[str, Any]] = []
            for doc in results:
                entry: dict[str, Any] = {
                    "text": doc.get("text", ""),
                    "score": doc.get("score", 0.0),
                    "policy_id": doc.get("metadata", {}).get("policy_id", ""),
                    "policy_number": doc.get("metadata", {}).get("policy_number", ""),
                    "coverage_type": doc.get("metadata", {}).get("coverage_type", ""),
                    "max_coverage": doc.get("metadata", {}).get("max_coverage", 0.0),
                    "deductible": doc.get("metadata", {}).get("deductible", 0.0),
                    "copay_percent": doc.get("metadata", {}).get("copay_percent", 0.0),
                    "excluded_conditions": json.loads(
                        doc.get("metadata", {}).get("excluded_conditions", "[]")
                    ),
                    "covered_procedures": json.loads(
                        doc.get("metadata", {}).get("covered_procedures", "[]")
                    ),
                    "metadata": doc.get("metadata", {}),
                }
                enriched.append(entry)
            logger.info("Policy search returned %d results", len(enriched))
            return enriched
        except Exception as exc:
            logger.error("Error during policy search: %s", exc, exc_info=True)
            return []

    def get_policy_by_id(self, policy_id: str) -> dict[str, Any]:
        """
        Retrieve a specific policy document by its unique ID.

        Args:
            policy_id: The unique policy identifier.

        Returns:
            Dict with full policy details, or empty dict if not found.
        """
        if not policy_id:
            logger.warning("get_policy_by_id called with empty policy_id")
            return {}

        try:
            logger.info("Looking up policy by id='%s'", policy_id)
            collection = self.vectorstore.client.get_collection(
                name=self.vectorstore.collection_name
            )
            result = collection.get(ids=[policy_id], include=["documents", "metadatas"])
            if not result["documents"]:
                logger.warning("Policy with id='%s' not found", policy_id)
                return {}

            metadata = result["metadatas"][0] if result["metadatas"] else {}
            return {
                "policy_id": policy_id,
                "text": result["documents"][0],
                "policy_number": metadata.get("policy_number", ""),
                "holder_name": metadata.get("holder_name", ""),
                "coverage_type": metadata.get("coverage_type", ""),
                "deductible": metadata.get("deductible", 0.0),
                "max_coverage": metadata.get("max_coverage", 0.0),
                "copay_percent": metadata.get("copay_percent", 0.0),
                "covered_procedures": json.loads(
                    metadata.get("covered_procedures", "[]")
                ),
                "excluded_conditions": json.loads(
                    metadata.get("excluded_conditions", "[]")
                ),
                "effective_date": metadata.get("effective_date", ""),
                "expiry_date": metadata.get("expiry_date", ""),
                "is_active": metadata.get("is_active", True),
                "metadata": metadata,
            }
        except Exception as exc:
            logger.error(
                "Error retrieving policy id='%s': %s", policy_id, exc, exc_info=True
            )
            return {}

    def get_coverage_for_procedure(
        self, procedure_code: str, policy_id: str
    ) -> dict[str, Any]:
        """
        Check whether a specific procedure is covered under a given policy.

        Args:
            procedure_code: CPT or similar procedure code to check.
            policy_id: The policy to check coverage against.

        Returns:
            Dict with keys: covered (bool), coverage_details, exclusion_reason,
            estimated_coverage_amount, copay_percent.
        """
        if not procedure_code or not policy_id:
            logger.warning(
                "get_coverage_for_procedure called with missing args: "
                "procedure_code='%s', policy_id='%s'",
                procedure_code,
                policy_id,
            )
            return {
                "covered": False,
                "coverage_details": "Invalid input parameters",
                "exclusion_reason": None,
                "estimated_coverage_amount": 0.0,
                "copay_percent": 0.0,
            }

        try:
            logger.info(
                "Checking coverage for procedure='%s' under policy='%s'",
                procedure_code,
                policy_id,
            )
            policy = self.get_policy_by_id(policy_id)
            if not policy:
                return {
                    "covered": False,
                    "coverage_details": f"Policy '{policy_id}' not found",
                    "exclusion_reason": "Policy not found",
                    "estimated_coverage_amount": 0.0,
                    "copay_percent": 0.0,
                }

            covered_procedures: list[str] = policy.get("covered_procedures", [])
            excluded_conditions: list[str] = policy.get("excluded_conditions", [])
            is_covered = any(
                procedure_code.upper() == cp.upper() for cp in covered_procedures
            )
            exclusion_reason: str | None = None
            if not is_covered:
                # Check if explicitly excluded
                if any(
                    procedure_code.upper() in ex.upper() for ex in excluded_conditions
                ):
                    exclusion_reason = "Procedure falls under excluded conditions"
                else:
                    exclusion_reason = "Procedure not in covered procedures list"

            copay_percent = float(policy.get("copay_percent", 0.0))
            max_coverage = float(policy.get("max_coverage", 0.0))
            # Simple estimate: copay applies to max coverage slot
            estimated_amount = max_coverage * (1.0 - copay_percent / 100.0) if is_covered else 0.0

            return {
                "covered": is_covered,
                "coverage_details": (
                    f"Procedure {procedure_code} is {'covered' if is_covered else 'not covered'} "
                    f"under policy {policy.get('policy_number', policy_id)} "
                    f"({policy.get('coverage_type', 'Unknown')} plan)"
                ),
                "exclusion_reason": exclusion_reason,
                "estimated_coverage_amount": round(estimated_amount, 2),
                "copay_percent": copay_percent,
                "max_coverage": max_coverage,
                "deductible": float(policy.get("deductible", 0.0)),
            }
        except Exception as exc:
            logger.error(
                "Error checking coverage for procedure='%s', policy='%s': %s",
                procedure_code,
                policy_id,
                exc,
                exc_info=True,
            )
            return {
                "covered": False,
                "coverage_details": f"Error checking coverage: {exc}",
                "exclusion_reason": str(exc),
                "estimated_coverage_amount": 0.0,
                "copay_percent": 0.0,
            }
