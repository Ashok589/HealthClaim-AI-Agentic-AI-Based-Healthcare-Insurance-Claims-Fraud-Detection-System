"""
ChromaDB vector store wrapper for the RAG pipeline.
Provides a unified interface for indexing and retrieving documents.
"""

from __future__ import annotations

import os
from typing import Any

from app.rag.embeddings import EmbeddingGenerator
from app.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    """
    Wrapper around ChromaDB providing document storage and similarity search.

    Supports three collections: 'policies', 'medical_codes', 'claim_history'.
    """

    SUPPORTED_COLLECTIONS = {"policies", "medical_codes", "claim_history"}

    def __init__(self, persist_dir: str, collection_name: str) -> None:
        """
        Initialize the VectorStore with a ChromaDB backend.

        Args:
            persist_dir: Directory where ChromaDB will persist data.
            collection_name: Name of the ChromaDB collection to use.
        """
        try:
            import chromadb  # noqa: PLC0415
            from chromadb.config import Settings  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "chromadb is not installed. Run: pip install chromadb"
            ) from exc

        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._embedding_generator = EmbeddingGenerator()

        os.makedirs(persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "VectorStore initialized: collection='%s', persist_dir='%s'",
            collection_name,
            persist_dir,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def collection(self):
        """Return the underlying ChromaDB collection."""
        return self._collection

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_documents(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]],
        ids: list[str],
    ) -> None:
        """
        Add or upsert documents into the vector store.

        Args:
            texts: List of document texts to index.
            metadatas: List of metadata dicts corresponding to each text.
            ids: List of unique string IDs for each document.
        """
        if not texts:
            logger.warning("add_documents called with empty texts list")
            return
        if not (len(texts) == len(metadatas) == len(ids)):
            raise ValueError(
                f"texts ({len(texts)}), metadatas ({len(metadatas)}), "
                f"and ids ({len(ids)}) must have the same length"
            )

        try:
            logger.info(
                "Adding %d documents to collection '%s'", len(texts), self.collection_name
            )
            embeddings = self._embedding_generator.generate_batch_embeddings(texts)

            # Sanitize metadatas: ChromaDB only allows str/int/float/bool values
            sanitized_meta: list[dict[str, Any]] = []
            for meta in metadatas:
                clean: dict[str, Any] = {}
                for k, v in meta.items():
                    if isinstance(v, (str, int, float, bool)):
                        clean[k] = v
                    else:
                        import json  # noqa: PLC0415
                        clean[k] = json.dumps(v)
                sanitized_meta.append(clean)

            self._collection.upsert(
                documents=texts,
                embeddings=embeddings if embeddings and all(e for e in embeddings) else None,
                metadatas=sanitized_meta,
                ids=ids,
            )
            logger.info(
                "Successfully upserted %d documents into '%s'",
                len(texts),
                self.collection_name,
            )
        except Exception as exc:
            logger.error(
                "Error adding documents to '%s': %s", self.collection_name, exc, exc_info=True
            )
            raise

    def similarity_search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Perform semantic similarity search over the collection.

        Args:
            query: Query text string.
            top_k: Number of top results to return.

        Returns:
            List of dicts with keys: id, text, score, metadata.
        """
        if not query or not query.strip():
            logger.warning("similarity_search called with empty query")
            return []

        try:
            logger.info(
                "Similarity search in '%s': query='%s', top_k=%d",
                self.collection_name,
                query[:80],
                top_k,
            )
            query_embedding = self._embedding_generator.generate_embedding(query)
            query_params: dict[str, Any] = {
                "n_results": top_k,
                "include": ["documents", "metadatas", "distances"],
            }
            if query_embedding:
                query_params["query_embeddings"] = [query_embedding]
            else:
                query_params["query_texts"] = [query]

            results = self._collection.query(**query_params)

            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            ids = results.get("ids", [[]])[0]

            output: list[dict[str, Any]] = []
            for doc_id, text, meta, dist in zip(ids, documents, metadatas, distances):
                # Convert cosine distance to similarity score (0-1)
                score = max(0.0, 1.0 - dist)
                output.append(
                    {
                        "id": doc_id,
                        "text": text,
                        "metadata": meta,
                        "score": round(score, 4),
                    }
                )
            logger.info(
                "Similarity search returned %d results from '%s'",
                len(output),
                self.collection_name,
            )
            return output
        except Exception as exc:
            logger.error(
                "Error during similarity search in '%s': %s",
                self.collection_name,
                exc,
                exc_info=True,
            )
            return []

    def delete_collection(self) -> None:
        """Delete the entire collection from ChromaDB."""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info("Deleted collection '%s'", self.collection_name)
        except Exception as exc:
            logger.error(
                "Error deleting collection '%s': %s", self.collection_name, exc, exc_info=True
            )
            raise

    def get_collection_stats(self) -> dict[str, Any]:
        """
        Return statistics about the current collection.

        Returns:
            Dict with keys: collection_name, document_count, persist_dir.
        """
        try:
            count = self._collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_dir": self.persist_dir,
            }
        except Exception as exc:
            logger.error(
                "Error getting stats for collection '%s': %s",
                self.collection_name,
                exc,
                exc_info=True,
            )
            return {
                "collection_name": self.collection_name,
                "document_count": -1,
                "persist_dir": self.persist_dir,
                "error": str(exc),
            }
