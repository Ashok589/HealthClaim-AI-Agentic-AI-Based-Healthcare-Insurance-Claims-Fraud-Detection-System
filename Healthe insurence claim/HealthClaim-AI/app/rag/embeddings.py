"""
Embedding generation using sentence-transformers for RAG pipeline.
Supports lazy loading and LRU caching for efficiency.
"""

from __future__ import annotations

import functools
import hashlib
from typing import TYPE_CHECKING, Any

from app.utils.logger import get_logger

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)

_DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingGenerator:
    """
    Generates text embeddings using sentence-transformers.

    Features:
    - Lazy model loading on first use
    - LRU cache for repeated single-query embeddings
    - Batch processing support
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL_NAME) -> None:
        """
        Initialize the EmbeddingGenerator.

        Args:
            model_name: Sentence-transformers model name. Defaults to all-MiniLM-L6-v2.
        """
        self._model_name = model_name
        self._model: SentenceTransformer | None = None
        logger.info("EmbeddingGenerator configured with model='%s'", model_name)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> "SentenceTransformer":
        """Lazy-load the sentence-transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # noqa: PLC0415

                logger.info(
                    "Loading sentence-transformer model '%s'...", self._model_name
                )
                self._model = SentenceTransformer(self._model_name)
                logger.info("Model '%s' loaded successfully", self._model_name)
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is not installed. "
                    "Run: pip install sentence-transformers"
                ) from exc
        return self._model

    @staticmethod
    def _text_hash(text: str) -> str:
        """Return SHA-256 hex digest of text for cache keying."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @functools.lru_cache(maxsize=1024)
    def _cached_embedding(self, text_hash: str, text: str) -> tuple[float, ...]:
        """
        Internal cached embedding call. Returns a tuple (hashable) for lru_cache.

        Args:
            text_hash: SHA-256 hash of text (used as cache key alongside text).
            text: The original text.

        Returns:
            Tuple of floats representing the embedding.
        """
        model = self._load_model()
        embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return tuple(float(v) for v in embedding)

    def generate_embedding(self, text: str) -> list[float]:
        """
        Generate a single text embedding, using cache for repeated queries.

        Args:
            text: Input text to embed.

        Returns:
            List of floats representing the embedding vector.
        """
        if not text or not text.strip():
            logger.warning("generate_embedding called with empty text")
            return []

        text = text.strip()
        try:
            text_hash = self._text_hash(text)
            embedding_tuple = self._cached_embedding(text_hash, text)
            return list(embedding_tuple)
        except Exception as exc:
            logger.error("Error generating embedding: %s", exc, exc_info=True)
            return []

    def generate_batch_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of input strings to embed.

        Returns:
            List of embedding vectors (each a list of floats).
        """
        if not texts:
            logger.warning("generate_batch_embeddings called with empty list")
            return []

        # Filter and track empty strings
        valid_texts = [t.strip() for t in texts]
        try:
            model = self._load_model()
            logger.info("Generating batch embeddings for %d texts", len(valid_texts))
            embeddings = model.encode(
                valid_texts,
                batch_size=32,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            result = []
            for idx, emb in enumerate(embeddings):
                vec = [float(v) for v in emb]
                result.append(vec)
                # Populate the LRU cache for individual texts
                h = self._text_hash(valid_texts[idx])
                try:
                    self._cached_embedding.__wrapped__(self, h, valid_texts[idx])  # type: ignore[attr-defined]
                except Exception:
                    pass
            logger.info("Batch embedding complete, %d vectors generated", len(result))
            return result
        except Exception as exc:
            logger.error("Error generating batch embeddings: %s", exc, exc_info=True)
            return [[] for _ in texts]

    def get_embedding_dimension(self) -> int:
        """Return the dimension of embeddings produced by the current model."""
        try:
            model = self._load_model()
            return model.get_sentence_embedding_dimension()  # type: ignore[return-value]
        except Exception as exc:
            logger.error("Could not determine embedding dimension: %s", exc)
            return 384  # Default for all-MiniLM-L6-v2

    def clear_cache(self) -> None:
        """Clear the LRU embedding cache."""
        self._cached_embedding.cache_clear()
        logger.info("Embedding cache cleared")
