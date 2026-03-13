"""Sentence-transformer embedding adapter for local use."""

import logging

from otterwiki_semantic_search.embeddings.base import EmbeddingFunction

log = logging.getLogger(__name__)

# all-MiniLM-L6-v2 produces 384-dimensional vectors
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    """Embedding function using sentence-transformers (runs locally).

    Uses the all-MiniLM-L6-v2 model by default, which produces
    384-dimensional normalized vectors.
    """

    def __init__(self, model_name=DEFAULT_MODEL):
        """Initialize the sentence-transformer model.

        Args:
            model_name: HuggingFace model name. Defaults to all-MiniLM-L6-v2.
        """
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._dimensionality = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using the sentence-transformer model.

        Returns normalized vectors suitable for inner product similarity.
        """
        embeddings = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return embeddings.tolist()

    @property
    def dimensionality(self) -> int:
        return self._dimensionality
