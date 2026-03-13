"""Abstract interface for embedding functions."""

from abc import ABC, abstractmethod


class EmbeddingFunction(ABC):
    """Interface for text embedding functions.

    Implementations produce dense vector representations of text,
    suitable for similarity search via inner product or cosine distance.
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into vectors.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (each a list of floats).
            All vectors must have the same dimensionality.
        """

    @property
    @abstractmethod
    def dimensionality(self) -> int:
        """Return the dimensionality of the embedding vectors."""
