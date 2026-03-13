"""Abstract interface for vector store backends."""

from abc import ABC, abstractmethod


class VectorBackend(ABC):
    """Interface for vector store backends.

    Backends manage storage, retrieval, and search of embedded document
    chunks. Each chunk has an ID, text, embedding vector, and metadata dict.
    """

    @abstractmethod
    def upsert(self, ids: list[str], texts: list[str], metadatas: list[dict],
               embeddings: list[list[float]] | None = None):
        """Insert or update chunks.

        Args:
            ids: Unique chunk IDs.
            texts: Chunk text content.
            metadatas: Per-chunk metadata dicts.
            embeddings: Pre-computed embedding vectors. If None, the backend
                is responsible for computing embeddings (e.g., ChromaDB does
                this internally).
        """

    @abstractmethod
    def delete(self, where: dict):
        """Delete chunks matching a metadata filter.

        Args:
            where: Metadata filter dict (e.g., {"page_path": "some/page"}).
        """

    @abstractmethod
    def query(self, query_texts: list[str] | None = None,
              query_embeddings: list[list[float]] | None = None,
              n_results: int = 5) -> dict:
        """Search for similar chunks.

        Provide either query_texts or query_embeddings, not both.

        Args:
            query_texts: Text queries to embed and search. Used by ChromaDB
                which handles embedding internally.
            query_embeddings: Pre-computed query embedding vectors. Used by
                FAISS where embedding is handled externally.
            n_results: Number of results to return.

        Returns:
            Dict with keys: ids, documents, metadatas, distances.
            Each value is a list of lists (one inner list per query).
        """

    @abstractmethod
    def count(self) -> int:
        """Return the number of chunks stored."""

    @abstractmethod
    def reset(self):
        """Delete all data and recreate the collection/index."""
