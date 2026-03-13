"""ChromaDB vector store backend."""

import logging

from otterwiki_semantic_search.backends.base import VectorBackend

log = logging.getLogger(__name__)


class ChromaBackend(VectorBackend):
    """Vector store backend using ChromaDB.

    ChromaDB handles embedding internally using its default sentence-transformer
    model (all-MiniLM-L6-v2). Callers pass text; ChromaDB computes embeddings.
    """

    def __init__(self, client, collection_name="otterwiki_pages"):
        """Initialize with a ChromaDB client.

        Args:
            client: A chromadb.Client instance (PersistentClient or HttpClient).
            collection_name: Name of the ChromaDB collection.
        """
        self._client = client
        self._collection_name = collection_name
        self._collection = client.get_or_create_collection(collection_name)

    @property
    def collection(self):
        return self._collection

    @property
    def client(self):
        return self._client

    def upsert(self, ids, texts, metadatas, embeddings=None):
        """Upsert chunks into ChromaDB.

        ChromaDB computes embeddings internally from the document texts,
        so the embeddings parameter is ignored.
        """
        self._collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
        )

    def delete(self, where):
        """Delete chunks matching metadata filter."""
        self._collection.delete(where=where)

    def query(self, query_texts=None, query_embeddings=None, n_results=5):
        """Search ChromaDB using query text (embedding done internally)."""
        kwargs = {"n_results": n_results}
        if query_texts is not None:
            kwargs["query_texts"] = query_texts
        elif query_embeddings is not None:
            kwargs["query_embeddings"] = query_embeddings
        return self._collection.query(**kwargs)

    def count(self):
        return self._collection.count()

    def reset(self):
        """Delete and recreate the collection."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            self._collection_name
        )
