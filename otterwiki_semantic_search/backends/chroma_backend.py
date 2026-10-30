"""ChromaDB vector store backend."""

import logging

from otterwiki_semantic_search.backends.base import VectorBackend

try:
    from chromadb.errors import NotFoundError
except ImportError:  # older chromadb without NotFoundError
    class NotFoundError(Exception):
        pass

log = logging.getLogger(__name__)

_BATCH_SIZE = 5000  # below Chroma's max batch size


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

    def _call(self, method, **kwargs):
        """Call a collection method, re-fetching the collection once if it
        no longer exists (e.g. another process deleted and recreated it)."""
        try:
            return getattr(self._collection, method)(**kwargs)
        except NotFoundError:
            log.warning("Collection %s not found, re-fetching", self._collection_name)
            self._collection = self._client.get_or_create_collection(
                self._collection_name
            )
            return getattr(self._collection, method)(**kwargs)

    def upsert(self, ids, texts, metadatas, embeddings=None):
        """Upsert chunks into ChromaDB.

        ChromaDB computes embeddings internally from the document texts,
        so the embeddings parameter is ignored.
        """
        self._call(
            "upsert",
            ids=ids,
            documents=texts,
            metadatas=metadatas,
        )

    def delete(self, where):
        """Delete chunks matching metadata filter."""
        self._call("delete", where=where)

    def query(self, query_texts=None, query_embeddings=None, n_results=5):
        """Search ChromaDB using query text (embedding done internally)."""
        kwargs = {"n_results": n_results}
        if query_texts is not None:
            kwargs["query_texts"] = query_texts
        elif query_embeddings is not None:
            kwargs["query_embeddings"] = query_embeddings
        return self._call("query", **kwargs)

    def count(self):
        return self._call("count")

    def reindex_atomic(self, chunks, embedding_fn=None):
        """Replace the collection's contents with chunks.

        Upserts the new chunks, then deletes IDs that are no longer present.
        The collection itself is never dropped, so its ID stays stable for
        other processes holding a handle to it, and searches never see an
        empty collection mid-reindex. ChromaDB embeds internally, so
        embedding_fn is ignored.
        """
        existing = set(self._call("get", include=[])["ids"])
        for i in range(0, len(chunks), _BATCH_SIZE):
            batch = chunks[i : i + _BATCH_SIZE]
            self.upsert(
                ids=[c["id"] for c in batch],
                texts=[c["text"] for c in batch],
                metadatas=[c["metadata"] for c in batch],
            )
        stale = list(existing - {c["id"] for c in chunks})
        for i in range(0, len(stale), _BATCH_SIZE):
            self._call("delete", ids=stale[i : i + _BATCH_SIZE])

    def reset(self):
        """Delete and recreate the collection."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            self._collection_name
        )
