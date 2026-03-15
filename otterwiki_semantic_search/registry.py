"""Per-wiki FAISS backend registry for multi-tenant deployments.

Maps wiki slugs to FAISSBackend instances. Each wiki gets its own
subdirectory under the configured FAISS_INDEX_DIR base path.

Slug is derived from the storage path of the current request's wiki:
    os.path.basename(storage.path)  e.g. /srv/wikis/dev -> "dev"
"""

import logging
import os
import threading

log = logging.getLogger(__name__)


class BackendRegistry:
    """Thread-safe registry mapping wiki slug -> FAISSBackend.

    Backends are lazily created on first access. All backends share
    the same embedding function.
    """

    def __init__(self, base_dir, embedding_fn):
        """Initialize the registry.

        Args:
            base_dir: Root directory for per-wiki index subdirectories.
            embedding_fn: Shared EmbeddingFunction instance.
        """
        self._base_dir = base_dir
        self._embedding_fn = embedding_fn
        self._backends = {}
        self._lock = threading.Lock()

    @property
    def embedding_fn(self):
        return self._embedding_fn

    def get(self, slug):
        """Get or create a FAISSBackend for the given wiki slug.

        Args:
            slug: Wiki identifier (e.g. "dev", "research").

        Returns:
            FAISSBackend instance for this wiki.
        """
        backend = self._backends.get(slug)
        if backend is not None:
            return backend

        with self._lock:
            # Double-check after acquiring lock
            backend = self._backends.get(slug)
            if backend is not None:
                return backend

            from otterwiki_semantic_search.backends.faiss_backend import FAISSBackend

            index_dir = os.path.join(self._base_dir, slug)
            backend = FAISSBackend(index_dir, self._embedding_fn)
            self._backends[slug] = backend
            log.info("Created FAISS backend for wiki '%s' at %s", slug, index_dir)
            return backend

    def get_for_current_request(self):
        """Resolve the backend for the current request's wiki.

        Derives the slug from _state["storage"].path using
        os.path.basename().

        Returns:
            FAISSBackend for the current wiki.

        Raises:
            RuntimeError: If no storage is available in _state.
        """
        from otterwiki_semantic_search import _state

        storage = _state.get("storage")
        if storage is None:
            raise RuntimeError("No storage in _state — cannot resolve wiki slug")

        slug = os.path.basename(storage.path)
        if not slug:
            raise RuntimeError(f"Cannot derive slug from storage path: {storage.path}")

        return self.get(slug)

    def slug_for_storage(self, storage=None):
        """Return the wiki slug for a storage instance (or current _state storage).

        Args:
            storage: Optional storage instance. If None, uses _state["storage"].

        Returns:
            Wiki slug string.
        """
        if storage is None:
            from otterwiki_semantic_search import _state
            storage = _state.get("storage")

        if storage is None:
            raise RuntimeError("No storage available to derive slug")

        slug = os.path.basename(storage.path)
        if not slug:
            raise RuntimeError(f"Cannot derive slug from storage path: {storage.path}")
        return slug

    def all_backends(self):
        """Return a dict of slug -> backend for all initialized backends."""
        with self._lock:
            return dict(self._backends)
