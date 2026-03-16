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

    def _slug_from_path(self, path):
        """Derive wiki slug from a storage path.

        Handles two layouts:
          /srv/data/wikis/{slug}/repo  -> {slug}  (VPS standard)
          /srv/wikis/{slug}            -> {slug}  (legacy / test)
        """
        name = os.path.basename(path)
        if name == "repo":
            name = os.path.basename(os.path.dirname(path))
        return name

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
            if backend.count() == 0:
                from otterwiki_semantic_search import _state
                storage = _state.get("storage")
                app = _state.get("app")
                if storage is not None and self._slug_from_path(storage.path) == slug:
                    self._schedule_reindex(slug, backend, storage, app)
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

        slug = self._slug_from_path(storage.path)
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

        slug = self._slug_from_path(storage.path)
        if not slug:
            raise RuntimeError(f"Cannot derive slug from storage path: {storage.path}")
        return slug

    def _schedule_reindex(self, slug, backend, storage, app):
        import threading
        from otterwiki_semantic_search import index

        def _do_reindex():
            app_config = app.config if app else None
            log.info("Auto-reindexing new wiki '%s'", slug)
            try:
                index.reindex_all(storage, app_config, backend=backend)
                log.info("Auto-reindex complete for wiki '%s'", slug)
            except Exception:
                log.exception("Auto-reindex failed for wiki '%s'", slug)

        t = threading.Thread(target=_do_reindex, daemon=True, name=f"reindex-{slug}")
        t.start()

    def all_backends(self):
        """Return a dict of slug -> backend for all initialized backends."""
        with self._lock:
            return dict(self._backends)
