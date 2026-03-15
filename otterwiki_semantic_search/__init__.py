"""
Otterwiki Semantic Search Plugin

Adds vector-based semantic search to An Otter Wiki.
Supports ChromaDB (default) and FAISS backends.
"""

import logging
import os

from flask import Blueprint

from otterwiki.plugins import hookimpl, plugin_manager

log = logging.getLogger(__name__)

search_bp = Blueprint("otterwiki_semantic_search", __name__, url_prefix="/api/v1")

_DEFAULT_STATE = {
    "app": None,
    "storage": None,
    "db": None,
    # Legacy keys for backward compatibility
    "client": None,
    "collection": None,
    "collection_name": "otterwiki_pages",
    # New backend abstraction
    "backend": None,
    "embedding_fn": None,
    "backend_type": None,
    "available": False,
    # Multi-tenant FAISS registry (replaces singleton backend for FAISS)
    "registry": None,
}

# Shared state populated during setup()
_state = dict(_DEFAULT_STATE)


def reset_state():
    """Reset shared state to defaults. Used by tests."""
    _state.update(_DEFAULT_STATE)


def get_filename(pagepath):
    """Convert a URL page path to the on-disk filename."""
    app = _state["app"]
    retain_case = app.config.get("RETAIN_PAGE_NAME_CASE", False) if app else False
    p = pagepath if retain_case else pagepath.lower()
    parts = [part for part in p.split("/") if part]
    p = "/".join(parts)
    if not p.endswith(".md"):
        p = f"{p}.md"
    return p


def _init_backend():
    """Initialize the vector backend based on VECTOR_BACKEND env var.

    Supported values:
    - "chroma" (default): Uses ChromaDB with internal embedding.
    - "faiss": Uses FAISS with an external embedding function.
    """
    backend_type = os.environ.get("VECTOR_BACKEND", "chroma").lower()
    _state["backend_type"] = backend_type

    if backend_type == "faiss":
        _init_faiss()
    else:
        _init_chromadb()


def _init_chromadb():
    """Initialize ChromaDB backend."""
    import chromadb
    from otterwiki_semantic_search.backends.chroma_backend import ChromaBackend

    mode = os.environ.get("CHROMADB_MODE", "server")
    collection_name = os.environ.get("CHROMA_COLLECTION", "otterwiki_pages")
    _state["collection_name"] = collection_name

    if mode == "local":
        path = os.environ.get("CHROMADB_PATH", "/app-data/chroma")
        client = chromadb.PersistentClient(path=path)
    else:
        host = os.environ.get("CHROMADB_HOST", "localhost")
        port = int(os.environ.get("CHROMADB_PORT", "8000"))
        client = chromadb.HttpClient(host=host, port=port)

    backend = ChromaBackend(client, collection_name)

    # Keep legacy state for backward compatibility
    _state["client"] = client
    _state["collection"] = backend.collection

    _state["backend"] = backend
    _state["embedding_fn"] = None  # ChromaDB handles embedding internally
    _state["available"] = True
    log.info("ChromaDB backend initialized in %s mode (collection: %s)", mode, collection_name)


def _init_faiss():
    """Initialize FAISS backend registry for multi-tenant per-wiki indexes.

    Creates a BackendRegistry that lazily creates FAISSBackend instances
    per wiki slug under {FAISS_INDEX_DIR}/{slug}/. Also creates a default
    backend for the current storage (backward compatibility).
    """
    base_dir = os.environ.get("FAISS_INDEX_DIR", "/app-data/faiss")

    embedding_fn = _create_embedding_fn()
    from otterwiki_semantic_search.registry import BackendRegistry

    registry = BackendRegistry(base_dir, embedding_fn)
    _state["registry"] = registry
    _state["embedding_fn"] = embedding_fn

    # Create the default backend for the current wiki (backward compat)
    storage = _state.get("storage")
    if storage is not None:
        backend = registry.get_for_current_request()
        _state["backend"] = backend

    _state["available"] = True
    log.info("FAISS multi-tenant registry initialized (base_dir: %s)", base_dir)


def _create_embedding_fn():
    """Create the embedding function based on EMBEDDING_MODEL env var.

    Supported values:
    - "local" (default): sentence-transformers all-MiniLM-L6-v2
    - "onnx": ChromaDB's ONNX MiniLM-L6-v2 (no torch dependency)
    - "bedrock": AWS Bedrock titan-embed-text-v2
    """
    model = os.environ.get("EMBEDDING_MODEL", "local").lower()

    if model == "bedrock":
        from otterwiki_semantic_search.embeddings.bedrock import BedrockEmbeddingFunction

        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        dimensions = int(os.environ.get("BEDROCK_EMBED_DIMENSIONS", "1024"))
        return BedrockEmbeddingFunction(region_name=region, dimensions=dimensions)
    elif model == "onnx":
        from otterwiki_semantic_search.embeddings.onnx_embedding import ONNXEmbeddingFunction

        return ONNXEmbeddingFunction()
    else:
        from otterwiki_semantic_search.embeddings.sentence_transformer import (
            SentenceTransformerEmbeddingFunction,
        )

        model_name = os.environ.get(
            "SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2"
        )
        return SentenceTransformerEmbeddingFunction(model_name=model_name)


class HookListener:
    """Listens for page lifecycle hooks to update the vector index."""

    def _resolve_backend(self):
        """Resolve the per-wiki backend from the registry, or fall back to singleton."""
        registry = _state.get("registry")
        if registry is not None:
            try:
                return registry.get_for_current_request()
            except RuntimeError:
                pass
        return _state.get("backend")

    @hookimpl
    def page_saved(self, pagepath, content, author, message):
        try:
            from otterwiki_semantic_search import index

            backend = self._resolve_backend()
            index.upsert_page(pagepath, content, backend=backend)
        except Exception:
            log.exception("Hook page_saved failed for %s", pagepath)

    @hookimpl
    def page_deleted(self, pagepath, author, message):
        try:
            from otterwiki_semantic_search import index

            backend = self._resolve_backend()
            index.delete_page(pagepath, backend=backend)
        except Exception:
            log.exception("Hook page_deleted failed for %s", pagepath)

    @hookimpl
    def page_renamed(self, old_pagepath, new_pagepath, author, message):
        try:
            from otterwiki_semantic_search import index

            backend = self._resolve_backend()
            index.delete_page(old_pagepath, backend=backend)
            storage = _state.get("storage")
            if storage:
                filename = get_filename(new_pagepath)
                if storage.exists(filename):
                    content = storage.load(filename)
                    index.upsert_page(new_pagepath, content, backend=backend)
        except Exception:
            log.exception(
                "Hook page_renamed failed for %s -> %s", old_pagepath, new_pagepath
            )


class OtterwikiSemanticSearchPlugin:
    @hookimpl
    def setup(self, app, db, storage):
        _state["app"] = app
        _state["storage"] = storage
        _state["db"] = db

        # Warn if API key is not configured
        if not os.environ.get("OTTERWIKI_API_KEY"):
            log.warning("OTTERWIKI_API_KEY not set — semantic search endpoints will reject all requests")

        # Initialize vector backend
        try:
            _init_backend()
        except Exception:
            log.warning("Vector backend initialization failed. Semantic search will be unavailable.", exc_info=True)
            _state["available"] = False

        # Import auth + routes to register them on the blueprint
        import otterwiki_semantic_search.auth  # noqa: F401
        import otterwiki_semantic_search.routes  # noqa: F401

        # Register blueprint
        app.register_blueprint(search_bp)

        # Register hook listener if page lifecycle hooks are available
        if _state["available"]:
            try:
                if hasattr(plugin_manager.hook, "page_saved"):
                    plugin_manager.register(HookListener())
                    log.info("Registered HookListener for page lifecycle hooks")
            except Exception:
                log.warning("Could not register HookListener", exc_info=True)

        # NOTE: SyncThread is deprecated in multi-tenant mode.
        # Page lifecycle hooks handle index updates directly.
        # See sync.py for details.


plugin_manager.register(OtterwikiSemanticSearchPlugin())
