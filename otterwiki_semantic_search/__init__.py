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
    "sync_thread": None,
}

# Shared state populated during setup()
_state = dict(_DEFAULT_STATE)


def reset_state():
    """Reset shared state to defaults. Used by tests."""
    sync_thread = _state.get("sync_thread")
    if sync_thread is not None:
        sync_thread.stop()
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
    """Initialize FAISS backend with configured embedding function."""
    index_dir = os.environ.get("FAISS_INDEX_DIR", "/app-data/faiss")

    embedding_fn = _create_embedding_fn()
    from otterwiki_semantic_search.backends.faiss_backend import FAISSBackend

    backend = FAISSBackend(index_dir, embedding_fn)

    _state["backend"] = backend
    _state["embedding_fn"] = embedding_fn
    _state["available"] = True
    log.info("FAISS backend initialized (index_dir: %s)", index_dir)


def _create_embedding_fn():
    """Create the embedding function based on EMBEDDING_MODEL env var.

    Supported values:
    - "local" (default): sentence-transformers all-MiniLM-L6-v2
    - "bedrock": AWS Bedrock titan-embed-text-v2
    """
    model = os.environ.get("EMBEDDING_MODEL", "local").lower()

    if model == "bedrock":
        from otterwiki_semantic_search.embeddings.bedrock import BedrockEmbeddingFunction

        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        dimensions = int(os.environ.get("BEDROCK_EMBED_DIMENSIONS", "1024"))
        return BedrockEmbeddingFunction(region_name=region, dimensions=dimensions)
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

    @hookimpl
    def page_saved(self, pagepath, content, author, message):
        try:
            from otterwiki_semantic_search import index

            index.upsert_page(pagepath, content)
        except Exception:
            log.exception("Hook page_saved failed for %s", pagepath)

    @hookimpl
    def page_deleted(self, pagepath, author, message):
        try:
            from otterwiki_semantic_search import index

            index.delete_page(pagepath)
        except Exception:
            log.exception("Hook page_deleted failed for %s", pagepath)

    @hookimpl
    def page_renamed(self, old_pagepath, new_pagepath, author, message):
        try:
            from otterwiki_semantic_search import index

            index.delete_page(old_pagepath)
            storage = _state.get("storage")
            if storage:
                filename = get_filename(new_pagepath)
                if storage.exists(filename):
                    content = storage.load(filename)
                    index.upsert_page(new_pagepath, content)
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

        # Start background sync thread
        if _state["available"]:
            interval = int(os.environ.get("CHROMA_SYNC_INTERVAL", "60"))
            state_path = os.environ.get(
                "CHROMA_SYNC_STATE_PATH", "/app-data/chroma_sync_state.json"
            )
            from otterwiki_semantic_search.sync import SyncThread

            sync = SyncThread(app, storage, interval=interval, state_path=state_path)
            sync.start()
            _state["sync_thread"] = sync


plugin_manager.register(OtterwikiSemanticSearchPlugin())
