"""
Otterwiki Semantic Search Plugin

Adds vector-based semantic search to An Otter Wiki using ChromaDB.
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
    "client": None,
    "collection": None,
    "collection_name": "otterwiki_pages",
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


def _init_chromadb():
    """Initialize ChromaDB client based on CHROMADB_MODE env var."""
    import chromadb

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

    _state["client"] = client
    _state["collection"] = client.get_or_create_collection(collection_name)
    _state["available"] = True
    log.info("ChromaDB initialized in %s mode (collection: %s)", mode, collection_name)


class ChromaHookListener:
    """Listens for page lifecycle hooks to update the ChromaDB index."""

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
            # Load new content from storage and index it
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

        # Initialize ChromaDB
        try:
            _init_chromadb()
        except Exception:
            log.warning("ChromaDB initialization failed. Semantic search will be unavailable.", exc_info=True)
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
                    plugin_manager.register(ChromaHookListener())
                    log.info("Registered ChromaHookListener for page lifecycle hooks")
            except Exception:
                log.warning("Could not register ChromaHookListener", exc_info=True)

        # Start background sync thread — it handles initial reindex on first
        # cycle if the collection is empty, so no separate reindex thread needed.
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
