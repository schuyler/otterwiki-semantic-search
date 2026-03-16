"""API routes for semantic search."""

import logging
import threading
import datetime

from flask import jsonify, request

from otterwiki_semantic_search import _state, search_bp
from otterwiki_semantic_search import index

log = logging.getLogger(__name__)

# Per-wiki reindex status. Keyed by slug, values are dicts with status + result.
_reindex_results = {}
_reindex_results_lock = threading.Lock()


def _resolve_backend():
    """Resolve the per-wiki backend from registry, or fall back to singleton."""
    registry = _state.get("registry")
    if registry is not None:
        try:
            return registry.get_for_current_request()
        except RuntimeError:
            log.warning("Cannot resolve wiki backend for current request")
            return None
    return _state.get("backend")  # ChromaDB / single-tenant only


def _slug_for_current_request(storage):
    """Derive wiki slug from current request's storage."""
    registry = _state.get("registry")
    if registry is not None and storage is not None:
        try:
            return registry.slug_for_storage(storage)
        except RuntimeError:
            pass
    return "default"


@search_bp.route("/semantic-search", methods=["GET"])
def semantic_search():
    if not _state.get("available"):
        return jsonify({"error": "Vector backend is not available"}), 503

    query = request.args.get("q")
    if not query:
        return jsonify({"error": "Missing required parameter: q"}), 422

    try:
        n = int(request.args.get("n", 5))
    except (ValueError, TypeError):
        n = 5
    n = max(1, min(n, index.MAX_SEARCH_RESULTS))

    try:
        max_chunks_per_page = int(request.args.get("max_chunks_per_page", 2))
    except (ValueError, TypeError):
        max_chunks_per_page = 2

    backend = _resolve_backend()
    results = index.search(query, n=n, backend=backend, max_chunks_per_page=max_chunks_per_page)
    return jsonify({
        "query": query,
        "results": results,
        "total": len(results),
    })


@search_bp.route("/reindex", methods=["POST"])
def reindex():
    if not _state.get("available"):
        return jsonify({"error": "Vector backend is not available"}), 503

    if index.is_reindex_in_progress():
        return jsonify({"error": "Reindex already in progress"}), 409

    storage = _state.get("storage")
    app = _state.get("app")
    app_config = app.config if app else None

    backend = _resolve_backend()
    slug = _slug_for_current_request(storage)

    with _reindex_results_lock:
        current = _reindex_results.get(slug, {})
        if current.get("status") == "in_progress":
            return jsonify({"error": "Reindex already in progress"}), 409
        _reindex_results[slug] = {
            "status": "in_progress",
            "started_at": datetime.datetime.utcnow().isoformat(),
        }

    def _run_reindex():
        try:
            result = index.reindex_all(storage, app_config, backend=backend)
            with _reindex_results_lock:
                _reindex_results[slug] = {
                    "status": "complete",
                    "pages_indexed": result.get("pages_indexed", 0),
                    "chunks_created": result.get("chunks_created", 0),
                    "completed_at": datetime.datetime.utcnow().isoformat(),
                }
        except Exception:
            log.exception("Background reindex failed for slug %s", slug)
            with _reindex_results_lock:
                _reindex_results[slug] = {
                    "status": "error",
                    "completed_at": datetime.datetime.utcnow().isoformat(),
                }

    t = threading.Thread(target=_run_reindex, daemon=True)
    t.start()

    return jsonify({"status": "started"}), 202


@search_bp.route("/reindex", methods=["GET"])
@search_bp.route("/reindex/status", methods=["GET"])
def reindex_status():
    if not _state.get("available"):
        return jsonify({"error": "Vector backend is not available"}), 503
    storage = _state.get("storage")
    slug = _slug_for_current_request(storage)
    with _reindex_results_lock:
        result = dict(_reindex_results.get(slug, {}))
    if not result:
        return jsonify({"status": "idle"})
    return jsonify(result)


@search_bp.route("/chroma-status", methods=["GET"])
def chroma_status():
    available = _state.get("available", False)
    backend = _resolve_backend() if available else None

    status = {
        "status": "ok" if available else "unavailable",
        "backend_type": _state.get("backend_type", "chroma"),
        "collection": _state.get("collection_name", "otterwiki_pages"),
    }

    if available and backend is not None:
        try:
            status["document_count"] = backend.count()
        except Exception:
            status["document_count"] = 0
    else:
        status["document_count"] = 0

    return jsonify(status)
