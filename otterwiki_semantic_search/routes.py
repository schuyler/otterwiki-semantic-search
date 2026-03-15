"""API routes for semantic search."""

from flask import jsonify, request

from otterwiki_semantic_search import _state, search_bp
from otterwiki_semantic_search import index


def _resolve_backend():
    """Resolve the per-wiki backend from registry, or fall back to singleton."""
    registry = _state.get("registry")
    if registry is not None:
        try:
            return registry.get_for_current_request()
        except RuntimeError:
            pass
    return _state.get("backend")


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

    backend = _resolve_backend()
    results = index.search(query, n=n, backend=backend)
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
    result = index.reindex_all(storage, app_config, backend=backend)
    return jsonify({
        "status": "ok",
        **result,
    })


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
