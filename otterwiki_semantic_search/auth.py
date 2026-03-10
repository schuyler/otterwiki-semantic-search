"""Bearer token authentication for the semantic search API."""

import hmac
import os

from flask import jsonify, request

from otterwiki_semantic_search import search_bp

# Endpoints exempt from auth — use function names, not full endpoint strings,
# so renaming the blueprint doesn't silently break exemptions.
_AUTH_EXEMPT_FUNCTIONS = {"chroma_status"}


@search_bp.before_request
def check_api_key():
    # Check if the view function is exempt
    if request.endpoint:
        func_name = request.endpoint.rsplit(".", 1)[-1]
        if func_name in _AUTH_EXEMPT_FUNCTIONS:
            return None

    api_key = os.environ.get("OTTERWIKI_API_KEY")
    if not api_key:
        return jsonify({"error": "Authentication not available"}), 401

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return jsonify({"error": "Missing or invalid Authorization header"}), 401

    token = auth_header[7:]  # len("Bearer ") == 7
    if not hmac.compare_digest(token, api_key):
        return jsonify({"error": "Invalid API key"}), 401

    return None
