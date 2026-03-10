"""Integration tests using Flask test client + ChromaDB."""

import pytest


class TestChromaStatus:
    def test_status_no_auth_required(self, test_client):
        rv = test_client.get("/api/v1/chroma-status")
        assert rv.status_code == 200
        data = rv.get_json()
        assert data["status"] in ("ok", "unavailable")
        assert "document_count" in data
        assert "collection" in data


class TestAuth:
    def test_missing_auth(self, test_client):
        rv = test_client.get("/api/v1/semantic-search?q=test")
        assert rv.status_code == 401

    def test_wrong_auth(self, test_client):
        rv = test_client.get(
            "/api/v1/semantic-search?q=test",
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert rv.status_code == 401

    def test_valid_auth(self, test_client, auth_headers):
        rv = test_client.get(
            "/api/v1/semantic-search?q=test", headers=auth_headers
        )
        # Should get 200 (or 503 if ChromaDB unavailable, but not 401)
        assert rv.status_code in (200, 503)

    def test_non_bearer_auth(self, test_client):
        rv = test_client.get(
            "/api/v1/semantic-search?q=test",
            headers={"Authorization": "Basic dXNlcjpwYXNz"},
        )
        assert rv.status_code == 401


class TestSemanticSearch:
    def test_missing_query(self, test_client, auth_headers):
        rv = test_client.get("/api/v1/semantic-search", headers=auth_headers)
        if rv.status_code == 422:
            data = rv.get_json()
            assert "error" in data
        else:
            # 503 if ChromaDB unavailable — still a valid outcome
            assert rv.status_code == 503

    def test_search_returns_structure(self, test_client, auth_headers):
        rv = test_client.get(
            "/api/v1/semantic-search?q=test", headers=auth_headers
        )
        if rv.status_code == 503:
            pytest.skip("ChromaDB not available")
        assert rv.status_code == 200
        data = rv.get_json()
        assert data["query"] == "test"
        assert isinstance(data["results"], list)
        assert isinstance(data["total"], int)

    def test_n_parameter_capped(self, test_client, auth_headers):
        rv = test_client.get(
            "/api/v1/semantic-search?q=test&n=99999", headers=auth_headers
        )
        # Should not crash — n is capped internally
        assert rv.status_code in (200, 503)

    def test_negative_n_parameter(self, test_client, auth_headers):
        rv = test_client.get(
            "/api/v1/semantic-search?q=test&n=-5", headers=auth_headers
        )
        assert rv.status_code in (200, 503)

    def test_non_integer_n_parameter(self, test_client, auth_headers):
        rv = test_client.get(
            "/api/v1/semantic-search?q=test&n=abc", headers=auth_headers
        )
        assert rv.status_code in (200, 503)


class TestReindex:
    def test_reindex_empty_wiki(self, test_client, auth_headers):
        rv = test_client.post("/api/v1/reindex", headers=auth_headers)
        if rv.status_code == 503:
            pytest.skip("ChromaDB not available")
        assert rv.status_code == 200
        data = rv.get_json()
        assert data["status"] == "ok"
        assert data["pages_indexed"] >= 0
        assert data["chunks_created"] >= 0

    def test_reindex_requires_auth(self, test_client):
        rv = test_client.post("/api/v1/reindex")
        assert rv.status_code == 401

    def test_reindex_with_pages(self, test_client, create_app, auth_headers):
        """Create pages in storage, reindex, then search."""
        storage = create_app.storage

        # Create some pages
        storage.store(
            "python.md",
            "---\ntitle: Python\ncategory: programming\n---\n"
            "Python is a versatile programming language used for web development, "
            "data science, machine learning, and automation. It has a clean syntax "
            "that makes it readable and easy to learn. Python supports multiple "
            "programming paradigms including object-oriented, functional, and "
            "procedural programming styles.",
            message="Add Python page",
            author=("Test", "test@test.com"),
        )
        storage.store(
            "cooking.md",
            "---\ntitle: Cooking Basics\ncategory: food\n---\n"
            "Cooking is the art of preparing food using heat. Basic techniques "
            "include boiling, frying, baking, and grilling. Understanding these "
            "fundamental methods is essential for any home cook. Temperature control "
            "and timing are the two most important factors in successful cooking.",
            message="Add Cooking page",
            author=("Test", "test@test.com"),
        )

        # Reindex
        rv = test_client.post("/api/v1/reindex", headers=auth_headers)
        if rv.status_code == 503:
            pytest.skip("ChromaDB not available")
        assert rv.status_code == 200

        data = rv.get_json()
        assert data["pages_indexed"] >= 2
        assert data["chunks_created"] >= 2

        # Search should find results
        rv = test_client.get(
            "/api/v1/semantic-search?q=programming+language",
            headers=auth_headers,
        )
        assert rv.status_code == 200
        data = rv.get_json()
        assert data["total"] > 0

        # Verify status shows documents
        rv = test_client.get("/api/v1/chroma-status")
        data = rv.get_json()
        assert data["document_count"] > 0
