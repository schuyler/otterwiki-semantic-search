"""Tests for the BackendRegistry multi-tenant FAISS backend resolution."""

import os
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest

from otterwiki_semantic_search.embeddings.base import EmbeddingFunction


class FakeEmbeddingFunction(EmbeddingFunction):
    """Deterministic embedding function for testing."""

    def __init__(self, dim=64):
        self._dim = dim

    def embed(self, texts):
        vectors = []
        for text in texts:
            rng = np.random.RandomState(hash(text) % (2**31))
            vec = rng.randn(self._dim).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            vectors.append(vec.tolist())
        return vectors

    @property
    def dimensionality(self):
        return self._dim


@pytest.fixture
def base_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def embedding_fn():
    return FakeEmbeddingFunction(dim=64)


@pytest.fixture
def registry(base_dir, embedding_fn):
    from otterwiki_semantic_search.registry import BackendRegistry
    return BackendRegistry(base_dir, embedding_fn)


class TestBackendRegistry:
    def test_creates_backend_per_slug(self, registry, base_dir):
        b1 = registry.get("wiki-alpha")
        b2 = registry.get("wiki-beta")

        assert b1 is not b2
        assert os.path.isdir(os.path.join(base_dir, "wiki-alpha"))
        assert os.path.isdir(os.path.join(base_dir, "wiki-beta"))

    def test_same_slug_returns_same_backend(self, registry):
        b1 = registry.get("dev")
        b2 = registry.get("dev")
        assert b1 is b2

    def test_per_wiki_isolation(self, registry, embedding_fn):
        """Data in one wiki's backend must not appear in another."""
        b1 = registry.get("wiki-a")
        b2 = registry.get("wiki-b")

        b1.upsert(
            ids=["page1::chunk_0"],
            texts=["Python programming"],
            metadatas=[{"page_path": "page1", "chunk_index": 0}],
        )

        assert b1.count() == 1
        assert b2.count() == 0

        # Search in wiki-b should find nothing
        query_emb = embedding_fn.embed(["Python"])
        results = b2.query(query_embeddings=query_emb, n_results=5)
        assert results["ids"] == [[]]

    def test_shared_embedding_fn(self, registry, embedding_fn):
        assert registry.embedding_fn is embedding_fn

    def test_all_backends(self, registry):
        registry.get("a")
        registry.get("b")
        registry.get("c")
        backends = registry.all_backends()
        assert set(backends.keys()) == {"a", "b", "c"}

    def test_get_for_current_request(self, registry, monkeypatch, base_dir):
        """get_for_current_request() derives slug from _state storage path."""
        import otterwiki_semantic_search

        mock_storage = MagicMock()
        mock_storage.path = "/srv/wikis/research"
        monkeypatch.setitem(otterwiki_semantic_search._state, "storage", mock_storage)

        backend = registry.get_for_current_request()
        assert os.path.isdir(os.path.join(base_dir, "research"))

        # Same storage path should yield same backend
        backend2 = registry.get_for_current_request()
        assert backend is backend2

    def test_get_for_current_request_no_storage(self, registry, monkeypatch):
        import otterwiki_semantic_search
        monkeypatch.setitem(otterwiki_semantic_search._state, "storage", None)

        with pytest.raises(RuntimeError, match="No storage"):
            registry.get_for_current_request()

    def test_slug_for_storage(self, registry):
        mock_storage = MagicMock()
        mock_storage.path = "/srv/wikis/dev"
        assert registry.slug_for_storage(mock_storage) == "dev"

    def test_slug_for_storage_no_storage(self, registry, monkeypatch):
        import otterwiki_semantic_search
        monkeypatch.setitem(otterwiki_semantic_search._state, "storage", None)
        with pytest.raises(RuntimeError, match="No storage"):
            registry.slug_for_storage(None)


class TestMultiTenantIndexOperations:
    """Test that index.py functions route to the correct per-wiki backend."""

    def test_upsert_with_explicit_backend(self, registry, embedding_fn):
        b1 = registry.get("wiki-x")
        b2 = registry.get("wiki-y")

        from otterwiki_semantic_search import index
        index.upsert_page("TestPage", "Some content about testing", backend=b1)

        assert b1.count() > 0
        assert b2.count() == 0

    def test_delete_with_explicit_backend(self, registry, embedding_fn):
        b1 = registry.get("wiki-del")

        from otterwiki_semantic_search import index
        index.upsert_page("DelPage", "Content to delete", backend=b1)
        assert b1.count() > 0

        index.delete_page("DelPage", backend=b1)
        assert b1.count() == 0

    def test_search_with_explicit_backend(self, registry, embedding_fn):
        b1 = registry.get("wiki-search")

        from otterwiki_semantic_search import index
        index.upsert_page("SearchPage", "Quantum computing and physics", backend=b1)

        results = index.search("quantum", n=5, backend=b1)
        assert len(results) > 0
        assert results[0]["name"] == "SearchPage"

    def test_reindex_with_explicit_backend(self, registry, embedding_fn):
        b1 = registry.get("wiki-reindex")

        mock_storage = MagicMock()
        mock_storage.list.return_value = (["page1.md", "page2.md"], [])
        mock_storage.load.side_effect = lambda f: f"Content of {f}"

        from otterwiki_semantic_search import index
        result = index.reindex_all(mock_storage, app_config={}, backend=b1)

        assert result["pages_indexed"] == 2
        assert result["chunks_created"] > 0
        assert b1.count() > 0


class TestSlugFromPath:
    """Test _slug_from_path handles VPS and legacy path layouts."""

    def test_vps_path_pattern(self, registry):
        """VPS layout: /srv/data/wikis/{slug}/repo -> {slug}"""
        assert registry._slug_from_path("/srv/data/wikis/research/repo") == "research"

    def test_legacy_path_pattern(self, registry):
        """Legacy layout: /srv/wikis/{slug} -> {slug}"""
        assert registry._slug_from_path("/srv/wikis/dev") == "dev"

    def test_slug_for_storage_vps_path(self, registry):
        """`slug_for_storage` derives correct slug from VPS path."""
        mock_storage = MagicMock()
        mock_storage.path = "/srv/data/wikis/research/repo"
        assert registry.slug_for_storage(mock_storage) == "research"

    def test_trailing_slash(self, registry):
        """Trailing slash is stripped before deriving the slug."""
        assert registry._slug_from_path("/srv/wikis/dev/") == "dev"

    def test_single_component_path(self, registry):
        """A path with no parent directory returns the single component."""
        assert registry._slug_from_path("/repo") == "repo"

    def test_empty_path(self, registry):
        """Empty string normalises to '.' and returns '.'."""
        result = registry._slug_from_path("")
        assert result == "."

    def test_resolve_backend_returns_none_outside_request(self, registry, monkeypatch):
        """`_resolve_backend` returns None (not stale backend) when outside request in FAISS mode."""
        import otterwiki_semantic_search
        from otterwiki_semantic_search import _state

        # Set up registry in state, clear storage (outside request context)
        monkeypatch.setitem(_state, "registry", registry)
        monkeypatch.setitem(_state, "storage", None)
        monkeypatch.setitem(_state, "backend", MagicMock())  # stale backend

        hook = otterwiki_semantic_search.HookListener()
        result = hook._resolve_backend()
        assert result is None


class TestFileLocking:
    """Verify that FAISS backend uses file locking."""

    def test_lock_file_created(self, base_dir, embedding_fn):
        from otterwiki_semantic_search.backends.faiss_backend import FAISSBackend

        index_dir = os.path.join(base_dir, "lock-test")
        backend = FAISSBackend(index_dir, embedding_fn)

        backend.upsert(
            ids=["p1::c0"],
            texts=["test"],
            metadatas=[{"page_path": "p1", "chunk_index": 0}],
        )

        assert os.path.exists(os.path.join(index_dir, ".lock"))

    def test_lock_path_set(self, base_dir, embedding_fn):
        from otterwiki_semantic_search.backends.faiss_backend import FAISSBackend

        index_dir = os.path.join(base_dir, "lock-path-test")
        backend = FAISSBackend(index_dir, embedding_fn)
        assert backend._lock_path == os.path.join(index_dir, ".lock")
