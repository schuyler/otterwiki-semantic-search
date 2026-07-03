"""Failing tests for the repository_changed hook — feature NOT YET IMPLEMENTED.

These tests verify the behavior of a `repository_changed(changed_files)` hookimpl
on HookListener that keeps the search index consistent when an external git push
fires `repository_changed` rather than `page_saved`/`page_deleted`.

All tests are expected to FAIL until the feature is implemented.
Run in isolation to avoid confusion with the known-flaky concurrency test:
    pytest tests/test_repository_changed_hook.py -v
"""

import tempfile
from unittest.mock import MagicMock, call

import numpy as np
import pytest

from otterwiki_semantic_search.embeddings.base import EmbeddingFunction


# ---------------------------------------------------------------------------
# Shared test infrastructure (mirrors test_registry.py / test_faiss_backend.py)
# ---------------------------------------------------------------------------

class FakeEmbeddingFunction(EmbeddingFunction):
    """Deterministic embedding function for testing — no model download needed."""

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
def faiss_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def embedding_fn():
    return FakeEmbeddingFunction(dim=64)


@pytest.fixture
def backend(faiss_dir, embedding_fn):
    from otterwiki_semantic_search.backends.faiss_backend import FAISSBackend
    return FAISSBackend(faiss_dir, embedding_fn)


@pytest.fixture
def mock_storage():
    storage = MagicMock()
    storage.path = "/srv/wikis/test"
    return storage


@pytest.fixture
def hook_listener(backend, mock_storage, monkeypatch):
    """HookListener wired to a real FAISSBackend and mock storage via _state."""
    import otterwiki_semantic_search

    monkeypatch.setitem(otterwiki_semantic_search._state, "backend", backend)
    monkeypatch.setitem(otterwiki_semantic_search._state, "storage", mock_storage)
    monkeypatch.setitem(otterwiki_semantic_search._state, "registry", None)
    monkeypatch.setitem(
        otterwiki_semantic_search._state, "embedding_fn", FakeEmbeddingFunction(dim=64)
    )
    monkeypatch.setitem(otterwiki_semantic_search._state, "app", None)

    return otterwiki_semantic_search.HookListener()


# ---------------------------------------------------------------------------
# 1. Hook registration
# ---------------------------------------------------------------------------

class TestRepositoryChangedHookRegistration:
    """HookListener must expose repository_changed as a pluggy hookimpl."""

    def test_hook_listener_has_repository_changed_method(self):
        """HookListener must have a repository_changed() method."""
        import otterwiki_semantic_search

        listener = otterwiki_semantic_search.HookListener()
        assert hasattr(listener, "repository_changed"), (
            "HookListener is missing the repository_changed method"
        )
        assert callable(listener.repository_changed)

    def test_repository_changed_is_decorated_with_hookimpl(self):
        """repository_changed must carry the otterwiki_impl marker from @hookimpl."""
        import otterwiki_semantic_search

        listener = otterwiki_semantic_search.HookListener()
        method = listener.repository_changed
        assert hasattr(method, "otterwiki_impl"), (
            "repository_changed is not decorated with @hookimpl — "
            "pluggy will not call it"
        )


# ---------------------------------------------------------------------------
# 2. Upsert on add / modify
# ---------------------------------------------------------------------------

class TestRepositoryChangedUpsert:
    """Existing or newly pushed .md files should be upserted into the index."""

    def test_added_md_file_upserts_into_index(self, hook_listener, mock_storage, backend):
        """Firing repository_changed for an existing .md file adds chunks to the index."""
        mock_storage.exists.return_value = True
        mock_storage.load.return_value = (
            "# External Push Page\n\nContent added by an external git push."
        )

        assert backend.count() == 0

        hook_listener.repository_changed(changed_files=["External/Push_Page.md"])

        assert backend.count() > 0, (
            "Expected chunks to be upserted after repository_changed "
            "with an existing .md file"
        )

    def test_modified_md_file_updates_index(self, hook_listener, mock_storage, backend):
        """repository_changed re-upserts a modified page, replacing its old chunks."""
        from otterwiki_semantic_search import index

        # Seed with initial content (simulating prior state)
        initial_content = "# My Page\n\nOriginal content before external push."
        index.upsert_page("My/Page", initial_content, backend=backend)
        assert backend.count() > 0

        # External push provides updated content
        mock_storage.exists.return_value = True
        mock_storage.load.return_value = (
            "# My Page\n\nUpdated content delivered by an external git push."
        )

        hook_listener.repository_changed(changed_files=["My/Page.md"])

        # Page must still be indexed after the re-upsert
        assert backend.count() > 0, (
            "Index should contain chunks for the updated page after repository_changed"
        )

    def test_storage_load_called_for_existing_md_file(
        self, hook_listener, mock_storage, backend
    ):
        """The hook must load the file content from storage to perform the upsert."""
        mock_storage.exists.return_value = True
        mock_storage.load.return_value = "# Some Page\n\nSome content."

        hook_listener.repository_changed(changed_files=["Some/Page.md"])

        assert mock_storage.load.called, (
            "storage.load() should have been called to fetch content for upsert"
        )


# ---------------------------------------------------------------------------
# 3. Delete when file is gone
# ---------------------------------------------------------------------------

class TestRepositoryChangedDelete:
    """Deleted .md files (no longer in storage) must be removed from the index."""

    def test_deleted_md_file_removes_chunks_from_index(
        self, hook_listener, mock_storage, backend
    ):
        """When a .md file is absent from storage, repository_changed deletes its chunks."""
        from otterwiki_semantic_search import index

        # Seed the index with the page that will be "deleted" by the push
        index.upsert_page(
            "Deleted/Page",
            "# Deleted Page\n\nThis will be removed by an external push.",
            backend=backend,
        )
        assert backend.count() > 0

        # Simulate the file having been deleted in the push
        mock_storage.exists.return_value = False

        hook_listener.repository_changed(changed_files=["Deleted/Page.md"])

        assert backend.count() == 0, (
            "Expected chunks to be removed after repository_changed "
            "with a file that no longer exists in storage"
        )

    def test_deleted_page_does_not_remove_other_pages(
        self, hook_listener, mock_storage, backend
    ):
        """Deleting one page via repository_changed must not remove other indexed pages."""
        from otterwiki_semantic_search import index

        index.upsert_page(
            "Keep/This", "# Keep This\n\nThis page should survive.", backend=backend
        )
        index.upsert_page(
            "Remove/This", "# Remove This\n\nThis page is being deleted.", backend=backend
        )
        total_count = backend.count()
        assert total_count >= 2

        # Only Remove/This.md is absent
        def exists_side_effect(filename):
            return "remove" not in filename.lower()

        mock_storage.exists.side_effect = exists_side_effect

        hook_listener.repository_changed(changed_files=["Remove/This.md"])

        remaining = backend.count()
        assert remaining > 0, (
            "Pages not mentioned in changed_files should remain in the index"
        )

    def test_delete_nonexistent_page_is_noop(
        self, hook_listener, mock_storage, backend
    ):
        """Firing repository_changed for a file that was never indexed must not crash."""
        mock_storage.exists.return_value = False

        # Should not raise even if the page was never in the index
        hook_listener.repository_changed(changed_files=["Never/Indexed.md"])

        assert backend.count() == 0


# ---------------------------------------------------------------------------
# 4. Non-.md files are ignored
# ---------------------------------------------------------------------------

class TestRepositoryChangedIgnoresNonMd:
    """Files that don't end in .md must not trigger any index operation."""

    def test_image_file_is_ignored(self, hook_listener, mock_storage, backend):
        """A PNG in changed_files must not touch the index or storage."""
        hook_listener.repository_changed(changed_files=["images/logo.png"])

        assert backend.count() == 0, (
            "Non-.md files must not trigger an index operation"
        )
        mock_storage.exists.assert_not_called()
        mock_storage.load.assert_not_called()

    def test_yaml_file_is_ignored(self, hook_listener, mock_storage, backend):
        """A YAML config file in changed_files must not touch the index."""
        hook_listener.repository_changed(changed_files=["_data/config.yaml"])

        assert backend.count() == 0
        mock_storage.load.assert_not_called()

    def test_html_template_is_ignored(self, hook_listener, mock_storage, backend):
        """An HTML template in changed_files must not touch the index."""
        hook_listener.repository_changed(changed_files=["templates/base.html"])

        assert backend.count() == 0
        mock_storage.load.assert_not_called()

    def test_mixed_files_only_processes_md(self, hook_listener, mock_storage, backend):
        """In a mixed list, only .md files should be processed."""
        mock_storage.exists.return_value = True
        mock_storage.load.return_value = "# Real Page\n\nActual markdown content."

        hook_listener.repository_changed(
            changed_files=[
                "images/photo.jpg",
                "Real/Page.md",
                "config.yaml",
                "templates/base.html",
            ]
        )

        # The .md file should have been indexed
        assert backend.count() > 0, (
            "The .md file in a mixed list should result in an index upsert"
        )
        # storage.load should only be called for the .md file (once)
        assert mock_storage.load.call_count == 1, (
            "storage.load() should be called exactly once (for the .md file only)"
        )

    def test_empty_changed_files_is_noop(self, hook_listener, mock_storage, backend):
        """An empty changed_files list must be a complete no-op."""
        hook_listener.repository_changed(changed_files=[])

        assert backend.count() == 0
        mock_storage.exists.assert_not_called()
        mock_storage.load.assert_not_called()
