"""Unit tests for the ChromaDB backend — real PersistentClient on a temp dir."""

import logging
from unittest.mock import MagicMock, patch

import pytest

chromadb = pytest.importorskip("chromadb")

from otterwiki_semantic_search.backends.chroma_backend import ChromaBackend

COLLECTION = "test_pages"


def _chunk(pagepath, i, text):
    return {
        "id": f"{pagepath}::chunk_{i}",
        "text": text,
        "metadata": {"page_path": pagepath},
    }


@pytest.fixture
def chroma_path(tmp_path):
    return str(tmp_path / "chroma")


@pytest.fixture
def backend(chroma_path):
    return ChromaBackend(chromadb.PersistentClient(path=chroma_path), COLLECTION)


def _ids(backend):
    return set(backend.collection.get(include=[])["ids"])


class TestReindexAcrossWorkers:
    def test_other_worker_sees_reindexed_data(self, backend, chroma_path):
        """A backend built before reindex (another worker) keeps working after it."""
        other = ChromaBackend(chromadb.PersistentClient(path=chroma_path), COLLECTION)
        backend.reindex_atomic([_chunk("python", 0, "Python is a programming language")])

        assert other.count() == 1
        result = other.query(query_texts=["programming"], n_results=1)
        assert result["ids"][0] == ["python::chunk_0"]

    def test_reindex_all_keeps_other_worker_handle_valid(self, backend, chroma_path):
        """The reindex_all code path never invalidates another worker's collection."""
        from otterwiki_semantic_search import index

        other = ChromaBackend(chromadb.PersistentClient(path=chroma_path), COLLECTION)
        storage = MagicMock()
        storage.list.return_value = (["python.md"], [])
        storage.load.return_value = "# Python\n\nPython is a programming language."

        result = index.reindex_all(storage, app_config={}, backend=backend)

        assert result["chunks_created"] > 0
        assert other.collection.id == backend.collection.id
        assert other.count() == result["chunks_created"]

    def test_collection_id_is_stable(self, backend):
        before = backend.collection.id
        backend.reindex_atomic([_chunk("a", 0, "alpha")])
        assert backend.collection.id == before

    def test_prunes_stale_chunks(self, backend):
        backend.upsert(
            ids=["a::chunk_0", "a::chunk_1", "b::chunk_0"],
            texts=["alpha", "alpha two", "beta"],
            metadatas=[{"page_path": "a"}, {"page_path": "a"}, {"page_path": "b"}],
        )
        backend.reindex_atomic([_chunk("a", 0, "alpha revised")])

        assert _ids(backend) == {"a::chunk_0"}
        doc = backend.collection.get(ids=["a::chunk_0"])["documents"][0]
        assert doc == "alpha revised"

    def test_new_chunks_are_written_before_stale_ones_are_deleted(self, backend):
        """Searches never see an empty collection mid-reindex."""
        backend.upsert(
            ids=["a::chunk_0", "b::chunk_0"],
            texts=["alpha", "beta"],
            metadatas=[{"page_path": "a"}, {"page_path": "b"}],
        )
        collection = backend.collection
        real_delete = collection.delete
        seen = []

        def spy_delete(*args, **kwargs):
            seen.append(collection.get(ids=["a::chunk_0"])["documents"])
            return real_delete(*args, **kwargs)

        with patch.object(collection, "delete", side_effect=spy_delete):
            backend.reindex_atomic([_chunk("a", 0, "alpha revised")])

        assert seen == [["alpha revised"]]
        assert _ids(backend) == {"a::chunk_0"}

    def test_reindex_with_no_chunks_empties_collection(self, backend):
        backend.upsert(ids=["a::chunk_0"], texts=["alpha"], metadatas=[{"page_path": "a"}])
        backend.reindex_atomic([])
        assert backend.count() == 0

    def test_reindex_empty_collection_with_no_chunks(self, backend):
        backend.reindex_atomic([])
        assert backend.count() == 0


class TestCollectionDeletedExternally:
    def test_count_recovers(self, backend, chroma_path):
        backend.upsert(ids=["a::chunk_0"], texts=["alpha"], metadatas=[{"page_path": "a"}])
        chromadb.PersistentClient(path=chroma_path).delete_collection(COLLECTION)
        assert backend.count() == 0

    def test_upsert_recovers(self, backend, chroma_path):
        chromadb.PersistentClient(path=chroma_path).delete_collection(COLLECTION)
        backend.upsert(ids=["a::chunk_0"], texts=["alpha"], metadatas=[{"page_path": "a"}])
        other = ChromaBackend(chromadb.PersistentClient(path=chroma_path), COLLECTION)
        assert other.count() == 1


class TestChromaStatusError:
    def test_backend_failure_reports_error(self, test_client, caplog):
        from otterwiki_semantic_search import _state

        broken = MagicMock()
        broken.count.side_effect = RuntimeError("boom")
        with patch.dict(_state, {"available": True, "backend": broken, "registry": None}):
            with caplog.at_level(logging.ERROR, logger="otterwiki_semantic_search.routes"):
                rv = test_client.get("/api/v1/chroma-status")

        assert rv.status_code == 200
        data = rv.get_json()
        assert data["status"] == "error"
        assert data["document_count"] is None
        assert "boom" not in rv.get_data(as_text=True)
        assert any(r.exc_info for r in caplog.records)
