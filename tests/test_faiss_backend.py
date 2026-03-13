"""Unit tests for the FAISS backend — no ChromaDB or Flask required."""

import json
import os
import tempfile

import numpy as np
import pytest

from otterwiki_semantic_search.embeddings.base import EmbeddingFunction


class FakeEmbeddingFunction(EmbeddingFunction):
    """Deterministic embedding function for testing.

    Produces normalized vectors where each text maps to a reproducible
    direction based on a hash of the text content.
    """

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


class TestFAISSBackendUpsert:
    def test_upsert_adds_vectors(self, backend):
        backend.upsert(
            ids=["page1::chunk_0", "page1::chunk_1"],
            texts=["Hello world", "Goodbye world"],
            metadatas=[
                {"page_path": "page1", "chunk_index": 0},
                {"page_path": "page1", "chunk_index": 1},
            ],
        )
        assert backend.count() == 2

    def test_upsert_replaces_existing_ids(self, backend):
        backend.upsert(
            ids=["page1::chunk_0"],
            texts=["Hello world"],
            metadatas=[{"page_path": "page1", "chunk_index": 0}],
        )
        assert backend.count() == 1

        # Upsert same ID with different text
        backend.upsert(
            ids=["page1::chunk_0"],
            texts=["Updated text"],
            metadatas=[{"page_path": "page1", "chunk_index": 0}],
        )
        assert backend.count() == 1

    def test_upsert_with_precomputed_embeddings(self, backend, embedding_fn):
        texts = ["Hello world"]
        embeddings = embedding_fn.embed(texts)
        backend.upsert(
            ids=["page1::chunk_0"],
            texts=texts,
            metadatas=[{"page_path": "page1", "chunk_index": 0}],
            embeddings=embeddings,
        )
        assert backend.count() == 1


class TestFAISSBackendDelete:
    def test_delete_by_page_path(self, backend):
        backend.upsert(
            ids=["page1::chunk_0", "page1::chunk_1", "page2::chunk_0"],
            texts=["Text A", "Text B", "Text C"],
            metadatas=[
                {"page_path": "page1", "chunk_index": 0},
                {"page_path": "page1", "chunk_index": 1},
                {"page_path": "page2", "chunk_index": 0},
            ],
        )
        assert backend.count() == 3

        backend.delete(where={"page_path": "page1"})
        assert backend.count() == 1

    def test_delete_nonexistent_page(self, backend):
        backend.upsert(
            ids=["page1::chunk_0"],
            texts=["Text A"],
            metadatas=[{"page_path": "page1", "chunk_index": 0}],
        )
        backend.delete(where={"page_path": "nonexistent"})
        assert backend.count() == 1

    def test_delete_all_entries(self, backend):
        backend.upsert(
            ids=["page1::chunk_0"],
            texts=["Text A"],
            metadatas=[{"page_path": "page1", "chunk_index": 0}],
        )
        backend.delete(where={"page_path": "page1"})
        assert backend.count() == 0


class TestFAISSBackendQuery:
    def test_query_returns_results(self, backend, embedding_fn):
        backend.upsert(
            ids=["page1::chunk_0", "page2::chunk_0"],
            texts=["Python programming language", "Cooking recipes"],
            metadatas=[
                {"page_path": "page1", "chunk_index": 0},
                {"page_path": "page2", "chunk_index": 0},
            ],
        )

        query_embeddings = embedding_fn.embed(["Python programming language"])
        results = backend.query(query_embeddings=query_embeddings, n_results=2)

        assert len(results["ids"][0]) == 2
        assert len(results["documents"][0]) == 2
        assert len(results["metadatas"][0]) == 2
        assert len(results["distances"][0]) == 2

    def test_query_with_text(self, backend):
        backend.upsert(
            ids=["page1::chunk_0"],
            texts=["Python programming language"],
            metadatas=[{"page_path": "page1", "chunk_index": 0}],
        )

        results = backend.query(query_texts=["Python programming"], n_results=1)
        assert len(results["ids"][0]) == 1
        assert results["ids"][0][0] == "page1::chunk_0"

    def test_query_empty_index(self, backend, embedding_fn):
        query_embeddings = embedding_fn.embed(["test query"])
        results = backend.query(query_embeddings=query_embeddings, n_results=5)
        assert results["ids"] == [[]]

    def test_query_n_results_clamped(self, backend, embedding_fn):
        """Requesting more results than available should not crash."""
        backend.upsert(
            ids=["page1::chunk_0"],
            texts=["Only one document"],
            metadatas=[{"page_path": "page1", "chunk_index": 0}],
        )

        query_embeddings = embedding_fn.embed(["test"])
        results = backend.query(query_embeddings=query_embeddings, n_results=100)
        assert len(results["ids"][0]) == 1

    def test_query_distances_are_ordered(self, backend, embedding_fn):
        """Results should be ordered by distance (ascending)."""
        backend.upsert(
            ids=["page1::chunk_0", "page2::chunk_0", "page3::chunk_0"],
            texts=["Python programming", "Java programming", "Cooking recipes"],
            metadatas=[
                {"page_path": "page1", "chunk_index": 0},
                {"page_path": "page2", "chunk_index": 0},
                {"page_path": "page3", "chunk_index": 0},
            ],
        )

        query_embeddings = embedding_fn.embed(["Python programming"])
        results = backend.query(query_embeddings=query_embeddings, n_results=3)

        distances = results["distances"][0]
        # First result should be exact match (lowest distance)
        assert distances[0] <= distances[-1]

    def test_query_exact_match_has_zero_distance(self, backend, embedding_fn):
        """Querying with the exact same text should yield ~0 distance."""
        text = "Python programming language"
        backend.upsert(
            ids=["page1::chunk_0"],
            texts=[text],
            metadatas=[{"page_path": "page1", "chunk_index": 0}],
        )

        query_embeddings = embedding_fn.embed([text])
        results = backend.query(query_embeddings=query_embeddings, n_results=1)

        assert abs(results["distances"][0][0]) < 1e-5


class TestFAISSBackendReset:
    def test_reset_clears_all_data(self, backend):
        backend.upsert(
            ids=["page1::chunk_0", "page2::chunk_0"],
            texts=["Text A", "Text B"],
            metadatas=[
                {"page_path": "page1", "chunk_index": 0},
                {"page_path": "page2", "chunk_index": 0},
            ],
        )
        assert backend.count() == 2
        backend.reset()
        assert backend.count() == 0


class TestFAISSBackendPersistence:
    def test_index_persists_to_disk(self, faiss_dir, embedding_fn):
        from otterwiki_semantic_search.backends.faiss_backend import FAISSBackend

        # Create and populate
        backend = FAISSBackend(faiss_dir, embedding_fn)
        backend.upsert(
            ids=["page1::chunk_0"],
            texts=["Persistent data"],
            metadatas=[{"page_path": "page1", "chunk_index": 0}],
        )

        # Verify files exist
        assert os.path.exists(os.path.join(faiss_dir, "index.faiss"))
        assert os.path.exists(os.path.join(faiss_dir, "embeddings.json"))

        # Create a new backend from the same directory
        backend2 = FAISSBackend(faiss_dir, embedding_fn)
        assert backend2.count() == 1

        # Search should still work
        query_embeddings = embedding_fn.embed(["Persistent data"])
        results = backend2.query(query_embeddings=query_embeddings, n_results=1)
        assert results["ids"][0][0] == "page1::chunk_0"

    def test_sidecar_json_structure(self, faiss_dir, embedding_fn):
        from otterwiki_semantic_search.backends.faiss_backend import FAISSBackend

        backend = FAISSBackend(faiss_dir, embedding_fn)
        backend.upsert(
            ids=["page1::chunk_0"],
            texts=["Hello world"],
            metadatas=[{"page_path": "page1", "chunk_index": 0, "category": "test"}],
        )

        # Read the sidecar directly
        with open(os.path.join(faiss_dir, "embeddings.json"), "r") as f:
            sidecar = json.load(f)

        assert isinstance(sidecar, list)
        assert len(sidecar) == 1
        assert sidecar[0]["id"] == "page1::chunk_0"
        assert sidecar[0]["text"] == "Hello world"
        assert sidecar[0]["metadata"]["page_path"] == "page1"


class TestFAISSBackendDeduplication:
    """Test that search results can be deduplicated by page."""

    def test_multiple_chunks_per_page(self, backend, embedding_fn):
        """Multiple chunks from the same page should all be searchable."""
        backend.upsert(
            ids=["page1::chunk_0", "page1::chunk_1", "page1::chunk_2"],
            texts=[
                "Introduction to Python",
                "Python data types",
                "Python functions",
            ],
            metadatas=[
                {"page_path": "page1", "chunk_index": 0},
                {"page_path": "page1", "chunk_index": 1},
                {"page_path": "page1", "chunk_index": 2},
            ],
        )

        query_embeddings = embedding_fn.embed(["Introduction to Python"])
        results = backend.query(query_embeddings=query_embeddings, n_results=3)

        # All results should be from page1
        for meta in results["metadatas"][0]:
            assert meta["page_path"] == "page1"
