"""Unit tests for index.search() — mock backend, no ChromaDB."""
import pytest
from unittest.mock import MagicMock
from otterwiki_semantic_search import index as idx


def _make_results(rows):
    """rows: list of (doc_id, document_text, metadata_dict, distance)"""
    return {
        "ids": [[r[0] for r in rows]],
        "documents": [[r[1] for r in rows]],
        "metadatas": [[r[2] for r in rows]],
        "distances": [[r[3] for r in rows]],
    }


def _mock_backend(rows):
    backend = MagicMock()
    backend.query.return_value = _make_results(rows)
    return backend


class TestSearchEnrichment:
    def test_result_has_text_field(self):
        long_text = "word " * 60  # 300 chars
        meta = {
            "page_path": "Test/Page",
            "chunk_index": 0,
            "section": "Intro",
            "section_path": "My Doc > Intro",
            "total_chunks": 3,
            "page_word_count": 500,
        }
        backend = _mock_backend([("Test/Page::chunk_0", long_text, meta, 0.1)])
        results = idx.search("anything", n=5, backend=backend)
        assert len(results) == 1
        assert results[0]["text"] == long_text
        assert len(results[0]["snippet"]) <= 153  # 150 + "..."
        assert results[0]["snippet"].endswith("...")

    def test_result_has_section_metadata(self):
        meta = {
            "page_path": "Test/Page",
            "chunk_index": 0,
            "section": "Intro",
            "section_path": "My Doc > Intro",
            "total_chunks": 3,
            "page_word_count": 500,
        }
        backend = _mock_backend([("Test/Page::chunk_0", "some text", meta, 0.1)])
        results = idx.search("anything", n=5, backend=backend)
        assert results[0]["section"] == "Intro"
        assert results[0]["section_path"] == "My Doc > Intro"
        assert results[0]["chunk_index"] == 0
        assert results[0]["total_chunks"] == 3
        assert results[0]["page_word_count"] == 500

    def test_snippet_still_present_for_short_text(self):
        meta = {"page_path": "P", "chunk_index": 0}
        backend = _mock_backend([("P::chunk_0", "short text", meta, 0.1)])
        results = idx.search("q", n=5, backend=backend)
        assert results[0]["snippet"] == "short text"
        assert "..." not in results[0]["snippet"]

    def test_missing_section_metadata_handled(self):
        meta = {"page_path": "P"}  # old-format, no section fields
        backend = _mock_backend([("P::chunk_0", "text", meta, 0.1)])
        results = idx.search("q", n=5, backend=backend)
        assert results[0]["section"] is None
        assert results[0]["section_path"] is None

    def test_snippet_strips_prefix_with_brackets_in_title(self):
        # section_path contains ']' — old split-on-"] " approach would mangle this
        section_path = "Python [v3] Guide > Intro"
        prefix = f"[{section_path}] "
        body = "This is the intro content for Python v3."
        doc_text = prefix + body
        meta = {
            "page_path": "Python [v3] Guide",
            "chunk_index": 0,
            "section": "Intro",
            "section_path": section_path,
            "total_chunks": 1,
            "page_word_count": 100,
        }
        backend = _mock_backend([("Python [v3] Guide::chunk_0", doc_text, meta, 0.1)])
        results = idx.search("q", n=5, backend=backend)
        assert len(results) == 1
        # Snippet must equal the body, not a mangled fragment
        assert results[0]["snippet"] == body
        # Full text is preserved
        assert results[0]["text"] == doc_text


class TestConfigurableDedup:
    def _make_page_rows(self, page, count, base_dist=0.1):
        return [
            (f"{page}::chunk_{i}", f"text {i}", {
                "page_path": page, "chunk_index": i,
                "section": "", "section_path": page,
                "total_chunks": count, "page_word_count": 400,
            }, base_dist * (i + 1))
            for i in range(count)
        ]

    def test_default_max_chunks_per_page_is_2(self):
        rows = self._make_page_rows("P", 4)
        backend = _mock_backend(rows)
        results = idx.search("q", n=10, backend=backend)
        assert len(results) == 2
        assert results[0]["distance"] == pytest.approx(0.1)
        assert results[1]["distance"] == pytest.approx(0.2)

    def test_max_chunks_per_page_1(self):
        rows = self._make_page_rows("P", 4)
        backend = _mock_backend(rows)
        results = idx.search("q", n=10, max_chunks_per_page=1, backend=backend)
        assert len(results) == 1

    def test_max_chunks_per_page_5(self):
        rows = self._make_page_rows("P", 4)
        backend = _mock_backend(rows)
        results = idx.search("q", n=10, max_chunks_per_page=5, backend=backend)
        assert len(results) == 4

    def test_max_chunks_per_page_capped_at_5(self):
        rows = self._make_page_rows("P", 8)
        backend = _mock_backend(rows)
        results = idx.search("q", n=10, max_chunks_per_page=99, backend=backend)
        assert len(results) == 5

    def test_dedup_across_multiple_pages(self):
        rows_a = self._make_page_rows("A", 3, base_dist=0.1)
        rows_b = self._make_page_rows("B", 3, base_dist=0.15)
        backend = _mock_backend(rows_a + rows_b)
        results = idx.search("q", n=10, max_chunks_per_page=2, backend=backend)
        assert len(results) == 4
        # Sorted by distance: A:0.1, B:0.15, A:0.2, B:0.30
        assert results[0]["path"] == "A"
        assert results[1]["path"] == "B"

    def test_n_limits_final_result_count(self):
        rows = self._make_page_rows("P", 4)
        backend = _mock_backend(rows)
        results = idx.search("q", n=1, max_chunks_per_page=4, backend=backend)
        assert len(results) == 1
