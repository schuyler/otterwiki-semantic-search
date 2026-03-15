"""Unit tests for the ONNX embedding adapter — mocked, no ONNX model required."""

import sys
from unittest.mock import MagicMock, patch

import pytest


DIMENSIONALITY = 384


@pytest.fixture(autouse=True)
def mock_chromadb_onnx():
    """Mock ChromaDB's ONNXMiniLM_L6_V2 so tests don't need the ONNX model."""
    mock_ef_module = MagicMock()

    def fake_call(texts):
        return [[0.1] * DIMENSIONALITY for _ in texts]

    mock_instance = MagicMock()
    mock_instance.side_effect = fake_call
    mock_ef_module.ONNXMiniLM_L6_V2.return_value = mock_instance

    with patch.dict(
        sys.modules,
        {"chromadb.utils.embedding_functions": mock_ef_module},
    ):
        # Clear cached import so each test gets a fresh ONNXEmbeddingFunction
        sys.modules.pop("otterwiki_semantic_search.embeddings.onnx_embedding", None)
        yield mock_ef_module, mock_instance


class TestONNXEmbeddingFunction:
    def test_embed_single_text(self, mock_chromadb_onnx):
        """Test embedding a single text string."""
        _, mock_instance = mock_chromadb_onnx

        from otterwiki_semantic_search.embeddings.onnx_embedding import (
            ONNXEmbeddingFunction,
        )

        fn = ONNXEmbeddingFunction()
        result = fn.embed(["Hello world"])

        assert len(result) == 1
        assert len(result[0]) == DIMENSIONALITY
        mock_instance.assert_called_once_with(["Hello world"])

    def test_embed_multiple_texts(self, mock_chromadb_onnx):
        """Test embedding multiple texts in a single batch."""
        _, mock_instance = mock_chromadb_onnx

        from otterwiki_semantic_search.embeddings.onnx_embedding import (
            ONNXEmbeddingFunction,
        )

        fn = ONNXEmbeddingFunction()
        result = fn.embed(["Text A", "Text B", "Text C"])

        assert len(result) == 3
        assert all(len(v) == DIMENSIONALITY for v in result)
        mock_instance.assert_called_with(["Text A", "Text B", "Text C"])

    def test_dimensionality_property(self, mock_chromadb_onnx):
        """Test that dimensionality returns 384 (MiniLM-L6-v2)."""
        from otterwiki_semantic_search.embeddings.onnx_embedding import (
            ONNXEmbeddingFunction,
        )

        fn = ONNXEmbeddingFunction()
        assert fn.dimensionality == DIMENSIONALITY

    def test_delegates_to_chromadb_class(self, mock_chromadb_onnx):
        """Verify that embed() delegates to the ChromaDB ONNXMiniLM_L6_V2 __call__."""
        mock_ef_module, mock_instance = mock_chromadb_onnx

        from otterwiki_semantic_search.embeddings.onnx_embedding import (
            ONNXEmbeddingFunction,
        )

        fn = ONNXEmbeddingFunction()

        # Verify the class was instantiated
        mock_ef_module.ONNXMiniLM_L6_V2.assert_called_once()

        fn.embed(["test"])
        mock_instance.assert_called_once_with(["test"])

    def test_conforms_to_embedding_function_interface(self, mock_chromadb_onnx):
        """Verify that ONNXEmbeddingFunction is a proper EmbeddingFunction subclass."""
        from otterwiki_semantic_search.embeddings.base import EmbeddingFunction
        from otterwiki_semantic_search.embeddings.onnx_embedding import (
            ONNXEmbeddingFunction,
        )

        fn = ONNXEmbeddingFunction()

        assert isinstance(fn, EmbeddingFunction)
        assert hasattr(fn, "embed")
        assert hasattr(fn, "dimensionality")
