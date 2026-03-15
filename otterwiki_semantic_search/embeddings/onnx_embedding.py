"""ONNX MiniLM-L6-v2 embedding adapter using ChromaDB's bundled model."""

import logging

from otterwiki_semantic_search.embeddings.base import EmbeddingFunction

log = logging.getLogger(__name__)

# all-MiniLM-L6-v2 produces 384-dimensional vectors
DIMENSIONALITY = 384


class ONNXEmbeddingFunction(EmbeddingFunction):
    """Embedding function using ChromaDB's ONNX MiniLM-L6-v2 model.

    Wraps chromadb.utils.embedding_functions.ONNXMiniLM_L6_V2 to match
    the EmbeddingFunction interface expected by the FAISS backend.

    This avoids the ~2GB torch + sentence-transformers dependency by
    using onnxruntime (already installed as a chromadb dependency).
    The model is downloaded and cached automatically on first use.
    """

    def __init__(self):
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

        self._delegate = ONNXMiniLM_L6_V2()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using the ONNX MiniLM-L6-v2 model.

        Returns normalized 384-dimensional vectors.
        """
        return self._delegate(texts)

    @property
    def dimensionality(self) -> int:
        return DIMENSIONALITY
