"""Unit tests for the Bedrock embedding adapter — mocked, no AWS required."""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_boto3():
    """Mock boto3 module so tests don't require AWS credentials."""
    mock_module = MagicMock()
    with patch.dict(sys.modules, {"boto3": mock_module}):
        yield mock_module


class TestBedrockEmbeddingFunction:
    def test_embed_single_text(self, mock_boto3):
        """Test embedding a single text string."""
        from otterwiki_semantic_search.embeddings.bedrock import (
            BedrockEmbeddingFunction,
            DEFAULT_DIMENSIONALITY,
        )

        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        fake_embedding = [0.1] * DEFAULT_DIMENSIONALITY
        mock_response_body = MagicMock()
        mock_response_body.read.return_value = json.dumps(
            {"embedding": fake_embedding}
        ).encode()
        mock_client.invoke_model.return_value = {"body": mock_response_body}

        fn = BedrockEmbeddingFunction()
        result = fn.embed(["Hello world"])

        assert len(result) == 1
        assert len(result[0]) == DEFAULT_DIMENSIONALITY
        assert result[0] == fake_embedding

        # Verify the API call
        mock_client.invoke_model.assert_called_once()
        call_kwargs = mock_client.invoke_model.call_args[1]
        assert call_kwargs["modelId"] == "amazon.titan-embed-text-v2:0"
        body = json.loads(call_kwargs["body"])
        assert body["inputText"] == "Hello world"
        assert body["dimensions"] == DEFAULT_DIMENSIONALITY
        assert body["normalize"] is True

    def test_embed_multiple_texts(self, mock_boto3):
        """Test embedding multiple texts (one API call per text)."""
        from otterwiki_semantic_search.embeddings.bedrock import (
            BedrockEmbeddingFunction,
        )

        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        dim = 1024
        embeddings_data = [[0.1] * dim, [0.2] * dim, [0.3] * dim]

        def mock_invoke(modelId, body, contentType, accept):
            parsed = json.loads(body)
            idx = ["Text A", "Text B", "Text C"].index(parsed["inputText"])
            resp_body = MagicMock()
            resp_body.read.return_value = json.dumps(
                {"embedding": embeddings_data[idx]}
            ).encode()
            return {"body": resp_body}

        mock_client.invoke_model.side_effect = mock_invoke

        fn = BedrockEmbeddingFunction(dimensions=dim)
        result = fn.embed(["Text A", "Text B", "Text C"])

        assert len(result) == 3
        assert mock_client.invoke_model.call_count == 3

    def test_dimensionality_property(self, mock_boto3):
        """Test that dimensionality returns the configured value."""
        from otterwiki_semantic_search.embeddings.bedrock import (
            BedrockEmbeddingFunction,
        )

        mock_boto3.client.return_value = MagicMock()

        fn_default = BedrockEmbeddingFunction()
        assert fn_default.dimensionality == 1024

        fn_small = BedrockEmbeddingFunction(dimensions=256)
        assert fn_small.dimensionality == 256

        fn_medium = BedrockEmbeddingFunction(dimensions=512)
        assert fn_medium.dimensionality == 512

    def test_custom_region(self, mock_boto3):
        """Test that region_name is passed to boto3 client."""
        from otterwiki_semantic_search.embeddings.bedrock import (
            BedrockEmbeddingFunction,
        )

        mock_boto3.client.return_value = MagicMock()

        BedrockEmbeddingFunction(region_name="us-west-2")
        mock_boto3.client.assert_called_with(
            "bedrock-runtime", region_name="us-west-2"
        )

    def test_no_region_uses_default(self, mock_boto3):
        """Test that no region_name uses boto3 default."""
        from otterwiki_semantic_search.embeddings.bedrock import (
            BedrockEmbeddingFunction,
        )

        mock_boto3.client.return_value = MagicMock()

        BedrockEmbeddingFunction()
        mock_boto3.client.assert_called_with("bedrock-runtime")
