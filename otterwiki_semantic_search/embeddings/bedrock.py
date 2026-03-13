"""Bedrock titan-embed-text-v2 embedding adapter."""

import json
import logging

from otterwiki_semantic_search.embeddings.base import EmbeddingFunction

log = logging.getLogger(__name__)

# Titan Text Embeddings V2 supports 256, 512, 1024 dimensions.
# Default is 1024.
DEFAULT_DIMENSIONALITY = 1024
MODEL_ID = "amazon.titan-embed-text-v2:0"


class BedrockEmbeddingFunction(EmbeddingFunction):
    """Embedding function using AWS Bedrock titan-embed-text-v2.

    Requires boto3 and valid AWS credentials (via env vars, instance
    profile, or any standard boto3 credential chain).
    """

    def __init__(self, region_name=None, dimensions=DEFAULT_DIMENSIONALITY):
        """Initialize the Bedrock embedding adapter.

        Args:
            region_name: AWS region for Bedrock. Defaults to boto3's default.
            dimensions: Output vector dimensionality (256, 512, or 1024).
        """
        import boto3

        self._dimensions = dimensions
        kwargs = {}
        if region_name:
            kwargs["region_name"] = region_name
        self._client = boto3.client("bedrock-runtime", **kwargs)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using Bedrock titan-embed-text-v2.

        Calls the model once per text (Titan embed does not support
        batch embedding in a single API call).
        """
        vectors = []
        for text in texts:
            body = json.dumps({
                "inputText": text,
                "dimensions": self._dimensions,
                "normalize": True,
            })
            response = self._client.invoke_model(
                modelId=MODEL_ID,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            result = json.loads(response["body"].read())
            vectors.append(result["embedding"])
        return vectors

    @property
    def dimensionality(self) -> int:
        return self._dimensions
