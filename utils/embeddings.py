"""
This module provides a factory for creating embedding providers and defines concrete implementations
for various embedding services like Cohere.
"""

import logging
from typing import List, Dict, Any, Optional
import cohere

from config.settings import settings
from interfaces.iembedding_provider import IEmbeddingProvider

logger = logging.getLogger(__name__)

class CohereEmbeddings(IEmbeddingProvider):
    """Cohere embeddings client with reranking for text embedding operations."""

    def __init__(self, model: str = "embed-english-v3.0"):
        if not settings.COHERE_API_KEY:
            raise ValueError("COHERE_API_KEY not found in environment")
        
        self.model = model
        self.client = cohere.Client(api_key=settings.COHERE_API_KEY)
        self._dimensions = 1024  # For embed-english-v3.0
        logger.info(f"Initialized Cohere embeddings: {model}")

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_texts(self, texts: List[str], input_type: str = "search_document", batch_size: int = 96) -> List[List[float]]:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            try:
                response = self.client.embed(
                    texts=batch_texts,
                    model=self.model,
                    input_type=input_type,
                )
                all_embeddings.extend(response.embeddings)
            except Exception as e:
                logger.error(f"Cohere embedding failed for batch {i//batch_size + 1}: {e}")
                all_embeddings.extend([[0.0] * self.dimensions] * len(batch_texts))
        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query], input_type="search_query")[0]

    def rerank_with_cohere(self, query: str, documents: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        if not documents:
            return []
        try:
            response = self.client.rerank(
                query=query,
                documents=documents,
                top_n=top_k,
                model="rerank-english-v3.0"
            )
            return [
                {
                    "document": documents[result.index],
                    "relevance_score": result.relevance_score,
                }
                for result in response.results
            ]
        except Exception as e:
            logger.error(f"Cohere reranking failed: {e}")
            return [{"document": doc, "relevance_score": 1.0 - (i * 0.1)} for i, doc in enumerate(documents[:top_k])]

class EmbeddingFactory:
    """Factory for creating embedding provider instances."""
    _providers = {
        "cohere": CohereEmbeddings
        # "openai": OpenAIEmbeddings, # Future implementation
    }

    @staticmethod
    def create_embedding_provider(provider_name: Optional[str] = None) -> IEmbeddingProvider:
        provider_name = provider_name or getattr(settings, 'EMBEDDING_PROVIDER', 'cohere')
        provider_class = EmbeddingFactory._providers.get(provider_name.lower())

        if not provider_class:
            raise ValueError(f"Unsupported embedding provider: {provider_name}")
        
        logger.info(f"Creating embedding provider: {provider_name}")
        return provider_class()

# Global default embedder instance, created by the factory
default_embedder = EmbeddingFactory.create_embedding_provider()