"""
Abstract interface for Embedding Providers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any

class IEmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed_texts(self, texts: List[str], input_type: str) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Generate an embedding for a single query."""
        pass

    @abstractmethod
    def rerank_with_cohere(self, query: str, documents: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """Rerank documents using Cohere's reranking API."""
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """The dimension of the embeddings produced by the model."""
        pass
