"""
Abstract interface for embedding components.
"""

from abc import ABC, abstractmethod
from typing import List, Any, Optional, Dict


class IEmbeddings(ABC):
    """Abstract interface for embedding functionality."""

    @abstractmethod
    def embed_texts(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process in each batch

        Returns:
            List of embedding vectors
        """
        raise NotImplementedError("embed_texts method must be implemented by concrete classes")

    @abstractmethod
    def embed_single_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        raise NotImplementedError("embed_single_text method must be implemented by concrete classes")

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a user query.

        Args:
            query: User query text

        Returns:
            Query embedding vector
        """
        raise NotImplementedError("embed_query method must be implemented by concrete classes")

    @abstractmethod
    def rerank_with_cohere(self, query: str, documents: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Rerank documents using Cohere reranking.

        Args:
            query: Search query
            documents: List of document texts to rerank
            top_k: Number of top results to return

        Returns:
            Reranked documents with relevance scores
        """
        raise NotImplementedError("rerank_with_cohere method must be implemented by concrete classes")

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current embedding model.

        Returns:
            Model information dictionary
        """
        raise NotImplementedError("get_model_info method must be implemented by concrete classes")