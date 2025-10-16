"""
Abstract interface for Vector Database Adapters.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class IVectorDatabase(ABC):
    """Abstract base class for vector database adapters."""

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the database."""
        pass

    @abstractmethod
    def retrieve_similar_chunks(self, query_embedding: List[float], top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve similar chunks from the vector store."""
        pass

    @abstractmethod
    def upsert_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> bool:
        """Add or update chunks in the vector store."""
        pass

    @abstractmethod
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector collection."""
        pass
