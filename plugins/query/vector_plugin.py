"""
Vector search query plugin.
"""

from typing import Dict, Any, List
from query.vector_query import VectorQueryInterface


class VectorPlugin:
    """Plugin for vector-based query handling."""

    def __init__(self):
        """Initialize vector plugin."""
        self.name = "vector"
        self.description = "Semantic vector search plugin"
        self._vector_interface = None

    def create_handler(self) -> VectorQueryInterface:
        """
        Create vector query handler instance.

        Returns:
            Vector query interface
        """
        if self._vector_interface is None:
            self._vector_interface = VectorQueryInterface()
        return self._vector_interface

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get plugin capabilities.

        Returns:
            Dictionary of capabilities
        """
        return {
            "query_types": ["semantic_search", "similarity_search"],
            "supports_filtering": False,
            "supports_hybrid": False,
            "max_results": 10,
            "embedding_model": "nomic-embed-text-v1.5"
        }

    def can_handle_query(self, query: str, context: Dict[str, Any]) -> float:
        """
        Determine if this plugin can handle the query.

        Args:
            query: User query
            context: Query context

        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Simple heuristic: high confidence for descriptive queries
        query_lower = query.lower()
        descriptive_keywords = [
            'what', 'how', 'describe', 'explain', 'tell me about',
            'what are', 'how does', 'what is'
        ]

        if any(keyword in query_lower for keyword in descriptive_keywords):
            return 0.8
        elif len(query.split()) > 3:  # Longer queries tend to be semantic
            return 0.6
        else:
            return 0.3

    def process_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process query using vector search.

        Args:
            query: User query
            context: Query context

        Returns:
            Query result
        """
        handler = self.create_handler()
        return handler.query(query)