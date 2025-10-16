"""
Graph query plugin.
"""

from typing import Dict, Any, List
from query.graph_query import GraphQueryInterface


class GraphPlugin:
    """Plugin for graph-based query handling."""

    def __init__(self):
        """Initialize graph plugin."""
        self.name = "graph"
        self.description = "Structured graph query plugin"
        self._graph_interface = None

    def create_handler(self) -> GraphQueryInterface:
        """
        Create graph query handler instance.

        Returns:
            Graph query interface
        """
        if self._graph_interface is None:
            from knowledge_graph.neo4j_manager import Neo4jManager
            neo4j_manager = Neo4jManager()
            self._graph_interface = GraphQueryInterface(neo4j_manager)
        return self._graph_interface

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get plugin capabilities.

        Returns:
            Dictionary of capabilities
        """
        return {
            "query_types": ["relationship_queries", "structural_queries", "pattern_matching"],
            "supports_filtering": True,
            "supports_hybrid": True,
            "max_results": 50,
            "query_language": "cypher"
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
        # Simple heuristic: high confidence for relationship/structural queries
        query_lower = query.lower()
        graph_keywords = [
            'relationship', 'connected', 'related', 'structure',
            'components', 'architecture', 'hierarchy', 'dependencies',
            'links', 'connections', 'graph', 'network'
        ]

        if any(keyword in query_lower for keyword in graph_keywords):
            return 0.9
        elif 'how' in query_lower and 'work' in query_lower:
            return 0.7
        elif any(word in query_lower for word in ['what', 'which', 'who']):
            return 0.6
        else:
            return 0.2

    def process_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process query using graph search.

        Args:
            query: User query
            context: Query context

        Returns:
            Query result
        """
        handler = self.create_handler()
        return handler.query(query)