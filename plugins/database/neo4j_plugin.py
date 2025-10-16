"""
Neo4j database adapter plugin.
"""

from typing import Any
from interfaces import IDatabaseManager
from knowledge_graph.neo4j_manager import Neo4jManager


class Neo4jPlugin:
    """Plugin for Neo4j database adapter."""

    def __init__(self):
        """Initialize Neo4j plugin."""
        self.name = "neo4j"
        self.description = "Neo4j graph database adapter"

    def create_manager(self) -> IDatabaseManager:
        """
        Create Neo4j database manager instance.

        Returns:
            Neo4j database manager
        """
        return Neo4jManager()

    def get_capabilities(self) -> dict:
        """
        Get plugin capabilities.

        Returns:
            Dictionary of capabilities
        """
        return {
            "graph_queries": True,
            "cypher_support": True,
            "schema_constraints": True,
            "full_text_search": False,
            "vector_search": False
        }

    def is_available(self) -> bool:
        """
        Check if Neo4j is available.

        Returns:
            True if Neo4j can be used
        """
        try:
            # Check if Neo4j connection parameters are configured
            from config.settings import settings
            return bool(settings.NEO4J_URI and settings.NEO4J_USER)
        except Exception:
            return False