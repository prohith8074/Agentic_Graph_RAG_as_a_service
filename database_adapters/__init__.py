"""
Database adapters module for pluggable graph database support.
Provides unified interfaces for different graph databases (Neo4j, Neptune, etc.).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from enum import Enum

class QueryLanguage(Enum):
    """Supported query languages for graph databases."""
    CYPHER = "cypher"
    GREMLIN = "gremlin"
    SPARQL = "sparql"

class GraphDatabaseAdapter(ABC):
    """Abstract base class for graph database adapters."""

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the database."""
        pass

    @abstractmethod
    def disconnect(self):
        """Close database connection."""
        pass

    @abstractmethod
    def execute_query(self, query: str, language: QueryLanguage = QueryLanguage.CYPHER) -> List[Dict[str, Any]]:
        """Execute a query and return results."""
        pass

    @abstractmethod
    def create_constraints(self) -> bool:
        """Create necessary database constraints."""
        pass

    @abstractmethod
    def clear_database(self) -> bool:
        """Clear all data from database."""
        pass

    @abstractmethod
    def populate_graph(self, graph_data: Dict[str, Any]) -> bool:
        """Populate database with knowledge graph data."""
        pass

    @abstractmethod
    def get_schema_info(self) -> Dict[str, Any]:
        """Get database schema information."""
        pass

    @abstractmethod
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        pass

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on database."""
        pass

    @property
    @abstractmethod
    def supported_languages(self) -> List[QueryLanguage]:
        """Return list of supported query languages."""
        pass

    @property
    @abstractmethod
    def database_type(self) -> str:
        """Return database type identifier."""
        pass