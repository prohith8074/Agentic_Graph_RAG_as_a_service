"""
Abstract interface for database management components.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class IDatabaseManager(ABC):
    """Abstract interface for database operations."""

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the database.

        Returns:
            True if connection successful
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close database connection."""
        pass

    @abstractmethod
    def create_constraints(self) -> bool:
        """
        Create database constraints and schema.

        Returns:
            True if constraints created successfully
        """
        pass

    @abstractmethod
    def clear_database(self) -> bool:
        """
        Clear all data from database.

        Returns:
            True if database cleared successfully
        """
        pass

    @abstractmethod
    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute database query.

        Args:
            query: Query string
            parameters: Query parameters

        Returns:
            Query results
        """
        pass