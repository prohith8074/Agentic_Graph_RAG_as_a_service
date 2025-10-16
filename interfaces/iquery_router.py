"""
Abstract interface for query routing components.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, AsyncGenerator, Optional


class IQueryRouter(ABC):
    """Abstract interface for intelligent query routing."""

    @abstractmethod
    async def route_query(self, query: str) -> Dict[str, Any]:
        """
        Route query to appropriate backend(s) and return results.

        Args:
            query: User query string

        Returns:
            Query result with answer and metadata
        """
        pass

    @abstractmethod
    def get_routing_stats(self) -> Dict[str, Any]:
        """
        Get routing statistics and system status.

        Returns:
            Dictionary with routing statistics
        """
        pass

    @abstractmethod
    async def route_query_advanced(self, query: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Advanced routing with streaming and multi-step reasoning.

        Args:
            query: User query string

        Yields:
            Query results with intermediate steps
        """
        pass