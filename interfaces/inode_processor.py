"""
Abstract interface for node processing components.
"""

from abc import ABC, abstractmethod
from typing import List, Any, Optional


class INodeProcessor(ABC):
    """Abstract interface for node processing functionality."""

    @abstractmethod
    def validate_nodes(self, nodes: List[Any]) -> bool:
        """
        Validate processing nodes.

        Args:
            nodes: List of processing nodes

        Returns:
            True if all nodes are valid
        """
        raise NotImplementedError("validate_nodes method must be implemented by concrete classes")

    @abstractmethod
    def process_nodes(self, nodes: List[Any]) -> Optional[List[Any]]:
        """
        Process and transform nodes.

        Args:
            nodes: Input processing nodes

        Returns:
            Processed nodes or None if processing failed
        """
        raise NotImplementedError("process_nodes method must be implemented by concrete classes")

    @abstractmethod
    def filter_nodes(self, nodes: List[Any], criteria: dict) -> List[Any]:
        """
        Filter nodes based on criteria.

        Args:
            nodes: Input processing nodes
            criteria: Filtering criteria

        Returns:
            Filtered list of nodes
        """
        raise NotImplementedError("filter_nodes method must be implemented by concrete classes")