"""
Abstract interface for graph building components.
"""

from abc import ABC, abstractmethod
from typing import List, Any, Optional, Dict


class IGraphBuilder(ABC):
    """Abstract interface for graph building functionality."""

    @abstractmethod
    def merge_and_deduplicate_graphs(self, graphs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge multiple knowledge graphs and remove duplicates.

        Args:
            graphs: List of knowledge graphs to merge

        Returns:
            Merged and deduplicated graph
        """
        raise NotImplementedError("merge_and_deduplicate_graphs method must be implemented by concrete classes")

    @abstractmethod
    def validate_graph_structure(self, graph: Dict[str, Any]) -> bool:
        """
        Validate knowledge graph structure.

        Args:
            graph: Knowledge graph to validate

        Returns:
            True if graph structure is valid
        """
        raise NotImplementedError("validate_graph_structure method must be implemented by concrete classes")

    @abstractmethod
    def get_graph_statistics(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph.

        Args:
            graph: Knowledge graph to analyze

        Returns:
            Dictionary with graph statistics
        """
        raise NotImplementedError("get_graph_statistics method must be implemented by concrete classes")