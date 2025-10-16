"""
Abstract interface for entity extraction components.
"""

from abc import ABC, abstractmethod
from typing import List, Any, Optional, AsyncGenerator, Dict


class IEntityExtractor(ABC):
    """Abstract interface for entity extraction functionality."""

    @abstractmethod
    async def process_chunks_batch(self, chunks: List[str]) -> Optional[List[Dict[str, Any]]]:
        """
        Process batch of text chunks to extract entities and relationships.

        Args:
            chunks: List of text chunks to process

        Returns:
            List of knowledge graphs or None if extraction failed
        """
        raise NotImplementedError("process_chunks_batch method must be implemented by concrete classes")

    @abstractmethod
    async def extract_entities_and_relationships(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract entities and relationships from text.

        Args:
            text: Input text to analyze

        Returns:
            Knowledge graph dictionary or None if extraction failed
        """
        raise NotImplementedError("extract_entities_and_relationships method must be implemented by concrete classes")

    @abstractmethod
    def validate_extracted_graph(self, graph: Dict[str, Any]) -> bool:
        """
        Validate extracted knowledge graph structure.

        Args:
            graph: Knowledge graph to validate

        Returns:
            True if graph is valid
        """
        raise NotImplementedError("validate_extracted_graph method must be implemented by concrete classes")