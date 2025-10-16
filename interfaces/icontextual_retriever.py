"""
Abstract interface for contextual retrieval components.
"""

from abc import ABC, abstractmethod
from typing import List, Any, Optional, AsyncGenerator


class IContextualRetriever(ABC):
    """Abstract interface for contextual retrieval functionality."""

    @abstractmethod
    async def contextualize_chunks(self, nodes: List[Any]) -> Optional[List[Any]]:
        """
        Generate contextual summaries for text chunks.

        Args:
            nodes: Processing nodes to contextualize

        Returns:
            Contextualized chunks or None if failed
        """
        raise NotImplementedError("contextualize_chunks method must be implemented by concrete classes")

    @abstractmethod
    def get_sliding_window_context(self, chunk: str, document: str,
                                 window_size: int = 2048) -> str:
        """
        Get sliding window context around a chunk.

        Args:
            chunk: Target chunk
            document: Full document text
            window_size: Size of context window

        Returns:
            Contextual text around chunk
        """
        raise NotImplementedError("get_sliding_window_context method must be implemented by concrete classes")

    @abstractmethod
    async def generate_contextual_summary(self, text: str) -> str:
        """
        Generate contextual summary for text using LLM.

        Args:
            text: Input text to summarize

        Returns:
            Contextual summary
        """
        raise NotImplementedError("generate_contextual_summary method must be implemented by concrete classes")