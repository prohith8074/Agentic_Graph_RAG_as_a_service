"""
Abstract interface for memory management components.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class IMemoryManager(ABC):
    """Abstract interface for memory management systems."""

    @abstractmethod
    def store_conversation_context(self, session_id: str, context: Dict[str, Any]) -> bool:
        """
        Store conversation context.

        Args:
            session_id: Unique session identifier
            context: Context data to store

        Returns:
            True if stored successfully
        """
        pass

    @abstractmethod
    def get_conversation_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve conversation context.

        Args:
            session_id: Unique session identifier

        Returns:
            Context data or None if not found
        """
        pass

    @abstractmethod
    def add_to_conversation_history(self, session_id: str, message: Dict[str, Any]) -> bool:
        """
        Add message to conversation history.

        Args:
            session_id: Unique session identifier
            message: Message data

        Returns:
            True if added successfully
        """
        pass

    @abstractmethod
    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get conversation history for session.

        Args:
            session_id: Unique session identifier

        Returns:
            List of conversation messages
        """
        pass

    @abstractmethod
    def store_long_term_memory(self, content: str, metadata: Dict[str, Any],
                              collection_name: str = "long_term_memory") -> bool:
        """
        Store content in long-term memory.

        Args:
            content: Content to store
            metadata: Associated metadata
            collection_name: Memory collection name

        Returns:
            True if stored successfully
        """
        pass

    @abstractmethod
    def retrieve_long_term_memory(self, query: str, collection_name: str = "long_term_memory",
                                 top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant content from long-term memory.

        Args:
            query: Search query
            collection_name: Memory collection name
            top_k: Number of results to return

        Returns:
            List of relevant content with metadata
        """
        pass

    @abstractmethod
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive memory usage statistics.

        Returns:
            Dictionary with memory statistics
        """
        pass