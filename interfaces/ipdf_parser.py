"""
Abstract interface for PDF parsing components.
"""

from abc import ABC, abstractmethod
from typing import List, Any, Optional
from pathlib import Path


class IPDFParser(ABC):
    """Abstract interface for PDF parsing functionality."""

    @abstractmethod
    def parse_pdf(self, pdf_path: str) -> Optional[List[Any]]:
        """
        Parse PDF document and extract structured content.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of parsed documents or None if parsing failed
        """
        raise NotImplementedError("parse_pdf method must be implemented by concrete classes")

    @abstractmethod
    def extract_nodes(self, documents: List[Any]) -> Optional[List[Any]]:
        """
        Convert parsed documents to processing nodes.

        Args:
            documents: Parsed document objects

        Returns:
            List of processing nodes or None if extraction failed
        """
        raise NotImplementedError("extract_nodes method must be implemented by concrete classes")