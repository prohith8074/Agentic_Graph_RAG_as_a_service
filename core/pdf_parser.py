"""
Concrete implementation of IPDFParser using LlamaParse.
"""

import logging
from typing import List, Any, Optional
from interfaces import IPDFParser
from llama_parse import LlamaParse
from llama_index.core import Document
from config.settings import settings
from utils.opik_tracer import opik_tracer

logger = logging.getLogger(__name__)


class PDFParser(IPDFParser):
    """PDF parser implementation using LlamaParse."""

    def __init__(self):
        """Initialize PDF parser with LlamaParse."""
        self.llama_parse = LlamaParse(
            api_key=settings.LLAMA_CLOUD_API_KEY,
            result_type="markdown",
            verbose=True,
            language="en"
        )
        logger.info("PDF Parser initialized with LlamaParse")

    def parse_pdf(self, pdf_path: str) -> Optional[List[Any]]:
        """
        Parse PDF document using LlamaParse.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of parsed documents or None if parsing failed
        """
        try:
            logger.info(f"Parsing PDF: {pdf_path}")

            # Parse with LlamaParse
            documents = self.llama_parse.load_data(pdf_path)

            if not documents:
                logger.error("No documents returned from LlamaParse")
                return None

            # Track with Opik
            opik_tracer.track_llm_call(
                provider="llamaparse",
                model="llama-parse",
                prompt=f"Parse PDF: {pdf_path}",
                response=f"Extracted {len(documents)} documents",
                metadata={"operation": "pdf_parsing", "pdf_path": pdf_path}
            )

            logger.info(f"Successfully parsed PDF into {len(documents)} documents")
            return documents

        except Exception as e:
            logger.error(f"PDF parsing failed: {e}")
            opik_tracer.log_error("pdf_parsing_error", str(e), {"pdf_path": pdf_path})
            return None

    def extract_nodes(self, documents: List[Any]) -> Optional[List[Any]]:
        """
        Convert parsed documents to processing nodes.

        Args:
            documents: Parsed document objects from LlamaParse

        Returns:
            List of processing nodes or None if extraction failed
        """
        try:
            if not documents:
                logger.warning("No documents provided for node extraction")
                return None

            nodes = []

            for doc in documents:
                # Convert LlamaParse documents to LlamaIndex nodes
                if hasattr(doc, 'text'):
                    # Create Document objects for LlamaIndex
                    llama_doc = Document(text=doc.text, metadata=doc.metadata or {})

                    # For LlamaParse documents, create nodes with proper structure
                    # LlamaParse returns documents with text and metadata
                    node = {
                        'id_': f"{doc.doc_id or 'doc'}_{0}",  # Use 0 since we don't chunk
                        'text': doc.text,
                        'metadata': doc.metadata or {},
                        'embedding': None  # Will be populated later
                    }
                    nodes.append(node)

            if not nodes:
                logger.warning("No nodes extracted from documents")
                return None

            # Track node extraction
            opik_tracer.track_pipeline_step(
                "node_extraction",
                {"documents": len(documents)},
                {"nodes": len(nodes)},
                0.0
            )

            logger.info(f"Extracted {len(nodes)} nodes from {len(documents)} documents")

            # Debug: Print first node structure
            if nodes:
                logger.info(f"Sample node structure: {list(nodes[0].keys())}")

            return nodes

        except Exception as e:
            logger.error(f"Node extraction failed: {e}")
            # Use standard logging instead of opik_tracer.log_error
            logger.error(f"OpikTracer node extraction error: {e}")
            return None

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: Input text
            chunk_size: Size of each chunk
            overlap: Overlap between chunks

        Returns:
            List of text chunks
        """
        if not text:
            return []

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Adjust end to not split words
            if end < len(text):
                # Find last space within chunk
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space

            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)

            # Move start position with overlap
            start = end - overlap

            # Ensure we don't get stuck
            if start >= len(text) or len(chunk) < overlap:
                break

        return chunks