"""
Node processing module for extracting and processing text chunks from parsed documents.
Handles text extraction, metadata processing, and node formatting.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class NodeProcessor:
    """Handles processing and formatting of document nodes."""

    @staticmethod
    def extract_text_and_metadata(nodes: List) -> List[Dict[str, Any]]:
        """
        Extract text content and metadata from nodes.

        Args:
            nodes: List of node objects from LlamaIndex

        Returns:
            List of dictionaries containing id, metadata, and text
        """
        try:
            extracted_text = []

            for i, node in enumerate(nodes):
                logger.debug(f"Processing node #{i+1}: {node.id_}")

                # Extract basic information
                node_id = node.id_
                metadata = node.metadata
                text_content = node.text.replace('\n', ' ')

                # Create structured node data
                node_data = {
                    'id_': node_id,
                    'metadata': metadata,
                    'text': text_content
                }

                extracted_text.append(node_data)

            logger.info(f"Successfully processed {len(extracted_text)} nodes")
            return extracted_text

        except Exception as e:
            logger.error(f"Error processing nodes: {e}")
            raise

    @staticmethod
    def print_node_summary(nodes: List[Dict[str, Any]], max_nodes: int = 5) -> None:
        """
        Print a summary of extracted nodes for debugging.

        Args:
            nodes: List of processed node dictionaries
            max_nodes: Maximum number of nodes to print
        """
        for i, node in enumerate(nodes[:max_nodes]):
            print(f"\n--- Node #{i+1} ---")
            print(f"ID: {node['id_']}")
            print(f"Metadata: {node['metadata']}")
            print(f"Text Snippet: \"{node['text'][:200]}...\"")

        if len(nodes) > max_nodes:
            print(f"\n... and {len(nodes) - max_nodes} more nodes")

    @staticmethod
    def validate_nodes(nodes: List[Dict[str, Any]]) -> bool:
        """
        Validate that nodes have required fields.

        Args:
            nodes: List of node dictionaries to validate

        Returns:
            True if all nodes are valid, False otherwise
        """
        required_fields = ['id_', 'metadata', 'text']

        for i, node in enumerate(nodes):
            missing_fields = [field for field in required_fields if field not in node]
            if missing_fields:
                logger.error(f"Node {i} missing required fields: {missing_fields}")
                return False

            if not node['text'].strip():
                logger.warning(f"Node {i} has empty text content")

        return True