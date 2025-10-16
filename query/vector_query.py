"""
Vector-based query interface that uses a database adapter for all database operations.
This class orchestrates the process of embedding a query, retrieving context, and generating an answer.
"""

import logging
from typing import List, Dict, Any, Optional

from database_adapters.database_factory import get_vector_adapter
from utils.embeddings import default_embedder
from config.settings import settings

logger = logging.getLogger(__name__)

class VectorQueryInterface:
    """Handles vector-based queries by delegating to a vector database adapter."""

    def __init__(self):
        """Initialize vector query interface."""
        self.embedder = default_embedder
        self.adapter = get_vector_adapter()
        if not self.adapter:
            logger.error("Vector database adapter not initialized - vector queries will fail")

    def retrieve_similar_chunks(self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve similar chunks by embedding the query and calling the adapter.
        """
        if not self.adapter:
            raise ConnectionError("Vector database adapter is not available.")
        
        try:
            query_embedding = self.embedder.embed_query(query)
            return self.adapter.retrieve_similar_chunks(query_embedding, top_k, filters)
        except Exception as e:
            logger.error(f"Error retrieving similar chunks via adapter: {e}", exc_info=True)
            return []

    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Generate an answer using the Cohere LLM based on the retrieved context.

        ### Tools Used
        *   `cohere.Client`

        Args:
            query: The user's query.
            context_chunks: A list of context chunks retrieved from the vector store.

        Returns:
            The generated answer.
        """
        if not context_chunks:
            return "No relevant information found in the vector store."

        context_text = "\n".join([chunk['text'] for chunk in context_chunks])
        prompt = f"""Based on the following context, answer the question:

Context:
{context_text}

Question: {query}

Answer:"""

        try:
            # Initialize the Cohere client and generate an answer.
            import cohere
            co = cohere.Client(api_key=settings.COHERE_API_KEY)
            response = co.chat(
                message=prompt,
                model=settings.COHERE_MODEL,
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {e}"

    def query(self, user_query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the complete vector query pipeline using the adapter.
        """
        context = context or {}
        top_k = context.get('top_k', 5)
        filters = context.get('filters')

        try:
            if not self.adapter:
                 return {
                    'answer': "Vector database is not connected. Please check the configuration.",
                    'chunks': [], 'query': user_query, 'num_chunks': 0
                }

            stats = self.get_collection_stats()
            if stats.get('total_vectors', 0) == 0:
                logger.warning("Vector collection is empty")
                return {
                    'answer': "Vector database is empty. Please process a document first.",
                    'chunks': [], 'query': user_query, 'num_chunks': 0
                }

            chunks = self.retrieve_similar_chunks(user_query, top_k, filters)
            if not chunks:
                return {
                    'answer': "No relevant information found in the vector store for the given query and filters.",
                    'chunks': [], 'query': user_query, 'num_chunks': 0
                }

            answer = self.generate_answer(user_query, chunks)

            return {
                'answer': answer,
                'chunks': chunks,
                'query': user_query,
                'num_chunks': len(chunks)
            }
        except Exception as e:
            logger.error(f"Error in vector query pipeline: {e}", exc_info=True)
            return {
                'answer': f"Error processing vector query: {e}",
                'chunks': [], 'query': user_query, 'error': str(e), 'num_chunks': 0
            }

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector collection via the adapter.
        """
        if not self.adapter:
            return {'error': 'Vector database adapter not initialized'}
        try:
            return self.adapter.get_collection_stats()
        except Exception as e:
            logger.error(f"Error getting collection stats via adapter: {e}")
            return {'error': str(e)}
