"""
Qdrant vector database adapter.
"""

import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, UpdateStatus

from interfaces.ivector_database import IVectorDatabase
from config.settings import settings

logger = logging.getLogger(__name__)

class QdrantAdapter(IVectorDatabase):
    """Adapter for interacting with a Qdrant vector database."""

    def __init__(self):
        self.client = None
        self.collection_name = settings.QDRANT_COLLECTION_NAME

    def connect(self) -> bool:
        try:
            from database_adapters.database_factory import get_qdrant_client
            self.client = get_qdrant_client()
            if self.client:
                logger.info("Successfully connected Qdrant adapter to global client.")
                return True
            else:
                logger.error("Failed to get Qdrant client for adapter.")
                return False
        except Exception as e:
            logger.error(f"Qdrant connection failed: {e}")
            return False

    def retrieve_similar_chunks(self, query_embedding: List[float], top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not self.client:
            raise ConnectionError("Qdrant client is not connected.")
        
        search_params = {
            "collection_name": self.collection_name,
            "query_vector": query_embedding,
            "limit": top_k,
            "with_payload": True
        }

        if filters:
            logger.info(f"Applying filters to vector search: {filters}")
            search_params["query_filter"] = Filter(**filters)

        search_results = self.client.search(**search_params)

        results = [
            {
                'text': hit.payload.get('document', ''),
                'score': hit.score,
                'metadata': hit.payload.get('metadata', {})
            }
            for hit in search_results
        ]
        return results

    def upsert_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> bool:
        if not self.client:
            raise ConnectionError("Qdrant client is not connected.")

        points = [
            PointStruct(id=hash(chunk['contextualized_text']) % (2**63), vector=embedding, payload=chunk)
            for chunk, embedding in zip(chunks, embeddings)
            if embedding is not None and any(v != 0 for v in embedding)
        ]

        if not points:
            logger.warning("No valid points to upsert.")
            return True # Nothing to do

        operation_info = self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True
        )
        
        return operation_info.status == UpdateStatus.COMPLETED

    def get_collection_stats(self) -> Dict[str, Any]:
        if not self.client:
            raise ConnectionError("Qdrant client is not connected.")
        
        collection_info = self.client.get_collection(self.collection_name)
        return {
            'total_vectors': collection_info.points_count,
            'collection_name': self.collection_name,
            'vector_size': settings.QDRANT_VECTOR_SIZE
        }
