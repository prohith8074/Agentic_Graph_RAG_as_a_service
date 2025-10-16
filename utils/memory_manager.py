"""
Memory management system with Redis for short-term memory and Qdrant for long-term memory.
Provides caching, conversation history, and context management.
"""

import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
import redis
from cachetools import TTLCache
import hashlib

from config.settings import settings
from utils.opik_tracer import opik_tracer

logger = logging.getLogger(__name__)

class MemoryManager:
    """Manages short-term and long-term memory for the RAG system."""

    def __init__(self):
        """Initialize memory manager with Redis and local cache."""
        self.redis_client = None
        self.local_cache = TTLCache(maxsize=1000, ttl=settings.CACHE_TTL)

        # Initialize Redis connection (Redis Cloud - 30MB free tier)
        try:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                password=settings.REDIS_PASSWORD,
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info(f"Redis Cloud connection established: {settings.REDIS_HOST}:{settings.REDIS_PORT}")

            # Check memory usage (Redis Cloud 30MB limit)
            memory_info = self.redis_client.info('memory')
            used_memory = memory_info.get('used_memory', 0)
            max_memory = 30 * 1024 * 1024  # 30MB in bytes

            logger.info(f"Redis memory usage: {used_memory}/{max_memory} bytes ({(used_memory/max_memory)*100:.1f}%)")

        except redis.ConnectionError as e:
            logger.warning(f"Redis Cloud connection failed: {e}, using local cache only")
            self.redis_client = None

    def _get_cache_key(self, prefix: str, identifier: str) -> str:
        """Generate a consistent cache key."""
        return f"{prefix}:{identifier}"

    def _hash_content(self, content: str) -> str:
        """Generate hash for content deduplication."""
        return hashlib.md5(content.encode()).hexdigest()

    # Short-term Memory (Redis)
    def store_conversation_context(self, session_id: str, context: Dict[str, Any]) -> bool:
        """
        Store conversation context in Redis.

        Args:
            session_id: Unique session identifier
            context: Context data to store

        Returns:
            True if stored successfully
        """
        try:
            key = self._get_cache_key("conversation", session_id)
            context["timestamp"] = time.time()

            if self.redis_client:
                self.redis_client.setex(
                    key,
                    settings.SHORT_TERM_MEMORY_TTL,
                    json.dumps(context)
                )
                return True
            else:
                # Fallback to local cache
                self.local_cache[key] = context
                return True
        except Exception as e:
            logger.error(f"Error storing conversation context: {e}")
            return False

    def get_conversation_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve conversation context from Redis.

        Args:
            session_id: Unique session identifier

        Returns:
            Context data or None if not found
        """
        try:
            key = self._get_cache_key("conversation", session_id)

            if self.redis_client:
                data = self.redis_client.get(key)
                if data:
                    return json.loads(data)
            else:
                # Check local cache
                return self.local_cache.get(key)

        except Exception as e:
            logger.error(f"Error retrieving conversation context: {e}")

        return None

    def add_to_conversation_history(self, session_id: str, message: Dict[str, Any]) -> bool:
        """
        Add message to conversation history with memory management.

        Args:
            session_id: Unique session identifier
            message: Message data (query, response, etc.)

        Returns:
            True if added successfully
        """
        try:
            key = self._get_cache_key("history", session_id)

            if self.redis_client:
                # Check Redis memory usage before adding
                memory_info = self.redis_client.info('memory')
                used_memory = memory_info.get('used_memory', 0)
                max_memory = 28 * 1024 * 1024  # 28MB threshold (30MB limit - 2MB buffer)

                # If approaching memory limit, clean up old conversations
                if used_memory > max_memory:
                    logger.info(f"Redis memory usage high ({used_memory}/{max_memory}), cleaning up old conversations")
                    self._cleanup_old_conversations()

                # Get existing history
                history_data = self.redis_client.get(key)
                if history_data:
                    history = json.loads(history_data)
                else:
                    history = []

                # Add new message
                history.append({
                    **message,
                    "timestamp": time.time()
                })

                # Keep only recent messages (limit per session)
                if len(history) > settings.MAX_CONVERSATION_HISTORY:
                    history = history[-settings.MAX_CONVERSATION_HISTORY:]

                # Store updated history
                self.redis_client.setex(
                    key,
                    settings.SHORT_TERM_MEMORY_TTL,
                    json.dumps(history)
                )

                # Log memory usage after addition
                final_memory = self.redis_client.info('memory').get('used_memory', 0)
                logger.debug(f"Memory usage after addition: {final_memory} bytes")

                return True
            else:
                # Local cache fallback
                history = self.local_cache.get(key, [])
                history.append({**message, "timestamp": time.time()})
                if len(history) > settings.MAX_CONVERSATION_HISTORY:
                    history = history[-settings.MAX_CONVERSATION_HISTORY:]
                self.local_cache[key] = history
                return True

        except Exception as e:
            logger.error(f"Error adding to conversation history: {e}")
            return False

    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session.

        Args:
            session_id: Unique session identifier

        Returns:
            List of conversation messages
        """
        try:
            key = self._get_cache_key("history", session_id)

            if self.redis_client:
                data = self.redis_client.get(key)
                if data:
                    return json.loads(data)
            else:
                return self.local_cache.get(key, [])

        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")

        return []

    # Long-term Memory (Qdrant integration)
    def store_long_term_memory(self, content: str, metadata: Dict[str, Any],
                              collection_name: str = "long_term_memory") -> bool:
        """
        Store content in long-term memory (Qdrant).

        Args:
            content: Content to store
            metadata: Associated metadata
            collection_name: Qdrant collection name

        Returns:
            True if stored successfully
        """
        try:
            # Validate inputs
            if content is None:
                raise ValueError("Content cannot be None")
            if metadata is None:
                raise ValueError("Metadata cannot be None")

            # Import here to avoid circular imports
            from qdrant_client.models import Distance, VectorParams, PointStruct
            import cohere
            from database_adapters.database_factory import get_qdrant_client

            # Use the global Qdrant client initialized at startup
            qdrant_client = get_qdrant_client()
            if not qdrant_client:
                logger.error("Qdrant client not initialized - cannot store long-term memory")
                return False

            # Create collection if it doesn't exist
            collection_name = f"{settings.QDRANT_COLLECTION_NAME}_{collection_name}"
            try:
                qdrant_client.get_collection(collection_name)
            except Exception:
                # Collection doesn't exist, create it
                qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=settings.QDRANT_VECTOR_SIZE,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {collection_name}")

            # Generate embedding using Cohere
            co = cohere.Client(api_key=settings.COHERE_API_KEY)
            response = co.embed(
                texts=[content],
                model="embed-english-v3.0",
                input_type="search_document",
                embedding_types=['float']
            )

            # Handle Cohere response structure properly
            if hasattr(response.embeddings, 'float_'):
                embeddings = response.embeddings.float_
            else:
                embeddings = response.embeddings

            embedding = embeddings[0]

            # Create point for Qdrant
            point = PointStruct(
                id=hash(content) % (2**63),  # Generate deterministic ID
                vector=embedding,
                payload={
                    "content": content,
                    "metadata": metadata
                }
            )

            # Upsert point
            operation_info = qdrant_client.upsert(
                collection_name=collection_name,
                points=[point]
            )

            logger.info(f"Stored content in long-term memory: {len(content)} chars, operation: {operation_info}")
            return True

        except Exception as e:
            logger.error(f"Error storing long-term memory: {e}")
            return False

    def retrieve_long_term_memory(self, query: str, collection_name: str = "long_term_memory",
                                 top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant content from long-term memory.

        Args:
            query: Search query
            collection_name: Qdrant collection name
            top_k: Number of results to return

        Returns:
            List of relevant content with metadata
        """
        try:
            # Import here to avoid circular imports
            import cohere
            from database_adapters.database_factory import get_qdrant_client

            # Use the global Qdrant client initialized at startup
            qdrant_client = get_qdrant_client()
            if not qdrant_client:
                logger.error("Qdrant client not initialized - cannot retrieve long-term memory")
                return []

            # Generate embedding for query
            co = cohere.Client(api_key=settings.COHERE_API_KEY)
            response = co.embed(
                texts=[query],
                model="embed-english-v3.0",
                input_type="search_query",
                embedding_types=['float']
            )

            # Handle Cohere response structure properly
            if hasattr(response.embeddings, 'float_'):
                embeddings = response.embeddings.float_
            else:
                embeddings = response.embeddings

            query_embedding = embeddings[0]

            # Search Qdrant
            collection_name = f"{settings.QDRANT_COLLECTION_NAME}_{collection_name}"
            search_results = qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=top_k
            )

            # Format results
            memories = []
            for i, hit in enumerate(search_results):
                memories.append({
                    'content': hit.payload.get('content', ''),
                    'metadata': hit.payload.get('metadata', {}),
                    'relevance_score': hit.score,
                    'rank': i + 1
                })

            return memories

        except Exception as e:
            logger.error(f"Error retrieving long-term memory: {e}")
            return []

    def _cleanup_old_conversations(self) -> int:
        """
        Clean up old conversations when Redis memory usage is high.

        Returns:
            Number of conversations removed
        """
        if not self.redis_client:
            return 0

        try:
            # Get all conversation history keys
            conversation_keys = []
            for key in self.redis_client.scan_iter("history:*"):
                conversation_keys.append(key)

            # Sort by TTL (oldest first)
            keys_with_ttl = []
            for key in conversation_keys:
                ttl = self.redis_client.ttl(key)
                if ttl > 0:  # Only include keys that haven't expired
                    keys_with_ttl.append((key, ttl))

            # Sort by TTL (oldest first)
            keys_with_ttl.sort(key=lambda x: x[1])

            # Remove oldest conversations until memory usage is acceptable
            removed_count = 0
            memory_info = self.redis_client.info('memory')
            used_memory = memory_info.get('used_memory', 0)
            target_memory = 25 * 1024 * 1024  # 25MB target (5MB buffer)

            for key, _ in keys_with_ttl:
                if used_memory <= target_memory:
                    break

                self.redis_client.delete(key)
                removed_count += 1

                # Check memory after each deletion
                memory_info = self.redis_client.info('memory')
                used_memory = memory_info.get('used_memory', 0)

            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old conversations. Memory usage: {used_memory} bytes")

            return removed_count

        except Exception as e:
            logger.error(f"Error during conversation cleanup: {e}")
            return 0

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive memory usage statistics.

        Returns:
            Dictionary with memory statistics
        """
        stats = {
            "redis_available": self.redis_client is not None,
            "local_cache_size": len(self.local_cache),
            "local_cache_maxsize": self.local_cache.maxsize
        }

        if self.redis_client:
            try:
                memory_info = self.redis_client.info('memory')
                stats.update({
                    "redis_memory_used": memory_info.get('used_memory', 0),
                    "redis_memory_peak": memory_info.get('used_memory_peak', 0),
                    "redis_memory_available": 30 * 1024 * 1024,  # 30MB limit
                    "redis_memory_usage_percent": (memory_info.get('used_memory', 0) / (30 * 1024 * 1024)) * 100
                })
            except Exception as e:
                stats["redis_error"] = str(e)

        return stats

    # Caching for Database Queries
    def cache_query_result(self, query_key: str, result: Any, ttl: Optional[int] = None) -> bool:
        """
        Cache database query results.

        Args:
            query_key: Unique key for the query
            result: Query result to cache
            ttl: Time to live in seconds (optional)

        Returns:
            True if cached successfully
        """
        try:
            cache_key = self._get_cache_key("query", query_key)
            cache_data = {
                "result": result,
                "timestamp": time.time()
            }

            ttl_value = ttl or settings.CACHE_TTL

            if self.redis_client:
                self.redis_client.setex(
                    cache_key,
                    ttl_value,
                    json.dumps(cache_data)
                )
            else:
                self.local_cache[cache_key] = cache_data

            return True

        except Exception as e:
            logger.error(f"Error caching query result: {e}")
            return False

    def get_cached_query_result(self, query_key: str) -> Optional[Any]:
        """
        Retrieve cached query result.

        Args:
            query_key: Unique key for the query

        Returns:
            Cached result or None if not found/expired
        """
        try:
            cache_key = self._get_cache_key("query", query_key)

            if self.redis_client:
                data = self.redis_client.get(cache_key)
                if data:
                    cache_data = json.loads(data)
                    return cache_data["result"]
            else:
                cache_data = self.local_cache.get(cache_key)
                if cache_data:
                    return cache_data["result"]

        except Exception as e:
            logger.error(f"Error retrieving cached query: {e}")

        return None

    def invalidate_cache(self, pattern: str = "*") -> bool:
        """
        Invalidate cache entries matching a pattern.

        Args:
            pattern: Pattern to match for invalidation

        Returns:
            True if invalidation successful
        """
        try:
            if self.redis_client:
                # Use SCAN for pattern matching
                keys_to_delete = []
                for key in self.redis_client.scan_iter(f"query:{pattern}"):
                    keys_to_delete.append(key)

                if keys_to_delete:
                    self.redis_client.delete(*keys_to_delete)

                logger.info(f"Invalidated {len(keys_to_delete)} cache entries")
                return True
            else:
                # For local cache, clear all entries (simple implementation)
                keys_to_delete = [k for k in self.local_cache.keys() if pattern in k]
                for key in keys_to_delete:
                    del self.local_cache[key]

                logger.info(f"Invalidated {len(keys_to_delete)} local cache entries")
                return True

        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")
            return False

    # Context Integration for LLM Responses
    def build_enhanced_context(self, session_id: str, current_query: str,
                              retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build enhanced context for LLM responses including short-term memory.

        Args:
            session_id: Session identifier
            current_query: Current user query
            retrieved_chunks: Retrieved context chunks

        Returns:
            Enhanced context dictionary
        """
        try:
            # Get conversation context
            conversation_context = self.get_conversation_context(session_id) or {}

            # Get recent conversation history
            conversation_history = self.get_conversation_history(session_id)

            # Extract relevant history (last few exchanges)
            recent_history = []
            for msg in conversation_history[-4:]:  # Last 2 exchanges (query + response)
                if 'query' in msg and 'answer' in msg:
                    recent_history.append({
                        'user_query': msg['query'],
                        'assistant_response': msg['answer'][:200] + "..." if len(msg['answer']) > 200 else msg['answer']
                    })

            # Build context
            enhanced_context = {
                'current_query': current_query,
                'conversation_context': conversation_context,
                'recent_history': recent_history,
                'retrieved_chunks': retrieved_chunks,
                'session_id': session_id,
                'timestamp': time.time()
            }

            return enhanced_context

        except Exception as e:
            logger.error(f"Error building enhanced context: {e}")
            return {
                'current_query': current_query,
                'retrieved_chunks': retrieved_chunks,
                'error': str(e)
            }

# Global memory manager instance
memory_manager = MemoryManager()

# Cache decorator
def cache_result(ttl: Optional[int] = None):
    """
    Decorator to cache function results to reduce database queries.

    Args:
        ttl: Time to live in seconds (optional)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"

            # Check cache first
            cached_result = memory_manager.get_cached_query_result(cache_key)
            if cached_result is not None:
                opik_tracer.track_llm_call(
                    provider="cache",
                    model="memory_manager",
                    prompt=f"Cache hit for {func.__name__}",
                    response="Retrieved from cache",
                    metadata={"function": func.__name__, "cache_hit": True}
                )
                return cached_result

            # Execute function
            result = func(*args, **kwargs)

            # Cache result
            memory_manager.cache_query_result(cache_key, result, ttl)

            opik_tracer.track_llm_call(
                provider="cache",
                model="memory_manager",
                prompt=f"Cache miss for {func.__name__}",
                response="Stored in cache",
                metadata={"function": func.__name__, "cache_hit": False}
            )

            return result

        return wrapper
    return decorator
