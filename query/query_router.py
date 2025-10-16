"""
Intelligent query router using Groq LLM to route between vector and graph databases.
Analyzes query intent and routes to the most appropriate retrieval system.
"""

import logging
from typing import Dict, Any, Optional
import groq

from config.settings import settings
from query.graph_query import GraphQueryInterface
from query.vector_query import VectorQueryInterface
from utils.opik_tracer import opik_tracer, track_query_routing

logger = logging.getLogger(__name__)

class QueryRouter:
    """Intelligent router that selects between graph and vector queries based on intent."""

    def __init__(self, neo4j_adapter=None, qdrant_client=None):
        """Initialize query router with Groq LLM."""
        self.llm = groq.Groq(api_key=settings.GROQ_API_KEY)

        # Use provided adapters or get from global manager
        if neo4j_adapter is None:
            from database_adapters.database_factory import db_manager
            neo4j_adapter = db_manager.get_adapter("neo4j")
        if qdrant_client is None:
            from database_adapters.database_factory import get_qdrant_client
            qdrant_client = get_qdrant_client()

        self.neo4j_adapter = neo4j_adapter
        self.qdrant_client = qdrant_client
        self.graph_query = GraphQueryInterface(self.neo4j_adapter)
        self.vector_query = VectorQueryInterface()

        # Track connection status
        self.neo4j_connected = self.neo4j_adapter is not None
        self.vector_available = self.qdrant_client is not None

    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Analyze query intent using Groq LLM to determine best retrieval approach.

        Args:
            query: User query to analyze

        Returns:
            Dictionary with routing decision and confidence
        """
        prompt = f"""
        Analyze this query and determine whether it should be answered using:
        1. GRAPH DATABASE (Neo4j) - for structured queries about relationships, components, architecture, definitions, explanations
        2. VECTOR DATABASE (Qdrant) - for semantic search, general information retrieval, contextual questions

        Query: "{query}"

        Consider:
        - Questions about "what is X", "explain X", "components of X", "how does X work" → GRAPH
        - Questions about "tell me about", "describe", "information on", "find text about" → VECTOR
        - Questions about relationships, connections, hierarchies → GRAPH
        - Open-ended or broad questions → VECTOR

        Respond with JSON format:
        {{
            "method": "graph" or "vector",
            "confidence": 0.0-1.0,
            "reasoning": "brief explanation"
        }}
        """

        try:
            response = self.llm.complete(prompt)
            result_text = response.text.strip()

            # Check if response is empty or None
            if not result_text:
                logger.warning("Empty response from LLM, defaulting to vector")
                return {
                    "method": "vector",
                    "confidence": 0.5,
                    "reasoning": "Empty LLM response, defaulting to vector search"
                }

            # Check for common error responses
            if "error" in result_text.lower() or "failed" in result_text.lower():
                logger.warning(f"LLM returned error response: {result_text[:100]}")
                return {
                    "method": "vector",
                    "confidence": 0.5,
                    "reasoning": "LLM error response, defaulting to vector search"
                }

            # Track with Opik
            opik_tracer.track_llm_call(
                provider="groq",
                model=settings.GROQ_MODEL,
                prompt=prompt,
                response=result_text,
                metadata={"operation": "query_intent_analysis"}
            )

            # Extract JSON from response with better error handling
            import json
            import re

            # Find JSON object in response
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    logger.info(f"Query routing decision: {result}")
                    return result
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing failed: {e}")
                    logger.debug(f"LLM response content: {result_text[:200]}")
                    return {
                        "method": "vector",
                        "confidence": 0.5,
                        "reasoning": f"JSON parsing failed: {e}, defaulting to vector search"
                    }
            else:
                logger.warning("No JSON found in LLM response, defaulting to vector")
                return {
                    "method": "vector",
                    "confidence": 0.5,
                    "reasoning": "Could not determine intent, defaulting to vector search"
                }

        except Exception as e:
            logger.error(f"Error analyzing query intent: {e}")
            opik_tracer.log_error("query_intent_analysis_error", str(e), query=query)

            # Check if it's a connection/API error
            if "api" in str(e).lower() or "connection" in str(e).lower() or "auth" in str(e).lower():
                logger.warning("LLM service unavailable, using vector search as fallback")
                return {
                    "method": "vector",
                    "confidence": 0.8,  # High confidence for vector when LLM fails
                    "reasoning": f"LLM service error: {e}, using vector search"
                }
            else:
                return {
                    "method": "vector",
                    "confidence": 0.5,
                    "reasoning": f"Error in analysis: {e}, defaulting to vector search"
                }

    @track_query_routing
    def route_query(self, user_query: str) -> Dict[str, Any]:
        """
        Route query to appropriate interface based on intent analysis.

        Args:
            user_query: User query

        Returns:
            Query result with routing information
        """
        try:
            # Analyze query intent
            intent_analysis = self.analyze_query_intent(user_query)

            method = intent_analysis.get("method", "vector")
            confidence = intent_analysis.get("confidence", 0.5)
            reasoning = intent_analysis.get("reasoning", "Unknown")

            # Route to appropriate interface
            if method == "graph":
                logger.info(f"Routing to graph database (confidence: {confidence})")
                try:
                    # Check if Neo4j adapter is available
                    if not self.neo4j_adapter:
                        raise ConnectionError("Neo4j adapter not available")

                    answer = self.graph_query.query(user_query)
                    result_type = "graph"

                except Exception as e:
                    logger.warning(f"Graph query failed, falling back to vector: {e}")
                    opik_tracer.log_error("graph_query_fallback", str(e), query=user_query)
                    try:
                        vector_result = self.vector_query.query(user_query)
                        answer = vector_result.get('answer', 'Error in graph query, used vector fallback')
                        result_type = "vector_fallback"
                    except Exception as vector_error:
                        logger.error(f"Vector query also failed: {vector_error}")
                        answer = f"Both graph and vector queries failed. Graph error: {e}, Vector error: {vector_error}"
                        result_type = "error"

            else:  # vector
                logger.info(f"Routing to vector database (confidence: {confidence})")
                try:
                    vector_result = self.vector_query.query(user_query)
                    answer = vector_result.get('answer', 'No answer generated')
                    result_type = "vector"
                except Exception as vector_error:
                    logger.warning(f"Vector query failed: {vector_error}")
                    # Try to use graph as fallback
                    try:
                        if not self.neo4j_adapter:
                            raise ConnectionError("Neo4j adapter not available for fallback")

                        answer = self.graph_query.query(user_query)
                        result_type = "graph_fallback"
                        logger.info("Used graph database as fallback for vector query failure")
                    except Exception as graph_error:
                        logger.error(f"Both vector and graph queries failed: {vector_error}, {graph_error}")
                        answer = f"Query processing failed. Vector error: {str(vector_error)[:100]}, Graph error: {str(graph_error)[:100]}"
                        result_type = "error"

            result = {
                'query': user_query,
                'answer': answer,
                'routing': {
                    'method': method,
                    'confidence': confidence,
                    'reasoning': reasoning,
                    'actual_method': result_type
                },
                'chunks_retrieved': vector_result.get('num_chunks', 0) if 'vector_result' in locals() else 0
            }

            return result

        except Exception as e:
            logger.error(f"Error in query routing: {e}")
            opik_tracer.log_error("query_routing_error", str(e), query=user_query)
            return {
                'query': user_query,
                'answer': f"Error processing query: {e}",
                'routing': {
                    'method': 'error',
                    'confidence': 0.0,
                    'reasoning': str(e),
                    'actual_method': 'error'
                },
                'chunks_retrieved': 0
            }

    def get_routing_stats(self) -> Dict[str, Any]:
        """
        Get statistics about routing decisions and system status.

        Returns:
            Dictionary with routing and system statistics
        """
        try:
            # Get database stats with error handling
            neo4j_stats = {}
            vector_stats = {}

            try:
                neo4j_stats = self.neo4j_adapter.get_database_stats()
            except Exception as e:
                logger.warning(f"Could not get Neo4j stats: {e}")
                neo4j_stats = {}

            try:
                vector_stats = self.vector_query.get_collection_stats()
            except Exception as e:
                logger.warning(f"Could not get vector stats: {e}")
                vector_stats = {}

            # Check connection status
            self.neo4j_connected = self.neo4j_adapter is not None
            self.vector_available = vector_stats.get('total_vectors', 0) > 0

            return {
                'neo4j_connected': self.neo4j_connected,
                'neo4j_stats': neo4j_stats,
                'vector_stats': vector_stats,
                'router_config': {
                    'llm_model': settings.GROQ_MODEL,
                    'graph_available': bool(neo4j_stats),
                    'vector_available': self.vector_available
                }
            }
        except Exception as e:
            logger.error(f"Error getting routing stats: {e}")
            return {
                'neo4j_connected': False,
                'neo4j_stats': {},
                'vector_stats': {},
                'router_config': {
                    'llm_model': settings.GROQ_MODEL,
                    'graph_available': False,
                    'vector_available': False
                },
                'error': str(e)
            }