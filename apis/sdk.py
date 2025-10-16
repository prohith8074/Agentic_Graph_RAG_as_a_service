"""
Python SDK for the Lyzr Challenge RAG system.
Provides programmatic interfaces for easy integration.
"""

import requests
import json
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import asyncio

@dataclass
class QueryResult:
    """Query result from the RAG system."""
    query: str
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    reasoning_steps: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class LyzrRAGClient:
    """Python SDK client for Lyzr Challenge RAG system."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the RAG client.

        Args:
            base_url: Base URL of the RAG API server
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request to the API."""
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {e}")

    def query(self, query: str, session_id: Optional[str] = None,
             streaming: bool = False) -> Union[QueryResult, Dict[str, Any]]:
        """
        Execute a query against the RAG system.

        Args:
            query: Natural language query
            session_id: Optional session identifier for conversation continuity
            streaming: Whether to use streaming response

        Returns:
            QueryResult object or raw response dict
        """
        payload = {
            "query": query,
            "streaming": streaming
        }
        if session_id:
            payload["session_id"] = session_id

        if streaming:
            # For streaming, we'd need to implement SSE handling
            response = self._make_request("POST", "/query", json=payload)
            return response
        else:
            response = self._make_request("POST", "/query", json=payload)
            return QueryResult(
                query=response.get("query", ""),
                answer=response.get("final_answer", ""),
                confidence=response.get("confidence_score", 0.0),
                sources=response.get("sources", []),
                reasoning_steps=response.get("reasoning_chain", []),
                metadata=response.get("metadata", {})
            )

    def get_ontology_suggestions(self, focus_area: str = "general") -> Dict[str, Any]:
        """
        Get AI suggestions for ontology improvements.

        Args:
            focus_area: Area to focus on ('entities', 'relationships', 'general')

        Returns:
            Dictionary with suggestions
        """
        payload = {"focus_area": focus_area}
        return self._make_request("POST", "/ontology/suggest", json=payload)

    def apply_ontology_edit(self, suggestion: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply an ontology edit suggestion.

        Args:
            suggestion: Suggestion dictionary from get_ontology_suggestions

        Returns:
            Operation result
        """
        payload = {"suggestion": suggestion}
        return self._make_request("POST", "/ontology/edit", json=payload)

    def get_ontology_visualization(self) -> Dict[str, Any]:
        """
        Get ontology data formatted for visualization.

        Returns:
            Dictionary with nodes and edges for visualization
        """
        return self._make_request("GET", "/ontology/visualization")

    def health_check(self) -> Dict[str, Any]:
        """
        Check system health.

        Returns:
            Health status dictionary
        """
        return self._make_request("GET", "/health")

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics for all connected databases.

        Returns:
            Database statistics
        """
        return self._make_request("GET", "/database/stats")

    def switch_database(self, db_type: str) -> Dict[str, Any]:
        """
        Switch active database.

        Args:
            db_type: Database type ('neo4j')  # Note: Neptune support is disabled

        Returns:
            Operation result
        """
        return self._make_request("POST", f"/database/switch/{db_type}")

    def get_available_databases(self) -> Dict[str, Any]:
        """
        Get list of available database types.

        Returns:
            Dictionary of available databases
        """
        return self._make_request("GET", "/database/available")

    def clear_cache(self) -> Dict[str, Any]:
        """
        Clear query result cache.

        Returns:
            Operation result
        """
        return self._make_request("DELETE", "/cache")

class AsyncLyzrRAGClient:
    """Asynchronous version of the RAG client."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize async client."""
        import httpx
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    async def query_streaming(self, query: str, session_id: Optional[str] = None):
        """
        Execute a streaming query.

        Args:
            query: Natural language query
            session_id: Optional session identifier

        Yields:
            Streaming response chunks
        """
        import httpx

        payload = {
            "query": query,
            "streaming": True
        }
        if session_id:
            payload["session_id"] = session_id

        url = f"{self.base_url}/query"
        async with self.client.stream("POST", url, json=payload) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])  # Remove "data: " prefix
                        yield data
                    except json.JSONDecodeError:
                        continue

    async def query(self, query: str, session_id: Optional[str] = None) -> QueryResult:
        """
        Execute a regular (non-streaming) query asynchronously.

        Args:
            query: Natural language query
            session_id: Optional session identifier

        Returns:
            QueryResult object
        """
        payload = {
            "query": query,
            "streaming": False
        }
        if session_id:
            payload["session_id"] = session_id

        response = await self.client.post(f"{self.base_url}/query", json=payload)
        response.raise_for_status()
        data = response.json()

        return QueryResult(
            query=data.get("query", ""),
            answer=data.get("final_answer", ""),
            confidence=data.get("confidence_score", 0.0),
            sources=data.get("sources", []),
            reasoning_steps=data.get("reasoning_chain", []),
            metadata=data.get("metadata", {})
        )

# Convenience functions for quick usage
def quick_query(query: str, base_url: str = "http://localhost:8000") -> QueryResult:
    """
    Quick query function for simple use cases.

    Args:
        query: Query string
        base_url: API base URL

    Returns:
        QueryResult object
    """
    client = LyzrRAGClient(base_url)
    return client.query(query)

async def quick_query_async(query: str, base_url: str = "http://localhost:8000") -> QueryResult:
    """
    Quick async query function.

    Args:
        query: Query string
        base_url: API base URL

    Returns:
        QueryResult object
    """
    async with AsyncLyzrRAGClient(base_url) as client:
        return await client.query(query)

# Example usage:
"""
# Synchronous usage
client = LyzrRAGClient()
result = client.query("What are the components of a transformer?")
print(result.answer)

# Asynchronous streaming
async with AsyncLyzrRAGClient() as client:
    async for chunk in client.query_streaming("Explain attention mechanism"):
        if chunk["type"] == "result":
            print(chunk["data"]["final_answer"])
            break
        elif chunk["type"] == "status":
            print(f"Status: {chunk['data']['status']}")

# Quick query
result = quick_query("What is self-attention?")
"""