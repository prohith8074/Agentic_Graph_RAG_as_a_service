"""
Filter query plugin for structured filtering and constraints.
Uses an LLM to extract filters from natural language and applies them to vector search.
"""

import logging
import json
import re
from typing import Dict, Any, List

from config.settings import settings

logger = logging.getLogger(__name__)

class FilterPlugin:
    """Plugin for filter-based query handling."""

    def __init__(self):
        """Initialize filter plugin."""
        self.name = "filter"
        self.description = "Structured filtering and constraint-based queries on vector data"
        self._vector_plugin = None
        self._llm = None

    def _get_vector_plugin(self):
        """Lazy load vector plugin for filtering."""
        if self._vector_plugin is None:
            from .vector_plugin import VectorPlugin
            self._vector_plugin = VectorPlugin()
        return self._vector_plugin

    @property
    def llm(self):
        """Lazy load the LLM client."""
        if self._llm is None:
            import groq
            self._llm = groq.Groq(api_key=settings.GROQ_API_KEY)
        return self._llm

    def can_handle_query(self, query: str, context: Dict[str, Any]) -> float:
        """
        Determine if this plugin can handle the query.
        """
        query_lower = query.lower()
        filter_keywords = [
            'filter', 'where', 'only', 'excluding', 'except', 'with', 'having',
            'matching', 'containing', 'find all', 'list all', 'show me',
            'after', 'before', 'between', 'from'
        ]
        if any(keyword in query_lower for keyword in filter_keywords):
            return 0.9
        return 0.1

    def process_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process query by extracting filters and applying them to a vector search.
        """
        logger.info("FilterPlugin: Processing query...")
        # Extract the core semantic search part of the query and the filters
        core_query, filters = self._extract_filters(query)

        # Get the vector search plugin
        vector_plugin = self._get_vector_plugin()

        # Add extracted filters to the context for the vector plugin
        enhanced_context = {**context, 'filters': filters}

        # Perform a filtered vector search
        logger.info(f"Performing filtered vector search with core_query: '{core_query}' and filters: {filters}")
        result = vector_plugin.process_query(core_query, enhanced_context)

        # Add filter metadata to the final result
        result['method'] = 'filter + vector'
        result['applied_filters'] = filters
        result['original_query'] = query

        return result

    def _extract_filters(self, query: str) -> tuple[str, List[Dict[str, Any]]]:
        """
        Use an LLM to extract filtering constraints and the core semantic query from the user's input.
        """
        prompt = f"""
        You are a query parsing expert. Analyze the user's query and separate it into two parts:
        1. A `core_query` for semantic vector search.
        2. A structured `filters` object for metadata filtering in a Qdrant database.

        The available metadata fields for filtering are: `year`, `author`, `document_type`, `page_number`.

        The filter structure should follow Qdrant's format. Use `must`, `should`, and `must_not`. Conditions can be `match` (for text), `range` (for numbers), `gte` (greater than or equal), `lte` (less than or equal).

        User Query: "{query}"

        Example 1:
        User Query: "show me papers about attention mechanism by Vaswani after 2016"
        Response:
        ```json
        {{
            "core_query": "attention mechanism",
            "filters": {{
                "must": [
                    {{ "key": "metadata.author", "match": {{ "value": "Vaswani" }} }},
                    {{ "key": "metadata.year", "range": {{ "gte": 2017 }} }}
                ]
            }}
        }}
        ```

        Example 2:
        User Query: "what are the main components of transformers, excluding documents before 2017"
        Response:
        ```json
        {{
            "core_query": "main components of transformers",
            "filters": {{
                "must_not": [
                    {{ "key": "metadata.year", "range": {{ "lt": 2017 }} }}
                ]
            }}
        }}
        ```

        Now, parse the given user query.
        Response:
        """
        try:
            response = self.llm.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="openai/gpt-oss-120b",
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            response_text = response.choices[0].message.content
            parsed_json = json.loads(response_text)
            
            core_query = parsed_json.get("core_query", query) # Fallback to original query
            filters = parsed_json.get("filters", {})
            
            logger.info(f"LLM extracted filters: {filters} and core query: '{core_query}'")
            return core_query, filters

        except Exception as e:
            logger.error(f"Failed to extract filters using LLM: {e}")
            # Fallback to using the whole query for semantic search with no filters
            return query, {}
