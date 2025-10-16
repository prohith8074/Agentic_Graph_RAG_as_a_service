"""
Graph-based query interface using Cohere for Cypher query generation.
Handles natural language to Cypher query conversion and Neo4j execution.
"""

import logging
import re
from typing import List, Dict, Any, Optional
import cohere

from config.settings import settings
from knowledge_graph.neo4j_manager import Neo4jManager

logger = logging.getLogger(__name__)

class GraphQueryInterface:
    """Handles graph-based queries using Cohere for Cypher generation."""

    def __init__(self, neo4j_adapter=None):
        """Initialize with Neo4j adapter."""
        if neo4j_adapter is None:
            from database_adapters.database_factory import db_manager
            neo4j_adapter = db_manager.get_adapter("neo4j")
        self.neo4j_adapter = neo4j_adapter
        self.co = cohere.ClientV2(api_key=settings.COHERE_API_KEY)

    def clean_cypher_query(self, raw_text: str) -> str:
        """
        Extract and clean Cypher query text from Cohere's markdown-formatted response.

        Args:
            raw_text: Raw response text from Cohere

        Returns:
            Cleaned Cypher query
        """
        if not raw_text:
            return ""

        # Remove markdown code fences like ```cypher ... ```
        cleaned = re.sub(r"```(?:cypher)?", "", raw_text, flags=re.IGNORECASE)
        # Remove leading/trailing whitespace and non-Cypher commentary
        cleaned = cleaned.strip()

        # If the model added explanations before the query, extract only the first MATCH block
        match = re.search(r"(MATCH\s+[\s\S]*)", cleaned, flags=re.IGNORECASE)
        if match:
            cleaned = match.group(1).strip()

        # Basic sanity check
        if not cleaned.lower().startswith("match"):
            logger.warning("Cleaned query does not start with MATCH — double check the output")
            logger.debug(f"Query: {cleaned}")

        return cleaned

    def generate_cypher_query(self, user_query: str, schema: Dict[str, Any]) -> str:
        """
        Generate Cypher query from natural language using Cohere.

        Args:
            user_query: Natural language question
            schema: Database schema information

        Returns:
            Generated Cypher query
        """
        prompt = f"""
        You are an expert Neo4j developer. Convert the following question into a Cypher query.
        Return ONLY the Cypher query, with no explanations or markdown.

        Rules:
        - Nodes are labeled `Entity`.
        - Each Entity has properties: `name` and `type`.
        - Use case-insensitive CONTAINS when searching by name.
        - For questions like "Explain X" or "What is X", find the node and its related entities.
        - Return meaningful fields like node names, relationship types, and related nodes.

        Graph Schema:
        Node Labels: {schema.get('node_labels', [])}
        Relationship Types: {schema.get('relationship_types', [])}

        Example:
        Question: What is the Transformer model?
        Cypher Query:
        MATCH (n:Entity)
        WHERE toLower(n.name) CONTAINS 'transformer'
        OPTIONAL MATCH (n)-[r]-(m)
        RETURN n.name AS name, n.type AS type, type(r) AS relation, m.name AS related_name, m.type AS related_type

        Now convert this question:
        "{user_query}"

        Cypher Query:
        """

        try:
            response = self.co.chat(
                messages=[{
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": prompt
                    }]
                }],
                temperature=0.3,
                model=settings.COHERE_MODEL,
            )

            raw_query = response.message.content[0].text.strip()
            return self.clean_cypher_query(raw_query)

        except Exception as e:
            logger.error(f"Error generating Cypher query: {e}")
            return ""

    def synthesize_answer(self, user_query: str, records: List[Dict[str, Any]]) -> str:
        """
        Synthesize a natural language answer from Neo4j query results using Cohere.

        Args:
            user_query: Original user question
            records: Query results from Neo4j

        Returns:
            Natural language answer
        """
        if not records:
            return "I couldn't find any relevant information in the knowledge graph."

        data_json = str(records)

        synthesis_prompt = f"""
        You are a knowledgeable assistant. Answer the user's question *only* using the information below.
        Be factual, concise, and directly relevant to the question.

        --- Knowledge Graph Data ---
        {data_json}
        ----------------------------

        Question: "{user_query}"

        Answer:
        """

        try:
            response = self.co.chat(
                messages=[{
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": synthesis_prompt
                    }]
                }],
                temperature=0.3,
                model=settings.COHERE_MODEL,
            )

            return response.message.content[0].text.strip()

        except Exception as e:
            logger.error(f"Error synthesizing answer: {e}")
            return f"Error generating answer: {e}"

    def query(self, user_query: str) -> Dict[str, Any]:
        """
        Execute a complete graph query pipeline.

        Args:
            user_query: Natural language question

        Returns:
            Dictionary with answer and metadata
        """
        try:
            # Get database schema
            schema = self.neo4j_adapter.get_schema_info()
            if not schema:
                return {
                    'answer': "Unable to connect to knowledge graph database.",
                    'query': user_query,
                    'method': 'graph',
                    'error': 'connection_failed'
                }

            # Generate Cypher query
            cypher_query = self.generate_cypher_query(user_query, schema)
            if not cypher_query:
                return {
                    'answer': "Failed to generate database query.",
                    'query': user_query,
                    'method': 'graph',
                    'error': 'query_generation_failed'
                }

            logger.info(f"Generated Cypher Query: {cypher_query}")

            # Execute query
            records = self.neo4j_adapter.execute_query(cypher_query)
            if not records:
                return {
                    'answer': "No results found for that question.",
                    'query': user_query,
                    'method': 'graph',
                    'records_found': 0
                }

            # Synthesize answer
            answer = self.synthesize_answer(user_query, records)
            return {
                'answer': answer,
                'query': user_query,
                'method': 'graph',
                'records_found': len(records),
                'cypher_query': cypher_query
            }

        except Exception as e:
            logger.error(f"Error in graph query pipeline: {e}")
            return {
                'answer': f"Error processing query: {e}",
                'query': user_query,
                'method': 'graph',
                'error': str(e)
            }
