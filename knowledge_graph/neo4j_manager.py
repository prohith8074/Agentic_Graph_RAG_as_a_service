"""
Neo4j database integration module.
Handles connection, population, and querying of the knowledge graph in Neo4j.
"""

import logging
from typing import Dict, Any, List
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from config.settings import settings

logger = logging.getLogger(__name__)

class Neo4jManager:
    """Manages Neo4j database operations for the knowledge graph."""

    def __init__(self):
        """Initialize Neo4j connection."""
        self.uri = settings.NEO4J_URI
        self.user = settings.NEO4J_USER
        self.password = settings.NEO4J_PASSWORD
        self.driver = None

    def connect(self) -> bool:
        """
        Establish connection to Neo4j database.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j database")
            return True
        except ServiceUnavailable as e:
            logger.error(f"Neo4j connection failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to Neo4j: {e}")
            return False

    def disconnect(self):
        """Close the Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            logger.info("Disconnected from Neo4j database")

    def create_constraints(self) -> bool:
        """
        Create necessary constraints in Neo4j database.

        Returns:
            True if successful, False otherwise
        """
        if not self.driver:
            logger.error("No active Neo4j connection")
            return False

        try:
            with self.driver.session() as session:
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
                logger.info("Created unique constraint on Entity.name")
                return True
        except Exception as e:
            logger.error(f"Error creating constraints: {e}")
            return False

    def clear_database(self) -> bool:
        """
        Clear all data from the database (use with caution).

        Returns:
            True if successful, False otherwise
        """
        if not self.driver:
            logger.error("No active Neo4j connection")
            return False

        try:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
                logger.info("Cleared all data from Neo4j database")
                return True
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            return False

    def populate_graph(self, graph_data: Dict[str, Any]) -> bool:
        """
        Populate Neo4j with knowledge graph data using MERGE for deduplication.

        Args:
            graph_data: Dictionary containing entities and relationships

        Returns:
            True if successful, False otherwise
        """
        if not self.driver:
            logger.error("No active Neo4j connection")
            return False

        try:
            with self.driver.session() as session:
                # Populate entities
                entities = graph_data.get("entities", [])
                logger.info(f"Adding {len(entities)} unique entities...")

                for entity in entities:
                    # Filter properties to avoid Neo4j storage issues
                    props = {}
                    for k, v in entity.items():
                        if k not in ["name"]:
                            # Skip fields that cause Neo4j storage issues
                            if k in ["embedding", "embeddings"] or k.startswith("_"):
                                logger.debug(f"Skipping field '{k}' for entity '{entity.get('name', 'unknown')}' - not compatible with Neo4j")
                                continue

                            # Only store simple types that Neo4j can handle
                            if isinstance(v, (str, int, float, bool)):
                                props[k] = v
                            elif isinstance(v, (list, dict)):
                                # For complex types, try to store as JSON string if it's not too nested
                                try:
                                    import json
                                    # Check if it's a simple list/dict (not nested)
                                    if isinstance(v, list) and v and all(isinstance(item, (str, int, float, bool)) for item in v):
                                        props[k] = v  # Store simple lists directly
                                    elif isinstance(v, dict) and v and all(isinstance(val, (str, int, float, bool)) for val in v.values()):
                                        props[k] = v  # Store simple dicts directly
                                    else:
                                        # For complex nested structures, convert to JSON string
                                        props[k] = json.dumps(v, default=str)
                                except (TypeError, ValueError) as e:
                                    logger.warning(f"Could not serialize field '{k}' for entity '{entity.get('name', 'unknown')}': {e}")
                                    props[k] = str(v)  # Fallback to string representation

                    session.run(
                        """
                        MERGE (e:Entity {name: $name})
                        SET e += $props
                        """,
                        name=entity["name"], props=props
                    )

                # Populate relationships
                relationships = graph_data.get("relationships", [])
                logger.info(f"Adding {len(relationships)} relationships...")

                for rel in relationships:
                    sanitized_label = "".join(filter(str.isalnum, rel["label"]))
                    if not sanitized_label:
                        continue
                    query = f"""
                    MATCH (src:Entity {{name: $source}})
                    MATCH (tgt:Entity {{name: $target}})
                    MERGE (src)-[r:`{sanitized_label}`]->(tgt)
                    """
                    session.run(query, source=rel["source"], target=rel["target"])

            logger.info("Successfully populated Neo4j with knowledge graph")
            return True

        except Exception as e:
            logger.error(f"Error populating Neo4j graph: {e}")
            return False

    def execute_query(self, cypher_query: str) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query and return results.

        Args:
            cypher_query: Cypher query string

        Returns:
            List of result records
        """
        if not self.driver:
            logger.error("No active Neo4j connection")
            raise ConnectionError("No active Neo4j connection")

        try:
            with self.driver.session() as session:
                result = session.run(cypher_query)
                records = [record.data() for record in result]
                logger.debug(f"Query executed successfully, returned {len(records)} records")
                return records
        except Exception as e:
            logger.error(f"Error executing Cypher query: {e}")
            raise

    def get_schema_info(self) -> Dict[str, Any]:
        """
        Get schema information from the Neo4j database.

        Returns:
            Dictionary with node labels and relationship types
        """
        if not self.driver:
            logger.error("No active Neo4j connection")
            return {}

        try:
            with self.driver.session() as session:
                node_labels = [record["label"] for record in session.run("CALL db.labels()")]
                rel_types = [record["relationshipType"] for record in session.run("CALL db.relationshipTypes()")]

                return {
                    "node_labels": node_labels,
                    "relationship_types": rel_types
                }
        except Exception as e:
            logger.error(f"Error getting schema info: {e}")
            return {}

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with database statistics
        """
        if not self.driver:
            logger.error("No active Neo4j connection")
            return {}

        try:
            with self.driver.session() as session:
                # Get node counts by label
                node_counts = {}
                for record in session.run("CALL db.labels()"):
                    label = record["label"]
                    count_result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                    count = count_result.single()["count"]
                    node_counts[label] = count

                # Get relationship counts by type
                rel_counts = {}
                for record in session.run("CALL db.relationshipTypes()"):
                    rel_type = record["relationshipType"]
                    count_result = session.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count")
                    count = count_result.single()["count"]
                    rel_counts[rel_type] = count

                return {
                    "node_counts": node_counts,
                    "relationship_counts": rel_counts,
                    "total_nodes": sum(node_counts.values()),
                    "total_relationships": sum(rel_counts.values())
                }
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}