"""
Neo4j graph database adapter.
Provides support for Neo4j with Cypher query language.
Refactored from original neo4j_manager.py to use the adapter interface.
"""

import logging
from typing import Dict, Any, List
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from . import GraphDatabaseAdapter, QueryLanguage
from config.settings import settings

logger = logging.getLogger(__name__)

class Neo4jAdapter(GraphDatabaseAdapter):
    """Neo4j graph database adapter with Cypher support."""

    def __init__(self):
        """Initialize Neo4j adapter with settings."""
        self.uri = settings.NEO4J_URI
        self.user = settings.NEO4J_USER
        self.password = settings.NEO4J_PASSWORD
        self.driver = None

    def connect(self) -> bool:
        """Establish connection to Neo4j database."""
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

    def execute_query(self, query: str, language: QueryLanguage = QueryLanguage.CYPHER) -> List[Dict[str, Any]]:
        """Execute Cypher query and return results."""
        if language != QueryLanguage.CYPHER:
            raise ValueError(f"Neo4j adapter only supports {QueryLanguage.CYPHER.value}")

        if not self.driver:
            raise ConnectionError("No active Neo4j connection")

        try:
            with self.driver.session() as session:
                result = session.run(query)
                records = [record.data() for record in result]
                logger.debug(f"Query executed successfully, returned {len(records)} records")
                return records
        except Exception as e:
            logger.error(f"Error executing Cypher query: {e}")
            raise

    def create_constraints(self) -> bool:
        """Create necessary database constraints."""
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
        """Clear all data from the database."""
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
        """Populate Neo4j with knowledge graph data."""
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

                            # Only store primitive types or lists of primitive types directly
                            if isinstance(v, (str, int, float, bool)):
                                props[k] = v
                            elif isinstance(v, list) and all(isinstance(item, (str, int, float, bool)) for item in v):
                                props[k] = v
                            else:
                                # For any other complex type (dict, list of dicts, etc.), serialize to JSON string
                                try:
                                    import json
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
                    # Sanitize relationship label
                    sanitized_label = "".join(filter(str.isalnum, rel["label"]))
                    if not sanitized_label:
                        continue

                    # Filter relationship properties to avoid Neo4j storage issues
                    rel_props = {}
                    for k, v in rel.items():
                        if k not in ["source", "target", "label"]:
                            # Skip fields that cause Neo4j storage issues
                            if k in ["embedding", "embeddings"] or k.startswith("_"):
                                logger.debug(f"Skipping field '{k}' for relationship '{rel.get('label', 'unknown')}' - not compatible with Neo4j")
                                continue

                            # Only store primitive types or lists of primitive types directly
                            if isinstance(v, (str, int, float, bool)):
                                rel_props[k] = v
                            elif isinstance(v, list) and all(isinstance(item, (str, int, float, bool)) for item in v):
                                rel_props[k] = v
                            else:
                                # For any other complex type (dict, list of dicts, etc.), serialize to JSON string
                                try:
                                    import json
                                    rel_props[k] = json.dumps(v, default=str)
                                except (TypeError, ValueError) as e:
                                    logger.warning(f"Could not serialize field '{k}' for relationship '{rel.get('label', 'unknown')}': {e}")
                                    rel_props[k] = str(v)  # Fallback to string representation

                    if rel_props:
                        query = f"""
                        MATCH (src:Entity {{name: $source}})
                        MATCH (tgt:Entity {{name: $target}})
                        MERGE (src)-[r:`{sanitized_label}`]->(tgt)
                        SET r += $props
                        """
                        session.run(query, source=rel["source"], target=rel["target"], props=rel_props)
                    else:
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
            import traceback
            traceback.print_exc()
            return False

    def get_schema_info(self) -> Dict[str, Any]:
        """Get database schema information."""
        if not self.driver:
            logger.error("No active Neo4j connection")
            return {}

        try:
            with self.driver.session() as session:
                node_labels = [record["label"] for record in session.run("CALL db.labels()")]
                rel_types = [record["relationshipType"] for record in session.run("CALL db.relationshipTypes()")]

                return {
                    "node_labels": node_labels,
                    "relationship_types": rel_types,
                    "database_type": "neo4j"
                }
        except Exception as e:
            logger.error(f"Error getting schema info: {e}")
            return {}

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
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
                    "total_relationships": sum(rel_counts.values()),
                    "database_type": "neo4j"
                }
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on database."""
        if not self.driver:
            return {
                "status": "disconnected",
                "database_type": "neo4j"
            }

        try:
            with self.driver.session() as session:
                result = session.run("MATCH (n) RETURN count(n) as count LIMIT 1")
                count = result.single()["count"]

                return {
                    "status": "healthy",
                    "database_type": "neo4j",
                    "node_count": count
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "database_type": "neo4j",
                "error": str(e)
            }

    @property
    def supported_languages(self) -> List[QueryLanguage]:
        """Return supported query languages."""
        return [QueryLanguage.CYPHER]

    @property
    def database_type(self) -> str:
        """Return database type identifier."""
        return "neo4j"