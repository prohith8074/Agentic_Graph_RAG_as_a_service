# """
# AWS Neptune graph database adapter.
# Provides support for AWS Neptune with Gremlin query language.
# NOTE: This implementation is commented out as per user request.
# To enable, uncomment the code in this file, in neptune_plugin.py, 
# and in the corresponding __init__.py files.
# """

# import logging
# from typing import Dict, Any, List, Optional
# from gremlin_python.driver import client, serializer
# from gremlin_python.driver.protocol import GremlinServerError
# from gremlin_python.process.anonymous_traversal import traversal
# from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection

# from . import GraphDatabaseAdapter, QueryLanguage
# from config.settings import settings

# logger = logging.getLogger(__name__)

# class NeptuneAdapter(GraphDatabaseAdapter):
#     """AWS Neptune graph database adapter with Gremlin support."""

#     def __init__(self):
#         """Initialize Neptune adapter with settings."""
#         self.endpoint = settings.NEPTUNE_ENDPOINT
#         self.port = settings.NEPTUNE_PORT
#         self.use_ssl = True
#         self.connection = None
#         self.g = None

#     def connect(self) -> bool:
#         """Establish connection to Neptune database."""
#         if not self.endpoint:
#             logger.error("Neptune endpoint not configured.")
#             return False
        
#         try:
#             connection_url = f"wss://{self.endpoint}:{self.port}/gremlin"
#             self.connection = DriverRemoteConnection(connection_url, 'g')
#             self.g = traversal().withRemote(self.connection)
            
#             # Test connection by getting a vertex count
#             count = self.g.V().count().next()
#             logger.info(f"Successfully connected to Neptune at {self.endpoint}. Found {count} vertices.")
#             return True
#         except Exception as e:
#             logger.error(f"Failed to connect to Neptune: {e}")
#             if self.connection:
#                 self.connection.close()
#             return False

#     def disconnect(self):
#         """Close Neptune connection."""
#         if self.connection:
#             self.connection.close()
#             logger.info("Disconnected from Neptune")

#     def execute_query(self, query: str, language: QueryLanguage = QueryLanguage.GREMLIN) -> List[Dict[str, Any]]:
#         """Execute Gremlin query and return results."""
#         if language != QueryLanguage.GREMLIN:
#             raise ValueError(f"Neptune adapter only supports {QueryLanguage.GREMLIN.value}")

#         if not self.g:
#             raise ConnectionError("No active Neptune connection")

#         try:
#             result = self.g.inject(query).next() # This is a simplified way to run raw Gremlin
#             # A more robust implementation would parse the query or use a proper execution method
#             return result
#         except Exception as e:
#             logger.error(f"Error executing Gremlin query: {e}")
#             raise

#     def populate_graph(self, graph_data: Dict[str, Any]) -> bool:
#         """Populate Neptune with knowledge graph data."""
#         if not self.g:
#             raise ConnectionError("No active Neptune connection")

#         try:
#             # Populate entities (vertices)
#             for entity in graph_data.get("entities", []):
#                 t = self.g.addV(entity.get('type', 'Entity')).property('name', entity['name'])
#                 for key, value in entity.items():
#                     if key not in ['name', 'type']:
#                         t.property(key, value)
#                 t.next()

#             # Populate relationships (edges)
#             for rel in graph_data.get("relationships", []):
#                 self.g.V().has('name', rel['source']).addE(rel['label']).to(
#                     self.g.V().has('name', rel['target'])
#                 ).next()
            
#             logger.info("Successfully populated Neptune with knowledge graph")
#             return True
#         except Exception as e:
#             logger.error(f"Error populating Neptune graph: {e}")
#             return False

#     @property
#     def supported_languages(self) -> List[QueryLanguage]:
#         return [QueryLanguage.GREMLIN]

#     @property
#     def database_type(self) -> str:
#         return "neptune"

#     # Other methods like clear_database, get_schema_info, etc. would be implemented here.
#     def clear_database(self) -> bool:
#         self.g.V().drop().iterate()
#         return True

#     def get_schema_info(self) -> Dict[str, Any]:
#         return {"message": "Schema info not implemented for Neptune adapter"}

#     def get_database_stats(self) -> Dict[str, Any]:
#         v_count = self.g.V().count().next()
#         e_count = self.g.E().count().next()
#         return {"vertex_count": v_count, "edge_count": e_count}

#     def health_check(self) -> Dict[str, Any]:
#         try:
#             self.g.V().limit(1).count().next()
#             return {"status": "healthy"}
#         except Exception as e:
#             return {"status": "unhealthy", "error": str(e)}

#     def create_constraints(self) -> bool:
#         logger.info("Neptune does not support constraints in the same way as Neo4j. Skipping.")
#         return True