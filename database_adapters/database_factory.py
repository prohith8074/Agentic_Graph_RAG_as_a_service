"""
Database factory for creating and managing all database adapters.
"""

import logging
from typing import Dict, Any, Optional

from config.settings import settings
from interfaces.ivector_database import IVectorDatabase
from . import GraphDatabaseAdapter
from .neo4j_adapter import Neo4jAdapter
from .vector.qdrant_adapter import QdrantAdapter

logger = logging.getLogger(__name__)

class DatabaseFactory:
    """Factory for creating all database adapter instances."""

    _adapter_map = {
        "graph": {
            "neo4j": Neo4jAdapter,
        },
        "vector": {
            "qdrant": QdrantAdapter,
        }
    }

    @staticmethod
    def create_adapter(adapter_type: str, database_name: str, **kwargs) -> Optional[Any]:
        """Creates a database adapter based on its type and name."""
        adapter_class = DatabaseFactory._adapter_map.get(adapter_type, {}).get(database_name.lower())
        if not adapter_class:
            logger.error(f"Unsupported adapter: {adapter_type}/{database_name}")
            return None
        try:
            adapter = adapter_class(**kwargs)
            logger.info(f"Created {database_name} {adapter_type} adapter")
            return adapter
        except Exception as e:
            logger.error(f"Error creating {database_name} adapter: {e}")
            return None

class DatabaseManager:
    """Manager for handling multiple graph database connections."""
    def __init__(self):
        self.adapters: Dict[str, GraphDatabaseAdapter] = {}
        self.active_adapter: Optional[str] = None

    def register_adapter(self, name: str, adapter: GraphDatabaseAdapter):
        self.adapters[name] = adapter
        logger.info(f"Registered graph database adapter: {name}")

    def set_active_adapter(self, name: str):
        if name not in self.adapters:
            raise ValueError(f"Adapter {name} not registered")
        self.active_adapter = name
        logger.info(f"Set active graph adapter to: {name}")

    def get_adapter(self, name: Optional[str] = None) -> Optional[GraphDatabaseAdapter]:
        return self.adapters.get(name or self.active_adapter)

    def connect_all(self):
        for name, adapter in self.adapters.items():
            try:
                adapter.connect()
            except Exception as e:
                logger.error(f"Failed to connect graph adapter {name}: {e}")

    def disconnect_all(self):
        for adapter in self.adapters.values():
            adapter.disconnect()

# --- Global Instances ---
db_manager = DatabaseManager()
vector_adapter: Optional[IVectorDatabase] = None
qdrant_client = None # Managed by the QdrantAdapter now

def get_vector_adapter() -> Optional[IVectorDatabase]:
    return vector_adapter

def init_database_connections() -> bool:
    """Initialize all database connections."""
    global vector_adapter, qdrant_client
    try:
        # Init Graph DB
        neo4j_adapter = DatabaseFactory.create_adapter("graph", "neo4j")
        if neo4j_adapter:
            db_manager.register_adapter("neo4j", neo4j_adapter)
            db_manager.set_active_adapter("neo4j")
            db_manager.connect_all()
        
        # Init Vector DB
        init_qdrant_connection() # This sets the global qdrant_client
        vector_adapter = DatabaseFactory.create_adapter("vector", "qdrant")
        if vector_adapter:
            vector_adapter.connect()

        return neo4j_adapter is not None and vector_adapter is not None
    except Exception as e:
        logger.error(f"Database initialization failed: {e}", exc_info=True)
        return False

def init_qdrant_connection():
    """Initializes the global Qdrant client, falling back to in-memory if remote fails."""
    global qdrant_client
    from qdrant_client import QdrantClient
    if settings.QDRANT_URL:
        try:
            qdrant_client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=10)
            qdrant_client.get_collections() # Verify connection
            logger.info(f"Connected to remote Qdrant at {settings.QDRANT_URL}")
        except Exception as e:
            logger.warning(f"Could not connect to remote Qdrant: {e}. Falling back to in-memory instance.")
            qdrant_client = QdrantClient(":memory:")
    else:
        logger.info("QDRANT_URL not set. Using in-memory Qdrant instance.")
        qdrant_client = QdrantClient(":memory:")

def get_qdrant_client() -> Optional[Any]:
    return qdrant_client

def ensure_qdrant_collection(collection_name: str = None) -> bool:
    """
    Ensure Qdrant collection exists, creating it if necessary.
    """
    global qdrant_client
    if not qdrant_client:
        logger.error("Qdrant client not initialized")
        return False

    collection_name = collection_name or settings.QDRANT_COLLECTION_NAME

    try:
        qdrant_client.get_collection(collection_name=collection_name)
        logger.info(f"Qdrant collection '{collection_name}' already exists.")
        return True
    except Exception:
        logger.info(f"Qdrant collection '{collection_name}' not found. Creating it...")
        try:
            from qdrant_client.models import Distance, VectorParams
            qdrant_client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=settings.QDRANT_VECTOR_SIZE, distance=Distance.COSINE)
            )
            logger.info(f"Successfully created Qdrant collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create Qdrant collection '{collection_name}': {e}")
            return False

def shutdown_database_connections():
    """Shutdown all database connections."""
    logger.info("Shutting down database connections...")
    db_manager.disconnect_all()
    global vector_adapter
    if vector_adapter and hasattr(vector_adapter, 'disconnect'):
        vector_adapter.disconnect()