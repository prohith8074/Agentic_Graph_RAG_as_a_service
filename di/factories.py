"""
Component factories for creating modular service implementations.
"""

from typing import Type
from interfaces import (
    IPDFParser, IDatabaseManager, IQueryRouter,
    IMemoryManager, IEvaluationService
)
from core.pdf_parser import PDFParser
from knowledge_graph.neo4j_manager import Neo4jManager
from query.advanced_router import AdvancedQueryRouter
from utils.memory_manager import MemoryManager
from utils.opik_tracer import OpikTracer
from plugins.registry import PluginRegistry


class ComponentFactory:
    """Factory for creating component implementations."""

    def __init__(self, plugin_registry: PluginRegistry = None):
        """
        Initialize component factory.

        Args:
            plugin_registry: Plugin registry for dynamic loading
        """
        self.plugin_registry = plugin_registry or PluginRegistry()

    def create_pdf_parser(self) -> IPDFParser:
        """Create PDF parser implementation."""
        return PDFParser()

    def create_database_manager(self) -> IDatabaseManager:
        """Create database manager implementation."""
        # Use plugin system to select database adapter
        db_plugin = self.plugin_registry.get_database_plugin()
        if db_plugin:
            return db_plugin.create_manager()
        # Fallback to Neo4j
        return Neo4jManager()

    def create_query_router(self) -> IQueryRouter:
        """Create query router implementation."""
        return AdvancedQueryRouter()

    def create_memory_manager(self) -> IMemoryManager:
        """Create memory manager implementation."""
        return MemoryManager()

    def create_evaluation_service(self) -> IEvaluationService:
        """Create evaluation service implementation."""
        return OpikTracer()