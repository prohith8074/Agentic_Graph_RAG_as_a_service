"""
Plugin registry for dynamic component discovery and loading.
"""

import importlib
import pkgutil
from typing import Dict, Any, Optional, Type, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Registry for managing plugins and their discovery."""

    def __init__(self):
        """Initialize plugin registry."""
        self._database_plugins: Dict[str, Any] = {}
        self._query_plugins: Dict[str, Any] = {}
        self._loaded_plugins: Dict[str, Any] = {}

        # Default plugin configurations
        self._defaults = {
            'database': 'neo4j',
            'query': 'advanced_router'
        }

        # Auto-discover plugins
        self._discover_plugins()

    def _discover_plugins(self) -> None:
        """Auto-discover available plugins."""
        try:
            # Discover database plugins
            self._discover_database_plugins()

            # Discover query plugins
            self._discover_query_plugins()

            logger.info("Plugin discovery completed")

        except Exception as e:
            logger.warning(f"Plugin discovery failed: {e}")

    def _discover_database_plugins(self) -> None:
        """Discover database adapter plugins."""
        database_plugins = {
            'neo4j': 'plugins.database.neo4j_plugin',
            # 'neptune': 'plugins.database.neptune_plugin'  # Commented out - Neptune support disabled
        }

        for name, module_path in database_plugins.items():
            try:
                module = importlib.import_module(module_path)
                if hasattr(module, 'Neo4jPlugin'):
                    self._database_plugins[name] = module.Neo4jPlugin()
                # elif hasattr(module, 'NeptunePlugin'):
                #     self._database_plugins[name] = module.NeptunePlugin()
                    logger.debug(f"Loaded database plugin: {name}")
            except ImportError:
                logger.debug(f"Database plugin not available: {name}")
            except Exception as e:
                logger.warning(f"Failed to load database plugin {name}: {e}")

    def _discover_query_plugins(self) -> None:
        """Discover query handler plugins."""
        query_plugins = {
            'vector': 'plugins.query.vector_plugin',
            'graph': 'plugins.query.graph_plugin',
            'filter': 'plugins.query.filter_plugin'
        }

        for name, module_path in query_plugins.items():
            try:
                module = importlib.import_module(module_path)
                if hasattr(module, 'VectorPlugin'):
                    self._query_plugins[name] = module.VectorPlugin()
                elif hasattr(module, 'GraphPlugin'):
                    self._query_plugins[name] = module.GraphPlugin()
                elif hasattr(module, 'HybridPlugin'):
                    self._query_plugins[name] = module.HybridPlugin()
                elif hasattr(module, 'FilterPlugin'):
                    self._query_plugins[name] = module.FilterPlugin()
                    logger.debug(f"Loaded query plugin: {name}")
            except ImportError:
                logger.debug(f"Query plugin not available: {name}")
            except Exception as e:
                logger.warning(f"Failed to load query plugin {name}: {e}")

    def get_database_plugin(self, name: Optional[str] = None) -> Optional[Any]:
        """
        Get a database plugin by name.

        Args:
            name: Plugin name, uses default if None

        Returns:
            Database plugin instance or None
        """
        plugin_name = name or self._defaults['database']
        return self._database_plugins.get(plugin_name)

    def get_query_plugin(self, name: Optional[str] = None) -> Optional[Any]:
        """
        Get a query plugin by name.

        Args:
            name: Plugin name, uses default if None

        Returns:
            Query plugin instance or None
        """
        plugin_name = name or self._defaults['query']
        return self._query_plugins.get(plugin_name)

    def list_database_plugins(self) -> List[str]:
        """
        List available database plugins.

        Returns:
            List of plugin names
        """
        return list(self._database_plugins.keys())

    def list_query_plugins(self) -> List[str]:
        """
        List available query plugins.

        Returns:
            List of plugin names
        """
        return list(self._query_plugins.keys())

    def register_database_plugin(self, name: str, plugin: Any) -> None:
        """
        Register a database plugin.

        Args:
            name: Plugin name
            plugin: Plugin instance
        """
        self._database_plugins[name] = plugin
        logger.info(f"Registered database plugin: {name}")

    def register_query_plugin(self, name: str, plugin: Any) -> None:
        """
        Register a query plugin.

        Args:
            name: Plugin name
            plugin: Plugin instance
        """
        self._query_plugins[name] = plugin
        logger.info(f"Registered query plugin: {name}")

    def set_default_database(self, name: str) -> None:
        """
        Set default database plugin.

        Args:
            name: Plugin name
        """
        if name in self._database_plugins:
            self._defaults['database'] = name
            logger.info(f"Set default database plugin: {name}")
        else:
            raise ValueError(f"Database plugin not found: {name}")

    def set_default_query(self, name: str) -> None:
        """
        Set default query plugin.

        Args:
            name: Plugin name
        """
        if name in self._query_plugins:
            self._defaults['query'] = name
            logger.info(f"Set default query plugin: {name}")
        else:
            raise ValueError(f"Query plugin not found: {name}")