"""
# Database adapter plugins.
# NOTE: Neptune plugin is commented out since Neptune support is disabled.
"""

from .neo4j_plugin import Neo4jPlugin
# from .neptune_plugin import NeptunePlugin  # Commented out - Neptune support disabled

__all__ = ['Neo4jPlugin']  # 'NeptunePlugin' commented out - Neptune support disabled