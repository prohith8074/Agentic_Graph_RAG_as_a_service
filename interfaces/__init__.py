"""
Abstract interfaces for modular RAG system components.
Provides contracts for dependency injection and plugin system.
"""

from .ipdf_parser import IPDFParser
from .inode_processor import INodeProcessor
from .icontextual_retriever import IContextualRetriever
from .ientity_extractor import IEntityExtractor
from .igraph_builder import IGraphBuilder
from .idatabase_manager import IDatabaseManager
from .iquery_router import IQueryRouter
from .imemory_manager import IMemoryManager
from .iembeddings import IEmbeddings
from .ievaluation_service import IEvaluationService

__all__ = [
    'IPDFParser',
    'INodeProcessor',
    'IContextualRetriever',
    'IEntityExtractor',
    'IGraphBuilder',
    'IDatabaseManager',
    'IQueryRouter',
    'IMemoryManager',
    'IEmbeddings',
    'IEvaluationService'
]