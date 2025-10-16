"""
Dependency injection container and factories for modular components.
"""

from .container import DIContainer
from .factories import ComponentFactory

__all__ = ['DIContainer', 'ComponentFactory']