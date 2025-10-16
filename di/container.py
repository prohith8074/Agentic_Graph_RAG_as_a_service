"""
Dependency injection container for managing component lifecycles.
"""

from typing import Dict, Any, Type, Optional, TypeVar
from interfaces import (
    IPDFParser, IDatabaseManager, IQueryRouter,
    IMemoryManager, IEvaluationService
)

T = TypeVar('T')


class DIContainer:
    """Dependency injection container for component management."""

    def __init__(self):
        """Initialize the DI container."""
        self._services: Dict[Type, Any] = {}
        self._singletons: Dict[Type, Any] = {}
        self._factories: Dict[Type, callable] = {}

    def register(self, interface: Type[T], implementation: Type[T],
                singleton: bool = True) -> None:
        """
        Register a service implementation.

        Args:
            interface: Abstract interface type
            implementation: Concrete implementation type
            singleton: Whether to create as singleton
        """
        if singleton:
            self._singletons[interface] = None  # Will be created on first access
        else:
            self._services[interface] = implementation

    def register_factory(self, interface: Type[T], factory: callable) -> None:
        """
        Register a factory function for service creation.

        Args:
            interface: Abstract interface type
            factory: Factory function that returns implementation
        """
        self._factories[interface] = factory

    def register_instance(self, interface: Type[T], instance: T) -> None:
        """
        Register a pre-created instance.

        Args:
            interface: Abstract interface type
            instance: Pre-created instance
        """
        self._singletons[interface] = instance

    def resolve(self, interface: Type[T]) -> T:
        """
        Resolve a service implementation.

        Args:
            interface: Interface type to resolve

        Returns:
            Service implementation instance

        Raises:
            ValueError: If service not registered
        """
        # Check for pre-registered instance
        if interface in self._singletons:
            if self._singletons[interface] is None:
                # Create singleton instance
                if interface in self._factories:
                    self._singletons[interface] = self._factories[interface]()
                elif interface in self._services:
                    self._singletons[interface] = self._services[interface]()
                else:
                    raise ValueError(f"No factory or implementation registered for {interface}")
            return self._singletons[interface]

        # Check for factory
        if interface in self._factories:
            return self._factories[interface]()

        # Check for transient service
        if interface in self._services:
            return self._services[interface]()

        raise ValueError(f"Service not registered: {interface}")

    def has_service(self, interface: Type[T]) -> bool:
        """
        Check if a service is registered.

        Args:
            interface: Interface type to check

        Returns:
            True if service is registered
        """
        return (interface in self._singletons or
                interface in self._factories or
                interface in self._services)

    def clear(self) -> None:
        """Clear all registered services."""
        self._services.clear()
        self._singletons.clear()
        self._factories.clear()