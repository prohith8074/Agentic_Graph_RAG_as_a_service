"""
Abstract interface for evaluation and tracing services.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
from functools import wraps


class IEvaluationService(ABC):
    """Abstract interface for LLM evaluation and tracing."""

    @abstractmethod
    def is_enabled(self) -> bool:
        """
        Check if evaluation service is enabled.

        Returns:
            True if service is active
        """
        pass

    @abstractmethod
    def track_llm_call(self, provider: str, model: str, prompt: str, response: str,
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Track an LLM call with evaluation metrics.

        Args:
            provider: LLM provider (e.g., 'groq', 'cohere')
            model: Model name
            prompt: Input prompt
            response: LLM response
            metadata: Additional metadata
        """
        pass

    @abstractmethod
    def track_query_routing(self, query: str, routing_decision: Dict[str, Any],
                           final_answer: str) -> None:
        """
        Track query routing decisions.

        Args:
            query: User query
            routing_decision: Routing analysis result
            final_answer: Final answer provided
        """
        pass

    @abstractmethod
    def track_pipeline_step(self, step_name: str, input_data: Any,
                           output_data: Any, duration: float) -> None:
        """
        Track pipeline processing steps.

        Args:
            step_name: Name of the processing step
            input_data: Input to the step
            output_data: Output from the step
            duration: Processing duration in seconds
        """
        pass

    @abstractmethod
    def log_performance(self, func_name: str, duration: float, **metadata) -> None:
        """
        Log performance metrics.

        Args:
            func_name: Function name
            duration: Execution duration in seconds
            metadata: Additional performance metadata
        """
        pass

    @abstractmethod
    def log_error(self, error_type: str, error_message: str, **metadata) -> None:
        """
        Log errors with evaluation context.

        Args:
            error_type: Type of error
            error_message: Error message
            metadata: Additional error metadata
        """
        pass