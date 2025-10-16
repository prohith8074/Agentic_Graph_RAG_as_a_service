"""
Evaluation service implementing IEvaluationService interface with Opik integration.
"""

import logging
from typing import Dict, Any, Optional, Callable
from functools import wraps

from interfaces import IEvaluationService
from .llm_tracer import LLMTracer

logger = logging.getLogger(__name__)


class EvaluationService(IEvaluationService):
    """Evaluation service with Opik integration for LLM tracing and monitoring."""

    def __init__(self):
        """Initialize evaluation service."""
        self.tracer = LLMTracer()
        self._metrics_store: Dict[str, Any] = {}

    def is_enabled(self) -> bool:
        """Check if evaluation service is enabled."""
        return self.tracer.is_enabled()

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
        self.tracer.track_llm_call(provider, model, prompt, response, metadata)

        # Store metrics locally
        if metadata:
            call_id = f"{provider}_{model}_{hash(prompt) % 10000}"
            self._metrics_store[call_id] = {
                "provider": provider,
                "model": model,
                "timestamp": __import__('time').time(),
                "metrics": metadata
            }

    def track_query_routing(self, query: str, routing_decision: Dict[str, Any],
                           final_answer: str) -> None:
        """
        Track query routing decisions.

        Args:
            query: User query
            routing_decision: Routing analysis result
            final_answer: Final answer provided
        """
        # Calculate evaluation metrics for the query
        evaluation_metrics = self._evaluate_query_performance(
            query, routing_decision, final_answer
        )

        self.tracer.track_query_evaluation(
            query, routing_decision, final_answer, evaluation_metrics
        )

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
        metrics = {
            "input_size": len(str(input_data)) if input_data else 0,
            "output_size": len(str(output_data)) if output_data else 0,
            "efficiency": self._calculate_efficiency(duration, input_data, output_data)
        }

        self.tracer.track_pipeline_step(
            step_name, input_data, output_data, duration, metrics
        )

    def log_performance(self, func_name: str, duration: float, **metadata) -> None:
        """
        Log performance metrics.

        Args:
            func_name: Function name
            duration: Execution duration in seconds
            metadata: Additional performance metadata
        """
        logger.info(f"Performance - {func_name}: {duration:.2f}s")

        if self.is_enabled():
            try:
                __import__('opik').track(
                    name="performance_metric",
                    project=self.tracer._project_name,
                    metadata={"function": func_name, "duration": duration, **metadata}
                )
            except Exception as e:
                logger.warning(f"Failed to log performance: {e}")

    def log_error(self, error_type: str, error_message: str, **metadata) -> None:
        """
        Log errors with evaluation context.

        Args:
            error_type: Type of error
            error_message: Error message
            metadata: Additional error metadata
        """
        logger.error(f"{error_type}: {error_message}")

        if self.is_enabled():
            try:
                __import__('opik').track(
                    name="error_log",
                    project=self.tracer._project_name,
                    metadata={"error_type": error_type, **metadata}
                )
            except Exception as e:
                logger.warning(f"Failed to log error: {e}")

    def get_evaluation_metrics(self, time_range: Optional[int] = None) -> Dict[str, Any]:
        """
        Get evaluation metrics summary.

        Args:
            time_range: Time range in seconds to look back (optional)

        Returns:
            Dictionary with evaluation metrics
        """
        current_time = __import__('time').time()

        # Filter metrics by time range if specified
        relevant_metrics = {}
        if time_range:
            cutoff_time = current_time - time_range
            relevant_metrics = {
                k: v for k, v in self._metrics_store.items()
                if v.get('timestamp', 0) > cutoff_time
            }
        else:
            relevant_metrics = self._metrics_store

        # Calculate aggregate metrics
        if relevant_metrics:
            provider_counts = {}
            model_counts = {}
            avg_performance = []

            for metric in relevant_metrics.values():
                provider = metric.get('provider', 'unknown')
                model = metric.get('model', 'unknown')

                provider_counts[provider] = provider_counts.get(provider, 0) + 1
                model_counts[model] = model_counts.get(model, 0) + 1

                if 'duration' in metric.get('metrics', {}):
                    avg_performance.append(metric['metrics']['duration'])

            return {
                "total_calls": len(relevant_metrics),
                "provider_distribution": provider_counts,
                "model_distribution": model_counts,
                "avg_performance": sum(avg_performance) / len(avg_performance) if avg_performance else 0,
                "time_range": time_range
            }
        else:
            return {"total_calls": 0, "message": "No metrics available"}

    def _evaluate_query_performance(self, query: str, routing_decision: Dict[str, Any],
                                   final_answer: str) -> Dict[str, Any]:
        """
        Evaluate query performance and quality.

        Args:
            query: User query
            routing_decision: Routing decision details
            final_answer: Final answer

        Returns:
            Evaluation metrics
        """
        metrics = {
            "query_length": len(query),
            "answer_length": len(final_answer),
            "routing_confidence": routing_decision.get("confidence", 0.0),
            "method_used": routing_decision.get("method", "unknown"),
            "overall_score": 0.0
        }

        # Calculate overall score based on various factors
        confidence_weight = 0.4
        answer_quality_weight = 0.3
        efficiency_weight = 0.3

        # Confidence score
        confidence_score = metrics["routing_confidence"]

        # Answer quality score (simple heuristic)
        answer_quality = min(1.0, len(final_answer) / 100)  # Prefer substantial answers

        # Efficiency score (inverse of processing time if available)
        efficiency_score = 0.8  # Default good score

        if "processing_time" in routing_decision:
            # Penalize very slow responses
            proc_time = routing_decision["processing_time"]
            efficiency_score = max(0.1, min(1.0, 10.0 / (proc_time + 1)))

        # Calculate overall score
        metrics["overall_score"] = (
            confidence_weight * confidence_score +
            answer_quality_weight * answer_quality +
            efficiency_weight * efficiency_score
        )

        return metrics

    def _calculate_efficiency(self, duration: float, input_data: Any, output_data: Any) -> float:
        """
        Calculate processing efficiency.

        Args:
            duration: Processing duration
            input_data: Input data
            output_data: Output data

        Returns:
            Efficiency score (0.0 to 1.0)
        """
        if duration <= 0:
            return 1.0

        input_size = len(str(input_data)) if input_data else 0
        output_size = len(str(output_data)) if output_data else 0

        # Efficiency based on processing rate (characters per second)
        if input_size > 0:
            processing_rate = (input_size + output_size) / duration
            # Normalize to a reasonable scale
            efficiency = min(1.0, processing_rate / 1000)
            return efficiency

        return 0.5  # Neutral score