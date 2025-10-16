import logging
import os

# This print statement is for diagnostics to ensure this file is being executed.
print("--- EXECUTING LATEST OPIK TRACER FILE ---")

from typing import Dict, Any, Optional, Callable
from functools import wraps
import opik

from config.settings import settings

logger = logging.getLogger(__name__)

class OpikTracer:
    """Centralized Opik tracer for LLM observability."""

    def __init__(self):
        """Initialize the Opik tracer using environment variables."""
        logger.info("--- Initializing OpikTracer with environment variable configuration ---")
        self._initialized = False
        self._project_name = getattr(settings, 'OPIK_PROJECT_NAME', 'Lyzr_challenge')
        self._workspace = 'rohith2'

        opik_api_key = getattr(settings, 'OPIK_API_KEY', None)
        if opik_api_key:
            try:
                # Set credentials via environment variables for compatibility
                os.environ["OPIK_PROJECT_NAME"] = self._project_name
                os.environ["OPIK_WORKSPACE"] = self._workspace
                
                # Configure with only the API key
                opik.configure(api_key=opik_api_key)

                self._initialized = True
                logger.info(f"Opik tracer initialized for project '{self._project_name}' in workspace '{self._workspace}'")
            except Exception as e:
                logger.warning(f"Failed to initialize Opik tracer: {e}")
        else:
            logger.warning("OPIK_API_KEY not found, LLM tracing disabled")

    def is_enabled(self) -> bool:
        return self._initialized

    def track_llm_call(self, provider: str, model: str, prompt: str, response: str,
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        if not self.is_enabled(): return
        try:
            opik.track(name=f"{provider}_llm_call", metadata=metadata)
        except Exception as e:
            logger.warning(f"Failed to track LLM call with Opik: {e}")

    def track_query_routing(self, query: str, routing_decision: Dict[str, Any],
                           final_answer: str) -> None:
        if not self.is_enabled(): return
        try:
            opik.track(name="query_routing", metadata=routing_decision)
        except Exception as e:
            logger.warning(f"Failed to track query routing with Opik: {e}")

    def track_pipeline_step(self, step_name: str, input_data: Any,
                           output_data: Any, duration: float) -> None:
        if not self.is_enabled(): return
        try:
            opik.track(name=f"pipeline_{step_name}", metadata={"duration_seconds": duration})
        except Exception as e:
            logger.warning(f"Failed to track pipeline step with Opik: {e}")

    def log_error(self, error_type: str, error_message: str, **metadata) -> None:
        logger.error(f"{error_type}: {error_message}")
        if self.is_enabled():
            try:
                opik.track(name=error_type, metadata=metadata)
            except Exception as e:
                logger.warning(f"Failed to log error with Opik: {e}")

# Global tracer instance
opik_tracer = OpikTracer()

# Decorator for easy integration
def track_query_routing(func: Callable) -> Callable:
    """Decorator to track query routing decisions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # The first arg is 'self' of the class whose method is decorated
        query = args[1] if len(args) > 1 else kwargs.get('user_query', '')
        
        # Execute the original function to get the result
        result = func(*args, **kwargs)
        
        # If tracing is enabled, track the result
        if opik_tracer.is_enabled() and isinstance(result, dict):
            opik_tracer.track_query_routing(
                query=query,
                routing_decision=result.get('routing', {}),
                final_answer=result.get('answer', '')
            )
        return result
    return wrapper