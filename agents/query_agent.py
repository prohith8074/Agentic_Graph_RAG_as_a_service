"""
Intelligent query agent that routes queries to appropriate plugins.
"""

import logging
from typing import Dict, Any, List, Tuple
from .llm_orchestrator import LLMOrchestrator
from .strategy_selector import StrategySelector
from plugins.registry import PluginRegistry
from utils.opik_tracer import opik_tracer

logger = logging.getLogger(__name__)


class QueryAgent:
    """Intelligent agent for query routing and execution."""

    def __init__(self, llm_orchestrator: LLMOrchestrator = None,
                 plugin_registry: PluginRegistry = None):
        """
        Initialize query agent.

        Args:
            llm_orchestrator: LLM orchestration service
            plugin_registry: Plugin registry for component discovery
        """
        self.llm = llm_orchestrator or LLMOrchestrator()
        self.plugin_registry = plugin_registry or PluginRegistry()
        self.strategy_selector = StrategySelector(self.llm)

        logger.info("Query Agent initialized")

    def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a user query by selecting optimal strategy and executing it.

        Args:
            query: User query string
            context: Additional context information

        Returns:
            Query result with metadata
        """
        context = context or {}

        try:
            # Step 1: Analyze query and select strategy
            strategy_analysis = self.strategy_selector.analyze_query(query, context)

            # Step 2: Select appropriate plugin
            plugin_name, confidence = self._select_plugin(strategy_analysis)

            # Step 3: Execute query using selected plugin
            result = self._execute_with_plugin(plugin_name, query, context)

            # Step 4: Add evaluation metadata
            result.update({
                "strategy_analysis": strategy_analysis,
                "selected_plugin": plugin_name,
                "confidence": confidence,
                "agent": "query_agent"
            })

            # Track with Opik
            opik_tracer.track_query_routing(
                query=query,
                routing_decision={
                    "method": plugin_name,
                    "confidence": confidence,
                    "strategy": strategy_analysis
                },
                final_answer=result.get("answer", "")
            )

            return result

        except Exception as e:
            logger.error(f"Query agent processing error: {e}")
            return {
                "error": str(e),
                "answer": "I encountered an error processing your query.",
                "agent": "query_agent"
            }

    def _select_plugin(self, strategy_analysis: Dict[str, Any]) -> Tuple[str, float]:
        """
        Select the best plugin based on strategy analysis.

        Args:
            strategy_analysis: Strategy analysis results

        Returns:
            Tuple of (plugin_name, confidence_score)
        """
        strategy_type = strategy_analysis.get("recommended_strategy", "hybrid")
        confidence = strategy_analysis.get("confidence", 0.5)

        # Map strategy types to plugins
        strategy_plugin_map = {
            "vector": "vector",
            "graph": "graph",
            "hybrid": "hybrid",
            "filter": "filter"
        }

        plugin_name = strategy_plugin_map.get(strategy_type, "hybrid")

        # Verify plugin is available
        if not self.plugin_registry.get_query_plugin(plugin_name):
            logger.warning(f"Plugin {plugin_name} not available, falling back to hybrid")
            plugin_name = "hybrid"
            confidence = max(confidence * 0.8, 0.3)  # Reduce confidence for fallback

        return plugin_name, confidence

    def _execute_with_plugin(self, plugin_name: str, query: str,
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute query using specified plugin.

        Args:
            plugin_name: Name of plugin to use
            query: User query
            context: Query context

        Returns:
            Query execution result
        """
        plugin = self.plugin_registry.get_query_plugin(plugin_name)

        if not plugin:
            raise ValueError(f"Plugin {plugin_name} not found")

        try:
            # Different plugins have different interfaces
            if hasattr(plugin, 'process_query'):
                # For custom plugins (vector, graph, hybrid, filter)
                return plugin.process_query(query, context)
            elif hasattr(plugin, 'create_handler'):
                # For handler-based plugins
                handler = plugin.create_handler()
                if hasattr(handler, 'query'):
                    return handler.query(query)
                else:
                    raise AttributeError(f"Handler has no query method")

        except Exception as e:
            logger.error(f"Plugin execution error for {plugin_name}: {e}")
            raise

    def get_available_strategies(self) -> List[str]:
        """
        Get list of available query strategies.

        Returns:
            List of strategy names
        """
        return self.plugin_registry.list_query_plugins()

    def explain_decision(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Explain why a particular strategy/plugin was selected for a query.

        Args:
            query: User query
            context: Query context

        Returns:
            Explanation of decision process
        """
        context = context or {}

        strategy_analysis = self.strategy_selector.analyze_query(query, context)
        plugin_name, confidence = self._select_plugin(strategy_analysis)

        return {
            "query": query,
            "strategy_analysis": strategy_analysis,
            "selected_plugin": plugin_name,
            "confidence": confidence,
            "explanation": self._generate_explanation(strategy_analysis, plugin_name, confidence)
        }

    def _generate_explanation(self, strategy_analysis: Dict[str, Any],
                            plugin_name: str, confidence: float) -> str:
        """Generate human-readable explanation of agent decision."""
        strategy = strategy_analysis.get("recommended_strategy", "unknown")
        reasoning = strategy_analysis.get("reasoning", "No reasoning provided")

        explanations = {
            "vector": "Selected vector search for semantic similarity and descriptive queries.",
            "graph": "Selected graph search for relationship and structural queries.",
            "hybrid": "Selected hybrid search combining vector and graph approaches.",
            "filter": "Selected filtering for constraint-based and conditional queries."
        }

        plugin_explanation = explanations.get(plugin_name, f"Selected {plugin_name} plugin.")

        confidence_text = ".2f"

        return f"{plugin_explanation} Confidence: {confidence_text}. Reasoning: {reasoning}"