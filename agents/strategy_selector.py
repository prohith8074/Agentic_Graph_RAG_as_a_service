"""
Strategy selection for query routing based on LLM analysis.
"""

import logging
from typing import Dict, Any
from .llm_orchestrator import LLMOrchestrator

logger = logging.getLogger(__name__)


class StrategySelector:
    """Analyzes queries and recommends optimal retrieval strategies."""

    def __init__(self, llm_orchestrator: LLMOrchestrator):
        """
        Initialize strategy selector.

        Args:
            llm_orchestrator: LLM orchestration service
        """
        self.llm = llm_orchestrator

    def analyze_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze query to determine optimal retrieval strategy.

        Args:
            query: User query string
            context: Additional context information

        Returns:
            Strategy analysis results
        """
        context = context or {}

        # Create analysis prompt
        prompt = self._build_analysis_prompt(query, context)

        try:
            # Use LLM to analyze query
            response = self.llm.create_completion(
                prompt=prompt,
                provider="groq",
                model_type="reasoning",
                temperature=0.1,
                max_tokens=500
            )

            # Parse LLM response
            analysis = self._parse_analysis_response(response)

            # Add metadata
            analysis.update({
                "query": query,
                "context_provided": bool(context),
                "analysis_method": "llm_reasoning"
            })

            return analysis

        except Exception as e:
            logger.error(f"Strategy analysis failed: {e}")
            # Return fallback analysis
            return self._fallback_analysis(query, context)

    def _build_analysis_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """Build the analysis prompt for the LLM."""
        return f"""
Analyze this query and determine the optimal retrieval strategy:

Query: "{query}"

Context: {context.get('document_type', 'unknown') if context else 'none'}

Available Strategies:
1. VECTOR - Semantic similarity search for descriptive/explanatory queries
2. GRAPH - Structural relationship queries (what connects to what, hierarchies)
3. HYBRID - Combination of vector and graph for complex queries
4. FILTER - Constraint-based queries (show only X where Y)

Consider:
- Query intent and type
- Information structure needed
- Complexity level
- Expected result format

Respond with JSON:
{{
    "recommended_strategy": "vector|graph|hybrid|filter",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "complexity": "simple|moderate|complex",
    "expected_sources": ["vector", "graph", "both"],
    "fallback_strategy": "strategy_name"
}}
"""

    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM analysis response.

        Args:
            response: Raw LLM response

        Returns:
            Parsed analysis dictionary
        """
        try:
            import json
            import re

            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())

                # Validate required fields
                required_fields = ["recommended_strategy", "confidence", "reasoning"]
                for field in required_fields:
                    if field not in analysis:
                        analysis[field] = self._get_default_value(field)

                # Ensure confidence is float
                analysis["confidence"] = float(analysis.get("confidence", 0.5))

                return analysis
            else:
                logger.warning("No JSON found in LLM response, using fallback")
                return self._fallback_analysis("", {})

        except Exception as e:
            logger.error(f"Failed to parse analysis response: {e}")
            return self._fallback_analysis("", {})

    def _get_default_value(self, field: str) -> Any:
        """Get default values for missing analysis fields."""
        defaults = {
            "recommended_strategy": "hybrid",
            "confidence": 0.5,
            "reasoning": "Default fallback strategy",
            "complexity": "moderate",
            "expected_sources": ["vector", "graph"],
            "fallback_strategy": "vector"
        }
        return defaults.get(field, None)

    def _fallback_analysis(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback strategy analysis when LLM fails.

        Args:
            query: User query
            context: Query context

        Returns:
            Basic fallback analysis
        """
        query_lower = query.lower() if query else ""

        # Simple keyword-based strategy selection
        if any(word in query_lower for word in ['relationship', 'connected', 'structure', 'hierarchy']):
            strategy = "graph"
            confidence = 0.8
        elif any(word in query_lower for word in ['describe', 'what is', 'explain', 'how']):
            strategy = "vector"
            confidence = 0.7
        elif any(word in query_lower for word in ['filter', 'only', 'where', 'show me']):
            strategy = "filter"
            confidence = 0.8
        else:
            strategy = "hybrid"
            confidence = 0.6

        return {
            "recommended_strategy": strategy,
            "confidence": confidence,
            "reasoning": f"Keyword-based fallback: {strategy}",
            "complexity": "moderate",
            "expected_sources": ["vector", "graph"] if strategy == "hybrid" else [strategy],
            "fallback_strategy": "vector",
            "analysis_method": "keyword_fallback"
        }

    def get_strategy_explanation(self, analysis: Dict[str, Any]) -> str:
        """
        Generate human-readable explanation of strategy choice.

        Args:
            analysis: Strategy analysis results

        Returns:
            Explanation string
        """
        strategy = analysis.get("recommended_strategy", "unknown")
        confidence = analysis.get("confidence", 0.0)
        reasoning = analysis.get("reasoning", "No reasoning provided")

        explanations = {
            "vector": "Vector search for semantic similarity and descriptive queries",
            "graph": "Graph search for structural relationships and connected data",
            "hybrid": "Hybrid approach combining vector and graph search",
            "filter": "Filtered search with specific constraints and conditions"
        }

        strategy_explanation = explanations.get(strategy, f"Unknown strategy: {strategy}")

        return f"Selected {strategy_explanation} (confidence: {confidence:.2f}). {reasoning}"