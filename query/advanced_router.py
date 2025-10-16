"""
Advanced query router with multi-step reasoning, iterative refinement, and streaming responses.
Implements agentic retrieval with hybrid relevance scoring and reasoning chains.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, AsyncGenerator, Optional, Tuple
from dataclasses import dataclass
import groq

from config.settings import settings
from query.graph_query import GraphQueryInterface
from query.vector_query import VectorQueryInterface
from query.contextual_response import response_generator
from knowledge_graph.neo4j_manager import Neo4jManager
from utils.opik_tracer import opik_tracer
from utils.memory_manager import memory_manager
from plugins.query.filter_plugin import FilterPlugin
from query.fusion import HybridScorer

logger = logging.getLogger(__name__)

@dataclass
class ReasoningStep:
    """Represents a single step in the reasoning chain."""
    step_number: int
    action: str
    input_data: Any
    reasoning: str
    result: Any
    confidence: float
    timestamp: float

@dataclass
class QueryResult:
    """Enhanced query result with reasoning chain."""
    query: str
    final_answer: str
    reasoning_chain: List[ReasoningStep]
    method_used: str
    confidence_score: float
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class AdvancedQueryRouter:
    """Advanced router with multi-step reasoning and streaming responses."""

    def __init__(self, neo4j_adapter=None, qdrant_client=None):
        """Initialize advanced query router."""
        self.llm = groq.Groq(api_key=settings.GROQ_API_KEY)

        # Use provided adapters or get from global manager
        if neo4j_adapter is None:
            from database_adapters.database_factory import db_manager
            neo4j_adapter = db_manager.get_adapter("neo4j")
        if qdrant_client is None:
            from database_adapters.database_factory import get_qdrant_client
            qdrant_client = get_qdrant_client()

        self.neo4j_adapter = neo4j_adapter
        self.qdrant_client = qdrant_client
        self.graph_query = GraphQueryInterface(self.neo4j_adapter)
        self.vector_query = VectorQueryInterface()
        self.filter_plugin = FilterPlugin()
        self.hybrid_scorer = HybridScorer()

        # Reasoning state
        self.reasoning_chain: List[ReasoningStep] = []
        self.max_reasoning_steps = 5

    async def analyze_query_and_plan(self, query: str) -> Dict[str, Any]:
        """
        Analyze query to determine the optimal set of retrieval tools and a core query.
        """
        prompt = f""" 
        Analyze the user's query to create an execution plan. Decompose it into:
        1. A `core_query` for semantic vector search (remove filter conditions).
        2. A list of `tools` to use. Available tools are: `search_vector`, `search_graph`, `search_filter`.
        3. A structured `filters` object for the `search_filter` tool.

        User Query: "{query}"

        Example:
        User Query: "show me papers about attention mechanism by Vaswani after 2016"
        Response:
        ```json
        {{
            "core_query": "attention mechanism papers by Vaswani",
            "tools": ["search_filter", "search_vector"],
            "filters": {{
                "must": [
                    {{ "key": "metadata.author", "match": {{ "value": "Vaswani" }} }},
                    {{ "key": "metadata.year", "range": {{ "gte": 2017 }} }}
                ]
            }},
            "reasoning": "The user has specific filters (author, year) and a core semantic query, so a filtered vector search is optimal."
        }}
        ```

        Now, parse the given user query.
        Response:
        """
        try:
            response = self.llm.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=settings.GROQ_MODEL,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            response_text = response.choices[0].message.content
            plan = json.loads(response_text)
            
            # Ensure plan has required keys
            if 'tools' not in plan or 'core_query' not in plan:
                raise ValueError("LLM response missing 'tools' or 'core_query' keys.")

            logger.info(f"Generated retrieval plan: {plan}")
            return plan

        except Exception as e:
            logger.error(f"Failed to generate retrieval plan: {e}")
            # Fallback to a default hybrid plan
            return {
                "core_query": query,
                "tools": ["search_vector", "search_graph"],
                "filters": {},
                "reasoning": "Fallback to default hybrid search due to planning error."
            }

    def _default_complexity_analysis(self) -> Dict[str, Any]:
        """Return default complexity analysis."""
        return {
            "complexity": "moderate",
            "primary_method": "hybrid",
            "secondary_methods": ["graph", "vector"],
            "reasoning_steps": 2,
            "expected_sources": ["graph_entities", "vector_chunks"],
            "confidence": 0.7
        }

    async def execute_reasoning_step(self, step_number: int, action: str,
                                   input_data: Any, context: Dict[str, Any]) -> ReasoningStep:
        """
        Execute a single reasoning step.

        Args:
            step_number: Current step number
            action: Action to perform
            input_data: Input data for the action
            context: Current context

        Returns:
            ReasoningStep with results
        """
        reasoning_prompt = f"""
        Based on the current context and previous reasoning, perform this action:

        Action: {action}
        Input: {str(input_data)[:500]}
        Context: {str(context)[:500]}

        Provide:
        1. Your reasoning for this step
        2. What you'll do
        3. Expected outcome
        4. Confidence in this approach

        Respond with JSON:
        {{
            "reasoning": "your thought process",
            "plan": "what you'll do",
            "expected_outcome": "what you expect to find",
            "confidence": 0.0-1.0
        }}
        """

        try:
            response = self.llm.chat.completions.create(
                messages=[{"role": "user", "content": reasoning_prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                max_tokens=300
            )

            # Safely extract response content
            response_content = response.choices[0].message.content
            if not response_content or not response_content.strip():
                logger.error("Empty response from LLM for reasoning step")
                step_analysis = {
                    "reasoning": "No response from LLM",
                    "plan": "Fallback to default analysis",
                    "expected_outcome": "Use default reasoning",
                    "confidence": 0.5
                }
            else:
                response_content = response_content.strip()

                # Check for error responses
                if "error" in response_content.lower() or "failed" in response_content.lower():
                    logger.warning(f"LLM returned error for reasoning step: {response_content[:100]}")
                    step_analysis = {
                        "reasoning": "LLM service error",
                        "plan": "Use default analysis approach",
                        "expected_outcome": "Continue with basic reasoning",
                        "confidence": 0.4
                    }
                else:
                    try:
                        step_analysis = json.loads(response_content)
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON parsing failed for reasoning response: {e}")
                        logger.debug(f"Response content: {response_content[:200]}")
                        step_analysis = {
                            "reasoning": "Failed to parse LLM response",
                            "plan": "Use default analysis",
                            "expected_outcome": "Continue with default reasoning",
                            "confidence": 0.3
                        }

            # Execute the action
            result = await self._perform_action(action, input_data, context)

            step = ReasoningStep(
                step_number=step_number,
                action=action,
                input_data=input_data,
                reasoning=step_analysis.get("reasoning", "No reasoning provided"),
                result=result,
                confidence=step_analysis.get("confidence", 0.5),
                timestamp=asyncio.get_event_loop().time()
            )

            return step

        except Exception as e:
            logger.error(f"Error in reasoning step {step_number}: {e}")
            # Return a safe fallback step instead of crashing
            return ReasoningStep(
                step_number=step_number,
                action=action,
                input_data=input_data,
                reasoning=f"Error: {str(e)[:100]}",  # Limit error message length
                result={"answer": f"Error in reasoning step: {str(e)[:100]}", "chunks": []},
                confidence=0.0,
                timestamp=asyncio.get_event_loop().time()
            )

    async def _perform_action(self, action: str, input_data: Any, context: Dict[str, Any]) -> Any:
        """Perform the specified action."""
        if action == "analyze_query":
            return await self.analyze_query_complexity(input_data)
        elif action == "search_vector":
            vector_result = self.vector_query.query(input_data)
            return vector_result
        elif action == "search_graph":
            graph_result = self.graph_query.query(input_data)
            return graph_result
        elif action == "search_filter":
            return self.filter_plugin.process_query(input_data, context)
        elif action == "hybrid_search":
            return await self._hybrid_search_with_fusion(input_data, context)
        elif action == "refine_answer":
            return await self._refine_answer(input_data, context)
        else:
            return f"Unknown action: {action}"

    async def _hybrid_search(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform hybrid search combining vector and graph results."""
        # Execute both searches in parallel
        vector_task = asyncio.create_task(asyncio.to_thread(self.vector_query.query, query))
        graph_task = asyncio.create_task(asyncio.to_thread(self.graph_query.query, query))

        vector_result, graph_result = await asyncio.gather(vector_task, graph_task, return_exceptions=True)

        # Handle exceptions
        if isinstance(vector_result, Exception):
            vector_result = {"error": str(vector_result), "answer": ""}
        if isinstance(graph_result, Exception):
            graph_result = {"error": str(graph_result), "answer": ""}

        # Calculate hybrid relevance scores
        hybrid_result = self._calculate_hybrid_relevance(query, vector_result, graph_result, context)

        return hybrid_result

    async def _hybrid_search_with_fusion(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform hybrid search, fuse results with RRF, and synthesize a final answer."""
        # 1. Execute searches in parallel
        vector_task = asyncio.to_thread(self.vector_query.query, query, context)
        graph_task = asyncio.to_thread(self.graph_query.query, query)
        vector_result, graph_result = await asyncio.gather(vector_task, graph_task, return_exceptions=True)

        # 2. Collect results, handling potential errors
        vector_chunks = []
        if isinstance(vector_result, dict) and vector_result.get('chunks'):
            vector_chunks = vector_result['chunks']
        elif isinstance(vector_result, Exception):
            logger.error(f"Vector search failed during hybrid search: {vector_result}")

        graph_docs = []
        if isinstance(graph_result, dict) and graph_result.get('answer'):
            # Treat the graph answer as a single, high-value document
            graph_docs.append({'text': graph_result['answer'], 'source': 'graph', 'score': 0.9})
        elif isinstance(graph_result, Exception):
            logger.error(f"Graph search failed during hybrid search: {graph_result}")

        # 3. Fuse results using RRF
        fused_results = self.hybrid_scorer.reciprocal_rank_fusion([vector_chunks, graph_docs])
        top_sources = fused_results[:5] # Use top 5 fused results for synthesis

        if not top_sources:
            return {'answer': "No relevant results found after hybrid search.", 'sources': []}

        # 4. Synthesize the final answer
        final_answer = await self._synthesize_answers(query, top_sources)

        return {
            'answer': final_answer,
            'sources': top_sources,
            'method': 'hybrid_rrf_synthesis'
        }

    async def _synthesize_answers(self, query: str, fused_sources: List[Dict[str, Any]]) -> str:
        """Synthesize a final answer from a fused list of source documents."""
        context_text = "\n\n---\n\n".join([source.get('text', '') for source in fused_sources])
        synthesis_prompt = f"""Synthesize a comprehensive answer by combining information from the following sources, which are ranked by relevance. 

        User Query: {query}

        Combined Context:
        {context_text}

        Provide a direct and comprehensive answer based on the context.
        Answer:"""

        try:
            response = self.llm.chat.completions.create(
                messages=[{"role": "user", "content": synthesis_prompt}],
                model=settings.GROQ_MODEL,
                temperature=0.2,
                max_tokens=1000
            )
            synthesized_answer = response.choices[0].message.content.strip()

            opik_tracer.track_llm_call(
                provider="groq",
                model=settings.GROQ_MODEL,
                prompt=synthesis_prompt,
                response=synthesized_answer,
                metadata={"operation": "answer_synthesis"}
            )
            return synthesized_answer
        except Exception as e:
            logger.error(f"Error synthesizing answers: {e}")
            return "Failed to synthesize a final answer from the combined search results."

    async def route_query_advanced(self, user_query: str) -> AsyncGenerator[QueryResult, None]:
        """
        Orchestrates the agentic retrieval process: Plan -> Execute -> Fuse -> Synthesize.
        """
        self.reasoning_chain = []
        start_time = asyncio.get_event_loop().time()

        try:
            # 1. Plan: Analyze the query to create a retrieval plan
            yield await self._create_intermediate_result("Analyzing query and creating retrieval plan...", self.reasoning_chain)
            plan = await self.analyze_query_and_plan(user_query)
            self.reasoning_chain.append(ReasoningStep(1, "analyze_and_plan", user_query, plan.get('reasoning', 'N/A'), plan, 0.9, time.time()))

            # 2. Execute: Run all planned retrieval tools in parallel
            yield await self._create_intermediate_result(f"Executing tools: {plan['tools']}...", self.reasoning_chain)
            
            tasks = []
            for tool in plan.get("tools", []):
                if tool == "search_vector":
                    tasks.append(asyncio.to_thread(self.vector_query.query, plan['core_query']))
                elif tool == "search_graph":
                    tasks.append(asyncio.to_thread(self.graph_query.query, plan['core_query']))
                elif tool == "search_filter":
                    tasks.append(asyncio.to_thread(self.filter_plugin.process_query, user_query, {"filters": plan.get('filters', {})}))

            tool_results = await asyncio.gather(*tasks, return_exceptions=True)

            # 3. Fuse: Collect and fuse results using RRF
            yield await self._create_intermediate_result("Fusing results from all tools...", self.reasoning_chain)
            results_to_fuse = []
            for i, result in enumerate(tool_results):
                if isinstance(result, Exception):
                    logger.error(f"Tool execution failed: {plan['tools'][i]} with error: {result}")
                    continue
                if isinstance(result, dict) and result.get('chunks'):
                    results_to_fuse.append(result['chunks'])
                elif isinstance(result, dict) and result.get('answer'): # Handle graph results
                    results_to_fuse.append([{'text': result['answer'], 'source': 'graph'}])

            fused_sources = self.hybrid_scorer.reciprocal_rank_fusion(results_to_fuse)
            top_sources = fused_sources[:7] # Use top 7 for synthesis
            self.reasoning_chain.append(ReasoningStep(2, "fuse_results", {}, f"Fused {len(results_to_fuse)} result sets into {len(fused_sources)} documents.", top_sources, 0.9, time.time()))

            # 4. Synthesize: Generate the final answer
            yield await self._create_intermediate_result("Synthesizing final answer...", self.reasoning_chain)
            final_answer = await self._synthesize_answers(plan['core_query'], top_sources)
            self.reasoning_chain.append(ReasoningStep(3, "synthesize_answer", {}, "Generated final answer from fused context.", final_answer, 0.9, time.time()))

            # 5. Yield final result
            final_result = QueryResult(
                query=user_query,
                final_answer=final_answer,
                reasoning_chain=self.reasoning_chain,
                method_used=f"hybrid_fusion({', '.join(plan['tools'])})",
                confidence_score=self._calculate_overall_confidence(),
                sources=top_sources,
                metadata={
                    'plan': plan,
                    'total_steps': len(self.reasoning_chain),
                    'processing_time': time.time() - start_time
                }
            )
            yield final_result

        except Exception as e:
            logger.error(f"Error in advanced routing: {e}", exc_info=True)
            yield QueryResult(
                query=user_query, final_answer=f"An error occurred: {e}", reasoning_chain=self.reasoning_chain,
                method_used="error", confidence_score=0.0, sources=[], metadata={'error': str(e)}
            )

    async def _create_intermediate_result(self, status: str,
                                        reasoning_chain: List[ReasoningStep]) -> QueryResult:
        """Create intermediate result for streaming."""
        return QueryResult(
            query="",
            final_answer="",
            reasoning_chain=reasoning_chain,
            method_used="processing",
            confidence_score=0.0,
            sources=[],
            metadata={'status': status, 'intermediate': True}
        )

    def _extract_final_answer(self) -> str:
        """Extract final answer from reasoning chain."""
        if not self.reasoning_chain:
            return "No reasoning steps completed"

        # Get the last meaningful result
        for step in reversed(self.reasoning_chain):
            if step.result:
                if isinstance(step.result, dict):
                    return step.result.get('answer', str(step.result))
                elif isinstance(step.result, str):
                    return step.result

        return "Could not extract final answer"

    def _calculate_overall_confidence(self) -> float:
        """Calculate overall confidence from reasoning chain."""
        if not self.reasoning_chain:
            return 0.0

        confidences = [step.confidence for step in self.reasoning_chain if step.confidence > 0]
        return sum(confidences) / len(confidences) if confidences else 0.5

    def _collect_sources(self) -> List[Dict[str, Any]]:
        """Collect all sources from reasoning chain."""
        sources = []
        for step in self.reasoning_chain:
            if isinstance(step.result, dict):
                if 'chunks' in step.result:
                    sources.extend(step.result['chunks'])
                if 'sources' in step.result:
                    sources.extend(step.result['sources'])
        return sources