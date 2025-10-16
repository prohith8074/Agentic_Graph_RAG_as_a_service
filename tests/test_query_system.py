"""
Tests for query system components.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio
from query.advanced_router import AdvancedQueryRouter
from query.contextual_response import ContextualResponseGenerator

class TestAdvancedQueryRouter:
    """Test advanced query router functionality."""

    @pytest.fixture
    def router(self):
        """Create test router instance."""
        with patch('query.advanced_router.Groq') as mock_groq:
            mock_llm = Mock()
            mock_groq.return_value = mock_llm

            router = AdvancedQueryRouter()
            router.llm = mock_llm
            yield router

    def test_complexity_analysis(self, router):
        """Test query complexity analysis."""
        router.llm.complete.return_value = Mock()
        router.llm.complete.return_value.text = '''{"complexity": "moderate", "primary_method": "hybrid", "secondary_methods": ["graph", "vector"], "reasoning_steps": 2, "expected_sources": ["graph_entities", "vector_chunks"], "confidence": 0.7}'''

        result = asyncio.run(router.analyze_query_complexity("What is a transformer?"))

        assert result["complexity"] == "moderate"
        assert result["primary_method"] == "hybrid"
        assert result["confidence"] == 0.7

    def test_reasoning_step_execution(self, router):
        """Test reasoning step execution."""
        router.llm.complete.return_value = Mock()
        router.llm.complete.return_value.text = '''{"reasoning": "Analyzing hybrid search approach", "plan": "Execute both vector and graph searches", "expected_outcome": "Combined relevant results", "confidence": 0.8}'''

        step = asyncio.run(router.execute_reasoning_step(
            1, "hybrid_search", "test query", {"complexity": "moderate"}
        ))

        assert step.step_number == 1
        assert step.action == "hybrid_search"
        assert step.confidence == 0.8

    def test_hybrid_relevance_scoring(self, router):
        """Test hybrid relevance calculation."""
        vector_result = {"answer": "Vector answer", "chunks": [{"score": 0.9}]}
        graph_result = {"answer": "Graph answer", "sources": []}

        score = router._calculate_hybrid_relevance("test query", vector_result, graph_result, {"complexity": "moderate"})

        assert "hybrid_score" in score
        assert "vector_score" in score
        assert "graph_score" in score
        assert isinstance(score["hybrid_score"], float)

    def test_overall_confidence_calculation(self, router):
        """Test overall confidence calculation."""
        # Set up reasoning chain
        router.reasoning_chain = [
            Mock(confidence=0.8),
            Mock(confidence=0.6),
            Mock(confidence=0.9)
        ]

        confidence = router._calculate_overall_confidence()
        expected = (0.8 + 0.6 + 0.9) / 3

        assert abs(confidence - expected) < 0.001

    def test_empty_reasoning_chain_confidence(self, router):
        """Test confidence calculation with empty chain."""
        router.reasoning_chain = []
        confidence = router._calculate_overall_confidence()
        assert confidence == 0.5  # Default value

class TestContextualResponseGenerator:
    """Test contextual response generator."""

    @pytest.fixture
    def generator(self):
        """Create test generator instance."""
        with patch('query.contextual_response.cohere.ClientV2') as mock_cohere, \
             patch('query.contextual_response.Groq') as mock_groq:

            mock_cohere_client = Mock()
            mock_groq_client = Mock()

            mock_cohere.return_value = mock_cohere_client
            mock_groq.return_value = mock_groq_client

            generator = ContextualResponseGenerator()
            generator.cohere_client = mock_cohere_client
            generator.groq_client = mock_groq_client

            yield generator

    def test_cohere_response_generation(self, generator):
        """Test Cohere response generation."""
        # Mock Cohere response
        mock_response = Mock()
        mock_response.message.content = [Mock()]
        mock_response.message.content[0].text = "Test Cohere response"
        generator.cohere_client.chat.return_value = mock_response

        result = generator.generate_response("test query", {"retrieved_chunks": []})

        assert result["success"] is True
        assert result["answer"] == "Test Cohere response"
        assert result["method"] == "cohere"

    def test_groq_fallback_response(self, generator):
        """Test Groq fallback when Cohere fails."""
        # Make Cohere fail
        generator.cohere_client.chat.side_effect = Exception("Cohere failed")

        # Mock Groq response
        mock_groq_response = Mock()
        mock_groq_response.text = "Test Groq fallback response"
        generator.groq_client.complete.return_value = mock_groq_response

        result = generator.generate_response("test query", {"retrieved_chunks": []})

        assert result["success"] is True
        assert "Groq fallback" in result["answer"]
        assert result["method"] == "groq_fallback"

    def test_memory_integration(self, generator):
        """Test memory context integration."""
        from utils.memory_manager import memory_manager

        # Mock memory retrieval
        with patch.object(memory_manager, 'build_enhanced_context') as mock_memory:
            mock_memory.return_value = {
                'current_query': 'test query',
                'retrieved_chunks': [],
                'conversation_context': {'previous_topic': 'transformers'},
                'recent_history': [{'user_query': 'What is AI?', 'assistant_response': 'AI is...'}],
                'memory_used': True
            }

            # Mock Cohere response
            mock_response = Mock()
            mock_response.message.content = [Mock()]
            mock_response.message.content[0].text = "Contextual response with memory"
            generator.cohere_client.chat.return_value = mock_response

            result = generator.generate_response("test query", {"retrieved_chunks": []}, "session123")

            assert result["success"] is True
            mock_memory.assert_called_once_with("session123", "test query", [])

    def test_prompt_building(self, generator):
        """Test prompt building with different contexts."""
        context = {
            'retrieved_chunks': [
                {'text': 'Chunk 1 content', 'score': 0.9},
                {'text': 'Chunk 2 content', 'score': 0.8}
            ],
            'recent_history': [
                {'user_query': 'Previous question', 'assistant_response': 'Previous answer'}
            ]
        }

        prompt = generator._build_response_prompt("Test query", context)

        assert "Test query" in prompt
        assert "Chunk 1 content" in prompt
        assert "Chunk 2 content" in prompt
        assert "Previous question" in prompt
        assert "Previous answer" in prompt

    def test_answer_synthesis(self, generator):
        """Test answer synthesis from multiple options."""
        responses = [
            {"answer": "Answer 1"},
            {"answer": "Answer 2"},
            {"answer": "Answer 3"}
        ]

        # Mock synthesis response
        mock_response = Mock()
        mock_response.message.content = [Mock()]
        mock_response.message.content[0].text = "Synthesized answer"
        generator.cohere_client.chat.return_value = mock_response

        result = generator.generate_answer_synthesis("test query", responses, {})

        assert "Synthesized answer" in result
        generator.cohere_client.chat.assert_called_once()

class TestQueryResultProcessing:
    """Test query result processing and formatting."""

    def test_result_formatting(self):
        """Test query result formatting."""
        from utils.helpers import format_query_result

        result = {
            'query': 'Test query',
            'answer': 'Test answer',
            'routing': {
                'method': 'graph',
                'confidence': 0.85,
                'reasoning': 'Graph-based query'
            },
            'chunks_retrieved': 3
        }

        formatted = format_query_result(result)

        assert 'Test query' in formatted
        assert 'Test answer' in formatted
        assert 'graph' in formatted
        assert '0.85' in formatted
        assert '3' in formatted