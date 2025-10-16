"""
Test suite for the Lyzr Challenge RAG system.
Provides comprehensive testing for all components.
"""

import pytest
import unittest
from typing import Dict, Any
import tempfile
import os
from pathlib import Path

# Test configuration
class TestConfig:
    """Test configuration and utilities."""

    @staticmethod
    def get_test_data_path() -> Path:
        """Get path to test data directory."""
        return Path(__file__).parent / "data"

    @staticmethod
    def create_temp_dir() -> str:
        """Create a temporary directory for tests."""
        return tempfile.mkdtemp()

    @staticmethod
    def mock_settings() -> Dict[str, Any]:
        """Get mock settings for testing."""
        return {
            "LLAMA_CLOUD_API_KEY": "test_key",
            "COHERE_API_KEY": "test_key",
            "GROQ_API_KEY": "test_key",
            "NOMIC_API_KEY": "test_key",
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
            "CONTEXT_WINDOW_SIZE": 512,
            "MAX_RETRIES": 1,
            "RATE_LIMIT_DELAY": 0,
            "LOG_LEVEL": "DEBUG"
        }

# Test fixtures
@pytest.fixture
def temp_dir():
    """Temporary directory fixture."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    # Cleanup would go here

@pytest.fixture
def mock_ontology():
    """Mock ontology data for testing."""
    return {
        "entities": [
            {"name": "Transformer", "type": "Model", "definition": "Neural network architecture"},
            {"name": "Attention", "type": "Mechanism", "definition": "Focus mechanism"}
        ],
        "relationships": [
            {"source": "Transformer", "target": "Attention", "label": "USES"}
        ]
    }

@pytest.fixture
def mock_query_result():
    """Mock query result for testing."""
    return {
        "query": "What is a transformer?",
        "answer": "A neural network architecture...",
        "confidence": 0.9,
        "sources": [],
        "reasoning_steps": [],
        "metadata": {}
    }