"""
LLM orchestration for managing different language models and their interactions.
"""

import logging
from typing import Dict, Any, Optional, List
import groq
import cohere

from config.settings import settings
from utils.opik_tracer import opik_tracer

logger = logging.getLogger(__name__)


class LLMOrchestrator:
    """Orchestrates different LLM providers and models."""

    def __init__(self):
        """Initialize LLM orchestrator with all providers."""
        self.groq_client = groq.Groq(api_key=settings.GROQ_API_KEY)
        self.cohere_client = cohere.ClientV2(api_key=settings.COHERE_API_KEY)

        # Available models configuration
        self.models = {
            "groq": {
                "fast": "openai/gpt-oss-120b",
                "reasoning": "llama-3.3-70b-versatile",
                "coding": "llama-3.3-70b-instant"
            },
            "cohere": {
                "command": "command-a-03-2025",
                "rerank": "rerank-english-v3.0"
            }
        }

        logger.info("LLM Orchestrator initialized with Groq and Cohere")

    def create_completion(self, prompt: str, provider: str = "groq",
                         model_type: str = "fast", **kwargs) -> str:
        """
        Create a completion using specified provider and model.

        Args:
            prompt: Input prompt
            provider: 'groq' or 'cohere'
            model_type: Model type key from models config
            **kwargs: Additional parameters

        Returns:
            Generated response text
        """
        try:
            if provider == "groq":
                return self._groq_completion(prompt, model_type, **kwargs)
            elif provider == "cohere":
                return self._cohere_completion(prompt, model_type, **kwargs)
            else:
                raise ValueError(f"Unsupported provider: {provider}")

        except Exception as e:
            logger.error(f"LLM completion failed: {e}")
            return ""

    def _groq_completion(self, prompt: str, model_type: str, **kwargs) -> str:
        """Handle Groq completions."""
        model = self.models["groq"].get(model_type, self.models["groq"]["fast"])

        temperature = kwargs.get('temperature', 0.1)
        max_tokens = kwargs.get('max_tokens', 1000)

        try:
            response = self.groq_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )

            result = response.choices[0].message.content

            # Track with Opik
            opik_tracer.track_llm_call(
                provider="groq",
                model=model,
                prompt=prompt,
                response=result,
                metadata={"temperature": temperature, "max_tokens": max_tokens}
            )

            return result

        except Exception as e:
            logger.error(f"Groq completion error: {e}")
            raise

    def _cohere_completion(self, prompt: str, model_type: str, **kwargs) -> str:
        """Handle Cohere completions."""
        model = self.models["cohere"].get(model_type, self.models["cohere"]["command"])

        temperature = kwargs.get('temperature', 0.3)

        try:
            response = self.cohere_client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )

            result = response.message.content[0].text

            # Track with Opik
            opik_tracer.track_llm_call(
                provider="cohere",
                model=model,
                prompt=prompt,
                response=result,
                metadata={"temperature": temperature}
            )

            return result

        except Exception as e:
            logger.error(f"Cohere completion error: {e}")
            raise

    def rerank_documents(self, query: str, documents: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Rerank documents using Cohere's reranking.

        Args:
            query: Search query
            documents: List of documents to rerank
            top_k: Number of top results

        Returns:
            Reranked documents with scores
        """
        try:
            response = self.cohere_client.rerank(
                query=query,
                documents=documents,
                top_n=top_k,
                model=self.models["cohere"]["rerank"]
            )

            results = []
            for result in response.results:
                results.append({
                    "document": documents[result.index],
                    "relevance_score": result.relevance_score,
                    "rank": result.rank,
                    "index": result.index
                })

            # Track reranking operation
            opik_tracer.track_llm_call(
                provider="cohere",
                model=self.models["cohere"]["rerank"],
                prompt=f"Reranking query: {query[:50]}...",
                response=f"Reranked {len(documents)} documents",
                metadata={"total_docs": len(documents), "top_k": top_k}
            )

            return results

        except Exception as e:
            logger.error(f"Cohere reranking error: {e}")
            # Return original order with dummy scores
            return [
                {
                    "document": doc,
                    "relevance_score": 1.0 - (i * 0.1),
                    "rank": i + 1,
                    "index": i
                }
                for i, doc in enumerate(documents[:top_k])
            ]

    def get_available_models(self) -> Dict[str, Dict[str, str]]:
        """
        Get available models configuration.

        Returns:
            Models configuration dictionary
        """
        return self.models.copy()