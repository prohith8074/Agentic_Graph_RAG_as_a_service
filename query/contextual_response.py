"""
Contextual response generator with Cohere primary LLM and Groq fallback.
Integrates short-term memory context for enhanced responses.
"""

import logging
import time
from typing import Dict, Any, List, Optional
import cohere

from config.settings import settings
from utils.opik_tracer import opik_tracer
from utils.memory_manager import memory_manager

logger = logging.getLogger(__name__)

class ContextualResponseGenerator:
    """Generates contextual responses using Cohere with Groq fallback."""

    def __init__(self):
        """Initialize response generator."""
        self.cohere_client = cohere.ClientV2(api_key=settings.COHERE_API_KEY)
        self.groq_available = bool(settings.GROQ_API_KEY)

        # Import here to avoid circular imports
        try:
            import groq
            self.groq_client = groq.Groq(
                api_key=settings.GROQ_API_KEY
            )
        except ImportError:
            self.groq_client = None
            logger.warning("Groq client not available for fallback")

    def generate_response(self, query: str, context: Dict[str, Any],
                         session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate contextual response using available context and memory.

        Args:
            query: User query
            context: Retrieved context chunks and metadata
            session_id: Optional session identifier for memory

        Returns:
            Response dictionary with answer and metadata
        """
        try:
            # Build enhanced context with memory
            if session_id:
                enhanced_context = memory_manager.build_enhanced_context(
                    session_id, query, context.get('retrieved_chunks', [])
                )
            else:
                enhanced_context = {
                    'current_query': query,
                    'retrieved_chunks': context.get('retrieved_chunks', []),
                    'conversation_context': {},
                    'recent_history': []
                }

            # Generate response using Cohere (primary)
            response_data = self._generate_with_cohere(query, enhanced_context)

            if not response_data.get('success', False):
                # Fallback to Groq if Cohere fails
                logger.warning("Cohere generation failed, attempting Groq fallback")
                response_data = self._generate_with_groq(query, enhanced_context)

            # Store response in conversation history if session_id provided
            if session_id and response_data.get('success'):
                memory_manager.add_to_conversation_history(
                    session_id,
                    {
                        'query': query,
                        'answer': response_data.get('answer', ''),
                        'method': response_data.get('method', 'unknown'),
                        'confidence': response_data.get('confidence', 0.0)
                    }
                )

            return response_data

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                'success': False,
                'answer': f"Error generating response: {e}",
                'method': 'error',
                'confidence': 0.0,
                'error': str(e)
            }

    def _generate_with_cohere(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate response using Cohere LLM.

        Args:
            query: User query
            context: Enhanced context

        Returns:
            Response data dictionary
        """
        try:
            # Build comprehensive prompt
            prompt = self._build_response_prompt(query, context)

            # Generate response
            response = self.cohere_client.chat(
                messages=[{
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": prompt
                    }]
                }],
                temperature=0.3,
                model=settings.COHERE_MODEL,
            )

            answer = response.message.content[0].text.strip()

            # Rate limiting for Cohere API (10 calls/min on free tier)
            time.sleep(7)

            # Track with Opik
            opik_tracer.track_llm_call(
                provider="cohere",
                model=settings.COHERE_MODEL,
                prompt=prompt,
                response=answer,
                metadata={
                    "operation": "contextual_response",
                    "context_chunks": len(context.get('retrieved_chunks', [])),
                    "has_memory": bool(context.get('recent_history'))
                }
            )

            return {
                'success': True,
                'answer': answer,
                'method': 'cohere',
                'confidence': 0.9,  # Cohere generally provides high-quality responses
                'context_used': len(context.get('retrieved_chunks', [])),
                'memory_used': bool(context.get('recent_history'))
            }

        except Exception as e:
            logger.error(f"Cohere response generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'method': 'cohere_failed'
            }

    def _generate_with_groq(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate response using Groq LLM as fallback.

        Args:
            query: User query
            context: Enhanced context

        Returns:
            Response data dictionary
        """
        if not self.groq_client:
            return {
                'success': False,
                'answer': "Fallback LLM not available",
                'method': 'fallback_unavailable'
            }

        try:
            # Build simplified prompt for Groq
            prompt = self._build_response_prompt(query, context, simplified=True)

            # Generate response
            response = self.groq_client.complete(prompt)
            answer = response.text.strip()

            # Track with Opik
            opik_tracer.track_llm_call(
                provider="groq",
                model=settings.GROQ_MODEL,
                prompt=prompt,
                response=answer,
                metadata={
                    "operation": "contextual_response_fallback",
                    "context_chunks": len(context.get('retrieved_chunks', [])),
                    "has_memory": bool(context.get('recent_history'))
                }
            )

            return {
                'success': True,
                'answer': answer,
                'method': 'groq_fallback',
                'confidence': 0.7,  # Slightly lower confidence for fallback
                'context_used': len(context.get('retrieved_chunks', [])),
                'memory_used': bool(context.get('recent_history'))
            }

        except Exception as e:
            logger.error(f"Groq fallback response generation failed: {e}")
            return {
                'success': False,
                'answer': f"Both primary and fallback LLMs failed. Error: {e}",
                'method': 'all_failed',
                'error': str(e)
            }

    def _build_response_prompt(self, query: str, context: Dict[str, Any],
                              simplified: bool = False) -> str:
        """
        Build comprehensive response prompt.

        Args:
            query: User query
            context: Enhanced context
            simplified: Whether to use simplified prompt for fallback

        Returns:
            Formatted prompt string
        """
        retrieved_chunks = context.get('retrieved_chunks', [])
        recent_history = context.get('recent_history', [])
        conversation_context = context.get('conversation_context', {})

        # Format retrieved context
        context_text = ""
        if retrieved_chunks:
            context_parts = []
            for i, chunk in enumerate(retrieved_chunks[:5]):  # Limit to top 5 chunks
                content = chunk.get('text', chunk.get('content', ''))
                score = chunk.get('score', chunk.get('relevance_score', 0))
                context_parts.append(f"[Context {i+1}, relevance: {score:.3f}]\n{content[:500]}...")

            context_text = "\n\n".join(context_parts)

        # Format conversation history
        history_text = ""
        if recent_history:
            history_parts = []
            for item in recent_history:
                history_parts.append(f"User: {item.get('user_query', '')}")
                history_parts.append(f"Assistant: {item.get('assistant_response', '')}")
            history_text = "\n".join(history_parts)

        # Build prompt
        if simplified:
            # Simplified prompt for Groq fallback
            prompt = f"""
Based on the following context, answer the user's question.

Context:
{context_text}

Recent Conversation:
{history_text}

Question: {query}

Answer:""".strip()
        else:
            # Full prompt for Cohere
            prompt = f"""
You are a knowledgeable assistant helping users understand complex documents and relationships.
Use the provided context and conversation history to give a comprehensive, accurate answer.

CONTEXT INFORMATION:
{context_text}

CONVERSATION HISTORY:
{history_text}

CURRENT QUESTION: {query}

INSTRUCTIONS:
1. Use ONLY the information from the provided context to answer
2. Consider the conversation history for context and follow-up questions
3. Be specific and reference relevant parts of the context when possible
4. If the context doesn't contain enough information, say so clearly
5. Maintain consistency with previous responses in the conversation
6. Provide explanations that are clear and easy to understand

ANSWER:""".strip()

        return prompt

    def generate_answer_synthesis(self, query: str, multiple_responses: List[Dict[str, Any]],
                                context: Dict[str, Any]) -> str:
        """
        Synthesize multiple response options into a final answer.

        Args:
            query: Original query
            multiple_responses: List of response options
            context: Context information

        Returns:
            Synthesized final answer
        """
        try:
            # Prepare synthesis prompt
            responses_text = "\n".join([
                f"Option {i+1}: {resp.get('answer', '')}"
                for i, resp in enumerate(multiple_responses)
            ])

            synthesis_prompt = f"""
Synthesize the best answer from these response options:

Query: {query}

Response Options:
{responses_text}

Context: {str(context)[:500]}

Provide the most accurate and comprehensive answer, combining the best elements from all options.
"""

            # Use Cohere for synthesis
            response = self.cohere_client.chat(
                messages=[{
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": synthesis_prompt
                    }]
                }],
                temperature=0.2,  # Lower temperature for synthesis
                model=settings.COHERE_MODEL,
            )

            final_answer = response.message.content[0].text.strip()

            # Track synthesis
            opik_tracer.track_llm_call(
                provider="cohere",
                model=settings.COHERE_MODEL,
                prompt=synthesis_prompt,
                response=final_answer,
                metadata={"operation": "answer_synthesis"}
            )

            # Rate limiting for Cohere API (10 calls/min on free tier)
            time.sleep(7)

            return final_answer

        except Exception as e:
            logger.error(f"Error in answer synthesis: {e}")
            # Return the first response as fallback
            return multiple_responses[0].get('answer', '') if multiple_responses else ""

# Global response generator instance
response_generator = ContextualResponseGenerator()