"""
Contextual retrieval module using Cohere for generating contextual summaries.
Implements sliding window context and async processing with retry logic.
"""

import asyncio
import logging
import time
import nest_asyncio
from typing import List, Dict, Any
import cohere

from config.settings import settings

logger = logging.getLogger(__name__)

class ContextualRetriever:
    """Handles contextual retrieval and summarization using Cohere."""

    def __init__(self):
        """Initialize Cohere client and configuration."""
        self.co = cohere.ClientV2(api_key=settings.COHERE_API_KEY)
        self.context_window_size = settings.CONTEXT_WINDOW_SIZE
        self.max_retries = settings.MAX_RETRIES
        self.rate_limit_delay = settings.RATE_LIMIT_DELAY

    def get_sliding_window_context(self, chunk_text: str, full_text: str) -> str:
        """
        Create a sliding window context around a chunk.

        Args:
            chunk_text: The specific chunk to contextualize
            full_text: The full document text

        Returns:
            Contextual window text
        """
        chunk_start = full_text.find(chunk_text)
        if chunk_start == -1:
            return full_text

        chars_per_token = 4  # rough estimate
        char_window = self.context_window_size * chars_per_token // 2

        start = max(0, chunk_start - char_window)
        end = min(len(full_text), chunk_start + len(chunk_text) + char_window)
        return full_text[start:end]

    async def generate_contextual_summary(self, context_window: str, chunk_text: str) -> str:
        """
        Generate a contextual summary for a chunk using Cohere.

        Args:
            context_window: The contextual window text
            chunk_text: The specific chunk to summarize

        Returns:
            Contextual summary
        """
        prompt = f"""
        <document_context>
        {context_window}
        </document_context>

        Here is a specific chunk from that context:
        <chunk>
        {chunk_text}
        </chunk>

        Please write TWO single sentences that provide a succinct, high-level summary of the chunk within its context.
        Output only the two summary sentences, nothing else.
        """

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.co.chat(
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
                return response.message.content[0].text

            except Exception as e:
                err_msg = str(e)
                logger.warning(f"Attempt {attempt} - Cohere API error: {err_msg}")

                if "429" in err_msg or "rate limit" in err_msg.lower():
                    logger.info(f"Rate limit hit, waiting {self.rate_limit_delay} seconds...")
                    await asyncio.sleep(self.rate_limit_delay)
                else:
                    await asyncio.sleep(8)

        logger.error("Max retries reached, returning error placeholder")
        return "❌ Error: Could not generate summary"

    async def contextualize_chunks(self, nodes: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Add contextual summaries to all chunks.

        Args:
            nodes: List of node dictionaries with text content

        Returns:
            Nodes with added contextual summaries
        """
        full_text = ''.join([node['text'] for node in nodes])

        contextualized_chunks = []

        for i, node in enumerate(nodes):
            logger.info(f"Processing chunk {i+1}/{len(nodes)}")

            # Get sliding window context
            dynamic_context = self.get_sliding_window_context(node["text"], full_text)

            # Generate contextual summary
            summary = await self.generate_contextual_summary(dynamic_context, node["text"])

            logger.info(f"Completed chunk {i+1}: {summary[:50]}...")

            # Create contextualized chunk
            contextualized_chunk = {
                "id": node["id_"],
                "original_text": node["text"],
                "context_summary": summary,
                "contextualized_text": f"{summary}\n---\n{node['text']}"
            }

            contextualized_chunks.append(contextualized_chunk)

            # Rate limiting delay (Cohere free tier: 10 calls/min = 6 seconds between calls)
            # Adding extra buffer for safety to prevent rate limit errors
            await asyncio.sleep(7)

        logger.info(f"Successfully contextualized {len(contextualized_chunks)} chunks")
        return contextualized_chunks

    @staticmethod
    async def process_chunks_async(nodes: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Main entry point for async contextualization processing.

        Args:
            nodes: List of node dictionaries

        Returns:
            Contextualized chunks
        """
        processor = ContextualRetriever()
        return await processor.contextualize_chunks(nodes)