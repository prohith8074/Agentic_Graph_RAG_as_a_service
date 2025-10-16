"""
Entity extraction module using Cohere for knowledge graph construction.
Extracts entities and relationships from contextualized text chunks.
"""

import asyncio
import json
import logging
import time
from typing import List, Dict, Any, Optional
import cohere

from config.settings import settings

logger = logging.getLogger(__name__)

class EntityExtractor:
    """Handles entity and relationship extraction using Cohere."""

    def __init__(self):
        """Initialize Cohere client."""
        self.co = cohere.ClientV2(api_key=settings.COHERE_API_KEY)
        self.rate_limit_delay = settings.RATE_LIMIT_DELAY

    def clean_and_parse_json(self, llm_response_text: str, chunk_id: Any) -> Optional[Dict[str, Any]]:
        """
        Clean and parse JSON response from LLM.

        Args:
            llm_response_text: Raw LLM response text
            chunk_id: Identifier for the chunk being processed

        Returns:
            Parsed JSON dictionary or None if parsing fails
        """
        try:
            # Trim and locate JSON braces
            start = llm_response_text.find('{')
            end = llm_response_text.rfind('}') + 1
            if start == -1 or end == -1:
                logger.warning(f"No JSON object found in response for chunk #{chunk_id}")
                return None

            json_string = llm_response_text[start:end]
            return json.loads(json_string)

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for chunk #{chunk_id}: {e}")
            logger.debug(f"Raw text snippet: {llm_response_text[:200]}...")
            return None

    async def extract_entities_and_relationships(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract entities and relationships from text using Cohere.

        Args:
            text: Text chunk to process

        Returns:
            Dictionary containing entities and relationships
        """
        prompt = f"""
        You are a data extraction specialist. Your task is to extract structured information from the provided text. Do not add any information other than the given text.
        First, study the text and identify the valid entity types, their properties, and the relationships between them.

        Then, from the "Text to Extract From", extract all entities and their relationships.

        Rules for JSON output:
        - The JSON object must have two keys: "entities" and "relationships".
        - "entities" is a list of objects. Each object represents a single entity.
          - It MUST have a "name" (the unique name of the entity).
          - It MUST have a "type" that is one of the valid types identified from the text.
          - It MUST contain relevant properties based on the text (e.g., "definition", "purpose"). If the information for a property is not in the text, omit the key.
        - "relationships" is a list of objects.
          - Each object must have a "source" (entity name), "target" (entity name), and "label" (the relationship name).
          - Entity names used in "source" and "target" MUST EXACTLY MATCH a name from the "entities" list.
        - Hierarchies: Define is-a relationships and taxonomies.
        - Constraints: Define rules for entity/relationship validity.
        Note: Understand the given content and extract the entities and relationships (also include the edge case entities and relationships)
        --- Text to Extract From ---
        {text}
        --- End Text ---

        Respond ONLY with a valid JSON object.
        JSON Output:
        """

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

            raw_text = response.message.content[0].text
            parsed_data = self.clean_and_parse_json(raw_text, text[:50])

            return parsed_data

        except Exception as e:
            logger.error(f"Error extracting entities from text: {e}")
            return None

    async def process_chunks_batch(self, chunks: List[str], start_idx: int = 0) -> List[Dict[str, Any]]:
        """
        Process a batch of text chunks for entity extraction.

        Args:
            chunks: List of text chunks to process
            start_idx: Starting index for processing (for resuming)

        Returns:
            List of extracted knowledge graphs
        """
        extracted_data = []

        for i, chunk in enumerate(chunks):
            actual_idx = start_idx + i
            logger.info(f"Processing chunk {actual_idx + 1}/{len(chunks) + start_idx}")

            if actual_idx >= start_idx:
                result = await self.extract_entities_and_relationships(chunk)

                if result:
                    extracted_data.append(result)
                    logger.info(f"Successfully extracted data from chunk {actual_idx + 1}")
                else:
                    logger.warning(f"Failed to extract data from chunk {actual_idx + 1}")

                # Rate limiting (Cohere free tier: 10 calls/min = 6 seconds between calls)
                # Adding extra buffer for safety to prevent rate limit errors
                await asyncio.sleep(7)

        logger.info(f"Completed processing {len(extracted_data)} chunks")
        return extracted_data

    @staticmethod
    def validate_extracted_data(data: Dict[str, Any]) -> bool:
        """
        Validate extracted entity and relationship data.

        Args:
            data: Extracted data dictionary

        Returns:
            True if data is valid, False otherwise
        """
        if not isinstance(data, dict):
            return False

        if "entities" not in data or "relationships" not in data:
            return False

        if not isinstance(data["entities"], list) or not isinstance(data["relationships"], list):
            return False

        # Check that all entities have required fields
        entity_names = set()
        for entity in data["entities"]:
            if not isinstance(entity, dict):
                return False
            if "name" not in entity or "type" not in entity:
                return False
            entity_names.add(entity["name"])

        # Check that relationship entities exist
        for rel in data["relationships"]:
            if not isinstance(rel, dict):
                return False
            if "source" not in rel or "target" not in rel or "label" not in rel:
                return False
            if rel["source"] not in entity_names or rel["target"] not in entity_names:
                return False

        return True