"""
Knowledge graph construction module.
Handles merging and deduplication of multiple graph fragments.
"""

import logging
from typing import List, Dict, Any, Set, Tuple

from utils.embeddings import default_embedder

logger = logging.getLogger(__name__)

class GraphBuilder:
    """Handles construction and merging of knowledge graphs."""

    @staticmethod
    def merge_and_deduplicate_graphs(list_of_graphs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge multiple graph fragments into a single unified graph and deduplicate entities and relationships.

        Args:
            list_of_graphs: List of graph dictionaries from entity extraction

        Returns:
            Unified graph with deduplicated entities and relationships
        """
        final_entities = {}
        final_relationships: Set[Tuple[str, str, str]] = set()

        for graph in list_of_graphs:
            if not isinstance(graph, dict):
                continue

            # Add and deduplicate entities
            for entity in graph.get("entities", []):
                name = entity.get("name")
                if not name:
                    continue

                key = name.strip().lower()
                node_type = entity.get("type", "Unknown")

                # Flatten nested properties
                flat_props = {}
                for k, v in entity.items():
                    if k not in ["name", "type", "properties"]:
                        flat_props[k] = v
                if "properties" in entity and isinstance(entity["properties"], dict):
                    flat_props.update(entity["properties"])

                if key not in final_entities:
                    final_entities[key] = {
                        "name": name.strip(),
                        "type": node_type,
                        **flat_props
                    }
                else:
                    # Merge new properties into existing entity
                    final_entities[key].update(flat_props)

            # Add relationships, avoiding duplicates
            for rel in graph.get("relationships", []):
                src = rel.get("source")
                tgt = rel.get("target")
                label = rel.get("label")

                if not (src and tgt and label):
                    continue

                src_key = src.strip().lower()
                tgt_key = tgt.strip().lower()

                if src_key in final_entities and tgt_key in final_entities:
                    # Normalize and sanitize relationship label
                    sanitized_label = label.strip().replace(" ", "_").upper()
                    final_relationships.add((
                        final_entities[src_key]["name"],
                        sanitized_label,
                        final_entities[tgt_key]["name"]
                    ))

        # Convert relationship set back to list of dicts
        rel_list = [
            {"source": s, "label": l, "target": t}
            for (s, l, t) in final_relationships
        ]

        # Generate embeddings for entities and relationships
        # Note: Embeddings are used for vector similarity search in Qdrant, not for Neo4j storage
        logger.info("Generating embeddings for vector similarity search (Qdrant)...")

        try:
            # Embed entities for vector search (these won't be stored in Neo4j)
            entity_texts = [str(e) for e in final_entities.values()]
            entity_embeddings = default_embedder.embed_texts(entity_texts)

            # Embed relationships for vector search (these won't be stored in Neo4j)
            relationship_texts = [f"{r['source']}-{r['label']}->{r['target']}" for r in rel_list]
            relationship_embeddings = default_embedder.embed_texts(relationship_texts)

            logger.info(f"Generated embeddings for {len(entity_embeddings)} entities and {len(relationship_embeddings)} relationships")

            # Store embeddings separately for vector operations (not in graph entities/relationships)
            # This prevents Neo4j storage issues while preserving vector search capabilities

        except Exception as e:
            logger.warning(f"Failed to generate embeddings: {e}. Continuing without vector embeddings.")

        merged_graph = {
            "entities": list(final_entities.values()),
            "relationships": rel_list
        }

        logger.info(f"Merged graph: {len(merged_graph['entities'])} entities, "
                   f"{len(merged_graph['relationships'])} relationships")

        return merged_graph

    @staticmethod
    def validate_graph_structure(graph: Dict[str, Any]) -> bool:
        """
        Validate the structure of a knowledge graph.

        Args:
            graph: Graph dictionary to validate

        Returns:
            True if graph structure is valid
        """
        if not isinstance(graph, dict):
            logger.error("Graph is not a dictionary")
            return False

        if "entities" not in graph or "relationships" not in graph:
            logger.error("Graph missing 'entities' or 'relationships' keys")
            return False

        if not isinstance(graph["entities"], list) or not isinstance(graph["relationships"], list):
            logger.error("'entities' and 'relationships' must be lists")
            return False

        # Validate entities
        entity_names = set()
        for entity in graph["entities"]:
            if not isinstance(entity, dict):
                logger.error("Entity is not a dictionary")
                return False
            if "name" not in entity:
                logger.error("Entity missing 'name' field")
                return False
            entity_names.add(entity["name"].lower())

        # Validate relationships
        for rel in graph["relationships"]:
            if not isinstance(rel, dict):
                logger.error("Relationship is not a dictionary")
                return False
            if not all(k in rel for k in ["source", "target", "label"]):
                logger.error("Relationship missing required fields")
                return False
            if rel["source"].lower() not in entity_names:
                logger.error(f"Relationship source '{rel['source']}' not in entities")
                return False
            if rel["target"].lower() not in entity_names:
                logger.error(f"Relationship target '{rel['target']}' not in entities")
                return False

        return True

    @staticmethod
    def get_graph_statistics(graph: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate statistics about the knowledge graph.

        Args:
            graph: Graph dictionary

        Returns:
            Dictionary with graph statistics
        """
        if not GraphBuilder.validate_graph_structure(graph):
            return {"error": "Invalid graph structure"}

        entities = graph["entities"]
        relationships = graph["relationships"]

        # Entity type distribution
        entity_types = {}
        for entity in entities:
            entity_type = entity.get("type", "Unknown")
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1

        # Relationship type distribution
        relationship_types = {}
        for rel in relationships:
            rel_type = rel.get("label", "Unknown")
            relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1

        # Connectivity statistics
        entity_degrees = {}
        for rel in relationships:
            src = rel["source"]
            tgt = rel["target"]
            entity_degrees[src] = entity_degrees.get(src, 0) + 1
            entity_degrees[tgt] = entity_degrees.get(tgt, 0) + 1

        return {
            "num_entities": len(entities),
            "num_relationships": len(relationships),
            "entity_types": entity_types,
            "relationship_types": relationship_types,
            "avg_degree": sum(entity_degrees.values()) / len(entity_degrees) if entity_degrees else 0,
            "max_degree": max(entity_degrees.values()) if entity_degrees else 0,
            "isolated_entities": len([e for e in entities if e["name"] not in entity_degrees])
        }