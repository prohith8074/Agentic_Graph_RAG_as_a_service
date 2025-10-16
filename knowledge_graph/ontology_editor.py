"""
Visual ontology editor with LLM-assisted refinement capabilities.
Provides tools for ontology management, validation, and refinement.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import groq
import cohere

from config.settings import settings
from knowledge_graph.graph_builder import GraphBuilder
from utils.opik_tracer import opik_tracer

logger = logging.getLogger(__name__)

@dataclass
class OntologyVersion:
    """Represents a version of the ontology."""
    version_id: str
    timestamp: float
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    parent_version: Optional[str] = None

@dataclass
class OntologyEdit:
    """Represents a single edit operation."""
    edit_id: str
    operation: str  # 'add_entity', 'remove_entity', 'modify_entity', 'add_relationship', etc.
    target_type: str  # 'entity' or 'relationship'
    target_id: str
    old_value: Any
    new_value: Any
    reasoning: str
    confidence: float
    timestamp: float
    llm_suggestions: List[Dict[str, Any]] = None

class OntologyEditor:
    """Visual ontology editor with LLM-assisted refinement."""

    def __init__(self):
        """Initialize ontology editor."""
        self.llm = groq.Groq(api_key=settings.GROQ_API_KEY)

        self.co = cohere.ClientV2(api_key=settings.COHERE_API_KEY)

        # Ontology state
        self.current_ontology: Dict[str, Any] = {"entities": [], "relationships": []}
        self.versions: List[OntologyVersion] = []
        self.edit_history: List[OntologyEdit] = []
        self.current_version_id: Optional[str] = None

    def load_ontology(self, ontology_data: Dict[str, Any]) -> bool:
        """
        Load ontology data into the editor.

        Args:
            ontology_data: Ontology dictionary with entities and relationships

        Returns:
            True if loaded successfully
        """
        try:
            # Validate ontology structure
            if not GraphBuilder.validate_graph_structure(ontology_data):
                logger.error("Invalid ontology structure")
                return False

            self.current_ontology = ontology_data.copy()

            # Create initial version
            initial_version = OntologyVersion(
                version_id="v1.0",
                timestamp=0,  # Will be set properly in a real implementation
                entities=ontology_data["entities"].copy(),
                relationships=ontology_data["relationships"].copy(),
                metadata={"type": "initial_load"}
            )

            self.versions.append(initial_version)
            self.current_version_id = "v1.0"

            logger.info(f"Ontology loaded with {len(ontology_data['entities'])} entities and "
                       f"{len(ontology_data['relationships'])} relationships")
            return True

        except Exception as e:
            logger.error(f"Error loading ontology: {e}")
            return False

    async def suggest_improvements(self, focus_area: str = "general") -> Dict[str, Any]:
        """
        Use LLM to suggest ontology improvements.

        Args:
            focus_area: Area to focus on ('entities', 'relationships', 'general')

        Returns:
            Dictionary with improvement suggestions
        """
        ontology_summary = self._summarize_ontology()

        prompt = f"""
        Analyze this ontology and suggest specific improvements:

        Ontology Summary:
        {ontology_summary}

        Focus Area: {focus_area}

        Provide specific, actionable suggestions for:
        1. Missing entities that should be added
        2. Incorrect or incomplete entity types
        3. Missing relationships between entities
        4. Relationship types that need refinement
        5. Entity properties that need standardization

        For each suggestion, provide:
        - The issue or opportunity
        - Specific recommended changes
        - Rationale for the improvement
        - Confidence level (0-1)

        Respond with JSON:
        {{
            "suggestions": [
                {{
                    "type": "add_entity|modify_entity|add_relationship|modify_relationship",
                    "description": "brief description",
                    "target": "entity/relationship name or ID",
                    "changes": "specific changes needed",
                    "rationale": "why this improves the ontology",
                    "confidence": 0.0-1.0
                }}
            ],
            "overall_assessment": "summary of ontology quality"
        }}
        """

        try:
            response = self.llm.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=settings.GROQ_MODEL,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            suggestions_text = response.choices[0].message.content

            # Track with Opik
            opik_tracer.track_llm_call(
                provider="groq",
                model=settings.GROQ_MODEL,
                prompt=prompt,
                response=suggestions_text,
                metadata={"operation": "ontology_improvement_suggestions"}
            )

            suggestions = json.loads(suggestions_text)
            return suggestions

        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            return {"suggestions": [], "overall_assessment": f"Error: {str(e)}"}

    async def apply_suggestion(self, suggestion: Dict[str, Any]) -> Tuple[bool, str, Optional[OntologyEdit]]:
        """
        Apply a suggested improvement to the ontology.

        ### Tools Used
        *   `self._add_entity_suggestion`
        *   `self._modify_entity_suggestion`
        *   `self._add_relationship_suggestion`
        *   `self._modify_relationship_suggestion`
        *   `self.create_new_version`

        Args:
            suggestion: Suggestion dictionary from suggest_improvements

        Returns:
            Tuple of (success, message, edit_object)
        """
        try:
            suggestion_type = suggestion.get("type", "")
            target = suggestion.get("target", "")
            changes = suggestion.get("changes", "")
            rationale = suggestion.get("rationale", "")

            success, message, edit = False, "", None

            if suggestion_type == "add_entity":
                success, message, edit = await self._add_entity_suggestion(target, changes, rationale)
            elif suggestion_type == "modify_entity":
                success, message, edit = await self._modify_entity_suggestion(target, changes, rationale)
            elif suggestion_type == "add_relationship":
                success, message, edit = await self._add_relationship_suggestion(target, changes, rationale)
            elif suggestion_type == "modify_relationship":
                success, message, edit = await self._modify_relationship_suggestion(target, changes, rationale)
            else:
                return False, f"Unknown suggestion type: {suggestion_type}", None

            if success and edit:
                # Create a new version of the ontology after applying the suggestion
                self.create_new_version(edit)

            return success, message, edit

        except Exception as e:
            logger.error(f"Error applying suggestion: {e}")
            return False, f"Error applying suggestion: {str(e)}", None

    def create_new_version(self, edit: OntologyEdit) -> None:
        """Create a new version of the ontology."""
        if not self.current_version_id:
            return

        parent_version_id = self.current_version_id
        version_parts = parent_version_id.split('.')
        new_minor_version = int(version_parts[1]) + 1
        new_version_id = f"{version_parts[0]}.{new_minor_version}"

        new_version = OntologyVersion(
            version_id=new_version_id,
            timestamp=edit.timestamp,
            entities=self.current_ontology["entities"].copy(),
            relationships=self.current_ontology["relationships"].copy(),
            metadata={"edit_id": edit.edit_id, "reason": edit.reasoning},
            parent_version=parent_version_id
        )

        self.versions.append(new_version)
        self.current_version_id = new_version_id
        logger.info(f"Created new ontology version: {new_version_id}")

    def switch_to_version(self, version_id: str) -> Tuple[bool, str]:
        """Switch the current ontology to a specific version."""
        version_to_load = next((v for v in self.versions if v.version_id == version_id), None)

        if not version_to_load:
            return False, f"Version '{version_id}' not found."

        self.current_ontology = {
            "entities": version_to_load.entities.copy(),
            "relationships": version_to_load.relationships.copy()
        }
        self.current_version_id = version_id
        logger.info(f"Switched to ontology version: {version_id}")
        return True, f"Successfully switched to version {version_id}"

    def get_version_info(self, version_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific version."""
        version = next((v for v in self.versions if v.version_id == version_id), None)
        if version:
            return asdict(version)
        return None

    async def _add_entity_suggestion(self, entity_name: str, details: str, rationale: str) -> Tuple[bool, str, Optional[OntologyEdit]]:
        """Add a new entity based on suggestion."""
        # Check if entity already exists
        existing_entities = [e for e in self.current_ontology["entities"]
                           if e.get("name", "").lower() == entity_name.lower()]

        if existing_entities:
            return False, f"Entity '{entity_name}' already exists", None

        # Use LLM to generate entity details
        entity_prompt = f"""
        Create a new entity for the ontology:

        Entity Name: {entity_name}
        Suggested Details: {details}
        Rationale: {rationale}

        Provide:
        - type: appropriate entity type
        - properties: relevant properties with values
        - definition: brief definition

        Respond with JSON:
        {{
            "type": "entity_type",
            "properties": {{"key": "value"}},
            "definition": "entity definition"
        }}
        """

        try:
            response = self.llm.chat.completions.create(
                messages=[{"role": "user", "content": entity_prompt}],
                model=settings.GROQ_MODEL,
                response_format={"type": "json_object"},
            )
            entity_data = json.loads(response.choices[0].message.content)

            new_entity = {
                "name": entity_name,
                "type": entity_data.get("type", "Unknown"),
                "definition": entity_data.get("definition", ""),
                **entity_data.get("properties", {})
            }

            self.current_ontology["entities"].append(new_entity)

            # Record edit
            edit = OntologyEdit(
                edit_id=f"edit_{len(self.edit_history) + 1}",
                operation="add_entity",
                target_type="entity",
                target_id=entity_name,
                old_value=None,
                new_value=new_entity,
                reasoning=rationale,
                confidence=0.8,
                timestamp=0,  # Would be set to current time
                llm_suggestions=[{"source": "suggestion", "details": details}]
            )
            self.edit_history.append(edit)

            return True, f"Added entity '{entity_name}'", edit

        except Exception as e:
            return False, f"Error creating entity: {str(e)}", None

    async def _modify_entity_suggestion(self, entity_name: str, changes: str, rationale: str) -> Tuple[bool, str, Optional[OntologyEdit]]:
        """Modify an existing entity."""
        # Find entity
        entity_idx = None
        for i, entity in enumerate(self.current_ontology["entities"]):
            if entity.get("name", "").lower() == entity_name.lower():
                entity_idx = i
                break

        if entity_idx is None:
            return False, f"Entity '{entity_name}' not found", None

        old_entity = self.current_ontology["entities"][entity_idx].copy()

        # Use LLM to apply changes
        modify_prompt = f"""
        Modify this entity based on the suggested changes:

        Current Entity: {json.dumps(old_entity, indent=2)}
        Suggested Changes: {changes}
        Rationale: {rationale}

        Provide the modified entity as JSON.
        """

        try:
            response = self.llm.chat.completions.create(
                messages=[{"role": "user", "content": modify_prompt}],
                model=settings.GROQ_MODEL,
                response_format={"type": "json_object"},
            )
            modified_entity = json.loads(response.choices[0].message.content)

            self.current_ontology["entities"][entity_idx] = modified_entity

            # Record edit
            edit = OntologyEdit(
                edit_id=f"edit_{len(self.edit_history) + 1}",
                operation="modify_entity",
                target_type="entity",
                target_id=entity_name,
                old_value=old_entity,
                new_value=modified_entity,
                reasoning=rationale,
                confidence=0.7,
                timestamp=0
            )
            self.edit_history.append(edit)

            return True, f"Modified entity '{entity_name}'", edit

        except Exception as e:
            return False, f"Error modifying entity: {str(e)}", None

    async def _add_relationship_suggestion(self, relationship_desc: str, details: str, rationale: str) -> Tuple[bool, str, Optional[OntologyEdit]]:
        """Add a new relationship."""
        # Parse relationship description (e.g., "EntityA relates to EntityB")
        # This is a simplified implementation

        relationship_prompt = f"""
        Create a relationship based on the description:

        Description: {relationship_desc}
        Details: {details}
        Rationale: {rationale}

        Available entities: {[e.get("name") for e in self.current_ontology["entities"]]}

        Provide relationship as JSON:
        {{
            "source": "source_entity_name",
            "target": "target_entity_name",
            "label": "RELATIONSHIP_TYPE"
        }}
        """

        try:
            response = self.llm.chat.completions.create(
                messages=[{"role": "user", "content": relationship_prompt}],
                model=settings.GROQ_MODEL,
                response_format={"type": "json_object"},
            )
            relationship_data = json.loads(response.choices[0].message.content)

            # Validate entities exist
            source_name = relationship_data.get("source")
            target_name = relationship_data.get("target")

            source_exists = any(e.get("name") == source_name for e in self.current_ontology["entities"])
            target_exists = any(e.get("name") == target_name for e in self.current_ontology["entities"])

            if not (source_exists and target_exists):
                return False, f"Source or target entity does not exist: {source_name} -> {target_name}", None

            self.current_ontology["relationships"].append(relationship_data)

            # Record edit
            edit = OntologyEdit(
                edit_id=f"edit_{len(self.edit_history) + 1}",
                operation="add_relationship",
                target_type="relationship",
                target_id=f"{source_name}_{relationship_data.get('label')}_{target_name}",
                old_value=None,
                new_value=relationship_data,
                reasoning=rationale,
                confidence=0.8,
                timestamp=0
            )
            self.edit_history.append(edit)

            return True, f"Added relationship: {source_name} -> {relationship_data.get('label')} -> {target_name}", edit

        except Exception as e:
            return False, f"Error creating relationship: {str(e)}", None

    async def _modify_relationship_suggestion(self, relationship_id: str, changes: str, rationale: str) -> Tuple[bool, str, Optional[OntologyEdit]]:
        """Modify an existing relationship."""
        # Simplified implementation - would need better relationship identification
        return False, "Relationship modification not yet implemented", None

    def _summarize_ontology(self) -> str:
        """Create a summary of the current ontology."""
        entities = self.current_ontology.get("entities", [])
        relationships = self.current_ontology.get("relationships", [])

        # Entity type distribution
        entity_types = {}
        for entity in entities:
            e_type = entity.get("type", "Unknown")
            entity_types[e_type] = entity_types.get(e_type, 0) + 1

        # Relationship type distribution
        rel_types = {}
        for rel in relationships:
            r_type = rel.get("label", "Unknown")
            rel_types[r_type] = rel_types.get(r_type, 0) + 1

        summary = f"""
        Ontology Summary:
        - Total Entities: {len(entities)}
        - Entity Types: {entity_types}
        - Total Relationships: {len(relationships)}
        - Relationship Types: {rel_types}

        Sample Entities: {[e.get("name", "Unknown") for e in entities[:5]]}
        Sample Relationships: {len(relationships)} total
        """

        return summary

    def get_ontology_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the ontology."""
        entities = self.current_ontology.get("entities", [])
        relationships = self.current_ontology.get("relationships", [])

        # Entity statistics
        entity_types = {}
        entity_names = []
        for entity in entities:
            e_type = entity.get("type", "Unknown")
            entity_types[e_type] = entity_types.get(e_type, 0) + 1
            entity_names.append(entity.get("name", ""))

        # Relationship statistics
        rel_types = {}
        connected_entities = set()
        for rel in relationships:
            r_type = rel.get("label", "Unknown")
            rel_types[r_type] = rel_types.get(r_type, 0) + 1
            connected_entities.add(rel.get("source", ""))
            connected_entities.add(rel.get("target", ""))

        # Connectivity analysis
        isolated_entities = len(entities) - len(connected_entities)

        return {
            "total_entities": len(entities),
            "total_relationships": len(relationships),
            "entity_types": entity_types,
            "relationship_types": rel_types,
            "connected_entities": len(connected_entities),
            "isolated_entities": isolated_entities,
            "connectivity_ratio": len(connected_entities) / len(entities) if entities else 0,
            "versions_count": len(self.versions),
            "edits_count": len(self.edit_history)
        }

    def export_ontology(self, format: str = "json") -> str:
        """
        Export current ontology in specified format.

        Args:
            format: Export format ('json', 'cypher', etc.)

        Returns:
            Exported ontology as string
        """
        if format == "json":
            return json.dumps(self.current_ontology, indent=2)
        elif format == "cypher":
            return self._export_as_cypher()
        else:
            return json.dumps(self.current_ontology, indent=2)

    def _export_as_cypher(self) -> str:
        """Export ontology as Cypher CREATE statements."""
        cypher_statements = []

        # Create entities
        for entity in self.current_ontology.get("entities", []):
            name = entity.get("name", "").replace("'", "\\'")
            e_type = entity.get("type", "Entity")
            properties = {k: v for k, v in entity.items() if k not in ["name", "type"]}

            props_str = ", ".join([f"{k}: '{v}'" if isinstance(v, str) else f"{k}: {v}"
                                  for k, v in properties.items()])

            cypher = f"CREATE (:{e_type} {{name: '{name}'"
            if props_str:
                cypher += f", {props_str}"
            cypher += "})"
            cypher_statements.append(cypher)

        # Create relationships
        for rel in self.current_ontology.get("relationships", []):
            source = rel.get("source", "").replace("'", "\\'")
            target = rel.get("target", "").replace("'", "\\'")
            label = rel.get("label", "RELATED_TO")

            cypher = f"MATCH (a {{name: '{source}'}}), (b {{name: '{target}'}}) CREATE (a)-[:{label}]->(b)"
            cypher_statements.append(cypher)

        return ";\n".join(cypher_statements) + ";"

    def get_visualization_data(self) -> Dict[str, Any]:
        """
        Get data formatted for visualization (nodes and edges).

        Returns:
            Dictionary with nodes and edges for graph visualization
        """
        entities = self.current_ontology.get("entities", [])
        relationships = self.current_ontology.get("relationships", [])

        # Create nodes
        nodes = []
        for i, entity in enumerate(entities):
            nodes.append({
                "id": i + 1,
                "label": entity.get("name", f"Entity_{i+1}"),
                "type": entity.get("type", "Unknown"),
                "properties": entity
            })

        # Create edges
        edges = []
        entity_name_to_id = {node["label"]: node["id"] for node in nodes}

        for rel in relationships:
            source_name = rel.get("source")
            target_name = rel.get("target")

            if source_name in entity_name_to_id and target_name in entity_name_to_id:
                edges.append({
                    "source": entity_name_to_id[source_name],
                    "target": entity_name_to_id[target_name],
                    "label": rel.get("label", "RELATED"),
                    "properties": rel
                })

        return {
            "nodes": nodes,
            "edges": edges,
            "stats": self.get_ontology_stats()
        }