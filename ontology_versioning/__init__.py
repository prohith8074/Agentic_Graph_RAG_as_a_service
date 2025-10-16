"""
Ontology versioning system for tracking and managing ontology evolution.
Provides version control, rollback, and collaborative editing capabilities.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import json
import hashlib
import time
from pathlib import Path

@dataclass
class OntologyVersion:
    """Represents a version of the ontology."""
    version_id: str
    timestamp: float
    author: str
    message: str
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    parent_version: Optional[str] = None
    checksum: Optional[str] = None

@dataclass
class OntologyDiff:
    """Represents changes between ontology versions."""
    from_version: str
    to_version: str
    added_entities: List[Dict[str, Any]]
    removed_entities: List[Dict[str, Any]]
    modified_entities: List[Dict[str, Any]]
    added_relationships: List[Dict[str, Any]]
    removed_relationships: List[Dict[str, Any]]
    modified_relationships: List[Dict[str, Any]]

class OntologyVersionControl:
    """Version control system for ontologies."""

    def __init__(self, storage_path: str = "./ontology_versions"):
        """
        Initialize version control system.

        Args:
            storage_path: Path to store version files
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.versions: Dict[str, OntologyVersion] = {}
        self.current_version: Optional[str] = None

        # Load existing versions
        self._load_versions()

    def _load_versions(self):
        """Load existing versions from storage."""
        for version_file in self.storage_path.glob("*.json"):
            try:
                with open(version_file, 'r') as f:
                    version_data = json.load(f)
                    version = OntologyVersion(**version_data)
                    self.versions[version.version_id] = version
            except Exception as e:
                print(f"Error loading version {version_file}: {e}")

        # Set current version to latest
        if self.versions:
            self.current_version = max(self.versions.keys(),
                                     key=lambda v: self.versions[v].timestamp)

    def _save_version(self, version: OntologyVersion):
        """Save version to storage."""
        version_file = self.storage_path / f"{version.version_id}.json"
        with open(version_file, 'w') as f:
            json.dump(asdict(version), f, indent=2)

    def _calculate_checksum(self, entities: List[Dict], relationships: List[Dict]) -> str:
        """Calculate checksum for ontology content."""
        content = json.dumps({
            "entities": sorted([e.copy() for e in entities], key=lambda x: x.get("name", "")),
            "relationships": sorted([r.copy() for r in relationships], key=lambda x: (x.get("source", ""), x.get("target", "")))
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def create_version(self, entities: List[Dict[str, Any]],
                      relationships: List[Dict[str, Any]],
                      author: str = "system",
                      message: str = "Ontology update") -> str:
        """
        Create a new ontology version.

        Args:
            entities: Ontology entities
            relationships: Ontology relationships
            author: Author of the change
            message: Commit message

        Returns:
            New version ID
        """
        # Generate version ID
        timestamp = time.time()
        checksum = self._calculate_checksum(entities, relationships)
        version_id = f"v{int(timestamp)}"

        # Create version object
        version = OntologyVersion(
            version_id=version_id,
            timestamp=timestamp,
            author=author,
            message=message,
            entities=entities.copy(),
            relationships=relationships.copy(),
            metadata={
                "entity_count": len(entities),
                "relationship_count": len(relationships)
            },
            parent_version=self.current_version,
            checksum=checksum
        )

        # Save and register version
        self.versions[version_id] = version
        self._save_version(version)
        self.current_version = version_id

        return version_id

    def get_version(self, version_id: str) -> Optional[OntologyVersion]:
        """
        Get a specific ontology version.

        Args:
            version_id: Version identifier

        Returns:
            OntologyVersion object or None
        """
        return self.versions.get(version_id)

    def list_versions(self) -> List[Dict[str, Any]]:
        """
        List all available versions.

        Returns:
            List of version summaries
        """
        return [
            {
                "version_id": v.version_id,
                "timestamp": v.timestamp,
                "author": v.author,
                "message": v.message,
                "entity_count": len(v.entities),
                "relationship_count": len(v.relationships),
                "parent_version": v.parent_version
            }
            for v in sorted(self.versions.values(), key=lambda x: x.timestamp, reverse=True)
        ]

    def compare_versions(self, from_version: str, to_version: str) -> OntologyDiff:
        """
        Compare two ontology versions.

        Args:
            from_version: Source version ID
            to_version: Target version ID

        Returns:
            OntologyDiff object with changes
        """
        from_ver = self.versions.get(from_version)
        to_ver = self.versions.get(to_version)

        if not from_ver or not to_ver:
            raise ValueError("Version not found")

        # Compare entities
        from_entities = {e.get("name"): e for e in from_ver.entities}
        to_entities = {e.get("name"): e for e in to_ver.entities}

        added_entities = [e for name, e in to_entities.items() if name not in from_entities]
        removed_entities = [e for name, e in from_entities.items() if name not in to_entities]
        modified_entities = []

        for name in set(from_entities.keys()) & set(to_entities.keys()):
            if from_entities[name] != to_entities[name]:
                modified_entities.append({
                    "name": name,
                    "old": from_entities[name],
                    "new": to_entities[name]
                })

        # Compare relationships (simplified - using string representation)
        def rel_key(r):
            return f"{r.get('source')}_{r.get('label')}_{r.get('target')}"

        from_rels = {rel_key(r): r for r in from_ver.relationships}
        to_rels = {rel_key(r): r for r in to_ver.relationships}

        added_relationships = [r for key, r in to_rels.items() if key not in from_rels]
        removed_relationships = [r for key, r in from_rels.items() if key not in to_rels]
        modified_relationships = []

        return OntologyDiff(
            from_version=from_version,
            to_version=to_version,
            added_entities=added_entities,
            removed_entities=removed_entities,
            modified_entities=modified_entities,
            added_relationships=added_relationships,
            removed_relationships=removed_relationships,
            modified_relationships=modified_relationships
        )

    def rollback_to_version(self, version_id: str) -> bool:
        """
        Rollback to a specific version (creates new version).

        Args:
            version_id: Version to rollback to

        Returns:
            True if successful
        """
        target_version = self.versions.get(version_id)
        if not target_version:
            return False

        # Create new version with rolled back content
        return self.create_version(
            target_version.entities,
            target_version.relationships,
            author="system",
            message=f"Rollback to version {version_id}"
        ) != ""

    def merge_versions(self, base_version: str, branch_version: str,
                      author: str = "system") -> Optional[str]:
        """
        Merge two versions (simplified merge strategy).

        Args:
            base_version: Base version ID
            branch_version: Branch version ID
            author: Merge author

        Returns:
            New merged version ID or None if failed
        """
        # Simplified merge - just take the newer version
        base_ver = self.versions.get(base_version)
        branch_ver = self.versions.get(branch_version)

        if not base_ver or not branch_ver:
            return None

        # Use the newer version as the merged result
        newer_ver = branch_ver if branch_ver.timestamp > base_ver.timestamp else base_ver

        return self.create_version(
            newer_ver.entities,
            newer_ver.relationships,
            author=author,
            message=f"Merged {base_version} with {branch_version}"
        )

    def get_current_version(self) -> Optional[OntologyVersion]:
        """Get the current active version."""
        return self.versions.get(self.current_version) if self.current_version else None

    def validate_version(self, version_id: str) -> Dict[str, Any]:
        """
        Validate a version's integrity.

        Args:
            version_id: Version to validate

        Returns:
            Validation result
        """
        version = self.versions.get(version_id)
        if not version:
            return {"valid": False, "error": "Version not found"}

        # Check checksum
        current_checksum = self._calculate_checksum(version.entities, version.relationships)
        if current_checksum != version.checksum:
            return {"valid": False, "error": "Checksum mismatch"}

        # Check entity relationships
        entity_names = {e.get("name") for e in version.entities}
        invalid_relationships = []

        for rel in version.relationships:
            if rel.get("source") not in entity_names or rel.get("target") not in entity_names:
                invalid_relationships.append(rel)

        if invalid_relationships:
            return {
                "valid": False,
                "error": f"Invalid relationships found: {len(invalid_relationships)}"
            }

        return {
            "valid": True,
            "entity_count": len(version.entities),
            "relationship_count": len(version.relationships)
        }