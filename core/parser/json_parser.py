"""
JSON parser for configuration and data file analysis.

Extracts structured data entities from JSON files including objects, arrays,
key-value pairs, and configuration sections with comprehensive metadata.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import hashlib

from .base import BaseParser, ParseResult
from .registry import register_parser
from ..models.entities import (
    Entity, EntityType, Relation, RelationType, 
    SourceLocation, Visibility
)

logger = logging.getLogger(__name__)


@register_parser("json", [".json", ".jsonc", ".json5"])
class JSONParser(BaseParser):
    """
    Comprehensive JSON parser for configuration and data analysis.
    
    Features:
    - JSON objects and nested structures
    - Arrays and data collections
    - Key-value pairs with type detection
    - Configuration sections identification
    - Schema and metadata extraction
    - JSON5 and JSONC support (comments)
    """
    
    # Supported features
    SUPPORTED_FEATURES = [
        "objects", "arrays", "key_value_pairs", "nested_structures",
        "type_detection", "configuration_sections", "schema_analysis"
    ]
    
    def __init__(self):
        super().__init__("json")
        self.__version__ = "1.0.0"
        
        logger.debug("JSON parser initialized")
    
    def get_supported_extensions(self) -> List[str]:
        return [".json", ".jsonc", ".json5"]
    
    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.get_supported_extensions()
    
    def parse_file(self, file_path: Path) -> ParseResult:
        """
        Parse JSON file and extract entities and relations.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            ParseResult with extracted entities and relations
        """
        self._start_timing()
        
        # Validate file
        is_valid, error_msg = self.validate_file(file_path)
        if not is_valid:
            return self._create_error_result(file_path, error_msg)
        
        try:
            # Read file content
            content, file_hash, file_size = self._read_file_safe(file_path)
            
            # Extract entities and relations
            entities = self.extract_entities(None, content, file_path)  # No tree for JSON
            relations = self.extract_relations(None, content, entities, file_path)
            
            # Create successful result
            result = ParseResult(
                file_path=file_path,
                language=self.language,
                entities=entities,
                relations=relations,
                parse_time=self._get_elapsed_time(),
                file_size=file_size,
                file_hash=file_hash,
                parser_version=self.__version__
            )
            
            logger.debug(f"Successfully parsed {file_path}: {len(entities)} entities, {len(relations)} relations")
            return result
            
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            return self._create_error_result(file_path, str(e))
    
    def extract_entities(
        self, 
        tree: None,  # JSON doesn't use tree-sitter
        content: str,
        file_path: Path
    ) -> List[Entity]:
        """
        Extract JSON entities from parsed content.
        
        Args:
            tree: Not used for JSON (no AST)
            content: JSON content string
            file_path: Path to JSON file
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        try:
            # Parse JSON content (with error handling for JSONC/JSON5)
            json_data = self._parse_json_content(content)
            if json_data is None:
                return entities
            
            # Extract different entity types from root level
            entities.extend(self._extract_root_entities(json_data, content, file_path))
            entities.extend(self._extract_nested_entities(json_data, content, file_path))
            entities.extend(self._extract_configuration_sections(json_data, content, file_path))
            
            logger.debug(f"Extracted {len(entities)} entities from {file_path}")
            
        except Exception as e:
            logger.error(f"Entity extraction failed for {file_path}: {e}")
        
        return entities
    
    def extract_relations(
        self,
        tree: None,  # JSON doesn't use tree-sitter
        content: str,
        entities: List[Entity],
        file_path: Path
    ) -> List[Relation]:
        """
        Extract relationships between JSON entities.
        
        Args:
            tree: Not used for JSON (no AST)
            content: JSON content string
            entities: Previously extracted entities
            file_path: Path to JSON file
            
        Returns:
            List of extracted relations
        """
        if not entities:
            return []
        
        relations = []
        
        try:
            # Build entity lookup for quick access
            entity_lookup = self._build_entity_lookup(entities)
            
            # Extract different relation types
            relations.extend(self._extract_containment_relations(entities))
            relations.extend(self._extract_reference_relations(entities, content, file_path))
            relations.extend(self._extract_dependency_relations(entities, content, file_path))
            
            logger.debug(f"Extracted {len(relations)} relations from {file_path}")
            
        except Exception as e:
            logger.error(f"Relation extraction failed for {file_path}: {e}")
        
        return relations
    
    def _parse_json_content(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse JSON content with support for JSONC/JSON5 features"""
        try:
            # Try standard JSON first
            return json.loads(content)
        except json.JSONDecodeError:
            try:
                # Try to clean comments for JSONC support
                cleaned_content = self._remove_json_comments(content)
                return json.loads(cleaned_content)
            except Exception as e:
                logger.warning(f"Failed to parse JSON content: {e}")
                return None
    
    def _remove_json_comments(self, content: str) -> str:
        """Remove comments from JSONC content (improved implementation)"""
        import re
        
        # Remove single-line comments (// ...)
        # But be careful not to remove // inside strings
        result = []
        lines = content.split('\n')
        
        for line in lines:
            # Simple approach: remove // comments not in strings
            in_string = False
            escaped = False
            cleaned_line = ""
            i = 0
            
            while i < len(line):
                char = line[i]
                
                if not in_string:
                    # Check for start of comment
                    if char == '/' and i + 1 < len(line) and line[i + 1] == '/':
                        # Found comment, stop processing this line
                        break
                    elif char == '"':
                        in_string = True
                        cleaned_line += char
                    else:
                        cleaned_line += char
                else:
                    # Inside string
                    if escaped:
                        cleaned_line += char
                        escaped = False
                    elif char == '\\':
                        cleaned_line += char
                        escaped = True
                    elif char == '"':
                        cleaned_line += char
                        in_string = False
                    else:
                        cleaned_line += char
                
                i += 1
            
            result.append(cleaned_line.rstrip())
        
        # Remove block comments /* ... */
        text = '\n'.join(result)
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        
        return text
    
    def _extract_root_entities(
        self, 
        json_data: Dict[str, Any], 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract root-level JSON entities"""
        entities = []
        
        # Create root object entity
        root_entity = self._create_json_entity(
            name="root",
            entity_type=EntityType.MODULE,
            value=json_data,
            path="$",
            content=content,
            file_path=file_path,
            line_number=1
        )
        if root_entity:
            entities.append(root_entity)
        
        # Extract top-level properties
        if isinstance(json_data, dict):
            for key, value in json_data.items():
                entity = self._create_property_entity(
                    key, value, f"$.{key}", content, file_path
                )
                if entity:
                    entities.append(entity)
        
        return entities
    
    def _extract_nested_entities(
        self, 
        json_data: Dict[str, Any], 
        content: str, 
        file_path: Path,
        parent_path: str = "$",
        depth: int = 0
    ) -> List[Entity]:
        """Extract nested JSON entities recursively"""
        entities = []
        
        # Limit recursion depth to prevent infinite loops
        if depth > 10:
            return entities
        
        if isinstance(json_data, dict):
            for key, value in json_data.items():
                current_path = f"{parent_path}.{key}"
                
                if isinstance(value, dict):
                    # Nested object
                    entity = self._create_json_entity(
                        name=key,
                        entity_type=EntityType.CLASS,  # Use CLASS for JSON objects
                        value=value,
                        path=current_path,
                        content=content,
                        file_path=file_path,
                        line_number=self._estimate_line_number(current_path, content)
                    )
                    if entity:
                        entities.append(entity)
                    
                    # Recurse into nested object
                    entities.extend(
                        self._extract_nested_entities(value, content, file_path, current_path, depth + 1)
                    )
                
                elif isinstance(value, list):
                    # Array entity
                    entity = self._create_json_entity(
                        name=key,
                        entity_type=EntityType.VARIABLE,  # Use VARIABLE for arrays
                        value=value,
                        path=current_path,
                        content=content,
                        file_path=file_path,
                        line_number=self._estimate_line_number(current_path, content)
                    )
                    if entity:
                        entities.append(entity)
                    
                    # Process array elements if they're objects
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            item_path = f"{current_path}[{i}]"
                            entities.extend(
                                self._extract_nested_entities(item, content, file_path, item_path, depth + 1)
                            )
        
        return entities
    
    def _extract_configuration_sections(
        self, 
        json_data: Dict[str, Any], 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract common configuration sections"""
        entities = []
        
        # Common configuration section names
        config_sections = [
            "config", "configuration", "settings", "options", "env", "environment",
            "database", "db", "server", "api", "auth", "security", "logging",
            "cache", "redis", "mongodb", "postgres", "mysql", "docker", "k8s",
            "kubernetes", "aws", "azure", "gcp", "scripts", "dependencies",
            "devdependencies", "peerdependencies", "optionaldependencies", 
            "eslintconfig", "babel", "webpack", "jest", "husky", "lint-staged"
        ]
        
        if isinstance(json_data, dict):
            for key, value in json_data.items():
                # Check if this is a configuration section
                if (key.lower() in config_sections or 
                    key.endswith('Config') or 
                    key.endswith('Settings') or
                    key.endswith('Options')):
                    
                    entity = self._create_json_entity(
                        name=key,
                        entity_type=EntityType.NAMESPACE,  # Use NAMESPACE for config sections
                        value=value,
                        path=f"$.{key}",
                        content=content,
                        file_path=file_path,
                        line_number=self._estimate_line_number(f"$.{key}", content),
                        is_config_section=True
                    )
                    if entity:
                        entities.append(entity)
        
        return entities
    
    def _create_json_entity(
        self,
        name: str,
        entity_type: EntityType,
        value: Any,
        path: str,
        content: str,
        file_path: Path,
        line_number: int,
        is_config_section: bool = False
    ) -> Optional[Entity]:
        """Create a JSON entity from parsed data"""
        try:
            # Estimate location (JSON doesn't have precise line info)
            location = SourceLocation(
                file_path=file_path,
                start_line=line_number,
                end_line=line_number + self._estimate_entity_lines(value),
                start_column=0,
                end_column=100,  # Rough estimate
                start_byte=0,
                end_byte=len(str(value)[:1000])  # Limit for performance
            )
            
            # Create signature
            signature = self._create_json_signature(name, value, entity_type)
            
            # Determine value type and serialize safely
            value_type = type(value).__name__
            value_str = self._serialize_value_safely(value)
            
            # Build metadata
            metadata = {
                "json_path": path,
                "value_type": value_type,
                "value_length": len(value) if hasattr(value, '__len__') else 1,
                "is_nested": '.' in path and path != f"$.{name}",
                "is_config_section": is_config_section,
                "ast_node_type": "json_property"
            }
            
            # Add type-specific metadata
            if isinstance(value, dict):
                metadata.update({
                    "properties": list(value.keys()),
                    "property_count": len(value),
                    "nested_objects": sum(1 for v in value.values() if isinstance(v, dict)),
                    "nested_arrays": sum(1 for v in value.values() if isinstance(v, list))
                })
            elif isinstance(value, list):
                metadata.update({
                    "array_length": len(value),
                    "element_types": list(set(type(item).__name__ for item in value)),
                    "has_objects": any(isinstance(item, dict) for item in value),
                    "has_arrays": any(isinstance(item, list) for item in value)
                })
            
            # Create entity
            entity = Entity(
                id=f"json::{entity_type.value}::{path}::{line_number}",
                name=name,
                qualified_name=path,
                entity_type=entity_type,
                location=location,
                signature=signature,
                source_code=value_str,
                visibility=Visibility.PUBLIC,
                metadata=metadata
            )
            
            return entity
            
        except Exception as e:
            logger.warning(f"Failed to create JSON entity for {name}: {e}")
            return None
    
    def _create_property_entity(
        self,
        key: str,
        value: Any,
        path: str,
        content: str,
        file_path: Path
    ) -> Optional[Entity]:
        """Create entity for a JSON property"""
        # Determine entity type based on value
        if isinstance(value, dict):
            entity_type = EntityType.CLASS
        elif isinstance(value, list):
            entity_type = EntityType.VARIABLE
        else:
            entity_type = EntityType.CONSTANT
        
        return self._create_json_entity(
            name=key,
            entity_type=entity_type,
            value=value,
            path=path,
            content=content,
            file_path=file_path,
            line_number=self._estimate_line_number(path, content)
        )
    
    def _create_json_signature(self, name: str, value: Any, entity_type: EntityType) -> str:
        """Create a signature for JSON entity"""
        if isinstance(value, dict):
            return f'"{name}": {{ {len(value)} properties }}'
        elif isinstance(value, list):
            return f'"{name}": [ {len(value)} items ]'
        elif isinstance(value, str):
            preview = value[:50] + "..." if len(value) > 50 else value
            return f'"{name}": "{preview}"'
        else:
            return f'"{name}": {value}'
    
    def _serialize_value_safely(self, value: Any, max_length: int = 500) -> str:
        """Safely serialize JSON value for storage"""
        try:
            serialized = json.dumps(value, indent=2, ensure_ascii=False)
            if len(serialized) > max_length:
                return serialized[:max_length-3] + "..."
            return serialized
        except Exception:
            return str(value)[:max_length]
    
    def _estimate_line_number(self, path: str, content: str) -> int:
        """Estimate line number for a JSON path (rough approximation)"""
        try:
            # Simple heuristic: count occurrences of the key name
            key_name = path.split('.')[-1].replace('[', '').replace(']', '')
            if key_name and key_name != '$':
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    if f'"{key_name}"' in line:
                        return i
            return 1
        except Exception:
            return 1
    
    def _estimate_entity_lines(self, value: Any) -> int:
        """Estimate how many lines an entity spans"""
        if isinstance(value, dict):
            return min(len(value) + 2, 20)  # Cap at 20 lines
        elif isinstance(value, list):
            return min(len(value) + 2, 15)  # Cap at 15 lines
        else:
            return 1
    
    # Relation extraction methods
    
    def _build_entity_lookup(self, entities: List[Entity]) -> Dict[str, Entity]:
        """Build lookup table for entities by name and path"""
        lookup = {}
        for entity in entities:
            lookup[entity.name] = entity
            lookup[entity.qualified_name] = entity
            if "json_path" in entity.metadata:
                lookup[entity.metadata["json_path"]] = entity
        return lookup
    
    def _extract_containment_relations(self, entities: List[Entity]) -> List[Relation]:
        """Extract containment relations (objects contain properties)"""
        relations = []
        
        # Group entities by JSON path hierarchy
        for entity in entities:
            if "json_path" in entity.metadata:
                path = entity.metadata["json_path"]
                
                # Find parent path
                if '.' in path and path != "$":
                    path_parts = path.split('.')
                    if len(path_parts) > 1:
                        parent_path = '.'.join(path_parts[:-1])
                        
                        # Find parent entity
                        parent_entity = None
                        for candidate in entities:
                            if candidate.metadata.get("json_path") == parent_path:
                                parent_entity = candidate
                                break
                        
                        if parent_entity:
                            relation = Relation.create_contains_relation(
                                parent_entity.id,
                                entity.id,
                                context=f"JSON object {parent_entity.name} contains {entity.name}",
                                location=entity.location
                            )
                            relations.append(relation)
        
        return relations
    
    def _extract_reference_relations(
        self, 
        entities: List[Entity], 
        content: str, 
        file_path: Path
    ) -> List[Relation]:
        """Extract reference relations (e.g., $ref in JSON Schema)"""
        relations = []
        
        try:
            json_data = self._parse_json_content(content)
            if json_data:
                self._find_references_recursive(json_data, entities, relations, "$")
        except Exception as e:
            logger.warning(f"Failed to extract reference relations: {e}")
        
        return relations
    
    def _find_references_recursive(
        self, 
        data: Any, 
        entities: List[Entity], 
        relations: List[Relation], 
        current_path: str
    ) -> None:
        """Recursively find $ref and other references in JSON"""
        if isinstance(data, dict):
            for key, value in data.items():
                new_path = f"{current_path}.{key}"
                
                # Check for JSON Schema $ref
                if key == "$ref" and isinstance(value, str):
                    source_entity = self._find_entity_by_path(current_path, entities)
                    if source_entity:
                        relation_id = f"reference::{source_entity.id}::{value}"
                        relation = Relation(
                            id=relation_id,
                            relation_type=RelationType.REFERENCES,
                            source_entity_id=source_entity.id,
                            target_entity_id=f"json::reference::{value}",
                            context=f"References {value}",
                            location=source_entity.location
                        )
                        relations.append(relation)
                
                # Recurse
                self._find_references_recursive(value, entities, relations, new_path)
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_path = f"{current_path}[{i}]"
                self._find_references_recursive(item, entities, relations, new_path)
    
    def _extract_dependency_relations(
        self, 
        entities: List[Entity], 
        content: str, 
        file_path: Path
    ) -> List[Relation]:
        """Extract dependency relations (e.g., package.json dependencies)"""
        relations = []
        
        # Look for dependency entities
        dependency_keys = ["dependencies", "devDependencies", "peerDependencies", "optionalDependencies"]
        
        # Find root entity
        root_entity = None
        for entity in entities:
            if entity.entity_type == EntityType.MODULE and entity.name == "root":
                root_entity = entity
                break
        
        if not root_entity:
            return relations
        
        for entity in entities:
            if entity.name in dependency_keys or entity.metadata.get("is_config_section"):
                # Create dependency relations for packages
                if isinstance(entity.metadata.get("properties"), list):
                    for dep_name in entity.metadata["properties"]:
                        relation = Relation(
                            id=f"dependency::{root_entity.id}::{dep_name}",
                            relation_type=RelationType.DEPENDS_ON,
                            source_entity_id=root_entity.id,
                            target_entity_id=f"json::dependency::{dep_name}",
                            context=f"Depends on {dep_name}",
                            location=entity.location
                        )
                        relations.append(relation)
        
        return relations
    
    def _find_entity_by_path(self, path: str, entities: List[Entity]) -> Optional[Entity]:
        """Find entity by JSON path"""
        for entity in entities:
            if entity.metadata.get("json_path") == path:
                return entity
        return None