"""
YAML parser for configuration and data file analysis.

Extracts structured data entities from YAML files including objects, arrays,
key-value pairs, and configuration sections with comprehensive metadata.
"""

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


@register_parser("yaml", [".yaml", ".yml"])
class YAMLParser(BaseParser):
    """
    Comprehensive YAML parser for configuration and data analysis.
    
    Features:
    - YAML objects and nested structures
    - Arrays and data collections
    - Key-value pairs with type detection
    - Configuration sections identification
    - Schema and metadata extraction
    - Multi-document YAML support
    """
    
    # Supported features
    SUPPORTED_FEATURES = [
        "objects", "arrays", "key_value_pairs", "nested_structures",
        "type_detection", "configuration_sections", "schema_analysis",
        "multi_document", "anchors_aliases"
    ]
    
    def __init__(self):
        super().__init__("yaml")
        self.__version__ = "1.0.0"
        
        logger.debug("YAML parser initialized")
    
    def get_supported_extensions(self) -> List[str]:
        return [".yaml", ".yml"]
    
    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.get_supported_extensions()
    
    def parse_file(self, file_path: Path) -> ParseResult:
        """
        Parse YAML file and extract entities and relations.
        
        Args:
            file_path: Path to YAML file
            
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
            entities = self.extract_entities(None, content, file_path)  # No tree for YAML
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
        tree: None,  # YAML doesn't use tree-sitter
        content: str,
        file_path: Path
    ) -> List[Entity]:
        """
        Extract YAML entities from parsed content.
        
        Args:
            tree: Not used for YAML (no AST)
            content: YAML content string
            file_path: Path to YAML file
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        try:
            # Parse YAML content
            yaml_data = self._parse_yaml_content(content)
            if yaml_data is None:
                return entities
            
            # Handle multi-document YAML
            if isinstance(yaml_data, list) and len(yaml_data) > 1:
                # Multi-document YAML
                for i, doc in enumerate(yaml_data):
                    doc_entities = self._extract_document_entities(doc, content, file_path, i)
                    entities.extend(doc_entities)
            else:
                # Single document YAML
                if isinstance(yaml_data, list) and len(yaml_data) == 1:
                    yaml_data = yaml_data[0]
                
                # Extract different entity types from root level
                entities.extend(self._extract_root_entities(yaml_data, content, file_path))
                entities.extend(self._extract_nested_entities(yaml_data, content, file_path))
                entities.extend(self._extract_configuration_sections(yaml_data, content, file_path))
            
            logger.debug(f"Extracted {len(entities)} entities from {file_path}")
            
        except Exception as e:
            logger.error(f"Entity extraction failed for {file_path}: {e}")
        
        return entities
    
    def extract_relations(
        self,
        tree: None,  # YAML doesn't use tree-sitter
        content: str,
        entities: List[Entity],
        file_path: Path
    ) -> List[Relation]:
        """
        Extract relationships between YAML entities.
        
        Args:
            tree: Not used for YAML (no AST)
            content: YAML content string
            entities: Previously extracted entities
            file_path: Path to YAML file
            
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
    
    def _parse_yaml_content(self, content: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
        """Parse YAML content with support for multi-document files"""
        try:
            import yaml
            
            # Try to parse as multi-document YAML first
            documents = list(yaml.safe_load_all(content))
            
            if len(documents) == 0:
                return None
            elif len(documents) == 1:
                return documents[0]
            else:
                return documents
                
        except ImportError:
            logger.error("PyYAML is required for YAML parsing. Install with: pip install PyYAML")
            return None
        except Exception as e:
            logger.warning(f"Failed to parse YAML content: {e}")
            return None
    
    def _extract_document_entities(
        self, 
        yaml_data: Dict[str, Any], 
        content: str, 
        file_path: Path,
        doc_index: int
    ) -> List[Entity]:
        """Extract entities from a single YAML document"""
        entities = []
        doc_prefix = f"doc{doc_index}"
        
        # Create document root entity
        root_entity = self._create_yaml_entity(
            name=f"document_{doc_index}",
            entity_type=EntityType.MODULE,
            value=yaml_data,
            path=f"${doc_prefix}",
            content=content,
            file_path=file_path,
            line_number=1
        )
        if root_entity:
            entities.append(root_entity)
        
        # Extract nested entities with document prefix
        entities.extend(self._extract_nested_entities(
            yaml_data, content, file_path, f"${doc_prefix}", 0
        ))
        entities.extend(self._extract_configuration_sections(
            yaml_data, content, file_path, f"${doc_prefix}"
        ))
        
        return entities
    
    def _extract_root_entities(
        self, 
        yaml_data: Union[Dict[str, Any], List[Any]], 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract root-level YAML entities"""
        entities = []
        
        # Create root object entity
        root_entity = self._create_yaml_entity(
            name="root",
            entity_type=EntityType.MODULE,
            value=yaml_data,
            path="$",
            content=content,
            file_path=file_path,
            line_number=1
        )
        if root_entity:
            entities.append(root_entity)
        
        # Extract top-level properties
        if isinstance(yaml_data, dict):
            for key, value in yaml_data.items():
                entity = self._create_property_entity(
                    key, value, f"$.{key}", content, file_path
                )
                if entity:
                    entities.append(entity)
        elif isinstance(yaml_data, list):
            # Root-level array
            for i, item in enumerate(yaml_data):
                entity = self._create_property_entity(
                    f"item_{i}", item, f"$[{i}]", content, file_path
                )
                if entity:
                    entities.append(entity)
        
        return entities
    
    def _extract_nested_entities(
        self, 
        yaml_data: Union[Dict[str, Any], List[Any]], 
        content: str, 
        file_path: Path,
        parent_path: str = "$",
        depth: int = 0
    ) -> List[Entity]:
        """Extract nested YAML entities recursively"""
        entities = []
        
        # Limit recursion depth to prevent infinite loops
        if depth > 10:
            return entities
        
        if isinstance(yaml_data, dict):
            for key, value in yaml_data.items():
                current_path = f"{parent_path}.{key}"
                
                if isinstance(value, dict):
                    # Nested object
                    entity = self._create_yaml_entity(
                        name=key,
                        entity_type=EntityType.CLASS,  # Use CLASS for YAML objects
                        value=value,
                        path=current_path,
                        content=content,
                        file_path=file_path,
                        line_number=self._estimate_line_number(key, content)
                    )
                    if entity:
                        entities.append(entity)
                    
                    # Recurse into nested object
                    entities.extend(
                        self._extract_nested_entities(value, content, file_path, current_path, depth + 1)
                    )
                
                elif isinstance(value, list):
                    # Array entity
                    entity = self._create_yaml_entity(
                        name=key,
                        entity_type=EntityType.VARIABLE,  # Use VARIABLE for arrays
                        value=value,
                        path=current_path,
                        content=content,
                        file_path=file_path,
                        line_number=self._estimate_line_number(key, content)
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
        
        elif isinstance(yaml_data, list):
            for i, item in enumerate(yaml_data):
                current_path = f"{parent_path}[{i}]"
                
                if isinstance(item, dict):
                    entities.extend(
                        self._extract_nested_entities(item, content, file_path, current_path, depth + 1)
                    )
        
        return entities
    
    def _extract_configuration_sections(
        self, 
        yaml_data: Union[Dict[str, Any], List[Any]], 
        content: str, 
        file_path: Path,
        path_prefix: str = "$"
    ) -> List[Entity]:
        """Extract common configuration sections"""
        entities = []
        
        # Common configuration section names (similar to JSON parser)
        config_sections = [
            "config", "configuration", "settings", "options", "env", "environment",
            "database", "db", "server", "api", "auth", "security", "logging",
            "cache", "redis", "mongodb", "postgres", "mysql", "docker", "k8s",
            "kubernetes", "aws", "azure", "gcp", "scripts", "dependencies",
            "devdependencies", "peerdependencies", "optionaldependencies", 
            "eslintconfig", "babel", "webpack", "jest", "husky", "lint-staged",
            # YAML-specific config sections
            "services", "volumes", "networks", "deploy", "build", "image",
            "ports", "environment", "command", "entrypoint", "working_dir",
            "labels", "restart", "healthcheck", "secrets", "configs",
            # Kubernetes-specific sections
            "metadata", "spec", "template", "containers", "selector",
            "matchlabels", "resources", "limits", "requests", "probes",
            "livenessprobe", "readinessprobe", "volumemounts", "configmap",
            "secret", "service", "ingress", "deployment", "statefulset",
            "daemonset", "job", "cronjob", "namespace", "persistentvolume",
            "persistentvolumeclaim", "storageclass", "clusterrole",
            "clusterrolebinding", "role", "rolebinding", "serviceaccount"
        ]
        
        if isinstance(yaml_data, dict):
            for key, value in yaml_data.items():
                # Check if this is a configuration section
                if (key.lower() in config_sections or 
                    key.endswith('Config') or 
                    key.endswith('Settings') or
                    key.endswith('Options') or
                    key.endswith('_config') or
                    key.endswith('_settings')):
                    
                    entity = self._create_yaml_entity(
                        name=key,
                        entity_type=EntityType.NAMESPACE,  # Use NAMESPACE for config sections
                        value=value,
                        path=f"{path_prefix}.{key}",
                        content=content,
                        file_path=file_path,
                        line_number=self._estimate_line_number(key, content),
                        is_config_section=True
                    )
                    if entity:
                        entities.append(entity)
        
        return entities
    
    def _create_yaml_entity(
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
        """Create a YAML entity from parsed data"""
        try:
            # Estimate location (YAML doesn't have precise line info)
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
            signature = self._create_yaml_signature(name, value, entity_type)
            
            # Determine value type and serialize safely
            value_type = type(value).__name__
            value_str = self._serialize_value_safely(value)
            
            # Build metadata
            metadata = {
                "yaml_path": path,
                "value_type": value_type,
                "value_length": len(value) if hasattr(value, '__len__') else 1,
                "is_nested": '.' in path and not path.startswith(f"$.{name}") and '[' not in path,
                "is_config_section": is_config_section,
                "ast_node_type": "yaml_property"
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
                id=f"yaml::{entity_type.value}::{path}::{line_number}",
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
            logger.warning(f"Failed to create YAML entity for {name}: {e}")
            return None
    
    def _create_property_entity(
        self,
        key: str,
        value: Any,
        path: str,
        content: str,
        file_path: Path
    ) -> Optional[Entity]:
        """Create entity for a YAML property"""
        # Determine entity type based on value
        if isinstance(value, dict):
            entity_type = EntityType.CLASS
        elif isinstance(value, list):
            entity_type = EntityType.VARIABLE
        else:
            entity_type = EntityType.CONSTANT
        
        return self._create_yaml_entity(
            name=key,
            entity_type=entity_type,
            value=value,
            path=path,
            content=content,
            file_path=file_path,
            line_number=self._estimate_line_number(key, content)
        )
    
    def _create_yaml_signature(self, name: str, value: Any, entity_type: EntityType) -> str:
        """Create a signature for YAML entity"""
        if isinstance(value, dict):
            return f'{name}: {{ {len(value)} properties }}'
        elif isinstance(value, list):
            return f'{name}: [ {len(value)} items ]'
        elif isinstance(value, str):
            preview = value[:50] + "..." if len(value) > 50 else value
            return f'{name}: "{preview}"'
        else:
            return f'{name}: {value}'
    
    def _serialize_value_safely(self, value: Any, max_length: int = 500) -> str:
        """Safely serialize YAML value for storage"""
        try:
            import yaml
            serialized = yaml.dump(value, default_flow_style=False, allow_unicode=True)
            if len(serialized) > max_length:
                return serialized[:max_length-3] + "..."
            return serialized
        except ImportError:
            return str(value)[:max_length]
        except Exception:
            return str(value)[:max_length]
    
    def _estimate_line_number(self, key: str, content: str) -> int:
        """Estimate line number for a YAML key (rough approximation)"""
        try:
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                if f'{key}:' in line or f'"{key}":' in line or f"'{key}':" in line:
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
            if "yaml_path" in entity.metadata:
                lookup[entity.metadata["yaml_path"]] = entity
        return lookup
    
    def _extract_containment_relations(self, entities: List[Entity]) -> List[Relation]:
        """Extract containment relations (objects contain properties)"""
        relations = []
        
        # Group entities by YAML path hierarchy
        for entity in entities:
            if "yaml_path" in entity.metadata:
                path = entity.metadata["yaml_path"]
                
                # Find parent path
                if '.' in path and path != "$":
                    path_parts = path.split('.')
                    if len(path_parts) > 1:
                        parent_path = '.'.join(path_parts[:-1])
                        
                        # Find parent entity
                        parent_entity = None
                        for candidate in entities:
                            if candidate.metadata.get("yaml_path") == parent_path:
                                parent_entity = candidate
                                break
                        
                        if parent_entity:
                            relation = Relation.create_contains_relation(
                                parent_entity.id,
                                entity.id,
                                context=f"YAML object {parent_entity.name} contains {entity.name}",
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
        """Extract reference relations (e.g., YAML anchors and aliases)"""
        relations = []
        
        try:
            # Look for YAML anchors (&anchor) and aliases (*alias)
            lines = content.split('\n')
            anchors = {}  # anchor_name -> line_number
            
            for i, line in enumerate(lines, 1):
                # Find anchors (&name)
                if '&' in line:
                    import re
                    anchor_match = re.search(r'&([a-zA-Z_][a-zA-Z0-9_]*)', line)
                    if anchor_match:
                        anchor_name = anchor_match.group(1)
                        anchors[anchor_name] = i
                
                # Find aliases (*name)
                if '*' in line:
                    import re
                    alias_match = re.search(r'\*([a-zA-Z_][a-zA-Z0-9_]*)', line)
                    if alias_match:
                        alias_name = alias_match.group(1)
                        if alias_name in anchors:
                            # Create reference relation
                            relation_id = f"yaml_reference::{alias_name}::{i}"
                            relation = Relation(
                                id=relation_id,
                                relation_type=RelationType.REFERENCES,
                                source_entity_id=f"yaml::alias::{alias_name}::{i}",
                                target_entity_id=f"yaml::anchor::{alias_name}::{anchors[alias_name]}",
                                context=f"YAML alias *{alias_name} references &{alias_name}",
                                location=SourceLocation(
                                    file_path=file_path,
                                    start_line=i,
                                    end_line=i,
                                    start_column=0,
                                    end_column=100,
                                    start_byte=0,
                                    end_byte=100
                                )
                            )
                            relations.append(relation)
        
        except Exception as e:
            logger.warning(f"Failed to extract YAML reference relations: {e}")
        
        return relations
    
    def _extract_dependency_relations(
        self, 
        entities: List[Entity], 
        content: str, 
        file_path: Path
    ) -> List[Relation]:
        """Extract dependency relations (e.g., Docker Compose services, Kubernetes dependencies)"""
        relations = []
        
        # Look for dependency entities
        dependency_keys = [
            "dependencies", "devDependencies", "peerDependencies", "optionalDependencies",
            "services", "depends_on", "links", "external_links", "volumes_from"
        ]
        
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
                # Create dependency relations
                if isinstance(entity.metadata.get("properties"), list):
                    for dep_name in entity.metadata["properties"]:
                        relation = Relation(
                            id=f"dependency::{root_entity.id}::{dep_name}",
                            relation_type=RelationType.DEPENDS_ON,
                            source_entity_id=root_entity.id,
                            target_entity_id=f"yaml::dependency::{dep_name}",
                            context=f"Depends on {dep_name}",
                            location=entity.location
                        )
                        relations.append(relation)
        
        return relations
    
    def _find_entity_by_path(self, path: str, entities: List[Entity]) -> Optional[Entity]:
        """Find entity by YAML path"""
        for entity in entities:
            if entity.metadata.get("yaml_path") == path:
                return entity
        return None