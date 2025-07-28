"""
Core entity models for code parsing and storage.

Defines entities, AST nodes, and relations extracted from code.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, List, Dict
from pydantic import BaseModel, Field, ConfigDict, field_validator
import hashlib


class EntityType(Enum):
    """Types of code entities extracted"""
    # File structure
    PROJECT = "project"
    DIRECTORY = "directory"  
    FILE = "file"
    
    # Code entities
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    CONSTANT = "constant"
    
    # Language-specific
    IMPORT = "import"
    EXPORT = "export"
    DECORATOR = "decorator"
    TYPE_ALIAS = "type_alias"
    TYPE = "type"
    INTERFACE = "interface"
    ENUM = "enum"
    STRUCT = "struct"
    NAMESPACE = "namespace"
    TRAIT = "trait"
    IMPLEMENTATION = "implementation"
    MACRO = "macro"
    
    # HTML-specific
    HTML_ELEMENT = "html_element"
    HTML_COMPONENT = "html_component"
    HTML_FORM = "html_form"
    HTML_MEDIA = "html_media"
    HTML_LINK = "html_link"
    HTML_META = "html_meta"
    HTML_SCRIPT = "html_script"
    HTML_STYLE = "html_style"
    
    # CSS-specific
    CSS_RULE = "css_rule"
    CSS_SELECTOR = "css_selector"
    CSS_PROPERTY = "css_property"
    CSS_AT_RULE = "css_at_rule"
    CSS_MEDIA_QUERY = "css_media_query"
    CSS_KEYFRAMES = "css_keyframes"
    CSS_IMPORT = "css_import"
    CSS_VARIABLE = "css_variable"


class Visibility(Enum):
    """Entity visibility levels"""
    PUBLIC = "public"
    PRIVATE = "private"
    PROTECTED = "protected"
    PACKAGE_PRIVATE = "package_private"  # Java default visibility


class RelationType(Enum):
    """Comprehensive relation types for code entities"""
    # Structural relations
    CONTAINS = "contains"           # Module contains class
    BELONGS_TO = "belongs_to"       # Method belongs to class
    
    # Inheritance relations
    INHERITS = "inherits"          # Class inheritance
    IMPLEMENTS = "implements"      # Interface implementation
    EXTENDS = "extends"            # Extension
    MIXES_IN = "mixes_in"         # Mixin/trait
    
    # Usage relations
    CALLS = "calls"               # Function calls
    INSTANTIATES = "instantiates" # Creates instance
    USES_TYPE = "uses_type"      # Type reference
    IMPORTS = "imports"           # Import/require
    EXPORTS = "exports"           # Export/provide
    
    # Data flow relations
    READS = "reads"               # Reads variable
    WRITES = "writes"             # Writes variable
    RETURNS = "returns"           # Returns type
    ACCEPTS = "accepts"           # Accepts parameter
    
    # Special relations
    DECORATES = "decorates"       # Decorator/annotation
    OVERRIDES = "overrides"       # Method override
    TESTS = "tests"               # Test relationship
    REFERENCES = "references"     # General reference
    DEFINES = "defines"           # Definition relationship
    DEPENDS_ON = "depends_on"     # Dependency relationship


@dataclass(frozen=True)
class SourceLocation:
    """Precise source file location with byte and line information"""
    file_path: Path
    start_line: int
    end_line: int
    start_column: int
    end_column: int
    start_byte: int
    end_byte: int
    
    def __post_init__(self):
        """Validate location data"""
        if self.start_line > self.end_line:
            raise ValueError("start_line cannot be greater than end_line")
        if self.start_line == self.end_line and self.start_column > self.end_column:
            raise ValueError("start_column cannot be greater than end_column on same line")
        if self.start_byte > self.end_byte:
            raise ValueError("start_byte cannot be greater than end_byte")
    
    @property
    def location_id(self) -> str:
        """Generate unique location identifier"""
        return f"{self.file_path}:{self.start_line}:{self.start_column}"


class Entity(BaseModel):
    """Code entity with full metadata and source information"""
    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=False
    )
    
    # Identification
    id: str  # Format: "file://path::type::name::line"
    name: str
    qualified_name: str
    entity_type: EntityType
    
    # Location
    location: SourceLocation
    
    # Content
    signature: Optional[str] = None
    docstring: Optional[str] = None
    source_code: str
    source_hash: str = Field(default="")
    
    # Metadata
    visibility: Visibility = Visibility.PUBLIC
    is_async: bool = False
    is_test: bool = False
    is_deprecated: bool = False
    
    # Relationships
    parent_id: Optional[str] = None
    children_ids: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    last_modified: datetime = Field(default_factory=datetime.now)
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def __init__(self, **data):
        # Auto-generate source_hash if not provided
        if 'source_hash' not in data or not data['source_hash']:
            data['source_hash'] = self._generate_source_hash(data.get('source_code', ''))
        super().__init__(**data)
    
    @staticmethod
    def _generate_source_hash(source_code: str) -> str:
        """Generate SHA-256 hash of source code"""
        return hashlib.sha256(source_code.encode('utf-8')).hexdigest()[:16]
    
    @field_validator('id')
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Validate entity ID format"""
        if not v or '::' not in v:
            raise ValueError('Entity ID must contain "::" separators')
        return v
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate entity name is not empty"""
        if not v.strip():
            raise ValueError('Entity name cannot be empty')
        return v.strip()
    
    @field_validator('qualified_name')
    @classmethod
    def validate_qualified_name(cls, v: str) -> str:
        """Validate qualified name is not empty"""  
        if not v.strip():
            raise ValueError('Qualified name cannot be empty')
        return v.strip()
    
    def update_source(self, new_source: str) -> 'Entity':
        """Create updated entity with new source code"""
        new_hash = self._generate_source_hash(new_source)
        return self.model_copy(update={
            'source_code': new_source,
            'source_hash': new_hash,
            'last_modified': datetime.now()
        })
    
    @property
    def is_container(self) -> bool:
        """Check if entity can contain other entities"""
        return self.entity_type in {
            EntityType.PROJECT, 
            EntityType.DIRECTORY,
            EntityType.FILE,
            EntityType.MODULE,
            EntityType.CLASS,
            EntityType.INTERFACE
        }
    
    @property
    def language_hint(self) -> Optional[str]:
        """Infer programming language from file extension"""
        if not self.location.file_path:
            return None
        
        suffix = self.location.file_path.suffix.lower()
        language_map = {
            '.py': 'python',
            '.js': 'javascript', 
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.go': 'go',
            '.rs': 'rust',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.cs': 'csharp',
            '.rb': 'ruby',
            '.php': 'php',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala'
        }
        return language_map.get(suffix)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = self.model_dump()
        # Convert Path objects to strings for JSON serialization
        data['location'] = {
            **data['location'],
            'file_path': str(data['location']['file_path'])
        }
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Entity':
        """Create entity from dictionary"""
        # Convert file_path back to Path object
        if 'location' in data and 'file_path' in data['location']:
            data['location']['file_path'] = Path(data['location']['file_path'])
        return cls(**data)
    
    def to_qdrant_payload(self) -> Dict[str, Any]:
        """
        Convert entity to Qdrant payload format for storage and search.
        
        Returns:
            Dictionary suitable for Qdrant payload storage
        """
        return {
            # Core identification
            "entity_id": self.id,
            "entity_name": self.name,
            "qualified_name": self.qualified_name,
            "entity_type": self.entity_type.value,
            
            # Content for search
            "signature": self.signature or "",
            "docstring": self.docstring or "",
            "source_code": self.source_code,
            "source_hash": self.source_hash,
            
            # Location information
            "file_path": str(self.location.file_path),
            "start_line": self.location.start_line,
            "end_line": self.location.end_line,
            "start_column": self.location.start_column,
            "end_column": self.location.end_column,
            "start_byte": self.location.start_byte,
            "end_byte": self.location.end_byte,
            
            # Metadata for filtering
            "visibility": self.visibility.value,
            "is_async": self.is_async,
            "is_test": self.is_test,
            "is_deprecated": self.is_deprecated,
            "language": self.language_hint or "",
            
            # Relationships
            "parent_id": self.parent_id or "",
            "children_count": len(self.children_ids),
            "dependencies_count": len(self.dependencies),
            
            # Timestamps (as ISO strings for Qdrant)
            "created_at": self.created_at.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            
            # Additional searchable content
            "is_container": self.is_container,
            "location_id": self.location.location_id
        }


class ASTNode(BaseModel):
    """AST node representation for tree-sitter parsing"""
    model_config = ConfigDict(frozen=True)
    
    # Node identification
    node_id: str
    node_type: str  # Tree-sitter node type
    language: str
    
    # Location
    location: SourceLocation
    
    # Content
    text: str
    
    # Tree structure
    parent_id: Optional[str] = None
    children_ids: List[str] = Field(default_factory=list)
    
    # Metadata
    is_named: bool = True
    is_error: bool = False
    
    @field_validator('node_id')
    @classmethod
    def validate_node_id(cls, v: str) -> str:
        """Validate node ID is not empty"""
        if not v.strip():
            raise ValueError('Node ID cannot be empty')
        return v
    
    @field_validator('language')
    @classmethod
    def validate_language(cls, v: str) -> str:
        """Validate supported language"""
        supported = {
            'python', 'javascript', 'typescript', 'html', 'css', 'go', 'rust', 
            'java', 'cpp', 'c', 'csharp', 'ruby', 'php'
        }
        if v.lower() not in supported:
            raise ValueError(f'Unsupported language: {v}')
        return v.lower()
    
    def is_definition(self) -> bool:
        """Check if node represents a definition"""
        definition_types = {
            'function_definition', 'class_definition', 'method_definition',
            'variable_declaration', 'const_declaration', 'function_declaration',
            'interface_declaration', 'type_alias_declaration', 'enum_declaration'
        }
        return self.node_type in definition_types


class Relation(BaseModel):
    """Relationship between entities"""
    model_config = ConfigDict(frozen=True, use_enum_values=False)
    
    # Relationship identification
    id: str
    relation_type: RelationType
    
    # Entities involved
    source_entity_id: str
    target_entity_id: str
    
    # Context and location
    context: Optional[str] = None  # Where the relation occurs
    location: Optional[SourceLocation] = None  # Source location of relation
    
    # Strength and metadata
    strength: float = Field(default=1.0, ge=0.0, le=1.0)  # Relation strength
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    
    @field_validator('id')
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Validate relation ID"""
        if not v.strip():
            raise ValueError('Relation ID cannot be empty')
        return v
    
    @classmethod
    def create_call_relation(
        cls, 
        caller_id: str, 
        callee_id: str, 
        context: Optional[str] = None,
        location: Optional[SourceLocation] = None
    ) -> 'Relation':
        """Create a function call relation"""
        relation_id = f"call::{caller_id}::{callee_id}"
        return cls(
            id=relation_id,
            relation_type=RelationType.CALLS,
            source_entity_id=caller_id,
            target_entity_id=callee_id,
            context=context,
            location=location
        )
    
    @classmethod
    def create_import_relation(
        cls,
        importer_id: str,
        imported_id: str,
        context: Optional[str] = None,
        location: Optional[SourceLocation] = None
    ) -> 'Relation':
        """Create an import relation"""
        relation_id = f"import::{importer_id}::{imported_id}"
        return cls(
            id=relation_id,
            relation_type=RelationType.IMPORTS, 
            source_entity_id=importer_id,
            target_entity_id=imported_id,
            context=context,
            location=location
        )
    
    @classmethod
    def create_inheritance_relation(
        cls,
        child_id: str,
        parent_id: str,
        context: Optional[str] = None,
        location: Optional[SourceLocation] = None
    ) -> 'Relation':
        """Create an inheritance relation"""
        relation_id = f"inherits::{child_id}::{parent_id}"
        return cls(
            id=relation_id,
            relation_type=RelationType.INHERITS,
            source_entity_id=child_id,
            target_entity_id=parent_id,
            context=context,
            location=location
        )
    
    @classmethod
    def create_contains_relation(
        cls,
        container_id: str,
        contained_id: str,
        context: Optional[str] = None,
        location: Optional[SourceLocation] = None
    ) -> 'Relation':
        """Create a containment relation"""
        relation_id = f"contains::{container_id}::{contained_id}"
        return cls(
            id=relation_id,
            relation_type=RelationType.CONTAINS,
            source_entity_id=container_id,
            target_entity_id=contained_id,
            context=context,
            location=location
        )
    
    def to_qdrant_payload(self) -> Dict[str, Any]:
        """
        Convert relation to Qdrant payload format.
        
        Returns:
            Dictionary suitable for Qdrant payload storage
        """
        payload = {
            "relation_id": self.id,
            "relation_type": self.relation_type.value,
            "source_entity_id": self.source_entity_id,
            "target_entity_id": self.target_entity_id,
            "context": self.context or "",
            "strength": self.strength,
            "created_at": self.created_at.isoformat()
        }
        
        # Add location if available
        if self.location:
            payload.update({
                "source_file_path": str(self.location.file_path),
                "source_start_line": self.location.start_line,
                "source_end_line": self.location.end_line,
                "source_start_column": self.location.start_column,
                "source_end_column": self.location.end_column
            })
        
        return payload