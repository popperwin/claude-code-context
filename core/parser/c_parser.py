"""
C parser for comprehensive code analysis using Tree-sitter.

Extracts C language constructs including functions, structs, unions, enums,
typedefs, global variables, macros, and their relationships.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import tree_sitter

from .tree_sitter_base import TreeSitterBase
from .base import ParseResult
from .registry import register_parser
from ..models.entities import (
    Entity, EntityType, Relation, RelationType,
    SourceLocation, Visibility
)

logger = logging.getLogger(__name__)


@register_parser("c", [".c", ".h"])
class CParser(TreeSitterBase):
    """
    Comprehensive C parser for functions, structs, and other language constructs.
    
    Features:
    - Function declarations and definitions
    - Struct, union, and enum declarations
    - Typedef declarations
    - Global variable declarations
    - Macro definitions
    - Include statement analysis
    - Function call relationships
    - Type usage relationships
    """
    
    # Supported features
    SUPPORTED_FEATURES = [
        "functions", "structs", "unions", "enums", "typedefs",
        "variables", "macros", "includes", "function_calls", "type_usage"
    ]
    
    def __init__(self):
        super().__init__("c")
        self.__version__ = "1.0.0"
        
        logger.debug("C parser initialized")
    
    def get_supported_extensions(self) -> List[str]:
        return [".c", ".h"]
    
    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.get_supported_extensions()
    
    def extract_entities(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract C language entities from the AST"""
        entities = []
        
        if not tree.root_node:
            return entities
        
        try:
            # Extract different types of entities
            entities.extend(self._extract_functions(tree.root_node, content, file_path))
            entities.extend(self._extract_structs(tree.root_node, content, file_path))
            entities.extend(self._extract_unions(tree.root_node, content, file_path))
            entities.extend(self._extract_enums(tree.root_node, content, file_path))
            entities.extend(self._extract_typedefs(tree.root_node, content, file_path))
            entities.extend(self._extract_global_variables(tree.root_node, content, file_path))
            entities.extend(self._extract_macros(tree.root_node, content, file_path))
            entities.extend(self._extract_includes(tree.root_node, content, file_path))
            
            logger.debug(f"Extracted {len(entities)} entities from C file {file_path}")
            
        except Exception as e:
            logger.error(f"Error extracting entities from C file {file_path}: {e}")
        
        return entities
    
    def extract_relations(
        self,
        tree: tree_sitter.Tree,
        content: str,
        entities: List[Entity],
        file_path: Path
    ) -> List[Relation]:
        """Extract relationships between C entities"""
        if not tree.root_node or not entities:
            return []
        
        relations = []
        
        try:
            # Build entity lookup for quick access
            entity_lookup = self._build_entity_lookup(entities)
            
            # Extract different types of relations
            relations.extend(self._extract_function_calls(tree.root_node, entities, entity_lookup, content, file_path))
            relations.extend(self._extract_type_usage_relations(tree.root_node, entities, entity_lookup, content, file_path))
            relations.extend(self._extract_include_relations(tree.root_node, entities, content, file_path))
            
            logger.debug(f"Extracted {len(relations)} relations from C file {file_path}")
            
        except Exception as e:
            logger.error(f"Error extracting relations from C file {file_path}: {e}")
        
        return relations
    
    def _extract_functions(self, root_node: tree_sitter.Node, content: str, file_path: Path) -> List[Entity]:
        """Extract function declarations and definitions"""
        functions = []
        
        # Find function definition and declaration nodes  
        function_nodes = self._find_nodes_by_type(root_node, ["function_definition", "declaration"])
        
        processed_functions = set()
        
        for node in function_nodes:
            try:
                # Check if this is a function (has function_declarator)
                function_declarator = self.find_child_by_type(node, "function_declarator")
                if not function_declarator:
                    # For pointer functions like Node* create_node(...), check nested declarators
                    pointer_declarator = self.find_child_by_type(node, "pointer_declarator")
                    if pointer_declarator:
                        function_declarator = self.find_child_by_type(pointer_declarator, "function_declarator")
                
                if function_declarator:
                    is_definition = node.type == "function_definition"
                    func_entity = self._create_function_entity(node, content, file_path, is_definition)
                    if func_entity and func_entity.name not in processed_functions:
                        functions.append(func_entity)
                        processed_functions.add(func_entity.name)
            except Exception as e:
                logger.warning(f"Failed to extract function: {e}")
        
        return functions
    
    def _create_function_entity(
        self, 
        node: tree_sitter.Node, 
        content: str, 
        file_path: Path,
        is_definition: bool
    ) -> Optional[Entity]:
        """Create a function entity from AST node"""
        try:
            # Extract function name
            name_node = self.find_child_by_type(node, "function_declarator")
            if not name_node:
                # For pointer functions like Node* create_node(...), check nested declarators
                pointer_declarator = self.find_child_by_type(node, "pointer_declarator")
                if pointer_declarator:
                    name_node = self.find_child_by_type(pointer_declarator, "function_declarator")
            
            if not name_node:
                return None
            
            identifier_node = self.find_child_by_type(name_node, "identifier")
            if not identifier_node:
                return None
            
            func_name = self.get_node_text(identifier_node, content)
            
            # Extract return type
            return_type = "void"
            type_nodes = [child for child in node.children if child.type in ["primitive_type", "type_identifier"]]
            if type_nodes:
                return_type = self.get_node_text(type_nodes[0], content)
            
            # Extract parameters
            params_node = self.find_child_by_type(name_node, "parameter_list")
            parameters = []
            if params_node:
                parameters = self._extract_function_parameters(params_node, content)
            
            # Create signature
            param_str = ", ".join(parameters)
            signature = f"{return_type} {func_name}({param_str})"
            
            # Extract source code
            source_code = self.get_node_text(node, content)
            
            # Create location
            location = SourceLocation(
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                start_column=node.start_point[1],
                end_column=node.end_point[1],
                start_byte=node.start_byte,
                end_byte=node.end_byte
            )
            
            # Determine entity type
            entity_type = EntityType.FUNCTION if is_definition else EntityType.INTERFACE
            
            # Build metadata
            metadata = {
                "return_type": return_type,
                "parameters": parameters,
                "parameter_count": len(parameters),
                "is_definition": is_definition,
                "is_declaration": not is_definition,
                "ast_node_type": node.type,
                "has_body": is_definition
            }
            
            entity = Entity(
                id=f"file://{file_path}::c::{entity_type.value}::{func_name}::{location.start_line}",
                name=func_name,
                qualified_name=func_name,  # C doesn't have namespaces
                entity_type=entity_type,
                location=location,
                signature=signature,
                source_code=source_code,
                visibility=Visibility.PUBLIC,  # C functions are generally public
                metadata=metadata
            )
            
            return entity
            
        except Exception as e:
            logger.warning(f"Failed to create function entity: {e}")
            return None
    
    def _extract_function_parameters(self, params_node: tree_sitter.Node, content: str) -> List[str]:
        """Extract function parameter information"""
        parameters = []
        
        for child in params_node.children:
            if child.type == "parameter_declaration":
                param_text = self.get_node_text(child, content).strip()
                if param_text and param_text != ",":
                    parameters.append(param_text)
        
        return parameters
    
    def _extract_structs(self, root_node: tree_sitter.Node, content: str, file_path: Path) -> List[Entity]:
        """Extract struct declarations"""
        structs = []
        
        # Find struct declarations - both standalone and in typedefs
        struct_nodes = self._find_nodes_by_type(root_node, ["struct_specifier"])
        
        for node in struct_nodes:
            try:
                # Only process structs with names
                name_node = self.find_child_by_type(node, "type_identifier")
                if name_node:
                    struct_entity = self._create_struct_entity(node, content, file_path)
                    if struct_entity:
                        structs.append(struct_entity)
                elif node.parent and node.parent.type == "type_definition":
                    # Handle typedef struct { ... } Name; case
                    typedef_node = node.parent
                    typedef_name_nodes = [child for child in typedef_node.children if child.type == "type_identifier"]
                    if typedef_name_nodes:
                        # Use the last type_identifier as the struct name
                        name_node = typedef_name_nodes[-1]
                        struct_entity = self._create_typedef_struct_entity(node, name_node, content, file_path)
                        if struct_entity:
                            structs.append(struct_entity)
            except Exception as e:
                logger.warning(f"Failed to extract struct: {e}")
        
        return structs
    
    def _create_struct_entity(self, node: tree_sitter.Node, content: str, file_path: Path) -> Optional[Entity]:
        """Create a struct entity from AST node"""
        try:
            # Extract struct name
            name_node = self.find_child_by_type(node, "type_identifier")
            if not name_node:
                return None
            
            struct_name = self.get_node_text(name_node, content)
            
            # Extract fields
            fields = []
            body_node = self.find_child_by_type(node, "field_declaration_list")
            if body_node:
                fields = self._extract_struct_fields(body_node, content)
            
            # Create signature
            field_count = len(fields)
            signature = f"struct {struct_name} {{ {field_count} fields }}"
            
            # Extract source code
            source_code = self.get_node_text(node, content)
            
            # Create location
            location = SourceLocation(
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                start_column=node.start_point[1],
                end_column=node.end_point[1],
                start_byte=node.start_byte,
                end_byte=node.end_byte
            )
            
            # Build metadata
            metadata = {
                "fields": fields,
                "field_count": field_count,
                "ast_node_type": node.type,
                "is_struct": True
            }
            
            entity = Entity(
                id=self._generate_entity_id(EntityType.CLASS, struct_name, location),
                name=struct_name,
                qualified_name=struct_name,
                entity_type=EntityType.CLASS,  # Struct as class
                location=location,
                signature=signature,
                source_code=source_code,
                visibility=Visibility.PUBLIC,
                metadata=metadata
            )
            
            return entity
            
        except Exception as e:
            logger.warning(f"Failed to create struct entity: {e}")
            return None
    
    def _extract_struct_fields(self, body_node: tree_sitter.Node, content: str) -> List[str]:
        """Extract struct field information"""
        fields = []
        
        for child in body_node.children:
            if child.type == "field_declaration":
                field_text = self.get_node_text(child, content).strip()
                if field_text and not field_text.startswith("{") and not field_text.startswith("}"):
                    # Clean up the field text
                    field_text = field_text.replace(";", "").strip()
                    if field_text:
                        fields.append(field_text)
        
        return fields
    
    def _create_typedef_struct_entity(self, struct_node: tree_sitter.Node, name_node: tree_sitter.Node, content: str, file_path: Path) -> Optional[Entity]:
        """Create a struct entity from typedef struct"""
        try:
            struct_name = self.get_node_text(name_node, content)
            
            # Extract fields
            fields = []
            body_node = self.find_child_by_type(struct_node, "field_declaration_list")
            if body_node:
                fields = self._extract_struct_fields(body_node, content)
            
            # Create signature
            field_count = len(fields)
            signature = f"struct {struct_name} {{ {field_count} fields }}"
            
            # Extract source code
            source_code = self.get_node_text(struct_node, content)
            
            # Create location
            location = SourceLocation(
                file_path=file_path,
                start_line=struct_node.start_point[0] + 1,
                end_line=struct_node.end_point[0] + 1,
                start_column=struct_node.start_point[1],
                end_column=struct_node.end_point[1],
                start_byte=struct_node.start_byte,
                end_byte=struct_node.end_byte
            )
            
            # Build metadata
            metadata = {
                "fields": fields,
                "field_count": field_count,
                "ast_node_type": struct_node.type,
                "is_struct": True,
                "is_typedef_struct": True
            }
            
            entity = Entity(
                id=self._generate_entity_id(EntityType.CLASS, struct_name, location),
                name=struct_name,
                qualified_name=struct_name,
                entity_type=EntityType.CLASS,  # Struct as class
                location=location,
                signature=signature,
                source_code=source_code,
                visibility=Visibility.PUBLIC,
                metadata=metadata
            )
            
            return entity
            
        except Exception as e:
            logger.warning(f"Failed to create typedef struct entity: {e}")
            return None
    
    def _extract_unions(self, root_node: tree_sitter.Node, content: str, file_path: Path) -> List[Entity]:
        """Extract union declarations"""
        unions = []
        
        # Find union declarations
        union_nodes = self._find_nodes_by_type(root_node, ["union_specifier"])
        
        for node in union_nodes:
            try:
                # Only process unions with names
                name_node = self.find_child_by_type(node, "type_identifier")
                if name_node:
                    union_entity = self._create_union_entity(node, content, file_path)
                    if union_entity:
                        unions.append(union_entity)
                elif node.parent and node.parent.type == "type_definition":
                    # Handle typedef union { ... } Name; case
                    typedef_node = node.parent
                    typedef_name_nodes = [child for child in typedef_node.children if child.type == "type_identifier"]
                    if typedef_name_nodes:
                        # Use the last type_identifier as the union name
                        name_node = typedef_name_nodes[-1]
                        union_entity = self._create_typedef_union_entity(node, name_node, content, file_path)
                        if union_entity:
                            unions.append(union_entity)
            except Exception as e:
                logger.warning(f"Failed to extract union: {e}")
        
        return unions
    
    def _create_union_entity(self, node: tree_sitter.Node, content: str, file_path: Path) -> Optional[Entity]:
        """Create a union entity from AST node"""
        try:
            # Extract union name
            name_node = self.find_child_by_type(node, "type_identifier")
            if not name_node:
                return None
            
            union_name = self.get_node_text(name_node, content)
            
            # Extract fields
            fields = []
            body_node = self.find_child_by_type(node, "field_declaration_list")
            if body_node:
                fields = self._extract_struct_fields(body_node, content)  # Same as struct fields
            
            # Create signature
            field_count = len(fields)
            signature = f"union {union_name} {{ {field_count} fields }}"
            
            # Extract source code
            source_code = self.get_node_text(node, content)
            
            # Create location
            location = SourceLocation(
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                start_column=node.start_point[1],
                end_column=node.end_point[1],
                start_byte=node.start_byte,
                end_byte=node.end_byte
            )
            
            # Build metadata
            metadata = {
                "fields": fields,
                "field_count": field_count,
                "ast_node_type": node.type,
                "is_union": True
            }
            
            entity = Entity(
                id=self._generate_entity_id(EntityType.CLASS, union_name, location),
                name=union_name,
                qualified_name=union_name,
                entity_type=EntityType.CLASS,  # Union as class
                location=location,
                signature=signature,
                source_code=source_code,
                visibility=Visibility.PUBLIC,
                metadata=metadata
            )
            
            return entity
            
        except Exception as e:
            logger.warning(f"Failed to create union entity: {e}")
            return None
    
    def _create_typedef_union_entity(self, union_node: tree_sitter.Node, name_node: tree_sitter.Node, content: str, file_path: Path) -> Optional[Entity]:
        """Create a union entity from typedef union"""
        try:
            union_name = self.get_node_text(name_node, content)
            
            # Extract fields
            fields = []
            body_node = self.find_child_by_type(union_node, "field_declaration_list")
            if body_node:
                fields = self._extract_struct_fields(body_node, content)  # Same as struct fields
            
            # Create signature
            field_count = len(fields)
            signature = f"union {union_name} {{ {field_count} fields }}"
            
            # Extract source code
            source_code = self.get_node_text(union_node, content)
            
            # Create location
            location = SourceLocation(
                file_path=file_path,
                start_line=union_node.start_point[0] + 1,
                end_line=union_node.end_point[0] + 1,
                start_column=union_node.start_point[1],
                end_column=union_node.end_point[1],
                start_byte=union_node.start_byte,
                end_byte=union_node.end_byte
            )
            
            # Build metadata
            metadata = {
                "fields": fields,
                "field_count": field_count,
                "ast_node_type": union_node.type,
                "is_union": True,
                "is_typedef_union": True
            }
            
            entity = Entity(
                id=self._generate_entity_id(EntityType.CLASS, union_name, location),
                name=union_name,
                qualified_name=union_name,
                entity_type=EntityType.CLASS,  # Union as class
                location=location,
                signature=signature,
                source_code=source_code,
                visibility=Visibility.PUBLIC,
                metadata=metadata
            )
            
            return entity
            
        except Exception as e:
            logger.warning(f"Failed to create typedef union entity: {e}")
            return None
    
    def _extract_enums(self, root_node: tree_sitter.Node, content: str, file_path: Path) -> List[Entity]:
        """Extract enum declarations"""
        enums = []
        
        # Find enum declarations
        enum_nodes = self._find_nodes_by_type(root_node, ["enum_specifier"])
        
        for node in enum_nodes:
            try:
                # Only process enums with names
                name_node = self.find_child_by_type(node, "type_identifier")
                if name_node:
                    enum_entity = self._create_enum_entity(node, content, file_path)
                    if enum_entity:
                        enums.append(enum_entity)
                elif node.parent and node.parent.type == "type_definition":
                    # Handle typedef enum { ... } Name; case
                    typedef_node = node.parent
                    typedef_name_nodes = [child for child in typedef_node.children if child.type == "type_identifier"]
                    if typedef_name_nodes:
                        # Use the last type_identifier as the enum name
                        name_node = typedef_name_nodes[-1]
                        enum_entity = self._create_typedef_enum_entity(node, name_node, content, file_path)
                        if enum_entity:
                            enums.append(enum_entity)
            except Exception as e:
                logger.warning(f"Failed to extract enum: {e}")
        
        return enums
    
    def _create_enum_entity(self, node: tree_sitter.Node, content: str, file_path: Path) -> Optional[Entity]:
        """Create an enum entity from AST node"""
        try:
            # Extract enum name
            name_node = self.find_child_by_type(node, "type_identifier")
            if not name_node:
                return None
            
            enum_name = self.get_node_text(name_node, content)
            
            # Extract enum values
            values = []
            body_node = self.find_child_by_type(node, "enumerator_list")
            if body_node:
                values = self._extract_enum_values(body_node, content)
            
            # Create signature
            value_count = len(values)
            signature = f"enum {enum_name} {{ {value_count} values }}"
            
            # Extract source code
            source_code = self.get_node_text(node, content)
            
            # Create location
            location = SourceLocation(
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                start_column=node.start_point[1],
                end_column=node.end_point[1],
                start_byte=node.start_byte,
                end_byte=node.end_byte
            )
            
            # Build metadata
            metadata = {
                "values": values,
                "value_count": value_count,
                "ast_node_type": node.type,
                "is_enum": True
            }
            
            entity = Entity(
                id=self._generate_entity_id(EntityType.ENUM, enum_name, location),
                name=enum_name,
                qualified_name=enum_name,
                entity_type=EntityType.ENUM,
                location=location,
                signature=signature,
                source_code=source_code,
                visibility=Visibility.PUBLIC,
                metadata=metadata
            )
            
            return entity
            
        except Exception as e:
            logger.warning(f"Failed to create enum entity: {e}")
            return None
    
    def _extract_enum_values(self, body_node: tree_sitter.Node, content: str) -> List[str]:
        """Extract enum value information"""
        values = []
        
        for child in body_node.children:
            if child.type == "enumerator":
                value_text = self.get_node_text(child, content).strip()
                if value_text and value_text != ",":
                    values.append(value_text)
        
        return values
    
    def _create_typedef_enum_entity(self, enum_node: tree_sitter.Node, name_node: tree_sitter.Node, content: str, file_path: Path) -> Optional[Entity]:
        """Create an enum entity from typedef enum"""
        try:
            enum_name = self.get_node_text(name_node, content)
            
            # Extract enum values
            values = []
            body_node = self.find_child_by_type(enum_node, "enumerator_list")
            if body_node:
                values = self._extract_enum_values(body_node, content)
            
            # Create signature
            value_count = len(values)
            signature = f"enum {enum_name} {{ {value_count} values }}"
            
            # Extract source code
            source_code = self.get_node_text(enum_node, content)
            
            # Create location
            location = SourceLocation(
                file_path=file_path,
                start_line=enum_node.start_point[0] + 1,
                end_line=enum_node.end_point[0] + 1,
                start_column=enum_node.start_point[1],
                end_column=enum_node.end_point[1],
                start_byte=enum_node.start_byte,
                end_byte=enum_node.end_byte
            )
            
            # Build metadata
            metadata = {
                "values": values,
                "value_count": value_count,
                "ast_node_type": enum_node.type,
                "is_enum": True,
                "is_typedef_enum": True
            }
            
            entity = Entity(
                id=self._generate_entity_id(EntityType.ENUM, enum_name, location),
                name=enum_name,
                qualified_name=enum_name,
                entity_type=EntityType.ENUM,
                location=location,
                signature=signature,
                source_code=source_code,
                visibility=Visibility.PUBLIC,
                metadata=metadata
            )
            
            return entity
            
        except Exception as e:
            logger.warning(f"Failed to create typedef enum entity: {e}")
            return None
    
    def _extract_typedefs(self, root_node: tree_sitter.Node, content: str, file_path: Path) -> List[Entity]:
        """Extract typedef declarations"""
        typedefs = []
        
        # Find typedef declarations
        typedef_nodes = self._find_nodes_by_type(root_node, ["type_definition"])
        
        for node in typedef_nodes:
            try:
                typedef_entity = self._create_typedef_entity(node, content, file_path)
                if typedef_entity:
                    typedefs.append(typedef_entity)
            except Exception as e:
                logger.warning(f"Failed to extract typedef: {e}")
        
        return typedefs
    
    def _create_typedef_entity(self, node: tree_sitter.Node, content: str, file_path: Path) -> Optional[Entity]:
        """Create a typedef entity from AST node"""
        try:
            # Extract typedef name - handle both simple and function pointer typedefs
            typedef_name = None
            
            # Look for direct type_identifier children first (simple typedefs)
            name_nodes = [child for child in node.children if child.type == "type_identifier"]
            if name_nodes:
                typedef_name = self.get_node_text(name_nodes[-1], content)  # Last identifier is the new name
            else:
                # Look for function pointer typedef names in nested structure
                function_declarator = self.find_child_by_type(node, "function_declarator")
                if function_declarator:
                    # Function pointer: typedef type (*Name)(params)
                    parenthesized = self.find_child_by_type(function_declarator, "parenthesized_declarator")
                    if parenthesized:
                        pointer_declarator = self.find_child_by_type(parenthesized, "pointer_declarator")
                        if pointer_declarator:
                            name_identifier = self.find_child_by_type(pointer_declarator, "type_identifier")
                            if name_identifier:
                                typedef_name = self.get_node_text(name_identifier, content)
            
            if not typedef_name:
                return None
            
            # Extract base type - handle function pointers
            base_type = "unknown"
            for child in node.children:
                if child.type in ["primitive_type", "type_identifier", "struct_specifier", "union_specifier"]:
                    base_type = self.get_node_text(child, content)
                    break
                elif child.type == "function_declarator":
                    # Function pointer typedef
                    base_type = "function_pointer"
                    break
            
            # Create signature
            signature = f"typedef {base_type} {typedef_name}"
            
            # Extract source code
            source_code = self.get_node_text(node, content)
            
            # Create location
            location = SourceLocation(
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                start_column=node.start_point[1],
                end_column=node.end_point[1],
                start_byte=node.start_byte,
                end_byte=node.end_byte
            )
            
            # Build metadata
            metadata = {
                "base_type": base_type,
                "ast_node_type": node.type,
                "is_typedef": True
            }
            
            entity = Entity(
                id=self._generate_entity_id(EntityType.TYPE_ALIAS, typedef_name, location),
                name=typedef_name,
                qualified_name=typedef_name,
                entity_type=EntityType.TYPE_ALIAS,
                location=location,
                signature=signature,
                source_code=source_code,
                visibility=Visibility.PUBLIC,
                metadata=metadata
            )
            
            return entity
            
        except Exception as e:
            logger.warning(f"Failed to create typedef entity: {e}")
            return None
    
    def _extract_global_variables(self, root_node: tree_sitter.Node, content: str, file_path: Path) -> List[Entity]:
        """Extract global variable declarations"""
        variables = []
        
        # Find variable declarations at the top level
        declaration_nodes = self._find_nodes_by_type(root_node, ["declaration"])
        
        for node in declaration_nodes:
            try:
                # Skip if this is inside a function (not global)
                parent = node.parent
                while parent:
                    if parent.type in ["function_definition", "compound_statement"]:
                        break
                    parent = parent.parent
                else:
                    # This is a global variable - check if it has an identifier (not a function)
                    if not self.find_child_by_type(node, "function_declarator"):
                        var_entity = self._create_variable_entity(node, content, file_path)
                        if var_entity:
                            variables.append(var_entity)
            except Exception as e:
                logger.warning(f"Failed to extract variable: {e}")
        
        return variables
    
    def _create_variable_entity(self, node: tree_sitter.Node, content: str, file_path: Path) -> Optional[Entity]:
        """Create a variable entity from AST node"""
        try:
            # Extract variable name by recursively searching for identifier
            name_node = self._find_identifier_in_declarator(node)
            
            if not name_node:
                return None
            
            var_name = self.get_node_text(name_node, content)
            
            # Extract variable type by collecting type components
            var_type = self._extract_variable_type(node, content)
            
            # Create signature
            signature = f"{var_type} {var_name}"
            
            # Extract source code
            source_code = self.get_node_text(node, content)
            
            # Create location
            location = SourceLocation(
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                start_column=node.start_point[1],
                end_column=node.end_point[1],
                start_byte=node.start_byte,
                end_byte=node.end_byte
            )
            
            # Build metadata
            metadata = {
                "variable_type": var_type,
                "ast_node_type": node.type,
                "is_global": True
            }
            
            entity = Entity(
                id=self._generate_entity_id(EntityType.VARIABLE, var_name, location),
                name=var_name,
                qualified_name=var_name,
                entity_type=EntityType.VARIABLE,
                location=location,
                signature=signature,
                source_code=source_code,
                visibility=Visibility.PUBLIC,
                metadata=metadata
            )
            
            return entity
            
        except Exception as e:
            logger.warning(f"Failed to create variable entity: {e}")
            return None
    
    def _extract_macros(self, root_node: tree_sitter.Node, content: str, file_path: Path) -> List[Entity]:
        """Extract macro definitions"""
        macros = []
        
        # Find macro definitions - both simple and function-like macros
        macro_nodes = self._find_nodes_by_type(root_node, ["preproc_def", "preproc_function_def"])
        
        for node in macro_nodes:
            try:
                macro_entity = self._create_macro_entity(node, content, file_path)
                if macro_entity:
                    macros.append(macro_entity)
            except Exception as e:
                logger.warning(f"Failed to extract macro: {e}")
        
        return macros
    
    def _create_macro_entity(self, node: tree_sitter.Node, content: str, file_path: Path) -> Optional[Entity]:
        """Create a macro entity from AST node"""
        try:
            # Extract macro name
            name_node = self.find_child_by_type(node, "identifier")
            if not name_node:
                return None
            
            macro_name = self.get_node_text(name_node, content)
            
            # Handle function-like macros vs simple macros
            is_function_macro = node.type == "preproc_function_def"
            macro_params = ""
            macro_value = ""
            
            if is_function_macro:
                # Extract parameters for function-like macros
                params_node = self.find_child_by_type(node, "preproc_params")
                if params_node:
                    macro_params = self.get_node_text(params_node, content)
                
                # Extract macro body
                arg_node = self.find_child_by_type(node, "preproc_arg")
                if arg_node:
                    macro_value = self.get_node_text(arg_node, content)
            else:
                # Simple macro - extract value
                for child in node.children:
                    if child.type == "preproc_arg":
                        macro_value = self.get_node_text(child, content)
                        break
            
            # Create signature
            signature = f"#define {macro_name}"
            if macro_params:
                signature += macro_params  # Already includes parentheses
            if macro_value:
                signature += f" {macro_value}"
            
            # Extract source code
            source_code = self.get_node_text(node, content)
            
            # Create location
            location = SourceLocation(
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                start_column=node.start_point[1],
                end_column=node.end_point[1],
                start_byte=node.start_byte,
                end_byte=node.end_byte
            )
            
            # Build metadata
            metadata = {
                "macro_value": macro_value,
                "ast_node_type": node.type,
                "is_macro": True,
                "is_function_macro": is_function_macro,
                "macro_params": macro_params if is_function_macro else None
            }
            
            entity = Entity(
                id=self._generate_entity_id(EntityType.CONSTANT, macro_name, location),
                name=macro_name,
                qualified_name=macro_name,
                entity_type=EntityType.CONSTANT,  # Macro as constant
                location=location,
                signature=signature,
                source_code=source_code,
                visibility=Visibility.PUBLIC,
                metadata=metadata
            )
            
            return entity
            
        except Exception as e:
            logger.warning(f"Failed to create macro entity: {e}")
            return None
    
    def _extract_includes(self, root_node: tree_sitter.Node, content: str, file_path: Path) -> List[Entity]:
        """Extract include statements"""
        includes = []
        
        # Find include statements
        include_nodes = self._find_nodes_by_type(root_node, ["preproc_include"])
        
        for node in include_nodes:
            try:
                include_entity = self._create_include_entity(node, content, file_path)
                if include_entity:
                    includes.append(include_entity)
            except Exception as e:
                logger.warning(f"Failed to extract include: {e}")
        
        return includes
    
    def _create_include_entity(self, node: tree_sitter.Node, content: str, file_path: Path) -> Optional[Entity]:
        """Create an include entity from AST node"""
        try:
            # Extract include path
            path_node = None
            for child in node.children:
                if child.type in ["string_literal", "system_lib_string"]:
                    path_node = child
                    break
            
            if not path_node:
                return None
            
            include_path = self.get_node_text(path_node, content)
            
            # Clean up the path (remove quotes or brackets)
            clean_path = include_path.strip('"<>')
            
            # Create signature
            signature = f"#include {include_path}"
            
            # Extract source code
            source_code = self.get_node_text(node, content)
            
            # Create location
            location = SourceLocation(
                file_path=file_path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                start_column=node.start_point[1],
                end_column=node.end_point[1],
                start_byte=node.start_byte,
                end_byte=node.end_byte
            )
            
            # Determine if it's a system include or local include
            is_system = include_path.startswith('<')
            
            # Build metadata
            metadata = {
                "include_path": clean_path,
                "is_system_include": is_system,
                "is_local_include": not is_system,
                "ast_node_type": node.type
            }
            
            entity = Entity(
                id=self._generate_entity_id(EntityType.IMPORT, clean_path, location),
                name=clean_path,
                qualified_name=clean_path,
                entity_type=EntityType.IMPORT,
                location=location,
                signature=signature,
                source_code=source_code,
                visibility=Visibility.PUBLIC,
                metadata=metadata
            )
            
            return entity
            
        except Exception as e:
            logger.warning(f"Failed to create include entity: {e}")
            return None
    
    # Relation extraction methods
    
    def _extract_function_calls(
        self, 
        root_node: tree_sitter.Node, 
        entities: List[Entity], 
        entity_lookup: Dict[str, Entity], 
        content: str, 
        file_path: Path
    ) -> List[Relation]:
        """Extract function call relationships"""
        relations = []
        
        # Find function call expressions
        call_nodes = self._find_nodes_by_type(root_node, ["call_expression"])
        
        for node in call_nodes:
            try:
                # Find the function being called
                func_name_node = self.find_child_by_type(node, "identifier")
                if func_name_node:
                    called_func_name = self.get_node_text(func_name_node, content)
                    
                    # Find the calling function context
                    calling_func = self._find_containing_function(node, entities, content)
                    
                    if calling_func and called_func_name in entity_lookup:
                        called_func = entity_lookup[called_func_name]
                        
                        relation = Relation.create_call_relation(
                            calling_func.id,
                            called_func.id,
                            context=f"Function {calling_func.name} calls {called_func_name}",
                            location=self._create_source_location(node, file_path)
                        )
                        relations.append(relation)
                        
            except Exception as e:
                logger.warning(f"Failed to extract function call relation: {e}")
        
        return relations
    
    def _extract_type_usage_relations(
        self, 
        root_node: tree_sitter.Node, 
        entities: List[Entity], 
        entity_lookup: Dict[str, Entity], 
        content: str, 
        file_path: Path
    ) -> List[Relation]:
        """Extract type usage relationships"""
        relations = []
        
        # Find both global and local variable declarations that use custom types
        # Look for declarations and init_declarator nodes
        declaration_nodes = self._find_nodes_by_type(root_node, ["declaration", "init_declarator"])
        
        for node in declaration_nodes:
            try:
                # Handle both global declarations and local variable declarations
                type_name_node = None
                var_name_node = None
                
                if node.type == "declaration":
                    # Global or function parameter declarations
                    type_name_node = self.find_child_by_type(node, "type_identifier")
                    init_declarator = self.find_child_by_type(node, "init_declarator")
                    if init_declarator:
                        var_name_node = self.find_child_by_type(init_declarator, "identifier") or self._find_identifier_in_declarator(init_declarator)
                    else:
                        var_name_node = self.find_child_by_type(node, "identifier")
                
                elif node.type == "init_declarator":
                    # Local variable in function body - look at parent declaration
                    parent = node.parent
                    if parent and parent.type == "declaration":
                        type_name_node = self.find_child_by_type(parent, "type_identifier")
                        var_name_node = self.find_child_by_type(node, "identifier") or self._find_identifier_in_declarator(node)
                
                if type_name_node and var_name_node:
                    type_name = self.get_node_text(type_name_node, content)
                    var_name = self.get_node_text(var_name_node, content)
                    
                    if type_name in entity_lookup:
                        type_entity = entity_lookup[type_name]
                        
                        # Find the function containing this variable (if any)
                        source_entity_id = f"c::variable::{var_name}"
                        
                        # Create a uses relation
                        relation_id = f"uses::{var_name}::{type_name}::{node.start_point[0]}"
                        relation = Relation(
                            id=relation_id,
                            relation_type=RelationType.USES_TYPE,
                            source_entity_id=source_entity_id,
                            target_entity_id=type_entity.id,
                            context=f"Variable {var_name} uses type {type_name}",
                            location=self._create_source_location(node, file_path)
                        )
                        relations.append(relation)
                        
            except Exception as e:
                logger.warning(f"Failed to extract type usage relation: {e}")
        
        return relations
    
    def _extract_include_relations(
        self, 
        root_node: tree_sitter.Node, 
        entities: List[Entity], 
        content: str, 
        file_path: Path
    ) -> List[Relation]:
        """Extract include/import relationships"""
        relations = []
        
        # Find include entities
        include_entities = [e for e in entities if e.entity_type == EntityType.IMPORT]
        
        for include_entity in include_entities:
            try:
                # Create an import relation
                relation = Relation.create_import_relation(
                    f"c::module::{file_path.stem}",
                    include_entity.id,
                    context=f"File {file_path.name} includes {include_entity.name}",
                    location=include_entity.location
                )
                relations.append(relation)
                
            except Exception as e:
                logger.warning(f"Failed to extract include relation: {e}")
        
        return relations
    
    def _find_containing_function(self, node: tree_sitter.Node, entities: List[Entity], content: str) -> Optional[Entity]:
        """Find the function that contains the given node"""
        current = node.parent
        
        while current:
            if current.type == "function_definition":
                # Extract function name from this node
                name_node = self.find_child_by_type(current, "function_declarator")
                if name_node:
                    identifier_node = self.find_child_by_type(name_node, "identifier")
                    if identifier_node:
                        func_name = self.get_node_text(identifier_node, content)
                        # Find the entity with this name
                        for entity in entities:
                            if entity.name == func_name and entity.entity_type == EntityType.FUNCTION:
                                return entity
            current = current.parent
        
        return None
    
    def _build_entity_lookup(self, entities: List[Entity]) -> Dict[str, Entity]:
        """Build a lookup dictionary for entities by name"""
        lookup = {}
        for entity in entities:
            lookup[entity.name] = entity
        return lookup
    
    def _find_nodes_by_type(self, root_node: tree_sitter.Node, node_types: List[str]) -> List[tree_sitter.Node]:
        """Find all nodes of specified types in the AST"""
        matching_nodes = []
        
        def visit_node(node: tree_sitter.Node):
            if node.type in node_types:
                matching_nodes.append(node)
            for child in node.children:
                visit_node(child)
        
        visit_node(root_node)
        return matching_nodes
    
    def _find_child_by_type(self, node: tree_sitter.Node, child_type: str) -> Optional[tree_sitter.Node]:
        """Find first child node of specified type"""
        for child in node.children:
            if child.type == child_type:
                return child
        return None
    
    def _get_node_text(self, node: tree_sitter.Node, content: str) -> str:
        """Get text content of a node"""
        return content[node.start_byte:node.end_byte]
    
    def _create_source_location(self, node: tree_sitter.Node, file_path: Path) -> SourceLocation:
        """Create a SourceLocation from a node"""
        return SourceLocation(
            file_path=file_path,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            start_column=node.start_point[1],
            end_column=node.end_point[1],
            start_byte=node.start_byte,
            end_byte=node.end_byte
        )
    
    def _generate_entity_id(self, entity_type: EntityType, name: str, location: SourceLocation) -> str:
        """Generate a unique entity ID"""
        return f"c::{entity_type.value}::{name}::{location.start_line}"
    
    def _find_identifier_in_declarator(self, node: tree_sitter.Node) -> Optional[tree_sitter.Node]:
        """Recursively find identifier node in complex declarator structures"""
        # For complex declarations like "const char* app_name", the identifier
        # may be nested within pointer_declarator or other declarator types
        
        # Direct identifier
        if node.type == "identifier":
            return node
        
        # Check children for identifier
        for child in node.children:
            if child.type == "identifier":
                return child
            # Recursively search in nested declarators
            elif child.type in ["pointer_declarator", "declarator", "init_declarator"]:
                identifier = self._find_identifier_in_declarator(child)
                if identifier:
                    return identifier
        
        return None
    
    def _extract_variable_type(self, node: tree_sitter.Node, content: str) -> str:
        """Extract variable type from declaration node"""
        # Look for type specifiers in the declaration
        type_parts = []
        
        # Find primitive_type, type_identifier, or storage class specifiers
        for child in node.children:
            if child.type in ["primitive_type", "type_identifier", "storage_class_specifier"]:
                type_parts.append(self.get_node_text(child, content))
            elif child.type == "type_qualifier":
                # Handle const, volatile, etc.
                type_parts.append(self.get_node_text(child, content))
        
        # Handle pointer declarations
        pointer_declarator = self.find_child_by_type(node, "init_declarator")
        if pointer_declarator:
            declarator = self.find_child_by_type(pointer_declarator, "pointer_declarator")
            if declarator:
                # Count asterisks for pointer level
                asterisk_count = self.get_node_text(declarator, content).count("*")
                if asterisk_count > 0:
                    type_parts.append("*" * asterisk_count)
        
        return " ".join(type_parts) if type_parts else "unknown"