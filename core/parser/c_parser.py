"""
C parser for comprehensive code analysis using Tree-sitter.

Extracts C language constructs including functions, structs, unions, enums,
typedefs, global variables, macros, and their relationships.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import tree_sitter

from .base_tree_sitter import TreeSitterBase, TreeSitterParseResult
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
        
        # Query for function declarations and definitions
        query = self.language.query("""
            (function_definition
                type: (primitive_type) @return_type
                declarator: (function_declarator
                    declarator: (identifier) @name
                    parameters: (parameter_list) @params
                )
                body: (compound_statement) @body
            ) @function
            
            (declaration
                type: (primitive_type) @return_type
                declarator: (function_declarator
                    declarator: (identifier) @name
                    parameters: (parameter_list) @params
                )
            ) @function_decl
        """)
        
        captures = query.captures(root_node)
        processed_functions = set()
        
        for node, capture_name in captures:
            if capture_name in ["function", "function_decl"]:
                try:
                    func_entity = self._create_function_entity(node, content, file_path, capture_name == "function")
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
            name_node = self._find_child_by_type(node, "function_declarator")
            if not name_node:
                return None
            
            identifier_node = self._find_child_by_type(name_node, "identifier")
            if not identifier_node:
                return None
            
            func_name = self._get_node_text(identifier_node, content)
            
            # Extract return type
            return_type = "void"
            type_nodes = [child for child in node.children if child.type in ["primitive_type", "type_identifier"]]
            if type_nodes:
                return_type = self._get_node_text(type_nodes[0], content)
            
            # Extract parameters
            params_node = self._find_child_by_type(name_node, "parameter_list")
            parameters = []
            if params_node:
                parameters = self._extract_function_parameters(params_node, content)
            
            # Create signature
            param_str = ", ".join(parameters)
            signature = f"{return_type} {func_name}({param_str})"
            
            # Extract source code
            source_code = self._get_node_text(node, content)
            
            # Create location
            location = self._create_source_location(node, file_path)
            
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
                id=self._generate_entity_id(entity_type, func_name, location),
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
                param_text = self._get_node_text(child, content).strip()
                if param_text and param_text != ",":
                    parameters.append(param_text)
        
        return parameters
    
    def _extract_structs(self, root_node: tree_sitter.Node, content: str, file_path: Path) -> List[Entity]:
        """Extract struct declarations"""
        structs = []
        
        query = self.language.query("""
            (struct_specifier
                name: (type_identifier) @name
                body: (field_declaration_list) @body
            ) @struct
        """)
        
        captures = query.captures(root_node)
        
        for node, capture_name in captures:
            if capture_name == "struct":
                try:
                    struct_entity = self._create_struct_entity(node, content, file_path)
                    if struct_entity:
                        structs.append(struct_entity)
                except Exception as e:
                    logger.warning(f"Failed to extract struct: {e}")
        
        return structs
    
    def _create_struct_entity(self, node: tree_sitter.Node, content: str, file_path: Path) -> Optional[Entity]:
        """Create a struct entity from AST node"""
        try:
            # Extract struct name
            name_node = self._find_child_by_type(node, "type_identifier")
            if not name_node:
                return None
            
            struct_name = self._get_node_text(name_node, content)
            
            # Extract fields
            fields = []
            body_node = self._find_child_by_type(node, "field_declaration_list")
            if body_node:
                fields = self._extract_struct_fields(body_node, content)
            
            # Create signature
            field_count = len(fields)
            signature = f"struct {struct_name} {{ {field_count} fields }}"
            
            # Extract source code
            source_code = self._get_node_text(node, content)
            
            # Create location
            location = self._create_source_location(node, file_path)
            
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
                field_text = self._get_node_text(child, content).strip()
                if field_text and not field_text.startswith("{") and not field_text.startswith("}"):
                    # Clean up the field text
                    field_text = field_text.replace(";", "").strip()
                    if field_text:
                        fields.append(field_text)
        
        return fields
    
    def _extract_unions(self, root_node: tree_sitter.Node, content: str, file_path: Path) -> List[Entity]:
        """Extract union declarations"""
        unions = []
        
        query = self.language.query("""
            (union_specifier
                name: (type_identifier) @name
                body: (field_declaration_list) @body
            ) @union
        """)
        
        captures = query.captures(root_node)
        
        for node, capture_name in captures:
            if capture_name == "union":
                try:
                    union_entity = self._create_union_entity(node, content, file_path)
                    if union_entity:
                        unions.append(union_entity)
                except Exception as e:
                    logger.warning(f"Failed to extract union: {e}")
        
        return unions
    
    def _create_union_entity(self, node: tree_sitter.Node, content: str, file_path: Path) -> Optional[Entity]:
        """Create a union entity from AST node"""
        try:
            # Extract union name
            name_node = self._find_child_by_type(node, "type_identifier")
            if not name_node:
                return None
            
            union_name = self._get_node_text(name_node, content)
            
            # Extract fields
            fields = []
            body_node = self._find_child_by_type(node, "field_declaration_list")
            if body_node:
                fields = self._extract_struct_fields(body_node, content)  # Same as struct fields
            
            # Create signature
            field_count = len(fields)
            signature = f"union {union_name} {{ {field_count} fields }}"
            
            # Extract source code
            source_code = self._get_node_text(node, content)
            
            # Create location
            location = self._create_source_location(node, file_path)
            
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
    
    def _extract_enums(self, root_node: tree_sitter.Node, content: str, file_path: Path) -> List[Entity]:
        """Extract enum declarations"""
        enums = []
        
        query = self.language.query("""
            (enum_specifier
                name: (type_identifier) @name
                body: (enumerator_list) @body
            ) @enum
        """)
        
        captures = query.captures(root_node)
        
        for node, capture_name in captures:
            if capture_name == "enum":
                try:
                    enum_entity = self._create_enum_entity(node, content, file_path)
                    if enum_entity:
                        enums.append(enum_entity)
                except Exception as e:
                    logger.warning(f"Failed to extract enum: {e}")
        
        return enums
    
    def _create_enum_entity(self, node: tree_sitter.Node, content: str, file_path: Path) -> Optional[Entity]:
        """Create an enum entity from AST node"""
        try:
            # Extract enum name
            name_node = self._find_child_by_type(node, "type_identifier")
            if not name_node:
                return None
            
            enum_name = self._get_node_text(name_node, content)
            
            # Extract enum values
            values = []
            body_node = self._find_child_by_type(node, "enumerator_list")
            if body_node:
                values = self._extract_enum_values(body_node, content)
            
            # Create signature
            value_count = len(values)
            signature = f"enum {enum_name} {{ {value_count} values }}"
            
            # Extract source code
            source_code = self._get_node_text(node, content)
            
            # Create location
            location = self._create_source_location(node, file_path)
            
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
                value_text = self._get_node_text(child, content).strip()
                if value_text and value_text != ",":
                    values.append(value_text)
        
        return values
    
    def _extract_typedefs(self, root_node: tree_sitter.Node, content: str, file_path: Path) -> List[Entity]:
        """Extract typedef declarations"""
        typedefs = []
        
        query = self.language.query("""
            (type_definition
                type: (_) @base_type
                declarator: (type_identifier) @name
            ) @typedef
        """)
        
        captures = query.captures(root_node)
        
        for node, capture_name in captures:
            if capture_name == "typedef":
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
            # Extract typedef name
            name_nodes = [child for child in node.children if child.type == "type_identifier"]
            if not name_nodes:
                return None
            
            typedef_name = self._get_node_text(name_nodes[-1], content)  # Last identifier is the new name
            
            # Extract base type
            base_type = "unknown"
            for child in node.children:
                if child.type in ["primitive_type", "type_identifier", "struct_specifier", "union_specifier"]:
                    base_type = self._get_node_text(child, content)
                    break
            
            # Create signature
            signature = f"typedef {base_type} {typedef_name}"
            
            # Extract source code
            source_code = self._get_node_text(node, content)
            
            # Create location
            location = self._create_source_location(node, file_path)
            
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
        
        query = self.language.query("""
            (declaration
                type: (_) @type
                declarator: (identifier) @name
            ) @variable
        """)
        
        captures = query.captures(root_node)
        
        for node, capture_name in captures:
            if capture_name == "variable":
                # Skip if this is inside a function (not global)
                parent = node.parent
                while parent:
                    if parent.type in ["function_definition", "compound_statement"]:
                        break
                    parent = parent.parent
                else:
                    # This is a global variable
                    try:
                        var_entity = self._create_variable_entity(node, content, file_path)
                        if var_entity:
                            variables.append(var_entity)
                    except Exception as e:
                        logger.warning(f"Failed to extract variable: {e}")
        
        return variables
    
    def _create_variable_entity(self, node: tree_sitter.Node, content: str, file_path: Path) -> Optional[Entity]:
        """Create a variable entity from AST node"""
        try:
            # Extract variable name
            name_node = None
            for child in node.children:
                if child.type == "identifier":
                    name_node = child
                    break
                elif child.type in ["init_declarator", "declarator"]:
                    # Look for identifier in declarator
                    for subchild in child.children:
                        if subchild.type == "identifier":
                            name_node = subchild
                            break
            
            if not name_node:
                return None
            
            var_name = self._get_node_text(name_node, content)
            
            # Extract variable type
            var_type = "unknown"
            for child in node.children:
                if child.type in ["primitive_type", "type_identifier"]:
                    var_type = self._get_node_text(child, content)
                    break
            
            # Create signature
            signature = f"{var_type} {var_name}"
            
            # Extract source code
            source_code = self._get_node_text(node, content)
            
            # Create location
            location = self._create_source_location(node, file_path)
            
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
        
        query = self.language.query("""
            (preproc_def
                name: (identifier) @name
                value: (_)? @value
            ) @macro
        """)
        
        captures = query.captures(root_node)
        
        for node, capture_name in captures:
            if capture_name == "macro":
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
            name_node = self._find_child_by_type(node, "identifier")
            if not name_node:
                return None
            
            macro_name = self._get_node_text(name_node, content)
            
            # Extract macro value
            macro_value = ""
            for child in node.children:
                if child.type not in ["#", "identifier"]:  # Skip the # and name
                    macro_value = self._get_node_text(child, content)
                    break
            
            # Create signature
            signature = f"#define {macro_name}"
            if macro_value:
                signature += f" {macro_value}"
            
            # Extract source code
            source_code = self._get_node_text(node, content)
            
            # Create location
            location = self._create_source_location(node, file_path)
            
            # Build metadata
            metadata = {
                "macro_value": macro_value,
                "ast_node_type": node.type,
                "is_macro": True
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
        
        query = self.language.query("""
            (preproc_include
                path: (_) @path
            ) @include
        """)
        
        captures = query.captures(root_node)
        
        for node, capture_name in captures:
            if capture_name == "include":
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
            
            include_path = self._get_node_text(path_node, content)
            
            # Clean up the path (remove quotes or brackets)
            clean_path = include_path.strip('"<>')
            
            # Create signature
            signature = f"#include {include_path}"
            
            # Extract source code
            source_code = self._get_node_text(node, content)
            
            # Create location
            location = self._create_source_location(node, file_path)
            
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
        
        query = self.language.query("""
            (call_expression
                function: (identifier) @func_name
            ) @call
        """)
        
        captures = query.captures(root_node)
        
        for node, capture_name in captures:
            if capture_name == "call":
                try:
                    # Find the function being called
                    func_name_node = self._find_child_by_type(node, "identifier")
                    if func_name_node:
                        called_func_name = self._get_node_text(func_name_node, content)
                        
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
        
        query = self.language.query("""
            (declaration
                type: (type_identifier) @type_name
            ) @decl
        """)
        
        captures = query.captures(root_node)
        
        for node, capture_name in captures:
            if capture_name == "decl":
                try:
                    type_name_node = self._find_child_by_type(node, "type_identifier")
                    if type_name_node:
                        type_name = self._get_node_text(type_name_node, content)
                        
                        # Find the variable being declared
                        var_name_node = self._find_child_by_type(node, "identifier")
                        if var_name_node and type_name in entity_lookup:
                            var_name = self._get_node_text(var_name_node, content)
                            type_entity = entity_lookup[type_name]
                            
                            # Create a uses relation
                            relation_id = f"uses::{var_name}::{type_name}::{node.start_point[0]}"
                            relation = Relation(
                                id=relation_id,
                                relation_type=RelationType.USES,
                                source_entity_id=f"c::variable::{var_name}",
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
                name_node = self._find_child_by_type(current, "function_declarator")
                if name_node:
                    identifier_node = self._find_child_by_type(name_node, "identifier")
                    if identifier_node:
                        func_name = self._get_node_text(identifier_node, content)
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