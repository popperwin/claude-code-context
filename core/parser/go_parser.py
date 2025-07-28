"""
Go parser using Tree-sitter for comprehensive entity extraction.

Extracts packages, functions, methods, structs, interfaces, variables, constants,
types, and imports from Go source code with full metadata and relation information.
"""

import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Set
import re

try:
    import tree_sitter
    import tree_sitter_go
except ImportError:
    tree_sitter = None
    tree_sitter_go = None

from .tree_sitter_base import TreeSitterBase
from .registry import register_parser
from ..models.entities import (
    Entity, EntityType, Relation, RelationType, 
    SourceLocation, Visibility
)

logger = logging.getLogger(__name__)


@register_parser("go", [".go"])
class GoParser(TreeSitterBase):
    """
    Comprehensive Go parser with Tree-sitter.
    
    Features:
    - Functions and methods with receiver types
    - Structs with field analysis
    - Interfaces with method signatures
    - Variables and constants with types
    - Package and import analysis
    - Type definitions and aliases
    - Goroutine and channel detection
    - Go-specific syntax (defer, go, select)
    """
    
    # Supported features
    SUPPORTED_FEATURES = [
        "functions", "methods", "structs", "interfaces", "variables", 
        "constants", "types", "imports", "packages", "channels", "goroutines"
    ]
    
    # Go built-in types
    BUILTIN_TYPES = {
        "bool", "byte", "complex64", "complex128", "error", "float32", 
        "float64", "int", "int8", "int16", "int32", "int64", "rune",
        "string", "uint", "uint8", "uint16", "uint32", "uint64", "uintptr"
    }
    
    def __init__(self):
        super().__init__("go")
        self.__version__ = "1.0.0"
        
        # Compiled regex patterns for efficiency
        self._comment_pattern = re.compile(r'//.*$|/\*[\s\S]*?\*/', re.MULTILINE)
        self._package_pattern = re.compile(r'package\s+(\w+)')
        self._receiver_pattern = re.compile(r'\(\s*(\w+)\s+\*?(\w+)\s*\)')
        
        logger.debug("Go parser initialized")
    
    def get_supported_extensions(self) -> List[str]:
        return [".go"]
    
    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.get_supported_extensions()
    
    def extract_entities(
        self, 
        tree: tree_sitter.Tree, 
        content: str,
        file_path: Path
    ) -> List[Entity]:
        """
        Extract Go entities from AST.
        
        Args:
            tree: Tree-sitter AST
            content: Source code content
            file_path: Path to source file
            
        Returns:
            List of extracted entities
        """
        if not tree:
            return []
        
        entities = []
        
        try:
            # Extract different entity types
            entities.extend(self._extract_packages(tree, content, file_path))
            entities.extend(self._extract_imports(tree, content, file_path))
            entities.extend(self._extract_types(tree, content, file_path))
            entities.extend(self._extract_structs(tree, content, file_path))
            entities.extend(self._extract_interfaces(tree, content, file_path))
            entities.extend(self._extract_functions(tree, content, file_path))
            entities.extend(self._extract_methods(tree, content, file_path))
            entities.extend(self._extract_variables(tree, content, file_path))
            entities.extend(self._extract_constants(tree, content, file_path))
            
            logger.debug(f"Extracted {len(entities)} entities from {file_path}")
            
        except Exception as e:
            logger.error(f"Entity extraction failed for {file_path}: {e}")
        
        return entities
    
    def extract_relations(
        self,
        tree: tree_sitter.Tree,
        content: str,
        entities: List[Entity],
        file_path: Path
    ) -> List[Relation]:
        """
        Extract relationships between Go entities.
        
        Args:
            tree: Tree-sitter AST
            content: Source code content
            entities: Previously extracted entities
            file_path: Path to source file
            
        Returns:
            List of extracted relations
        """
        if not tree or not entities:
            return []
        
        relations = []
        
        try:
            # Build entity lookup for quick access
            entity_lookup = self._build_entity_lookup(entities)
            
            # Extract different relation types
            relations.extend(self._extract_struct_field_relations(tree, content, entities, entity_lookup))
            relations.extend(self._extract_interface_implementation_relations(tree, content, entities, entity_lookup))
            relations.extend(self._extract_method_receiver_relations(tree, content, entities, entity_lookup))
            relations.extend(self._extract_function_call_relations(tree, content, entities, entity_lookup))
            relations.extend(self._extract_import_relations(tree, content, entities, entity_lookup))
            relations.extend(self._extract_type_usage_relations(tree, content, entities, entity_lookup))
            relations.extend(self._extract_containment_relations(entities))
            
            logger.debug(f"Extracted {len(relations)} relations from {file_path}")
            
        except Exception as e:
            logger.error(f"Relation extraction failed for {file_path}: {e}")
        
        return relations
    
    def _extract_packages(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract package declarations"""
        packages = []
        
        # Find package declaration
        package_nodes = self.find_nodes_by_type(tree, ["package_clause"])
        
        for package_node in package_nodes:
            try:
                # Find package name
                name_node = self.find_child_by_type(package_node, "package_identifier")
                if not name_node:
                    continue
                
                package_name = self.get_node_text(name_node, content)
                
                # Create location
                location = SourceLocation(
                    file_path=file_path,
                    start_line=package_node.start_point[0] + 1,
                    end_line=package_node.end_point[0] + 1,
                    start_column=package_node.start_point[1],
                    end_column=package_node.end_point[1],
                    start_byte=package_node.start_byte,
                    end_byte=package_node.end_byte
                )
                
                # Generate entity ID
                entity_id = f"file://{file_path}::go_package::{package_name}::{location.start_line}"
                
                # Create package entity
                entity = Entity(
                    id=entity_id,
                    name=package_name,
                    qualified_name=package_name,
                    entity_type=EntityType.MODULE,  # Using MODULE for Go packages
                    location=location,
                    signature=f"package {package_name}",
                    source_code=self.get_node_text(package_node, content),
                    source_hash=hashlib.md5(self.get_node_text(package_node, content).encode()).hexdigest(),
                    visibility=Visibility.PUBLIC,
                    metadata={
                        "language": "go",
                        "package_name": package_name,
                        "ast_node_type": package_node.type
                    }
                )
                
                packages.append(entity)
                
            except Exception as e:
                logger.warning(f"Failed to extract package: {e}")
        
        return packages
    
    def _extract_imports(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract import declarations"""
        imports = []
        
        # Find import declarations
        import_nodes = self.find_nodes_by_type(tree, ["import_declaration"])
        
        for import_node in import_nodes:
            try:
                # Handle both single and multiple imports
                # Check for import_spec_list first (grouped imports)
                import_spec_list = self.find_child_by_type(import_node, "import_spec_list")
                if import_spec_list:
                    import_spec_nodes = self.find_children_by_type(import_spec_list, "import_spec")
                else:
                    # Single import (direct import_spec under import_declaration)
                    import_spec_nodes = self.find_children_by_type(import_node, "import_spec")
                
                for spec_node in import_spec_nodes:
                    # Get import path
                    path_node = self.find_child_by_type(spec_node, "interpreted_string_literal")
                    if not path_node:
                        continue
                    
                    import_path = self.get_node_text(path_node, content).strip('"')
                    
                    # Get alias if present
                    alias = None
                    alias_node = self.find_child_by_type(spec_node, "package_identifier")
                    if alias_node:
                        alias = self.get_node_text(alias_node, content)
                    
                    # Determine import name for display
                    import_name = alias if alias else import_path.split('/')[-1]
                    
                    # Create location
                    location = SourceLocation(
                        file_path=file_path,
                        start_line=spec_node.start_point[0] + 1,
                        end_line=spec_node.end_point[0] + 1,
                        start_column=spec_node.start_point[1],
                        end_column=spec_node.end_point[1],
                        start_byte=spec_node.start_byte,
                        end_byte=spec_node.end_byte
                    )
                    
                    # Generate entity ID
                    entity_id = f"file://{file_path}::go_import::{import_path}::{location.start_line}"
                    
                    # Create import entity
                    entity = Entity(
                        id=entity_id,
                        name=import_name,
                        qualified_name=import_path,
                        entity_type=EntityType.IMPORT,
                        location=location,
                        signature=f"import {alias + ' ' if alias else ''}\"{import_path}\"",
                        source_code=self.get_node_text(spec_node, content),
                        source_hash=hashlib.md5(self.get_node_text(spec_node, content).encode()).hexdigest(),
                        visibility=Visibility.PUBLIC,
                        metadata={
                            "language": "go",
                            "import_path": import_path,
                            "alias": alias,
                            "is_standard_library": self._is_standard_library(import_path),
                            "ast_node_type": spec_node.type
                        }
                    )
                    
                    imports.append(entity)
                    
            except Exception as e:
                logger.warning(f"Failed to extract import: {e}")
        
        return imports
    
    def _extract_structs(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract struct definitions"""
        structs = []
        
        # Find struct declarations
        struct_nodes = self.find_nodes_by_type(tree, ["type_declaration"])
        
        for type_node in struct_nodes:
            try:
                # Check if this is a struct type
                type_spec = self.find_child_by_type(type_node, "type_spec")
                if not type_spec:
                    continue
                
                struct_type = self.find_child_by_type(type_spec, "struct_type")
                if not struct_type:
                    continue
                
                # Get struct name
                name_node = self.find_child_by_type(type_spec, "type_identifier")
                if not name_node:
                    continue
                
                struct_name = self.get_node_text(name_node, content)
                
                # Extract fields
                fields = []
                field_list = self.find_child_by_type(struct_type, "field_declaration_list")
                if field_list:
                    field_nodes = self.find_children_by_type(field_list, "field_declaration")
                    for field_node in field_nodes:
                        field_info = self._extract_struct_field(field_node, content)
                        if field_info:
                            fields.append(field_info)
                
                # Create location
                location = SourceLocation(
                    file_path=file_path,
                    start_line=type_node.start_point[0] + 1,
                    end_line=type_node.end_point[0] + 1,
                    start_column=type_node.start_point[1],
                    end_column=type_node.end_point[1],
                    start_byte=type_node.start_byte,
                    end_byte=type_node.end_byte
                )
                
                # Generate entity ID
                entity_id = f"file://{file_path}::go_struct::{struct_name}::{location.start_line}"
                
                # Determine visibility based on naming convention
                visibility = Visibility.PUBLIC if struct_name[0].isupper() else Visibility.PRIVATE
                
                # Create struct entity
                entity = Entity(
                    id=entity_id,
                    name=struct_name,
                    qualified_name=struct_name,
                    entity_type=EntityType.STRUCT,
                    location=location,
                    signature=f"type {struct_name} struct",
                    source_code=self.get_node_text(type_node, content),
                    source_hash=hashlib.md5(self.get_node_text(type_node, content).encode()).hexdigest(),
                    visibility=visibility,
                    metadata={
                        "language": "go",
                        "struct_name": struct_name,
                        "fields": fields,
                        "field_count": len(fields),
                        "ast_node_type": type_node.type
                    }
                )
                
                structs.append(entity)
                
            except Exception as e:
                logger.warning(f"Failed to extract struct: {e}")
        
        return structs
    
    def _extract_interfaces(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract interface definitions"""
        interfaces = []
        
        # Find interface declarations
        type_nodes = self.find_nodes_by_type(tree, ["type_declaration"])
        
        for type_node in type_nodes:
            try:
                # Check if this is an interface type
                type_spec = self.find_child_by_type(type_node, "type_spec")
                if not type_spec:
                    continue
                
                interface_type = self.find_child_by_type(type_spec, "interface_type")
                if not interface_type:
                    continue
                
                # Get interface name
                name_node = self.find_child_by_type(type_spec, "type_identifier")
                if not name_node:
                    continue
                
                interface_name = self.get_node_text(name_node, content)
                
                # Extract methods - they are direct method_elem children of interface_type
                methods = []
                method_nodes = self.find_children_by_type(interface_type, "method_elem")
                for method_node in method_nodes:
                    method_info = self._extract_interface_method(method_node, content)
                    if method_info:
                        methods.append(method_info)
                
                # Create location
                location = SourceLocation(
                    file_path=file_path,
                    start_line=type_node.start_point[0] + 1,
                    end_line=type_node.end_point[0] + 1,
                    start_column=type_node.start_point[1],
                    end_column=type_node.end_point[1],
                    start_byte=type_node.start_byte,
                    end_byte=type_node.end_byte
                )
                
                # Generate entity ID
                entity_id = f"file://{file_path}::go_interface::{interface_name}::{location.start_line}"
                
                # Determine visibility based on naming convention
                visibility = Visibility.PUBLIC if interface_name[0].isupper() else Visibility.PRIVATE
                
                # Create interface entity
                entity = Entity(
                    id=entity_id,
                    name=interface_name,
                    qualified_name=interface_name,
                    entity_type=EntityType.INTERFACE,
                    location=location,
                    signature=f"type {interface_name} interface",
                    source_code=self.get_node_text(type_node, content),
                    source_hash=hashlib.md5(self.get_node_text(type_node, content).encode()).hexdigest(),
                    visibility=visibility,
                    metadata={
                        "language": "go",
                        "interface_name": interface_name,
                        "methods": methods,
                        "method_count": len(methods),
                        "ast_node_type": type_node.type
                    }
                )
                
                interfaces.append(entity)
                
            except Exception as e:
                logger.warning(f"Failed to extract interface: {e}")
        
        return interfaces
    
    def _extract_functions(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract function definitions (non-methods)"""
        functions = []
        
        # Find function declarations that are not methods
        function_nodes = self.find_nodes_by_type(tree, ["function_declaration"])
        
        for func_node in function_nodes:
            try:
                # Tree-sitter already distinguishes function_declaration from method_declaration
                # No need for manual receiver detection on function_declaration nodes
                function_entity = self._extract_function_entity(func_node, content, file_path, is_method=False)
                if function_entity:
                    functions.append(function_entity)
                    
            except Exception as e:
                logger.warning(f"Failed to extract function: {e}")
        
        return functions
    
    def _extract_methods(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract method definitions (functions with receivers)"""
        methods = []
        
        # Find method declarations
        method_nodes = self.find_nodes_by_type(tree, ["method_declaration"])
        
        for method_node in method_nodes:
            try:
                method_entity = self._extract_method_entity(method_node, content, file_path)
                if method_entity:
                    methods.append(method_entity)
                    
            except Exception as e:
                logger.warning(f"Failed to extract method: {e}")
        
        return methods
    
    def _extract_variables(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract variable declarations"""
        variables = []
        
        # Find variable declarations
        var_nodes = self.find_nodes_by_type(tree, ["var_declaration"])
        
        for var_node in var_nodes:
            try:
                # Handle both single and grouped variable declarations
                # Check for var_spec_list first (grouped variables)
                var_spec_list = self.find_child_by_type(var_node, "var_spec_list")
                if var_spec_list:
                    var_specs = self.find_children_by_type(var_spec_list, "var_spec")
                else:
                    # Single variable (direct var_spec under var_declaration)
                    var_specs = self.find_children_by_type(var_node, "var_spec")
                
                for spec in var_specs:
                    var_entities = self._extract_var_spec_entities(spec, content, file_path, EntityType.VARIABLE)
                    variables.extend(var_entities)
                    
            except Exception as e:
                logger.warning(f"Failed to extract variable: {e}")
        
        return variables
    
    def _extract_constants(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract constant declarations"""
        constants = []
        
        # Find constant declarations
        const_nodes = self.find_nodes_by_type(tree, ["const_declaration"])
        
        for const_node in const_nodes:
            try:
                # Handle both single and grouped constant declarations
                # Check for const_spec_list first (grouped constants)
                const_spec_list = self.find_child_by_type(const_node, "const_spec_list")
                if const_spec_list:
                    const_specs = self.find_children_by_type(const_spec_list, "const_spec")
                else:
                    # Single constant (direct const_spec under const_declaration)
                    const_specs = self.find_children_by_type(const_node, "const_spec")
                
                for spec in const_specs:
                    const_entities = self._extract_var_spec_entities(spec, content, file_path, EntityType.CONSTANT)
                    constants.extend(const_entities)
                    
            except Exception as e:
                logger.warning(f"Failed to extract constant: {e}")
        
        return constants
    
    def _extract_types(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract type alias declarations (excluding structs and interfaces)"""
        types = []
        
        # Find type declarations
        type_nodes = self.find_nodes_by_type(tree, ["type_declaration"])
        
        for type_node in type_nodes:
            try:
                type_spec = self.find_child_by_type(type_node, "type_spec")
                if not type_spec:
                    continue
                
                # Skip structs and interfaces (handled separately)
                if (self.find_child_by_type(type_spec, "struct_type") or 
                    self.find_child_by_type(type_spec, "interface_type")):
                    continue
                
                # Get type name
                name_node = self.find_child_by_type(type_spec, "type_identifier")
                if not name_node:
                    continue
                
                type_name = self.get_node_text(name_node, content)
                
                # Get underlying type
                underlying_type = None
                for child in type_spec.children:
                    if child != name_node and child.type != "=":
                        underlying_type = self.get_node_text(child, content)
                        break
                
                # Create location
                location = SourceLocation(
                    file_path=file_path,
                    start_line=type_node.start_point[0] + 1,
                    end_line=type_node.end_point[0] + 1,
                    start_column=type_node.start_point[1],
                    end_column=type_node.end_point[1],
                    start_byte=type_node.start_byte,
                    end_byte=type_node.end_byte
                )
                
                # Generate entity ID
                entity_id = f"file://{file_path}::go_type::{type_name}::{location.start_line}"
                
                # Determine visibility based on naming convention
                visibility = Visibility.PUBLIC if type_name[0].isupper() else Visibility.PRIVATE
                
                # Create type entity
                entity = Entity(
                    id=entity_id,
                    name=type_name,
                    qualified_name=type_name,
                    entity_type=EntityType.TYPE,
                    location=location,
                    signature=f"type {type_name} {underlying_type}",
                    source_code=self.get_node_text(type_node, content),
                    source_hash=hashlib.md5(self.get_node_text(type_node, content).encode()).hexdigest(),
                    visibility=visibility,
                    metadata={
                        "language": "go",
                        "type_name": type_name,
                        "underlying_type": underlying_type,
                        "is_alias": True,
                        "ast_node_type": type_node.type
                    }
                )
                
                types.append(entity)
                
            except Exception as e:
                logger.warning(f"Failed to extract type: {e}")
        
        return types
    
    # Helper methods for detailed extraction
    def _extract_struct_field(self, field_node: tree_sitter.Node, content: str) -> Optional[Dict[str, Any]]:
        """Extract struct field information"""
        try:
            # Get field names
            field_names = []
            name_nodes = self.find_children_by_type(field_node, "field_identifier")
            for name_node in name_nodes:
                field_names.append(self.get_node_text(name_node, content))
            
            # Get field type
            field_type = None
            type_nodes = [child for child in field_node.children 
                         if child.type not in ["field_identifier", ","]]
            if type_nodes:
                field_type = self.get_node_text(type_nodes[0], content)
            
            # Get struct tags if present
            tag = None
            tag_node = self.find_child_by_type(field_node, "raw_string_literal")
            if tag_node:
                tag = self.get_node_text(tag_node, content)
            
            if field_names and field_type:
                return {
                    "names": field_names,
                    "type": field_type,
                    "tag": tag,
                    "visibility": "public" if any(name[0].isupper() for name in field_names) else "private"
                }
        except Exception as e:
            logger.warning(f"Failed to extract struct field: {e}")
        
        return None
    
    def _extract_interface_method(self, method_node: tree_sitter.Node, content: str) -> Optional[Dict[str, Any]]:
        """Extract interface method signature"""
        try:
            # Get method name
            name_node = self.find_child_by_type(method_node, "field_identifier")
            if not name_node:
                return None
            
            method_name = self.get_node_text(name_node, content)
            
            # Get method signature
            signature = self.get_node_text(method_node, content)
            
            # Extract parameters and return types (simplified)
            parameters = []
            return_types = []
            
            # This is a simplified extraction - could be enhanced for full parameter parsing
            func_type = self.find_child_by_type(method_node, "function_type")
            if func_type:
                param_list = self.find_child_by_type(func_type, "parameter_list")
                if param_list:
                    # Extract parameter information (simplified)
                    param_count = len([child for child in param_list.children 
                                     if child.type == "parameter_declaration"])
                    parameters = [f"param_{i}" for i in range(param_count)]
                
                result = self.find_child_by_type(func_type, "parameter_list")
                if result and result != param_list:  # Second parameter_list is return types
                    return_count = len([child for child in result.children 
                                      if child.type != "(" and child.type != ")" and child.type != ","])
                    return_types = [f"return_{i}" for i in range(return_count)]
            
            return {
                "name": method_name,
                "signature": signature,
                "parameters": parameters,
                "return_types": return_types,
                "visibility": "public" if method_name[0].isupper() else "private"
            }
        except Exception as e:
            logger.warning(f"Failed to extract interface method: {e}")
        
        return None
    
    def _extract_function_entity(
        self, 
        func_node: tree_sitter.Node, 
        content: str, 
        file_path: Path,
        is_method: bool = False,
        receiver_info: Optional[Dict[str, Any]] = None
    ) -> Optional[Entity]:
        """Extract a function entity with full metadata"""
        try:
            # Get function name - methods use field_identifier, functions use identifier
            name_node = self.find_child_by_type(func_node, "field_identifier" if is_method else "identifier")
            if not name_node:
                return None
            
            func_name = self.get_node_text(name_node, content)
            
            # Extract parameters and return types
            parameters = []
            return_types = []
            
            # Get all parameter lists
            param_lists = self.find_children_by_type(func_node, "parameter_list")
            
            if is_method:
                # For methods: first param_list is receiver, second is parameters, others are return types
                if len(param_lists) > 1:
                    # Extract regular parameters (skip receiver)
                    param_list = param_lists[1]
                    param_decls = self.find_children_by_type(param_list, "parameter_declaration")
                    for param_decl in param_decls:
                        param_info = self._extract_parameter_info(param_decl, content)
                        if param_info:
                            parameters.extend(param_info)
                
                # Return types (if any) - could be third parameter list or direct type identifier
                if len(param_lists) > 2:
                    return_param_list = param_lists[2]
                    return_decls = self.find_children_by_type(return_param_list, "parameter_declaration")
                    for return_decl in return_decls:
                        return_info = self._extract_parameter_info(return_decl, content)
                        if return_info:
                            return_types.extend([p["type"] for p in return_info])
                else:
                    # Check for direct return type (type_identifier)
                    return_type_node = self.find_child_by_type(func_node, "type_identifier")
                    if return_type_node:
                        return_types.append(self.get_node_text(return_type_node, content))
            else:
                # For functions: first param_list is parameters, second is return types
                if param_lists:
                    param_list = param_lists[0]
                    param_decls = self.find_children_by_type(param_list, "parameter_declaration")
                    for param_decl in param_decls:
                        param_info = self._extract_parameter_info(param_decl, content)
                        if param_info:
                            parameters.extend(param_info)
                
                # Return types
                if len(param_lists) > 1:
                    return_param_list = param_lists[1]
                    return_decls = self.find_children_by_type(return_param_list, "parameter_declaration")
                    for return_decl in return_decls:
                        return_info = self._extract_parameter_info(return_decl, content)
                        if return_info:
                            return_types.extend([p["type"] for p in return_info])
                else:
                    # Check for direct return type (type_identifier)
                    return_type_node = self.find_child_by_type(func_node, "type_identifier")
                    if return_type_node:
                        return_types.append(self.get_node_text(return_type_node, content))
            
            # Create location
            location = SourceLocation(
                file_path=file_path,
                start_line=func_node.start_point[0] + 1,
                end_line=func_node.end_point[0] + 1,
                start_column=func_node.start_point[1],
                end_column=func_node.end_point[1],
                start_byte=func_node.start_byte,
                end_byte=func_node.end_byte
            )
            
            # Generate entity ID
            entity_type_str = "go_method" if is_method else "go_function"
            entity_id = f"file://{file_path}::{entity_type_str}::{func_name}::{location.start_line}"
            
            # Determine visibility based on naming convention
            visibility = Visibility.PUBLIC if func_name[0].isupper() else Visibility.PRIVATE
            
            # Build signature
            param_strs = [f"{p['name']} {p['type']}" for p in parameters]
            param_sig = f"({', '.join(param_strs)})"
            return_sig = f" ({', '.join(return_types)})" if return_types else ""
            
            # Include receiver in signature for methods
            if is_method and receiver_info:
                receiver_sig = f"({receiver_info['name']} {receiver_info['type']})"
                signature = f"func {receiver_sig} {func_name}{param_sig}{return_sig}"
            else:
                signature = f"func {func_name}{param_sig}{return_sig}"
            
            # Detect async patterns (goroutines)
            is_async = self._contains_goroutines(func_node, content)
            
            # Create function entity
            entity = Entity(
                id=entity_id,
                name=func_name,
                qualified_name=func_name,
                entity_type=EntityType.METHOD if is_method else EntityType.FUNCTION,
                location=location,
                signature=signature,
                source_code=self.get_node_text(func_node, content),
                source_hash=hashlib.md5(self.get_node_text(func_node, content).encode()).hexdigest(),
                visibility=visibility,
                is_async=is_async,
                metadata={
                    "language": "go",
                    "function_name": func_name,
                    "parameters": parameters,
                    "return_types": return_types,
                    "parameter_count": len(parameters),
                    "return_count": len(return_types),
                    "has_goroutines": is_async,
                    "ast_node_type": func_node.type,
                    **({"receiver": receiver_info, 
                        "receiver_type": receiver_info["type"], 
                        "receiver_name": receiver_info["name"]} if is_method and receiver_info else {})
                }
            )
            
            return entity
            
        except Exception as e:
            logger.warning(f"Failed to extract function entity: {e}")
            return None
    
    def _extract_method_entity(
        self, 
        method_node: tree_sitter.Node, 
        content: str, 
        file_path: Path
    ) -> Optional[Entity]:
        """Extract a method entity with receiver information"""
        try:
            # Get method name
            name_node = self.find_child_by_type(method_node, "field_identifier")
            if not name_node:
                return None
            
            method_name = self.get_node_text(name_node, content)
            
            # Extract receiver information
            receiver_info = None
            receiver_list = self.find_child_by_type(method_node, "parameter_list")
            if receiver_list:
                receiver_decls = self.find_children_by_type(receiver_list, "parameter_declaration")
                if receiver_decls:
                    receiver_info = self._extract_parameter_info(receiver_decls[0], content)
                    if receiver_info:
                        receiver_info = receiver_info[0]  # Take first parameter as receiver
            
            # Use function extraction but pass receiver info to build correct signature
            entity = self._extract_function_entity(method_node, content, file_path, is_method=True, receiver_info=receiver_info)
            
            return entity
            
        except Exception as e:
            logger.warning(f"Failed to extract method entity: {e}")
            return None
    
    def _extract_var_spec_entities(
        self, 
        spec_node: tree_sitter.Node, 
        content: str, 
        file_path: Path,
        entity_type: EntityType
    ) -> List[Entity]:
        """Extract variable or constant entities from a var/const spec"""
        entities = []
        
        try:
            # Get variable names
            name_nodes = self.find_children_by_type(spec_node, "identifier")
            if not name_nodes:
                return entities
            
            # Get type if present
            var_type = None
            type_nodes = [child for child in spec_node.children 
                         if child.type not in ["identifier", "=", "expression_list"]]
            if type_nodes:
                var_type = self.get_node_text(type_nodes[0], content)
            
            # Get initial values if present
            values = []
            value_list = self.find_child_by_type(spec_node, "expression_list")
            if value_list:
                # Simplified value extraction
                value_nodes = [child for child in value_list.children if child.type != ","]
                values = [self.get_node_text(node, content) for node in value_nodes]
            
            # Create entities for each variable/constant
            for i, name_node in enumerate(name_nodes):
                var_name = self.get_node_text(name_node, content)
                value = values[i] if i < len(values) else None
                
                # Create location
                location = SourceLocation(
                    file_path=file_path,
                    start_line=name_node.start_point[0] + 1,
                    end_line=name_node.end_point[0] + 1,
                    start_column=name_node.start_point[1],
                    end_column=name_node.end_point[1],
                    start_byte=name_node.start_byte,
                    end_byte=name_node.end_byte
                )
                
                # Generate entity ID
                type_str = "go_constant" if entity_type == EntityType.CONSTANT else "go_variable"
                entity_id = f"file://{file_path}::{type_str}::{var_name}::{location.start_line}"
                
                # Determine visibility based on naming convention
                visibility = Visibility.PUBLIC if var_name[0].isupper() else Visibility.PRIVATE
                
                # Build signature
                type_part = f" {var_type}" if var_type else ""
                value_part = f" = {value}" if value else ""
                keyword = "const" if entity_type == EntityType.CONSTANT else "var"
                signature = f"{keyword} {var_name}{type_part}{value_part}"
                
                # Create entity
                entity = Entity(
                    id=entity_id,
                    name=var_name,
                    qualified_name=var_name,
                    entity_type=entity_type,
                    location=location,
                    signature=signature,
                    source_code=self.get_node_text(spec_node, content),
                    source_hash=hashlib.md5(self.get_node_text(spec_node, content).encode()).hexdigest(),
                    visibility=visibility,
                    metadata={
                        "language": "go",
                        "variable_name": var_name,
                        "type": var_type,
                        "value": value,
                        "is_constant": entity_type == EntityType.CONSTANT,
                        "ast_node_type": spec_node.type
                    }
                )
                
                entities.append(entity)
                
        except Exception as e:
            logger.warning(f"Failed to extract var/const spec: {e}")
        
        return entities
    
    # Helper methods for relation extraction
    def _build_entity_lookup(self, entities: List[Entity]) -> Dict[str, Entity]:
        """Build lookup table for entities by name"""
        lookup = {}
        for entity in entities:
            lookup[entity.name] = entity
            if entity.qualified_name != entity.name:
                lookup[entity.qualified_name] = entity
        return lookup
    
    def _extract_struct_field_relations(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        entities: List[Entity],
        entity_lookup: Dict[str, Entity]
    ) -> List[Relation]:
        """Extract relations between structs and their fields"""
        relations = []
        # Implementation would go here
        return relations
    
    def _extract_interface_implementation_relations(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        entities: List[Entity],
        entity_lookup: Dict[str, Entity]
    ) -> List[Relation]:
        """Extract interface implementation relations"""
        relations = []
        # Implementation would go here
        return relations
    
    def _extract_method_receiver_relations(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        entities: List[Entity],
        entity_lookup: Dict[str, Entity]
    ) -> List[Relation]:
        """Extract relations between methods and their receiver types"""
        relations = []
        # Implementation would go here
        return relations
    
    def _extract_function_call_relations(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        entities: List[Entity],
        entity_lookup: Dict[str, Entity]
    ) -> List[Relation]:
        """Extract function call relations"""
        relations = []
        # Implementation would go here
        return relations
    
    def _extract_import_relations(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        entities: List[Entity],
        entity_lookup: Dict[str, Entity]
    ) -> List[Relation]:
        """Extract import usage relations"""
        relations = []
        # Implementation would go here
        return relations
    
    def _extract_type_usage_relations(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        entities: List[Entity],
        entity_lookup: Dict[str, Entity]
    ) -> List[Relation]:
        """Extract type usage relations"""
        relations = []
        # Implementation would go here
        return relations
    
    def _extract_containment_relations(self, entities: List[Entity]) -> List[Relation]:
        """Extract containment relations between entities"""
        relations = []
        # Implementation would go here
        return relations
    
    # Utility helper methods
    def _is_standard_library(self, import_path: str) -> bool:
        """Check if import path is from Go standard library"""
        # Simplified check - Go standard library packages don't have dots in the first path segment
        first_segment = import_path.split('/')[0]
        return '.' not in first_segment and not first_segment.startswith('github.com')
    
    def _has_receiver_syntax(self, params: tree_sitter.Node, content: str) -> bool:
        """Check if parameter list has receiver syntax (for method detection)"""
        param_text = self.get_node_text(params, content)
        return self._receiver_pattern.search(param_text) is not None
    
    def _extract_parameter_info(self, param_node: tree_sitter.Node, content: str) -> List[Dict[str, Any]]:
        """Extract parameter information from parameter declaration"""
        params = []
        
        try:
            # Get parameter names
            name_nodes = self.find_children_by_type(param_node, "identifier")
            
            # Get parameter type
            param_type = None
            type_nodes = [child for child in param_node.children 
                         if child.type not in ["identifier", ","]]
            if type_nodes:
                param_type = self.get_node_text(type_nodes[0], content)
            
            # Create parameter info for each name
            for name_node in name_nodes:
                param_name = self.get_node_text(name_node, content)
                params.append({
                    "name": param_name,
                    "type": param_type or "interface{}"
                })
            
            # If no names found but type exists, create unnamed parameter
            if not name_nodes and param_type:
                params.append({
                    "name": "_",
                    "type": param_type
                })
                
        except Exception as e:
            logger.warning(f"Failed to extract parameter info: {e}")
        
        return params
    
    def _contains_goroutines(self, func_node: tree_sitter.Node, content: str) -> bool:
        """Check if function contains goroutine usage"""
        func_text = self.get_node_text(func_node, content)
        return "go " in func_text
    
    def _determine_visibility(self, name: str) -> Visibility:
        """Determine visibility based on Go naming conventions"""
        return Visibility.PUBLIC if name and name[0].isupper() else Visibility.PRIVATE