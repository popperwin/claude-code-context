"""
Rust parser using Tree-sitter for comprehensive entity extraction.

Extracts modules, functions, structs, enums, traits, impls, constants,
types, and uses from Rust source code with full metadata and relation information.
"""

import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Set
import re

try:
    import tree_sitter
    import tree_sitter_rust
except ImportError:
    tree_sitter = None
    tree_sitter_rust = None

from .tree_sitter_base import TreeSitterBase
from .registry import register_parser
from ..models.entities import (
    Entity, EntityType, Relation, RelationType, 
    SourceLocation, Visibility
)

logger = logging.getLogger(__name__)


@register_parser("rust", [".rs"])
class RustParser(TreeSitterBase):
    """
    Rust parser using Tree-sitter for comprehensive entity extraction.
    
    Extracts:
    - Modules (mod declarations and file modules)
    - Functions (fn declarations)
    - Structs (struct definitions with fields)
    - Enums (enum definitions with variants)
    - Traits (trait definitions with methods)
    - Implementations (impl blocks)
    - Constants (const declarations)
    - Static variables (static declarations)
    - Type aliases (type declarations)
    - Use statements (imports)
    - Macros (macro_rules! definitions)
    """
    
    def __init__(self):
        super().__init__("rust")
        
        # Compiled regex patterns for efficiency
        self._comment_pattern = re.compile(r'//.*$|/\*[\s\S]*?\*/', re.MULTILINE)
        self._crate_pattern = re.compile(r'crate::')
        
        logger.debug("Rust parser initialized")
    
    def get_supported_extensions(self) -> List[str]:
        return [".rs"]
    
    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.get_supported_extensions()
    
    def extract_entities(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """
        Extract entities from Rust source code.
        
        Args:
            tree: Tree-sitter parse tree
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
            entities.extend(self._extract_modules(tree, content, file_path))
            entities.extend(self._extract_uses(tree, content, file_path))
            entities.extend(self._extract_structs(tree, content, file_path))
            entities.extend(self._extract_enums(tree, content, file_path))
            entities.extend(self._extract_traits(tree, content, file_path))
            entities.extend(self._extract_functions(tree, content, file_path))
            entities.extend(self._extract_impls(tree, content, file_path))
            entities.extend(self._extract_constants(tree, content, file_path))
            entities.extend(self._extract_static_vars(tree, content, file_path))
            entities.extend(self._extract_type_aliases(tree, content, file_path))
            entities.extend(self._extract_macros(tree, content, file_path))
            
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
        """Extract relations between entities"""
        if not tree or not entities:
            return []
        
        relations = []
        
        try:
            # Create entity lookup
            entity_lookup = {entity.name: entity for entity in entities}
            
            # Extract use relations
            relations.extend(self._extract_use_relations(tree, content, entities, file_path))
            
            # Extract inheritance relations (trait implementations)
            relations.extend(self._extract_impl_relations(tree, content, entities, file_path))
            
            # Extract call relations
            relations.extend(self._extract_call_relations(tree, content, entities, file_path))
            
        except Exception as e:
            logger.error(f"Relation extraction failed for {file_path}: {e}")
        
        return relations
    
    def _extract_modules(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract module declarations"""
        modules = []
        
        # Find module declarations
        mod_nodes = self.find_nodes_by_type(tree, ["mod_item"])
        
        for mod_node in mod_nodes:
            try:
                # Get module name
                name_node = self.find_child_by_type(mod_node, "identifier")
                if not name_node:
                    continue
                
                mod_name = self.get_node_text(name_node, content)
                
                # Check if it's a file module or inline module
                body_node = self.find_child_by_type(mod_node, "declaration_list")
                is_file_module = body_node is None
                
                # Create location
                location = SourceLocation(
                    file_path=file_path,
                    start_line=mod_node.start_point[0] + 1,
                    end_line=mod_node.end_point[0] + 1,
                    start_column=mod_node.start_point[1],
                    end_column=mod_node.end_point[1],
                    start_byte=mod_node.start_byte,
                    end_byte=mod_node.end_byte
                )
                
                # Extract module metadata
                visibility = self._determine_visibility(mod_node, content)
                source_code = self.get_node_text(mod_node, content)
                
                entity_id = f"file://{file_path}::rust_module::{mod_name}::{location.start_line}"
                module_entity = Entity(
                    id=entity_id,
                    name=mod_name,
                    entity_type=EntityType.MODULE,
                    location=location,
                    signature=f"mod {mod_name}",
                    source_code=source_code,
                    source_hash=hashlib.md5(source_code.encode('utf-8')).hexdigest()[:16],
                    visibility=visibility,
                    qualified_name=mod_name,  # TODO: Build full path
                    metadata={
                        "language": "rust",
                        "ast_node_type": "mod_item",
                        "module_name": mod_name,
                        "is_file_module": is_file_module,
                        "is_inline_module": not is_file_module
                    }
                )
                
                modules.append(module_entity)
                
            except Exception as e:
                logger.warning(f"Failed to extract module: {e}")
        
        return modules
    
    def _extract_uses(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract use statements (imports)"""
        uses = []
        
        # Find use declarations
        use_nodes = self.find_nodes_by_type(tree, ["use_declaration"])
        
        for use_node in use_nodes:
            try:
                # Extract the imported path directly from node text
                # This is more reliable than AST navigation for complex Rust use statements
                use_path = self.get_node_text(use_node, content).replace("use ", "").replace(";", "").strip()
                
                # Determine the imported name (last component)
                if "::" in use_path:
                    imported_name = use_path.split("::")[-1]
                else:
                    imported_name = use_path
                
                # Handle glob imports and use groups
                if imported_name == "*":
                    imported_name = f"{use_path.split('::')[-2]}::*"
                elif "{" in imported_name:
                    imported_name = use_path  # Keep full path for complex use statements
                
                # Create location
                location = SourceLocation(
                    file_path=file_path,
                    start_line=use_node.start_point[0] + 1,
                    end_line=use_node.end_point[0] + 1,
                    start_column=use_node.start_point[1],
                    end_column=use_node.end_point[1],
                    start_byte=use_node.start_byte,
                    end_byte=use_node.end_byte
                )
                
                source_code = self.get_node_text(use_node, content)
                
                # Determine if it's from standard library
                is_std = use_path.startswith("std::") or use_path.startswith("core::") or use_path.startswith("alloc::")
                is_external = not (use_path.startswith("crate::") or use_path.startswith("self::") or use_path.startswith("super::"))
                
                entity_id = f"file://{file_path}::rust_import::{imported_name}::{location.start_line}"
                use_entity = Entity(
                    id=entity_id,
                    name=imported_name,
                    entity_type=EntityType.IMPORT,
                    location=location,
                    signature=f"use {use_path}",
                    source_code=source_code,
                    source_hash=hashlib.md5(source_code.encode('utf-8')).hexdigest()[:16],
                    visibility=Visibility.PUBLIC,
                    qualified_name=use_path,
                    metadata={
                        "language": "rust",
                        "ast_node_type": "use_declaration",
                        "import_path": use_path,
                        "imported_name": imported_name,
                        "is_standard_library": is_std,
                        "is_external_crate": is_external and not is_std,
                        "is_glob_import": "*" in use_path,
                        "is_group_import": "{" in use_path
                    }
                )
                
                uses.append(use_entity)
                
            except Exception as e:
                logger.warning(f"Failed to extract use statement: {e}")
        
        return uses
    
    def _extract_structs(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract struct definitions"""
        structs = []
        
        # Find struct declarations
        struct_nodes = self.find_nodes_by_type(tree, ["struct_item"])
        
        for struct_node in struct_nodes:
            try:
                # Get struct name
                name_node = self.find_child_by_type(struct_node, "type_identifier")
                if not name_node:
                    continue
                
                struct_name = self.get_node_text(name_node, content)
                
                # Extract fields
                fields = []
                field_list = self.find_child_by_type(struct_node, "field_declaration_list")
                if field_list:
                    field_nodes = self.find_children_by_type(field_list, "field_declaration")
                    for field_node in field_nodes:
                        field_info = self._extract_struct_field(field_node, content)
                        if field_info:
                            fields.append(field_info)
                
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
                
                visibility = self._determine_visibility(struct_node, content)
                source_code = self.get_node_text(struct_node, content)
                
                entity_id = f"file://{file_path}::rust_struct::{struct_name}::{location.start_line}"
                struct_entity = Entity(
                    id=entity_id,
                    name=struct_name,
                    entity_type=EntityType.STRUCT,
                    location=location,
                    signature=f"struct {struct_name}",
                    source_code=source_code,
                    source_hash=hashlib.md5(source_code.encode('utf-8')).hexdigest()[:16],
                    visibility=visibility,
                    qualified_name=struct_name,  # TODO: Build full path
                    metadata={
                        "language": "rust",
                        "ast_node_type": "struct_item",
                        "struct_name": struct_name,
                        "fields": fields,
                        "field_count": len(fields),
                        "is_tuple_struct": self._is_tuple_struct(struct_node),
                        "is_unit_struct": self._is_unit_struct(struct_node)
                    }
                )
                
                structs.append(struct_entity)
                
            except Exception as e:
                logger.warning(f"Failed to extract struct: {e}")
        
        return structs
    
    def _extract_functions(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract function definitions"""
        functions = []
        
        # Find function declarations
        function_nodes = self.find_nodes_by_type(tree, ["function_item"])
        
        for func_node in function_nodes:
            try:
                # Get function name
                name_node = self.find_child_by_type(func_node, "identifier")
                if not name_node:
                    continue
                
                func_name = self.get_node_text(name_node, content)
                
                # Extract parameters
                parameters = []
                param_list = self.find_child_by_type(func_node, "parameters")
                if param_list:
                    param_nodes = self.find_children_by_type(param_list, "parameter")
                    for param_node in param_nodes:
                        param_info = self._extract_parameter_info(param_node, content)
                        if param_info:
                            parameters.extend(param_info)
                
                # Extract return type
                return_type = None
                type_node = self.find_child_by_type(func_node, "type_annotation")
                if type_node:
                    type_expr = self.find_child_by_type(type_node, ["primitive_type", "type_identifier", "generic_type"])
                    if type_expr:
                        return_type = self.get_node_text(type_expr, content)
                
                # Check if function is async
                is_async = self._has_modifier(func_node, "async", content)
                is_unsafe = self._has_modifier(func_node, "unsafe", content)
                is_const = self._has_modifier(func_node, "const", content)
                
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
                
                visibility = self._determine_visibility(func_node, content)
                source_code = self.get_node_text(func_node, content)
                
                # Build signature
                signature_parts = []
                if visibility == Visibility.PUBLIC:
                    signature_parts.append("pub")
                if is_unsafe:
                    signature_parts.append("unsafe")
                if is_async:
                    signature_parts.append("async")
                if is_const:
                    signature_parts.append("const")
                signature_parts.append("fn")
                signature_parts.append(func_name)
                
                param_str = ", ".join([f"{p['name']}: {p['type']}" for p in parameters])
                signature_parts.append(f"({param_str})")
                
                if return_type:
                    signature_parts.append(f"-> {return_type}")
                
                signature = " ".join(signature_parts)
                
                entity_id = f"file://{file_path}::rust_function::{func_name}::{location.start_line}"
                function_entity = Entity(
                    id=entity_id,
                    name=func_name,
                    entity_type=EntityType.FUNCTION,
                    location=location,
                    signature=signature,
                    source_code=source_code,
                    source_hash=hashlib.md5(source_code.encode('utf-8')).hexdigest()[:16],
                    visibility=visibility,
                    qualified_name=func_name,  # TODO: Build full path
                    is_async=is_async,
                    metadata={
                        "language": "rust",
                        "ast_node_type": "function_item",
                        "function_name": func_name,
                        "parameters": parameters,
                        "return_type": return_type,
                        "is_async": is_async,
                        "is_unsafe": is_unsafe,
                        "is_const": is_const,
                        "parameter_count": len(parameters)
                    }
                )
                
                functions.append(function_entity)
                
            except Exception as e:
                logger.warning(f"Failed to extract function: {e}")
        
        return functions
    
    def _extract_enums(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract enum definitions"""
        enums = []
        
        # Find enum declarations
        enum_nodes = self.find_nodes_by_type(tree, ["enum_item"])
        
        for enum_node in enum_nodes:
            try:
                # Get enum name
                name_node = self.find_child_by_type(enum_node, "type_identifier")
                if not name_node:
                    continue
                
                enum_name = self.get_node_text(name_node, content)
                
                # Extract variants
                variants = []
                variant_list = self.find_child_by_type(enum_node, "enum_variant_list")
                if variant_list:
                    variant_nodes = self.find_children_by_type(variant_list, "enum_variant")
                    for variant_node in variant_nodes:
                        variant_info = self._extract_enum_variant(variant_node, content)
                        if variant_info:
                            variants.append(variant_info)
                
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
                
                visibility = self._determine_visibility(enum_node, content)
                source_code = self.get_node_text(enum_node, content)
                
                entity_id = f"file://{file_path}::rust_enum::{enum_name}::{location.start_line}"
                enum_entity = Entity(
                    id=entity_id,
                    name=enum_name,
                    entity_type=EntityType.ENUM,
                    location=location,
                    signature=f"enum {enum_name}",
                    source_code=source_code,
                    source_hash=hashlib.md5(source_code.encode('utf-8')).hexdigest()[:16],
                    visibility=visibility,
                    qualified_name=enum_name,  # TODO: Build full path
                    metadata={
                        "language": "rust",
                        "ast_node_type": "enum_item",
                        "enum_name": enum_name,
                        "variants": variants,
                        "variant_count": len(variants)
                    }
                )
                
                enums.append(enum_entity)
                
            except Exception as e:
                logger.warning(f"Failed to extract enum: {e}")
        
        return enums
    
    def _extract_traits(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract trait definitions"""
        traits = []
        
        # Find trait declarations
        trait_nodes = self.find_nodes_by_type(tree, ["trait_item"])
        
        for trait_node in trait_nodes:
            try:
                # Get trait name
                name_node = self.find_child_by_type(trait_node, "type_identifier")
                if not name_node:
                    continue
                
                trait_name = self.get_node_text(name_node, content)
                
                # Extract methods
                methods = []
                declaration_list = self.find_child_by_type(trait_node, "declaration_list")
                if declaration_list:
                    # Look for function signatures and default implementations
                    # Try multiple possible node types for trait methods
                    method_nodes = self.find_children_by_type(declaration_list, ["function_signature_item", "function_item"])
                    
                    # If no methods found with specific types, try finding any function-like nodes
                    if not method_nodes:
                        # Look for any child that might be a method
                        for child in declaration_list.children:
                            if "function" in child.type or child.type == "declaration":
                                method_nodes.append(child)
                    
                    for method_node in method_nodes:
                        method_info = self._extract_trait_method(method_node, content)
                        if method_info:
                            methods.append(method_info)
                
                # Create location
                location = SourceLocation(
                    file_path=file_path,
                    start_line=trait_node.start_point[0] + 1,
                    end_line=trait_node.end_point[0] + 1,
                    start_column=trait_node.start_point[1],
                    end_column=trait_node.end_point[1],
                    start_byte=trait_node.start_byte,
                    end_byte=trait_node.end_byte
                )
                
                visibility = self._determine_visibility(trait_node, content)
                source_code = self.get_node_text(trait_node, content)
                
                entity_id = f"file://{file_path}::rust_trait::{trait_name}::{location.start_line}"
                trait_entity = Entity(
                    id=entity_id,
                    name=trait_name,
                    entity_type=EntityType.TRAIT,
                    location=location,
                    signature=f"trait {trait_name}",
                    source_code=source_code,
                    source_hash=hashlib.md5(source_code.encode('utf-8')).hexdigest()[:16],
                    visibility=visibility,
                    qualified_name=trait_name,  # TODO: Build full path
                    metadata={
                        "language": "rust",
                        "ast_node_type": "trait_item",
                        "trait_name": trait_name,
                        "methods": methods,
                        "method_count": len(methods)
                    }
                )
                
                traits.append(trait_entity)
                
            except Exception as e:
                logger.warning(f"Failed to extract trait: {e}")
        
        return traits
    
    def _extract_impls(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract impl blocks"""
        impls = []
        
        # Find impl declarations
        impl_nodes = self.find_nodes_by_type(tree, ["impl_item"])
        
        for impl_node in impl_nodes:
            try:
                # Determine if this is a trait impl or inherent impl
                has_for = any(self.get_node_text(child, content) == "for" for child in impl_node.children)
                
                trait_name = None
                impl_type = None
                
                if has_for:
                    # Trait implementation: impl TraitName for TypeName
                    # Find trait name (first type_identifier) and impl type (type_identifier after "for")
                    found_for = False
                    for child in impl_node.children:
                        if self.get_node_text(child, content) == "for":
                            found_for = True
                        elif child.type == "type_identifier":
                            if not found_for:
                                # This is the trait name
                                trait_name = self.get_node_text(child, content)
                            else:
                                # This is the implementation type
                                impl_type = self.get_node_text(child, content)
                                break
                else:
                    # Inherent implementation: impl TypeName or impl<T> TypeName
                    # Look for type_identifier or generic_type
                    type_node = self.find_child_by_type(impl_node, "type_identifier")
                    if not type_node:
                        type_node = self.find_child_by_type(impl_node, "generic_type")
                    if type_node:
                        impl_type = self.get_node_text(type_node, content)
                
                if not impl_type:
                    continue
                
                # Create a name for the impl
                if trait_name:
                    impl_name = f"{trait_name} for {impl_type}"
                    entity_type = EntityType.IMPLEMENTATION
                else:
                    impl_name = f"impl {impl_type}"
                    entity_type = EntityType.IMPLEMENTATION
                
                # Extract methods
                methods = []
                declaration_list = self.find_child_by_type(impl_node, "declaration_list")
                if declaration_list:
                    method_nodes = self.find_children_by_type(declaration_list, "function_item")
                    for method_node in method_nodes:
                        method_info = self._extract_impl_method(method_node, content)
                        if method_info:
                            methods.append(method_info)
                
                # Create location
                location = SourceLocation(
                    file_path=file_path,
                    start_line=impl_node.start_point[0] + 1,
                    end_line=impl_node.end_point[0] + 1,
                    start_column=impl_node.start_point[1],
                    end_column=impl_node.end_point[1],
                    start_byte=impl_node.start_byte,
                    end_byte=impl_node.end_byte
                )
                
                source_code = self.get_node_text(impl_node, content)
                
                entity_id = f"file://{file_path}::rust_impl::{impl_name}::{location.start_line}"
                impl_entity = Entity(
                    id=entity_id,
                    name=impl_name,
                    entity_type=entity_type,
                    location=location,
                    signature=impl_name,
                    source_code=source_code,
                    source_hash=hashlib.md5(source_code.encode('utf-8')).hexdigest()[:16],
                    visibility=Visibility.PUBLIC,  # impl blocks are always visible where the type is visible
                    qualified_name=impl_name,
                    metadata={
                        "language": "rust",
                        "ast_node_type": "impl_item",
                        "impl_type": impl_type,
                        "trait_name": trait_name,
                        "is_trait_impl": trait_name is not None,
                        "methods": methods,
                        "method_count": len(methods)
                    }
                )
                
                impls.append(impl_entity)
                
            except Exception as e:
                logger.warning(f"Failed to extract impl: {e}")
        
        return impls
    
    def _extract_constants(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract const declarations"""
        constants = []
        
        # Find const declarations
        const_nodes = self.find_nodes_by_type(tree, ["const_item"])
        
        for const_node in const_nodes:
            try:
                # Get constant name
                name_node = self.find_child_by_type(const_node, "identifier")
                if not name_node:
                    continue
                
                const_name = self.get_node_text(name_node, content)
                
                # Get type annotation
                const_type = None
                type_node = self.find_child_by_type(const_node, "type_annotation")
                if type_node:
                    type_expr = self.find_child_by_type(type_node, ["primitive_type", "type_identifier"])
                    if type_expr:
                        const_type = self.get_node_text(type_expr, content)
                
                # Get value (if present)
                value = None
                # Look for assignment expression
                for child in const_node.children:
                    if child.type in ["integer_literal", "string_literal", "boolean_literal", "call_expression"]:
                        value = self.get_node_text(child, content)
                        break
                
                # Create location
                location = SourceLocation(
                    file_path=file_path,
                    start_line=const_node.start_point[0] + 1,
                    end_line=const_node.end_point[0] + 1,
                    start_column=const_node.start_point[1],
                    end_column=const_node.end_point[1],
                    start_byte=const_node.start_byte,
                    end_byte=const_node.end_byte
                )
                
                visibility = self._determine_visibility(const_node, content)
                source_code = self.get_node_text(const_node, content)
                
                entity_id = f"file://{file_path}::rust_const::{const_name}::{location.start_line}"
                const_entity = Entity(
                    id=entity_id,
                    name=const_name,
                    entity_type=EntityType.CONSTANT,
                    location=location,
                    signature=f"const {const_name}: {const_type or '?'}",
                    source_code=source_code,
                    source_hash=hashlib.md5(source_code.encode('utf-8')).hexdigest()[:16],
                    visibility=visibility,
                    qualified_name=const_name,
                    metadata={
                        "language": "rust",
                        "ast_node_type": "const_item",
                        "const_name": const_name,
                        "type": const_type,
                        "value": value,
                        "is_constant": True
                    }
                )
                
                constants.append(const_entity)
                
            except Exception as e:
                logger.warning(f"Failed to extract constant: {e}")
        
        return constants
    
    def _extract_static_vars(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract static variable declarations"""
        statics = []
        
        # Find static declarations
        static_nodes = self.find_nodes_by_type(tree, ["static_item"])
        
        for static_node in static_nodes:
            try:
                # Get static name
                name_node = self.find_child_by_type(static_node, "identifier")
                if not name_node:
                    continue
                
                static_name = self.get_node_text(name_node, content)
                
                # Get type annotation
                static_type = None
                type_node = self.find_child_by_type(static_node, "type_annotation")
                if type_node:
                    type_expr = self.find_child_by_type(type_node, ["primitive_type", "type_identifier"])
                    if type_expr:
                        static_type = self.get_node_text(type_expr, content)
                
                # Check if mutable
                is_mutable = "mut" in self.get_node_text(static_node, content)
                
                # Create location
                location = SourceLocation(
                    file_path=file_path,
                    start_line=static_node.start_point[0] + 1,
                    end_line=static_node.end_point[0] + 1,
                    start_column=static_node.start_point[1],
                    end_column=static_node.end_point[1],
                    start_byte=static_node.start_byte,
                    end_byte=static_node.end_byte
                )
                
                visibility = self._determine_visibility(static_node, content)
                source_code = self.get_node_text(static_node, content)
                
                entity_id = f"file://{file_path}::rust_static::{static_name}::{location.start_line}"
                static_entity = Entity(
                    id=entity_id,
                    name=static_name,
                    entity_type=EntityType.VARIABLE,
                    location=location,
                    signature=f"static {static_name}: {static_type or '?'}",
                    source_code=source_code,
                    source_hash=hashlib.md5(source_code.encode('utf-8')).hexdigest()[:16],
                    visibility=visibility,
                    qualified_name=static_name,
                    metadata={
                        "language": "rust",
                        "ast_node_type": "static_item",
                        "static_name": static_name,
                        "type": static_type,
                        "is_mutable": is_mutable,
                        "is_static": True
                    }
                )
                
                statics.append(static_entity)
                
            except Exception as e:
                logger.warning(f"Failed to extract static variable: {e}")
        
        return statics
    
    def _extract_type_aliases(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract type alias declarations"""
        type_aliases = []
        
        # Find type alias declarations
        type_nodes = self.find_nodes_by_type(tree, ["type_item"])
        
        for type_node in type_nodes:
            try:
                # Get type alias name
                name_node = self.find_child_by_type(type_node, "type_identifier")
                if not name_node:
                    continue
                
                type_name = self.get_node_text(name_node, content)
                
                # Get the aliased type
                aliased_type = None
                for child in type_node.children:
                    if child.type in ["primitive_type", "type_identifier", "generic_type"]:
                        aliased_type = self.get_node_text(child, content)
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
                
                visibility = self._determine_visibility(type_node, content)
                source_code = self.get_node_text(type_node, content)
                
                entity_id = f"file://{file_path}::rust_type::{type_name}::{location.start_line}"
                type_entity = Entity(
                    id=entity_id,
                    name=type_name,
                    entity_type=EntityType.TYPE,
                    location=location,
                    signature=f"type {type_name} = {aliased_type or '?'}",
                    source_code=source_code,
                    source_hash=hashlib.md5(source_code.encode('utf-8')).hexdigest()[:16],
                    visibility=visibility,
                    qualified_name=type_name,
                    metadata={
                        "language": "rust",
                        "ast_node_type": "type_item",
                        "type_name": type_name,
                        "aliased_type": aliased_type,
                        "is_alias": True
                    }
                )
                
                type_aliases.append(type_entity)
                
            except Exception as e:
                logger.warning(f"Failed to extract type alias: {e}")
        
        return type_aliases
    
    def _extract_macros(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract macro definitions"""
        macros = []
        
        # Find macro declarations
        macro_nodes = self.find_nodes_by_type(tree, ["macro_definition"])
        
        for macro_node in macro_nodes:
            try:
                # Get macro name
                name_node = self.find_child_by_type(macro_node, "identifier")
                if not name_node:
                    continue
                
                macro_name = self.get_node_text(name_node, content)
                
                # Create location
                location = SourceLocation(
                    file_path=file_path,
                    start_line=macro_node.start_point[0] + 1,
                    end_line=macro_node.end_point[0] + 1,
                    start_column=macro_node.start_point[1],
                    end_column=macro_node.end_point[1],
                    start_byte=macro_node.start_byte,
                    end_byte=macro_node.end_byte
                )
                
                visibility = self._determine_visibility(macro_node, content)
                source_code = self.get_node_text(macro_node, content)
                
                entity_id = f"file://{file_path}::rust_macro::{macro_name}::{location.start_line}"
                macro_entity = Entity(
                    id=entity_id,
                    name=macro_name,
                    entity_type=EntityType.MACRO,
                    location=location,
                    signature=f"macro_rules! {macro_name}",
                    source_code=source_code,
                    source_hash=hashlib.md5(source_code.encode('utf-8')).hexdigest()[:16],
                    visibility=visibility,
                    qualified_name=macro_name,
                    metadata={
                        "language": "rust",
                        "ast_node_type": "macro_definition",
                        "macro_name": macro_name
                    }
                )
                
                macros.append(macro_entity)
                
            except Exception as e:
                logger.warning(f"Failed to extract macro: {e}")
        
        return macros
    
    # Helper methods
    def _determine_visibility(self, node: tree_sitter.Node, content: str) -> Visibility:
        """Determine visibility of a Rust item"""
        # Check for pub keyword
        for child in node.children:
            if child.type == "visibility_modifier":
                pub_text = self.get_node_text(child, content)
                if "pub" in pub_text:
                    return Visibility.PUBLIC
        
        # Rust defaults to private
        return Visibility.PRIVATE
    
    def _has_modifier(self, node: tree_sitter.Node, modifier: str, content: str) -> bool:
        """Check if a node has a specific modifier"""
        node_text = self.get_node_text(node, content)
        return modifier in node_text.split()
    
    def _is_tuple_struct(self, struct_node: tree_sitter.Node) -> bool:
        """Check if struct is a tuple struct"""
        return self.find_child_by_type(struct_node, "ordered_field_declaration_list") is not None
    
    def _is_unit_struct(self, struct_node: tree_sitter.Node) -> bool:
        """Check if struct is a unit struct"""
        return (self.find_child_by_type(struct_node, "field_declaration_list") is None and
                self.find_child_by_type(struct_node, "ordered_field_declaration_list") is None)
    
    def _extract_struct_field(self, field_node: tree_sitter.Node, content: str) -> Optional[Dict[str, Any]]:
        """Extract information about a struct field"""
        try:
            # Get field name
            name_node = self.find_child_by_type(field_node, "field_identifier")
            if not name_node:
                return None
            
            field_name = self.get_node_text(name_node, content)
            
            # Get field type
            field_type = None
            type_node = self.find_child_by_type(field_node, ["primitive_type", "type_identifier", "generic_type"])
            if type_node:
                field_type = self.get_node_text(type_node, content)
            
            # Check visibility
            visibility = self._determine_visibility(field_node, content)
            
            return {
                "name": field_name,
                "type": field_type,
                "visibility": visibility.value
            }
            
        except Exception as e:
            logger.warning(f"Failed to extract struct field: {e}")
            return None
    
    def _extract_enum_variant(self, variant_node: tree_sitter.Node, content: str) -> Optional[Dict[str, Any]]:
        """Extract information about an enum variant"""
        try:
            # Get variant name
            name_node = self.find_child_by_type(variant_node, "identifier")
            if not name_node:
                return None
            
            variant_name = self.get_node_text(name_node, content)
            
            # Check if it has fields (tuple variant) or named fields (struct variant)
            has_fields = (self.find_child_by_type(variant_node, "field_declaration_list") is not None or
                         self.find_child_by_type(variant_node, "ordered_field_declaration_list") is not None)
            
            return {
                "name": variant_name,
                "has_fields": has_fields
            }
            
        except Exception as e:
            logger.warning(f"Failed to extract enum variant: {e}")
            return None
    
    def _extract_trait_method(self, method_node: tree_sitter.Node, content: str) -> Optional[Dict[str, Any]]:
        """Extract information about a trait method"""
        try:
            # Get method name
            name_node = self.find_child_by_type(method_node, "identifier")
            if not name_node:
                return None
            
            method_name = self.get_node_text(name_node, content)
            
            # Check if it has a default implementation
            has_body = self.find_child_by_type(method_node, "block") is not None
            
            return {
                "name": method_name,
                "has_default_impl": has_body
            }
            
        except Exception as e:
            logger.warning(f"Failed to extract trait method: {e}")
            return None
    
    def _extract_impl_method(self, method_node: tree_sitter.Node, content: str) -> Optional[Dict[str, Any]]:
        """Extract information about an impl method"""
        try:
            # Get method name
            name_node = self.find_child_by_type(method_node, "identifier")
            if not name_node:
                return None
            
            method_name = self.get_node_text(name_node, content)
            
            # Check visibility
            visibility = self._determine_visibility(method_node, content)
            
            return {
                "name": method_name,
                "visibility": visibility.value
            }
            
        except Exception as e:
            logger.warning(f"Failed to extract impl method: {e}")
            return None
    
    def _extract_parameter_info(self, param_node: tree_sitter.Node, content: str) -> List[Dict[str, Any]]:
        """Extract parameter information from parameter declaration"""
        params = []
        
        try:
            # Get parameter name
            name_node = self.find_child_by_type(param_node, "identifier")
            if not name_node:
                return params
            
            param_name = self.get_node_text(name_node, content)
            
            # Get parameter type - look for type after the colon
            param_type = "unknown"
            
            # In Rust, parameters have structure: identifier : type
            # Find all children that are not identifiers or punctuation
            for child in param_node.children:
                if child.type in ["primitive_type", "type_identifier", "generic_type", "reference_type", "scoped_type_identifier"]:
                    param_type = self.get_node_text(child, content)
                    break
            
            # If still unknown, try looking deeper in the tree
            if param_type == "unknown":
                # Look for any node that might contain type information
                for child in param_node.children:
                    if child.type not in ["identifier", ":", "&", "mut"]:
                        # Check if this child has type information
                        type_child = self.find_child_by_type(child, ["primitive_type", "type_identifier", "generic_type", "reference_type", "scoped_type_identifier"])
                        if type_child:
                            param_type = self.get_node_text(type_child, content)
                            break
                        elif child.type not in ["(", ")", ",", " ", "\n"]:
                            # If it's not punctuation, it might be the type itself
                            param_type = self.get_node_text(child, content)
                            break
            
            params.append({
                "name": param_name,
                "type": param_type
            })
            
        except Exception as e:
            logger.warning(f"Failed to extract parameter: {e}")
        
        return params
    
    # Relation extraction methods
    def _extract_use_relations(
        self,
        tree: tree_sitter.Tree,
        content: str,
        entities: List[Entity],
        file_path: Path
    ) -> List[Relation]:
        """Extract use relations (imports)"""
        relations = []
        
        use_entities = [e for e in entities if e.entity_type == EntityType.IMPORT]
        
        for use_entity in use_entities:
            try:
                # Create a relation pointing to the imported entity
                target_entity_id = f"rust::entity::{use_entity.metadata.get('imported_name', use_entity.name)}"
                
                relation = Relation(
                    id=f"rust::relation::uses::{use_entity.id}",
                    source_entity_id=f"rust::file::{file_path.stem}",
                    target_entity_id=target_entity_id,
                    relation_type=RelationType.IMPORTS,
                    metadata={
                        "import_path": use_entity.metadata.get("import_path"),
                        "is_standard_library": use_entity.metadata.get("is_standard_library", False)
                    }
                )
                
                relations.append(relation)
                
            except Exception as e:
                logger.warning(f"Failed to extract use relation: {e}")
        
        return relations
    
    def _extract_impl_relations(
        self,
        tree: tree_sitter.Tree,
        content: str,
        entities: List[Entity],
        file_path: Path
    ) -> List[Relation]:
        """Extract trait implementation relations"""
        relations = []
        
        impl_entities = [e for e in entities if e.entity_type == EntityType.IMPLEMENTATION]
        
        for impl_entity in impl_entities:
            try:
                trait_name = impl_entity.metadata.get("trait_name")
                impl_type = impl_entity.metadata.get("impl_type")
                
                if trait_name and impl_type:
                    # Create IMPLEMENTS relation
                    relation = Relation(
                        id=f"rust::relation::implements::{impl_entity.id}",
                        source_entity_id=f"rust::entity::{impl_type}",
                        target_entity_id=f"rust::entity::{trait_name}",
                        relation_type=RelationType.IMPLEMENTS,
                        metadata={
                            "impl_type": impl_type,
                            "trait_name": trait_name
                        }
                    )
                    
                    relations.append(relation)
                
            except Exception as e:
                logger.warning(f"Failed to extract impl relation: {e}")
        
        return relations
    
    def _extract_call_relations(
        self,
        tree: tree_sitter.Tree,
        content: str,
        entities: List[Entity],
        file_path: Path
    ) -> List[Relation]:
        """Extract function call relations"""
        relations = []
        
        # This is complex - for now, return empty list
        # TODO: Implement call relation extraction
        
        return relations

