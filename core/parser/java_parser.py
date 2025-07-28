"""
Java parser implementation using Tree-sitter.
"""

import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import tree_sitter
except ImportError:
    tree_sitter = None

from .tree_sitter_base import TreeSitterBase
from .registry import register_parser
from ..models.entities import Entity, EntityType, SourceLocation, Visibility

logger = logging.getLogger(__name__)


@register_parser("java", [".java"])
class JavaParser(TreeSitterBase):
    """Java parser using Tree-sitter for comprehensive entity extraction."""
    
    def __init__(self):
        super().__init__("java")
        logger.debug("Java parser initialized")
    
    def get_supported_extensions(self) -> List[str]:
        return [".java"]
    
    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.get_supported_extensions()
    
    def extract_entities(self, tree: tree_sitter.Tree, content: str, file_path: Path) -> List[Entity]:
        """Extract Java entities from the AST."""
        entities = []
        
        try:
            # Extract different types of entities
            entities.extend(self._extract_packages(tree, content, file_path))
            entities.extend(self._extract_imports(tree, content, file_path))
            entities.extend(self._extract_classes(tree, content, file_path))
            entities.extend(self._extract_interfaces(tree, content, file_path))
            entities.extend(self._extract_enums(tree, content, file_path))
            entities.extend(self._extract_methods(tree, content, file_path))
            entities.extend(self._extract_fields(tree, content, file_path))
            entities.extend(self._extract_annotations(tree, content, file_path))
            
            logger.debug(f"Extracted {len(entities)} entities from {file_path}")
            return entities
            
        except Exception as e:
            logger.error(f"Failed to extract entities from {file_path}: {e}")
            return []
    
    def extract_relations(self, tree: tree_sitter.Tree, content: str, entities: List[Entity], file_path: Path) -> List:
        """Extract relations between Java entities."""
        # TODO: Implement relation extraction for Java
        # This is a placeholder for now - relations will be implemented in Phase 5
        return []
    
    def _extract_packages(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract package declarations"""
        packages = []
        
        # Find package declarations
        package_nodes = self.find_nodes_by_type(tree, ["package_declaration"])
        
        for package_node in package_nodes:
            try:
                # Get package name
                name_node = self.find_child_by_type(package_node, "scoped_identifier")
                if not name_node:
                    name_node = self.find_child_by_type(package_node, "identifier")
                
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
                
                source_code = self.get_node_text(package_node, content)
                
                entity_id = f"file://{file_path}::java_package::{package_name}::{location.start_line}"
                package_entity = Entity(
                    id=entity_id,
                    name=package_name,
                    entity_type=EntityType.MODULE,
                    location=location,
                    signature=f"package {package_name}",
                    source_code=source_code,
                    source_hash=hashlib.md5(source_code.encode('utf-8')).hexdigest()[:16],
                    visibility=Visibility.PUBLIC,
                    qualified_name=package_name,
                    metadata={
                        "language": "java",
                        "ast_node_type": "package_declaration",
                        "package_name": package_name
                    }
                )
                
                packages.append(package_entity)
                
            except Exception as e:
                logger.warning(f"Failed to extract package: {e}")
        
        return packages
    
    def _extract_imports(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract import statements"""
        imports = []
        
        # Find import declarations
        import_nodes = self.find_nodes_by_type(tree, ["import_declaration"])
        
        for import_node in import_nodes:
            try:
                # Get the full import text first
                full_import_text = self.get_node_text(import_node, content)
                
                # Check if static import
                is_static = "static" in full_import_text
                
                # Check if wildcard import
                is_wildcard = "*" in full_import_text
                
                # Get import name - handle wildcard imports specially
                import_name = None
                if is_wildcard:
                    # For wildcard imports, find the asterisk node or build the name manually
                    asterisk_node = self.find_child_by_type(import_node, "asterisk")
                    if asterisk_node:
                        # Find the scoped identifier before the asterisk
                        for child in import_node.children:
                            if child.type in ["scoped_identifier", "identifier"]:
                                base_name = self.get_node_text(child, content)
                                import_name = f"{base_name}.*"
                                break
                    
                    # Fallback: extract from the full text
                    if not import_name:
                        import_parts = full_import_text.strip().split()
                        for part in import_parts:
                            if "*" in part:
                                import_name = part.rstrip(";")
                                break
                
                if not import_name:
                    # Regular import - get the identifier/scoped_identifier
                    name_node = self.find_child_by_type(import_node, "scoped_identifier")
                    if not name_node:
                        name_node = self.find_child_by_type(import_node, "identifier")
                    
                    if name_node:
                        import_name = self.get_node_text(name_node, content)
                
                if not import_name:
                    continue
                
                # Create location
                location = SourceLocation(
                    file_path=file_path,
                    start_line=import_node.start_point[0] + 1,
                    end_line=import_node.end_point[0] + 1,
                    start_column=import_node.start_point[1],
                    end_column=import_node.end_point[1],
                    start_byte=import_node.start_byte,
                    end_byte=import_node.end_byte
                )
                
                source_code = self.get_node_text(import_node, content)
                
                entity_id = f"file://{file_path}::java_import::{import_name}::{location.start_line}"
                import_entity = Entity(
                    id=entity_id,
                    name=import_name,
                    entity_type=EntityType.IMPORT,
                    location=location,
                    signature=source_code.strip(),
                    source_code=source_code,
                    source_hash=hashlib.md5(source_code.encode('utf-8')).hexdigest()[:16],
                    visibility=Visibility.PUBLIC,
                    qualified_name=import_name,
                    metadata={
                        "language": "java",
                        "ast_node_type": "import_declaration",
                        "import_name": import_name,
                        "is_static": is_static,
                        "is_wildcard": is_wildcard
                    }
                )
                
                imports.append(import_entity)
                
            except Exception as e:
                logger.warning(f"Failed to extract import: {e}")
        
        return imports
    
    def _extract_classes(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract class definitions"""
        classes = []
        
        # Find class declarations
        class_nodes = self.find_nodes_by_type(tree, ["class_declaration"])
        
        for class_node in class_nodes:
            try:
                # Get class name
                name_node = self.find_child_by_type(class_node, "identifier")
                if not name_node:
                    continue
                
                class_name = self.get_node_text(name_node, content)
                
                # Get visibility
                visibility = self._determine_visibility(class_node, content)
                
                # Check modifiers
                is_abstract = self._has_modifier(class_node, "abstract", content)
                is_final = self._has_modifier(class_node, "final", content)
                is_static = self._has_modifier(class_node, "static", content)
                
                # Get superclass
                superclass = None
                superclass_node = self.find_child_by_type(class_node, "superclass")
                if superclass_node:
                    type_node = self.find_child_by_type(superclass_node, "type_identifier")
                    if type_node:
                        superclass = self.get_node_text(type_node, content)
                
                # Get interfaces
                interfaces = []
                super_interfaces_node = self.find_child_by_type(class_node, "super_interfaces")
                if super_interfaces_node:
                    # Look for type_list first, then type_identifier
                    type_list_node = self.find_child_by_type(super_interfaces_node, "type_list")
                    if type_list_node:
                        interface_nodes = self.find_children_by_type(type_list_node, "type_identifier")
                        interfaces = [self.get_node_text(node, content) for node in interface_nodes]
                    else:
                        # Fallback: direct type_identifier children
                        interface_nodes = self.find_children_by_type(super_interfaces_node, "type_identifier")
                        interfaces = [self.get_node_text(node, content) for node in interface_nodes]
                
                # Get type parameters (generics)
                type_parameters = []
                type_parameter_names = []  # Just the names for metadata
                type_params_node = self.find_child_by_type(class_node, "type_parameters")
                if type_params_node:
                    param_nodes = self.find_children_by_type(type_params_node, "type_parameter")
                    for param_node in param_nodes:
                        # Get the parameter name
                        param_name_node = self.find_child_by_type(param_node, "type_identifier")
                        if param_name_node:
                            param_name = self.get_node_text(param_name_node, content)
                            type_parameter_names.append(param_name)
                        
                        # Get the full parameter text including constraints
                        param_text = self.get_node_text(param_node, content)
                        type_parameters.append(param_text)
                
                # Extract methods and fields within the class
                methods = []
                fields = []
                class_body = self.find_child_by_type(class_node, "class_body")
                if class_body:
                    method_nodes = self.find_children_by_type(class_body, "method_declaration")
                    for method_node in method_nodes:
                        method_info = self._extract_method_info(method_node, content)
                        if method_info:
                            methods.append(method_info)
                    
                    field_nodes = self.find_children_by_type(class_body, "field_declaration")
                    for field_node in field_nodes:
                        field_infos = self._extract_field_info(field_node, content)
                        fields.extend(field_infos)
                
                # Create location
                location = SourceLocation(
                    file_path=file_path,
                    start_line=class_node.start_point[0] + 1,
                    end_line=class_node.end_point[0] + 1,
                    start_column=class_node.start_point[1],
                    end_column=class_node.end_point[1],
                    start_byte=class_node.start_byte,
                    end_byte=class_node.end_byte
                )
                
                source_code = self.get_node_text(class_node, content)
                
                # Build signature
                signature_parts = []
                if visibility != Visibility.PACKAGE_PRIVATE:
                    signature_parts.append(visibility.value)
                if is_abstract:
                    signature_parts.append("abstract")
                if is_final:
                    signature_parts.append("final")
                if is_static:
                    signature_parts.append("static")
                signature_parts.append("class")
                signature_parts.append(class_name)
                if type_parameters:
                    signature_parts.append(f"<{', '.join(type_parameters)}>")
                
                signature = " ".join(signature_parts)
                
                entity_id = f"file://{file_path}::java_class::{class_name}::{location.start_line}"
                class_entity = Entity(
                    id=entity_id,
                    name=class_name,
                    entity_type=EntityType.CLASS,
                    location=location,
                    signature=signature,
                    source_code=source_code,
                    source_hash=hashlib.md5(source_code.encode('utf-8')).hexdigest()[:16],
                    visibility=visibility,
                    qualified_name=class_name,  # TODO: Build full qualified name
                    metadata={
                        "language": "java",
                        "ast_node_type": "class_declaration",
                        "class_name": class_name,
                        "is_abstract": is_abstract,
                        "is_final": is_final,
                        "is_static": is_static,
                        "superclass": superclass,
                        "interfaces": interfaces,
                        "type_parameters": type_parameter_names,
                        "methods": methods,
                        "fields": fields,
                        "method_count": len(methods),
                        "field_count": len(fields)
                    }
                )
                
                classes.append(class_entity)
                
            except Exception as e:
                logger.warning(f"Failed to extract class: {e}")
        
        return classes
    
    def _extract_interfaces(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract interface definitions"""
        interfaces = []
        
        # Find interface declarations
        interface_nodes = self.find_nodes_by_type(tree, ["interface_declaration"])
        
        for interface_node in interface_nodes:
            try:
                # Get interface name
                name_node = self.find_child_by_type(interface_node, "identifier")
                if not name_node:
                    continue
                
                interface_name = self.get_node_text(name_node, content)
                
                # Get visibility
                visibility = self._determine_visibility(interface_node, content)
                
                # Get extends
                extends = []
                extends_interfaces_node = self.find_child_by_type(interface_node, "extends_interfaces")
                if extends_interfaces_node:
                    # Look for type_list first, then type_identifier
                    type_list_node = self.find_child_by_type(extends_interfaces_node, "type_list")
                    if type_list_node:
                        type_nodes = self.find_children_by_type(type_list_node, "type_identifier")
                        extends = [self.get_node_text(node, content) for node in type_nodes]
                    else:
                        # Fallback: direct type_identifier children
                        type_nodes = self.find_children_by_type(extends_interfaces_node, "type_identifier")
                        extends = [self.get_node_text(node, content) for node in type_nodes]
                
                # Get type parameters
                type_parameters = []
                type_parameter_names = []  # Just the names for metadata
                type_params_node = self.find_child_by_type(interface_node, "type_parameters")
                if type_params_node:
                    param_nodes = self.find_children_by_type(type_params_node, "type_parameter")
                    for param_node in param_nodes:
                        # Get the parameter name
                        param_name_node = self.find_child_by_type(param_node, "type_identifier")
                        if param_name_node:
                            param_name = self.get_node_text(param_name_node, content)
                            type_parameter_names.append(param_name)
                        
                        # Get the full parameter text including constraints
                        param_text = self.get_node_text(param_node, content)
                        type_parameters.append(param_text)
                
                # Extract methods within the interface
                methods = []
                interface_body = self.find_child_by_type(interface_node, "interface_body")
                if interface_body:
                    method_nodes = self.find_children_by_type(interface_body, "method_declaration")
                    for method_node in method_nodes:
                        method_info = self._extract_method_info(method_node, content)
                        if method_info:
                            methods.append(method_info)
                
                # Create location
                location = SourceLocation(
                    file_path=file_path,
                    start_line=interface_node.start_point[0] + 1,
                    end_line=interface_node.end_point[0] + 1,
                    start_column=interface_node.start_point[1],
                    end_column=interface_node.end_point[1],
                    start_byte=interface_node.start_byte,
                    end_byte=interface_node.end_byte
                )
                
                source_code = self.get_node_text(interface_node, content)
                
                # Build signature
                signature_parts = []
                if visibility != Visibility.PACKAGE_PRIVATE:
                    signature_parts.append(visibility.value)
                signature_parts.append("interface")
                signature_parts.append(interface_name)
                if type_parameters:
                    signature_parts.append(f"<{', '.join(type_parameters)}>")
                
                signature = " ".join(signature_parts)
                
                entity_id = f"file://{file_path}::java_interface::{interface_name}::{location.start_line}"
                interface_entity = Entity(
                    id=entity_id,
                    name=interface_name,
                    entity_type=EntityType.INTERFACE,
                    location=location,
                    signature=signature,
                    source_code=source_code,
                    source_hash=hashlib.md5(source_code.encode('utf-8')).hexdigest()[:16],
                    visibility=visibility,
                    qualified_name=interface_name,
                    metadata={
                        "language": "java",
                        "ast_node_type": "interface_declaration",
                        "interface_name": interface_name,
                        "extends": extends,
                        "type_parameters": type_parameter_names,
                        "methods": methods,
                        "method_count": len(methods)
                    }
                )
                
                interfaces.append(interface_entity)
                
            except Exception as e:
                logger.warning(f"Failed to extract interface: {e}")
        
        return interfaces
    
    def _extract_enums(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract enum definitions"""
        enums = []
        
        # Find enum declarations
        enum_nodes = self.find_nodes_by_type(tree, ["enum_declaration"])
        
        for enum_node in enum_nodes:
            try:
                # Get enum name
                name_node = self.find_child_by_type(enum_node, "identifier")
                if not name_node:
                    continue
                
                enum_name = self.get_node_text(name_node, content)
                
                # Get visibility
                visibility = self._determine_visibility(enum_node, content)
                
                # Get enum constants
                constants = []
                enum_body = self.find_child_by_type(enum_node, "enum_body")
                if enum_body:
                    constant_nodes = self.find_children_by_type(enum_body, "enum_constant")
                    for constant_node in constant_nodes:
                        constant_name_node = self.find_child_by_type(constant_node, "identifier")
                        if constant_name_node:
                            constants.append(self.get_node_text(constant_name_node, content))
                
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
                
                source_code = self.get_node_text(enum_node, content)
                
                # Build signature
                signature_parts = []
                if visibility != Visibility.PACKAGE_PRIVATE:
                    signature_parts.append(visibility.value)
                signature_parts.append("enum")
                signature_parts.append(enum_name)
                
                signature = " ".join(signature_parts)
                
                entity_id = f"file://{file_path}::java_enum::{enum_name}::{location.start_line}"
                enum_entity = Entity(
                    id=entity_id,
                    name=enum_name,
                    entity_type=EntityType.ENUM,
                    location=location,
                    signature=signature,
                    source_code=source_code,
                    source_hash=hashlib.md5(source_code.encode('utf-8')).hexdigest()[:16],
                    visibility=visibility,
                    qualified_name=enum_name,
                    metadata={
                        "language": "java",
                        "ast_node_type": "enum_declaration",
                        "enum_name": enum_name,
                        "constants": constants,
                        "constant_count": len(constants)
                    }
                )
                
                enums.append(enum_entity)
                
            except Exception as e:
                logger.warning(f"Failed to extract enum: {e}")
        
        return enums
    
    def _extract_methods(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract method definitions"""
        methods = []
        
        # Find method declarations
        method_nodes = self.find_nodes_by_type(tree, ["method_declaration"])
        
        for method_node in method_nodes:
            try:
                # Get method name
                name_node = self.find_child_by_type(method_node, "identifier")
                if not name_node:
                    continue
                
                method_name = self.get_node_text(name_node, content)
                
                # Get visibility
                visibility = self._determine_visibility(method_node, content)
                
                # Check modifiers
                is_static = self._has_modifier(method_node, "static", content)
                is_final = self._has_modifier(method_node, "final", content)
                is_abstract = self._has_modifier(method_node, "abstract", content)
                is_synchronized = self._has_modifier(method_node, "synchronized", content)
                
                # Extract parameters
                parameters = []
                params_node = self.find_child_by_type(method_node, "formal_parameters")
                if params_node:
                    param_nodes = self.find_children_by_type(params_node, "formal_parameter")
                    for param_node in param_nodes:
                        param_info = self._extract_parameter_info(param_node, content)
                        if param_info:
                            parameters.extend(param_info)
                
                # Extract return type
                return_type = None
                # Look for type before method name
                for child in method_node.children:
                    if child.type in ["type_identifier", "generic_type", "array_type", "void_type", "primitive_type"]:
                        return_type = self.get_node_text(child, content)
                        break
                
                # Extract throws
                throws = []
                throws_node = self.find_child_by_type(method_node, "throws")
                if throws_node:
                    exception_nodes = self.find_children_by_type(throws_node, "type_identifier")
                    throws = [self.get_node_text(node, content) for node in exception_nodes]
                
                # Create location
                location = SourceLocation(
                    file_path=file_path,
                    start_line=method_node.start_point[0] + 1,
                    end_line=method_node.end_point[0] + 1,
                    start_column=method_node.start_point[1],
                    end_column=method_node.end_point[1],
                    start_byte=method_node.start_byte,
                    end_byte=method_node.end_byte
                )
                
                source_code = self.get_node_text(method_node, content)
                
                # Build signature
                param_strs = [f"{p['type']} {p['name']}" for p in parameters]
                signature = f"{method_name}({', '.join(param_strs)})"
                if return_type:
                    signature = f"{return_type} {signature}"
                
                entity_id = f"file://{file_path}::java_method::{method_name}::{location.start_line}"
                method_entity = Entity(
                    id=entity_id,
                    name=method_name,
                    entity_type=EntityType.METHOD,
                    location=location,
                    signature=signature,
                    source_code=source_code,
                    source_hash=hashlib.md5(source_code.encode('utf-8')).hexdigest()[:16],
                    visibility=visibility,
                    qualified_name=method_name,
                    metadata={
                        "language": "java",
                        "ast_node_type": "method_declaration",
                        "method_name": method_name,
                        "parameters": parameters,
                        "return_type": return_type,
                        "is_static": is_static,
                        "is_final": is_final,
                        "is_abstract": is_abstract,
                        "is_synchronized": is_synchronized,
                        "throws": throws,
                        "parameter_count": len(parameters)
                    }
                )
                
                methods.append(method_entity)
                
            except Exception as e:
                logger.warning(f"Failed to extract method: {e}")
        
        return methods
    
    def _extract_fields(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract field definitions"""
        fields = []
        
        # Find field declarations
        field_nodes = self.find_nodes_by_type(tree, ["field_declaration"])
        
        for field_node in field_nodes:
            try:
                field_infos = self._extract_field_info(field_node, content)
                
                for field_info in field_infos:
                    # Create location
                    location = SourceLocation(
                        file_path=file_path,
                        start_line=field_node.start_point[0] + 1,
                        end_line=field_node.end_point[0] + 1,
                        start_column=field_node.start_point[1],
                        end_column=field_node.end_point[1],
                        start_byte=field_node.start_byte,
                        end_byte=field_node.end_byte
                    )
                    
                    source_code = self.get_node_text(field_node, content)
                    
                    entity_id = f"file://{file_path}::java_field::{field_info['name']}::{location.start_line}"
                    field_entity = Entity(
                        id=entity_id,
                        name=field_info['name'],
                        entity_type=EntityType.VARIABLE,
                        location=location,
                        signature=f"{field_info['type']} {field_info['name']}",
                        source_code=source_code,
                        source_hash=hashlib.md5(source_code.encode('utf-8')).hexdigest()[:16],
                        visibility=field_info['visibility'],
                        qualified_name=field_info['name'],
                        metadata={
                            "language": "java",
                            "ast_node_type": "field_declaration",
                            "field_name": field_info['name'],
                            "field_type": field_info['type'],
                            "is_static": field_info['is_static'],
                            "is_final": field_info['is_final'],
                            "is_volatile": field_info['is_volatile'],
                            "is_transient": field_info['is_transient']
                        }
                    )
                    
                    fields.append(field_entity)
                
            except Exception as e:
                logger.warning(f"Failed to extract field: {e}")
        
        return fields
    
    def _extract_annotations(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract annotation definitions"""
        annotations = []
        
        # Find annotation type declarations
        annotation_nodes = self.find_nodes_by_type(tree, ["annotation_type_declaration"])
        
        for annotation_node in annotation_nodes:
            try:
                # Get annotation name
                name_node = self.find_child_by_type(annotation_node, "identifier")
                if not name_node:
                    continue
                
                annotation_name = self.get_node_text(name_node, content)
                
                # Get visibility
                visibility = self._determine_visibility(annotation_node, content)
                
                # Create location
                location = SourceLocation(
                    file_path=file_path,
                    start_line=annotation_node.start_point[0] + 1,
                    end_line=annotation_node.end_point[0] + 1,
                    start_column=annotation_node.start_point[1],
                    end_column=annotation_node.end_point[1],
                    start_byte=annotation_node.start_byte,
                    end_byte=annotation_node.end_byte
                )
                
                source_code = self.get_node_text(annotation_node, content)
                
                entity_id = f"file://{file_path}::java_annotation::{annotation_name}::{location.start_line}"
                annotation_entity = Entity(
                    id=entity_id,
                    name=annotation_name,
                    entity_type=EntityType.CLASS,  # Annotations are special classes in Java
                    location=location,
                    signature=f"@interface {annotation_name}",
                    source_code=source_code,
                    source_hash=hashlib.md5(source_code.encode('utf-8')).hexdigest()[:16],
                    visibility=visibility,
                    qualified_name=annotation_name,
                    metadata={
                        "language": "java",
                        "ast_node_type": "annotation_type_declaration",
                        "annotation_name": annotation_name,
                        "is_annotation": True
                    }
                )
                
                annotations.append(annotation_entity)
                
            except Exception as e:
                logger.warning(f"Failed to extract annotation: {e}")
        
        return annotations
    
    def _determine_visibility(self, node: tree_sitter.Node, content: str) -> Visibility:
        """Determine the visibility of a Java entity"""
        modifiers_node = self.find_child_by_type(node, "modifiers")
        if modifiers_node:
            modifiers_text = self.get_node_text(modifiers_node, content)
            if "public" in modifiers_text:
                return Visibility.PUBLIC
            elif "private" in modifiers_text:
                return Visibility.PRIVATE
            elif "protected" in modifiers_text:
                return Visibility.PROTECTED
        
        # Default visibility in Java is package-private
        return Visibility.PACKAGE_PRIVATE
    
    def _has_modifier(self, node: tree_sitter.Node, modifier: str, content: str) -> bool:
        """Check if a node has a specific modifier"""
        modifiers_node = self.find_child_by_type(node, "modifiers")
        if modifiers_node:
            modifiers_text = self.get_node_text(modifiers_node, content)
            return modifier in modifiers_text
        return False
    
    def _extract_method_info(self, method_node: tree_sitter.Node, content: str) -> Optional[Dict[str, Any]]:
        """Extract information about a method"""
        try:
            # Get method name
            name_node = self.find_child_by_type(method_node, "identifier")
            if not name_node:
                return None
            
            method_name = self.get_node_text(name_node, content)
            
            # Get visibility
            visibility = self._determine_visibility(method_node, content)
            
            return {
                "name": method_name,
                "visibility": visibility.value
            }
            
        except Exception as e:
            logger.warning(f"Failed to extract method info: {e}")
            return None
    
    def _extract_field_info(self, field_node: tree_sitter.Node, content: str) -> List[Dict[str, Any]]:
        """Extract information about fields (can have multiple declarators)"""
        fields = []
        
        try:
            # Get field type
            field_type = "unknown"
            for child in field_node.children:
                if child.type in ["type_identifier", "generic_type", "array_type", "primitive_type"]:
                    field_type = self.get_node_text(child, content)
                    break
            
            # Get visibility and modifiers
            visibility = self._determine_visibility(field_node, content)
            is_static = self._has_modifier(field_node, "static", content)
            is_final = self._has_modifier(field_node, "final", content)
            is_volatile = self._has_modifier(field_node, "volatile", content)
            is_transient = self._has_modifier(field_node, "transient", content)
            
            # Get variable declarators
            declarators = self.find_children_by_type(field_node, "variable_declarator")
            for declarator in declarators:
                name_node = self.find_child_by_type(declarator, "identifier")
                if name_node:
                    field_name = self.get_node_text(name_node, content)
                    fields.append({
                        "name": field_name,
                        "type": field_type,
                        "visibility": visibility,
                        "is_static": is_static,
                        "is_final": is_final,
                        "is_volatile": is_volatile,
                        "is_transient": is_transient
                    })
            
        except Exception as e:
            logger.warning(f"Failed to extract field info: {e}")
        
        return fields
    
    def _extract_parameter_info(self, param_node: tree_sitter.Node, content: str) -> List[Dict[str, Any]]:
        """Extract parameter information from formal parameter"""
        params = []
        
        try:
            # Get parameter name
            name_node = self.find_child_by_type(param_node, "identifier")
            if not name_node:
                return params
            
            param_name = self.get_node_text(name_node, content)
            
            # Get parameter type
            param_type = "unknown"
            for child in param_node.children:
                if child.type in ["type_identifier", "generic_type", "array_type", "primitive_type"]:
                    param_type = self.get_node_text(child, content)
                    break
            
            params.append({
                "name": param_name,
                "type": param_type
            })
            
        except Exception as e:
            logger.warning(f"Failed to extract parameter: {e}")
        
        return params