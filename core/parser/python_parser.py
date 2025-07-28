"""
Python parser using Tree-sitter for comprehensive entity extraction.

Extracts functions, classes, methods, variables, imports, and relationships
from Python source code with full metadata and type information.
"""

import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Set
import re

try:
    import tree_sitter
    import tree_sitter_python
except ImportError:
    tree_sitter = None
    tree_sitter_python = None

from .tree_sitter_base import TreeSitterBase
from .registry import register_parser
from ..models.entities import (
    Entity, EntityType, Relation, RelationType, 
    SourceLocation, Visibility
)

logger = logging.getLogger(__name__)


@register_parser("python", [".py", ".pyi", ".pyw"])
class PythonParser(TreeSitterBase):
    """
    Comprehensive Python parser with Tree-sitter.
    
    Features:
    - Functions and methods with full signatures
    - Classes with inheritance detection
    - Variables and constants
    - Import analysis
    - Decorator support
    - Async/await patterns
    - Type hints extraction
    - Docstring parsing
    """
    
    # Supported features
    SUPPORTED_FEATURES = [
        "functions", "classes", "methods", "variables", "imports",
        "decorators", "async", "type_hints", "docstrings", "inheritance"
    ]
    
    def __init__(self):
        super().__init__("python")
        self.__version__ = "1.0.0"
        
        # Compiled regex patterns for efficiency
        self._docstring_pattern = re.compile(r'^[\s]*["\'][\s\S]*?["\'][\s]*$', re.MULTILINE)
        self._type_hint_pattern = re.compile(r':\s*([^=]+?)(?:\s*=|$)')
        
        logger.debug("Python parser initialized")
    
    def get_supported_extensions(self) -> List[str]:
        return [".py", ".pyi", ".pyw"]
    
    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.get_supported_extensions()
    
    def extract_entities(
        self, 
        tree: tree_sitter.Tree, 
        content: str,
        file_path: Path
    ) -> List[Entity]:
        """
        Extract Python entities from AST.
        
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
            entities.extend(self._extract_classes(tree, content, file_path))
            entities.extend(self._extract_functions(tree, content, file_path))
            entities.extend(self._extract_methods(tree, content, file_path))
            entities.extend(self._extract_variables(tree, content, file_path))
            entities.extend(self._extract_imports(tree, content, file_path))
            
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
        Extract relationships between Python entities.
        
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
            relations.extend(self._extract_inheritance_relations(tree, content, entities, entity_lookup))
            relations.extend(self._extract_call_relations(tree, content, entities, entity_lookup))
            relations.extend(self._extract_import_relations(tree, content, entities, entity_lookup))
            relations.extend(self._extract_containment_relations(entities))
            relations.extend(self._extract_decorator_relations(tree, content, entities, entity_lookup))
            
            logger.debug(f"Extracted {len(relations)} relations from {file_path}")
            
        except Exception as e:
            logger.error(f"Relation extraction failed for {file_path}: {e}")
        
        return relations
    
    def _extract_classes(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract class definitions"""
        classes = []
        
        # Find all class definitions
        class_nodes = self.find_nodes_by_type(tree, ["class_definition"])
        
        for class_node in class_nodes:
            try:
                class_entity = self._extract_class_entity(class_node, content, file_path)
                if class_entity:
                    classes.append(class_entity)
            except Exception as e:
                logger.warning(f"Failed to extract class at {class_node.start_point}: {e}")
        
        return classes
    
    def _extract_class_entity(
        self, 
        class_node: tree_sitter.Node, 
        content: str, 
        file_path: Path
    ) -> Optional[Entity]:
        """Extract a single class entity"""
        # Find class name
        name_node = self.find_child_by_type(class_node, "identifier")
        if not name_node:
            return None
        
        class_name = self.get_node_text(name_node, content)
        
        # Extract superclasses
        superclasses = []
        superclass_node = self.find_child_by_type(class_node, "argument_list")
        if superclass_node:
            for child in superclass_node.children:
                if child.type == "identifier":
                    superclasses.append(self.get_node_text(child, content))
                elif child.type == "attribute":
                    # Handle module.Class inheritance
                    superclasses.append(self.get_node_text(child, content))
        
        # Extract class body for docstring
        body_node = self.find_child_by_type(class_node, "block")
        docstring = self._extract_docstring(body_node, content) if body_node else None
        
        # Build qualified name (simple for now, can be enhanced with module detection)
        qualified_name = class_name
        
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
        
        # Extract signature
        signature_end = body_node.start_byte if body_node else class_node.end_byte
        signature = content[class_node.start_byte:signature_end].strip()
        if signature.endswith(':'):
            signature = signature[:-1]
        
        # Determine visibility
        visibility = self._determine_visibility(class_name)
        
        # Create entity
        entity = Entity(
            id=f"python::class::{qualified_name}::{location.start_line}",
            name=class_name,
            qualified_name=qualified_name,
            entity_type=EntityType.CLASS,
            location=location,
            signature=signature,
            docstring=docstring,
            source_code=self.get_node_text(class_node, content),
            visibility=visibility,
            metadata={
                "superclasses": superclasses,
                "is_abstract": "abc" in signature.lower() or "Abstract" in class_name,
                "ast_node_type": class_node.type
            }
        )
        
        return entity
    
    def _extract_functions(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract function definitions (not methods)"""
        functions = []
        
        # Find function definitions that are not inside classes
        for node in self.walk_tree(tree):
            if node.type == "function_definition":
                # Check if this function is at module level (not inside a class)
                if not self._is_inside_class(node):
                    try:
                        func_entity = self._extract_function_entity(node, content, file_path, is_method=False)
                        if func_entity:
                            functions.append(func_entity)
                    except Exception as e:
                        logger.warning(f"Failed to extract function at {node.start_point}: {e}")
        
        return functions
    
    def _extract_methods(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract method definitions (functions inside classes)"""
        methods = []
        
        # Find function definitions that are inside classes
        for node in self.walk_tree(tree):
            if node.type == "function_definition":
                # Check if this function is inside a class
                if self._is_inside_class(node):
                    try:
                        method_entity = self._extract_function_entity(node, content, file_path, is_method=True)
                        if method_entity:
                            methods.append(method_entity)
                    except Exception as e:
                        logger.warning(f"Failed to extract method at {node.start_point}: {e}")
        
        return methods
    
    def _extract_function_entity(
        self, 
        func_node: tree_sitter.Node, 
        content: str, 
        file_path: Path,
        is_method: bool = False
    ) -> Optional[Entity]:
        """Extract a function or method entity"""
        # Find function name
        name_node = self.find_child_by_type(func_node, "identifier")
        if not name_node:
            return None
        
        func_name = self.get_node_text(name_node, content)
        
        # Extract parameters
        params_node = self.find_child_by_type(func_node, "parameters")
        parameters = []
        if params_node:
            for child in params_node.children:
                if child.type == "identifier":
                    parameters.append(self.get_node_text(child, content))
                elif child.type == "typed_parameter":
                    # Handle typed parameters
                    param_name_node = self.find_child_by_type(child, "identifier")
                    if param_name_node:
                        param_name = self.get_node_text(param_name_node, content)
                        type_node = self.find_child_by_type(child, "type")
                        if type_node:
                            param_type = self.get_node_text(type_node, content)
                            parameters.append(f"{param_name}: {param_type}")
                        else:
                            parameters.append(param_name)
        
        # Extract return type
        return_type = None
        for child in func_node.children:
            if child.type == "type" and child.prev_sibling and child.prev_sibling.type == "->":
                return_type = self.get_node_text(child, content)
                break
        
        # Check if async
        is_async = False
        parent = func_node.parent
        if parent and parent.type == "decorated_definition":
            # Check for async in decorators or function definition
            func_text = content[parent.start_byte:func_node.start_byte + 20]  # Check beginning
            is_async = "async" in func_text
        else:
            # Check if function itself starts with async
            func_start = content[func_node.start_byte:func_node.start_byte + 20]
            is_async = func_start.strip().startswith("async")
        
        # Extract decorators
        decorators = []
        if parent and parent.type == "decorated_definition":
            for child in parent.children:
                if child.type == "decorator":
                    decorator_text = self.get_node_text(child, content)
                    decorators.append(decorator_text)
        
        # Extract docstring
        body_node = self.find_child_by_type(func_node, "block")
        docstring = self._extract_docstring(body_node, content) if body_node else None
        
        # Build signature
        signature_parts = []
        if decorators:
            signature_parts.extend(decorators)
        
        if is_async:
            signature_parts.append("async")
        
        signature_parts.append("def")
        signature_parts.append(func_name)
        
        if parameters:
            signature_parts.append(f"({', '.join(parameters)})")
        else:
            signature_parts.append("()")
        
        if return_type:
            signature_parts.append(f"-> {return_type}")
        
        signature = " ".join(signature_parts)
        
        # Build qualified name
        class_name = self._get_containing_class_name(func_node, content) if is_method else None
        qualified_name = f"{class_name}.{func_name}" if class_name else func_name
        
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
        
        # Determine visibility and other properties
        visibility = self._determine_visibility(func_name)
        is_test = func_name.startswith("test_") or "test" in func_name.lower()
        is_property = "@property" in " ".join(decorators)
        is_static = "@staticmethod" in " ".join(decorators)
        is_class_method = "@classmethod" in " ".join(decorators)
        
        # Determine entity type
        entity_type = EntityType.METHOD if is_method else EntityType.FUNCTION
        
        # Create entity
        entity = Entity(
            id=f"python::{entity_type.value}::{qualified_name}::{location.start_line}",
            name=func_name,
            qualified_name=qualified_name,
            entity_type=entity_type,
            location=location,
            signature=signature,
            docstring=docstring,
            source_code=self.get_node_text(func_node, content),
            visibility=visibility,
            is_async=is_async,
            is_test=is_test,
            metadata={
                "parameters": parameters,
                "return_type": return_type,
                "decorators": decorators,
                "is_property": is_property,
                "is_static": is_static,
                "is_class_method": is_class_method,
                "containing_class": class_name,
                "ast_node_type": func_node.type
            }
        )
        
        return entity
    
    def _extract_variables(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract variable and constant definitions"""
        variables = []
        
        # Find assignment statements
        assignment_nodes = self.find_nodes_by_type(tree, ["assignment"])
        
        for assign_node in assignment_nodes:
            try:
                # Skip assignments inside functions/methods (focus on module-level)
                if self._is_inside_function_or_method(assign_node):
                    continue
                
                var_entities = self._extract_variable_entities(assign_node, content, file_path)
                variables.extend(var_entities)
                    
            except Exception as e:
                logger.warning(f"Failed to extract variable at {assign_node.start_point}: {e}")
        
        return variables
    
    def _extract_variable_entities(
        self, 
        assign_node: tree_sitter.Node, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract variable entities from assignment node"""
        variables = []
        
        # Find left-hand side (variable names)
        left_side = None
        for child in assign_node.children:
            if child.type in ["identifier", "pattern_list", "attribute"]:
                left_side = child
                break
        
        if not left_side:
            return variables
        
        # Extract variable names
        var_names = []
        if left_side.type == "identifier":
            var_names = [self.get_node_text(left_side, content)]
        elif left_side.type == "pattern_list":
            # Multiple assignment: a, b = values
            for child in left_side.children:
                if child.type == "identifier":
                    var_names.append(self.get_node_text(child, content))
        elif left_side.type == "attribute":
            # Instance variable: self.var = value
            attr_name = None
            for child in left_side.children:
                if child.type == "identifier" and child != left_side.children[0]:
                    attr_name = self.get_node_text(child, content)
                    break
            if attr_name:
                var_names = [attr_name]
        
        # Find right-hand side (value)
        right_side = None
        for child in assign_node.children:
            if child.type not in ["identifier", "pattern_list", "attribute", ":", "="]:
                right_side = child
                break
        
        # Extract type annotation if present
        type_annotation = None
        for child in assign_node.children:
            if child.type == "type":
                type_annotation = self.get_node_text(child, content)
                break
        
        # Create entities for each variable
        for var_name in var_names:
            if not var_name:
                continue
            
            # Determine if this is a constant (uppercase name)
            is_constant = var_name.isupper() and "_" in var_name
            entity_type = EntityType.CONSTANT if is_constant else EntityType.VARIABLE
            
            # Create location
            location = SourceLocation(
                file_path=file_path,
                start_line=assign_node.start_point[0] + 1,
                end_line=assign_node.end_point[0] + 1,
                start_column=assign_node.start_point[1],
                end_column=assign_node.end_point[1],
                start_byte=assign_node.start_byte,
                end_byte=assign_node.end_byte
            )
            
            # Build signature
            signature_parts = [var_name]
            if type_annotation:
                signature_parts.append(f": {type_annotation}")
            if right_side:
                value_text = self.get_node_text(right_side, content)
                # Limit value length for signature
                if len(value_text) > 50:
                    value_text = value_text[:47] + "..."
                signature_parts.append(f" = {value_text}")
            
            signature = "".join(signature_parts)
            
            # Determine visibility
            visibility = self._determine_visibility(var_name)
            
            # Create entity
            entity = Entity(
                id=f"python::{entity_type.value}::{var_name}::{location.start_line}",
                name=var_name,
                qualified_name=var_name,
                entity_type=entity_type,
                location=location,
                signature=signature,
                source_code=self.get_node_text(assign_node, content),
                visibility=visibility,
                metadata={
                    "type_annotation": type_annotation,
                    "value": self.get_node_text(right_side, content) if right_side else None,
                    "is_constant": is_constant,
                    "ast_node_type": assign_node.type
                }
            )
            
            variables.append(entity)
        
        return variables
    
    def _extract_imports(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract import statements"""
        imports = []
        
        # Find import statements
        import_nodes = self.find_nodes_by_type(tree, ["import_statement", "import_from_statement"])
        
        for import_node in import_nodes:
            try:
                import_entities = self._extract_import_entities(import_node, content, file_path)
                imports.extend(import_entities)
            except Exception as e:
                logger.warning(f"Failed to extract import at {import_node.start_point}: {e}")
        
        return imports
    
    def _extract_import_entities(
        self, 
        import_node: tree_sitter.Node, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract import entities from import node"""
        imports = []
        
        if import_node.type == "import_statement":
            # import module, module2
            for child in import_node.children:
                if child.type == "dotted_name":
                    module_name = self.get_node_text(child, content)
                    import_entity = self._create_import_entity(
                        import_node, module_name, None, content, file_path
                    )
                    if import_entity:
                        imports.append(import_entity)
        
        elif import_node.type == "import_from_statement":
            # from module import name, name2
            module_name = None
            imported_names = []
            found_import_keyword = False
            
            for child in import_node.children:
                if child.type == "dotted_name" and module_name is None:
                    # First dotted_name is the module name
                    module_name = self.get_node_text(child, content)
                elif child.type == "import":
                    # Mark that we've found the 'import' keyword
                    found_import_keyword = True
                elif found_import_keyword and child.type == "dotted_name":
                    # After 'import' keyword, dotted_names are imported items
                    name = self.get_node_text(child, content)
                    imported_names.append((name, name, None))
                elif found_import_keyword and child.type == "aliased_import":
                    # Handle 'as' imports
                    name_child = self.find_child_by_type(child, "dotted_name")
                    alias_child = self.find_child_by_type(child, "identifier")
                    if name_child:
                        original_name = self.get_node_text(name_child, content)
                        alias_name = self.get_node_text(alias_child, content) if alias_child else None
                        # For aliased imports, use the alias as the name but store full info in metadata
                        import_name = alias_name if alias_name else original_name
                        imported_names.append((import_name, original_name, alias_name))
                elif child.type == "import_list":
                    # Handle import_list if present (some cases may have this)
                    for grandchild in child.children:
                        if grandchild.type == "dotted_name":
                            name = self.get_node_text(grandchild, content)
                            imported_names.append((name, name, None))
                        elif grandchild.type == "aliased_import":
                            name_child = self.find_child_by_type(grandchild, "dotted_name")
                            alias_child = self.find_child_by_type(grandchild, "identifier")
                            if name_child:
                                original_name = self.get_node_text(name_child, content)
                                alias_name = self.get_node_text(alias_child, content) if alias_child else None
                                import_name = alias_name if alias_name else original_name
                                imported_names.append((import_name, original_name, alias_name))
            
            # Create import entities
            if module_name:
                for import_info in imported_names:
                    if isinstance(import_info, tuple):
                        import_name, original_name, alias_name = import_info
                    else:
                        import_name = original_name = import_info
                        alias_name = None
                    
                    import_entity = self._create_import_entity(
                        import_node, module_name, import_name, content, file_path,
                        original_name=original_name, alias_name=alias_name
                    )
                    if import_entity:
                        imports.append(import_entity)
        
        return imports
    
    def _create_import_entity(
        self,
        import_node: tree_sitter.Node,
        module_name: str,
        imported_name: Optional[str],
        content: str,
        file_path: Path,
        original_name: Optional[str] = None,
        alias_name: Optional[str] = None
    ) -> Optional[Entity]:
        """Create an import entity"""
        # Build import name and signature
        if imported_name:
            name = imported_name
            if alias_name and original_name != alias_name:
                signature = f"from {module_name} import {original_name or imported_name} as {alias_name}"
            else:
                signature = f"from {module_name} import {imported_name}"
        else:
            name = module_name
            signature = f"import {module_name}"
        
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
        
        # Create entity
        entity = Entity(
            id=f"python::import::{name}::{location.start_line}",
            name=name,
            qualified_name=f"{module_name}.{imported_name}" if imported_name else module_name,
            entity_type=EntityType.IMPORT,
            location=location,
            signature=signature,
            source_code=self.get_node_text(import_node, content),
            visibility=Visibility.PUBLIC,  # Imports are typically public
            metadata={
                "module_name": module_name,
                "imported_name": imported_name,
                "original_name": original_name,
                "alias_name": alias_name,
                "is_from_import": imported_name is not None,
                "is_aliased": alias_name is not None,
                "ast_node_type": import_node.type
            }
        )
        
        return entity
    
    # Helper methods for relation extraction
    
    def _build_entity_lookup(self, entities: List[Entity]) -> Dict[str, Entity]:
        """Build lookup table for entities by name"""
        lookup = {}
        for entity in entities:
            lookup[entity.name] = entity
            lookup[entity.qualified_name] = entity
        return lookup
    
    def _extract_inheritance_relations(
        self,
        tree: tree_sitter.Tree,
        content: str,
        entities: List[Entity],
        entity_lookup: Dict[str, Entity]
    ) -> List[Relation]:
        """Extract class inheritance relations"""
        relations = []
        
        # Find all class definitions with superclasses
        class_nodes = self.find_nodes_by_type(tree, ["class_definition"])
        
        for class_node in class_nodes:
            try:
                # Find class name
                class_name_node = self.find_child_by_type(class_node, "identifier")
                if not class_name_node:
                    continue
                
                class_name = self.get_node_text(class_name_node, content)
                class_entity = entity_lookup.get(class_name)
                
                if not class_entity:
                    continue
                
                # Find superclasses
                superclass_node = self.find_child_by_type(class_node, "argument_list")
                if not superclass_node:
                    continue
                
                for child in superclass_node.children:
                    if child.type == "identifier":
                        parent_name = self.get_node_text(child, content)
                        
                        # Create inheritance relation
                        location = SourceLocation(
                            file_path=class_entity.location.file_path,
                            start_line=child.start_point[0] + 1,
                            end_line=child.end_point[0] + 1,
                            start_column=child.start_point[1],
                            end_column=child.end_point[1],
                            start_byte=child.start_byte,
                            end_byte=child.end_byte
                        )
                        
                        relation = Relation.create_inheritance_relation(
                            class_entity.id,
                            f"python::class::{parent_name}::external",  # May be external
                            context=f"class {class_name}({parent_name})",
                            location=location
                        )
                        relations.append(relation)
                        
            except Exception as e:
                logger.warning(f"Failed to extract inheritance for class at {class_node.start_point}: {e}")
        
        return relations
    
    def _extract_call_relations(
        self,
        tree: tree_sitter.Tree,
        content: str,
        entities: List[Entity],
        entity_lookup: Dict[str, Entity]
    ) -> List[Relation]:
        """Extract function/method call relations"""
        relations = []
        
        # Find all call expressions
        call_nodes = self.find_nodes_by_type(tree, ["call"])
        
        for call_node in call_nodes:
            try:
                # Find the function being called
                func_node = call_node.children[0] if call_node.children else None
                if not func_node:
                    continue
                
                called_name = None
                if func_node.type == "identifier":
                    called_name = self.get_node_text(func_node, content)
                elif func_node.type == "attribute":
                    # Method call: obj.method()
                    for child in func_node.children:
                        if child.type == "identifier" and child != func_node.children[0]:
                            called_name = self.get_node_text(child, content)
                            break
                
                if not called_name:
                    continue
                
                # Find the calling context (which function/method contains this call)
                calling_entity = self._find_containing_entity(call_node, entities)
                if not calling_entity:
                    continue
                
                # Create call relation
                location = SourceLocation(
                    file_path=calling_entity.location.file_path,
                    start_line=call_node.start_point[0] + 1,
                    end_line=call_node.end_point[0] + 1,
                    start_column=call_node.start_point[1],
                    end_column=call_node.end_point[1],
                    start_byte=call_node.start_byte,
                    end_byte=call_node.end_byte
                )
                
                relation = Relation.create_call_relation(
                    calling_entity.id,
                    f"python::function::{called_name}::external",  # May be external
                    context=f"{calling_entity.name} calls {called_name}",
                    location=location
                )
                relations.append(relation)
                
            except Exception as e:
                logger.warning(f"Failed to extract call relation at {call_node.start_point}: {e}")
        
        return relations
    
    def _extract_import_relations(
        self,
        tree: tree_sitter.Tree,
        content: str,
        entities: List[Entity],
        entity_lookup: Dict[str, Entity]
    ) -> List[Relation]:
        """Extract import relations"""
        relations = []
        
        # Find import entities
        import_entities = [e for e in entities if e.entity_type == EntityType.IMPORT]
        
        # Create import relations (file imports module)
        for import_entity in entities:
            if import_entity.entity_type == EntityType.IMPORT:
                module_name = import_entity.metadata.get("module_name", "")
                
                relation = Relation.create_import_relation(
                    f"python::file::{import_entity.location.file_path.stem}",
                    f"python::module::{module_name}",
                    context=import_entity.signature,
                    location=import_entity.location
                )
                relations.append(relation)
        
        return relations
    
    def _extract_containment_relations(self, entities: List[Entity]) -> List[Relation]:
        """Extract containment relations (class contains methods, etc.)"""
        relations = []
        
        # Group entities by file and location
        classes = [e for e in entities if e.entity_type == EntityType.CLASS]
        methods = [e for e in entities if e.entity_type == EntityType.METHOD]
        
        # Create containment relations
        for method in methods:
            containing_class_name = method.metadata.get("containing_class")
            if containing_class_name:
                # Find the class entity
                class_entity = None
                for cls in classes:
                    if cls.name == containing_class_name:
                        class_entity = cls
                        break
                
                if class_entity:
                    relation = Relation.create_contains_relation(
                        class_entity.id,
                        method.id,
                        context=f"class {class_entity.name} contains method {method.name}",
                        location=method.location
                    )
                    relations.append(relation)
        
        return relations
    
    def _extract_decorator_relations(
        self,
        tree: tree_sitter.Tree,
        content: str,
        entities: List[Entity],
        entity_lookup: Dict[str, Entity]
    ) -> List[Relation]:
        """Extract decorator relations"""
        relations = []
        
        # Find entities with decorators
        for entity in entities:
            if entity.entity_type in [EntityType.FUNCTION, EntityType.METHOD]:
                decorators = entity.metadata.get("decorators", [])
                
                for decorator in decorators:
                    # Extract decorator name (remove @ and parameters)
                    decorator_name = decorator.strip()
                    if decorator_name.startswith("@"):
                        decorator_name = decorator_name[1:]
                    
                    # Remove parameters if present
                    if "(" in decorator_name:
                        decorator_name = decorator_name[:decorator_name.index("(")]
                    
                    # Create decorator relation
                    relation = Relation(
                        id=f"decorates::{decorator_name}::{entity.id}",
                        relation_type=RelationType.DECORATES,
                        source_entity_id=f"python::decorator::{decorator_name}",
                        target_entity_id=entity.id,
                        context=f"@{decorator_name} decorates {entity.name}"
                    )
                    relations.append(relation)
        
        return relations
    
    # Utility helper methods
    
    def _extract_docstring(self, body_node: tree_sitter.Node, content: str) -> Optional[str]:
        """Extract docstring from function/class body"""
        if not body_node or not body_node.children:
            return None
        
        # Look for string literal as first statement
        for child in body_node.children:
            if child.type == "expression_statement":
                for grandchild in child.children:
                    if grandchild.type == "string":
                        docstring = self.get_node_text(grandchild, content)
                        # Clean up docstring (remove quotes and extra whitespace)
                        if docstring.startswith('"""') or docstring.startswith("'''"):
                            docstring = docstring[3:-3]
                        elif docstring.startswith('"') or docstring.startswith("'"):
                            docstring = docstring[1:-1]
                        return docstring.strip()
        
        return None
    
    def _determine_visibility(self, name: str) -> Visibility:
        """Determine visibility based on naming conventions"""
        if name.startswith("__") and name.endswith("__"):
            return Visibility.PUBLIC  # Magic methods are public
        elif name.startswith("__"):
            return Visibility.PRIVATE  # Name mangled
        elif name.startswith("_"):
            return Visibility.PROTECTED  # Convention for protected
        else:
            return Visibility.PUBLIC
    
    def _is_inside_class(self, node: tree_sitter.Node) -> bool:
        """Check if node is inside a class definition"""
        current = node.parent
        while current:
            if current.type == "class_definition":
                return True
            current = current.parent
        return False
    
    def _is_inside_function_or_method(self, node: tree_sitter.Node) -> bool:
        """Check if node is inside a function or method"""
        current = node.parent
        while current:
            if current.type == "function_definition":
                return True
            current = current.parent
        return False
    
    def _get_containing_class_name(self, node: tree_sitter.Node, content: str) -> Optional[str]:
        """Get the name of the class containing this node"""
        current = node.parent
        while current:
            if current.type == "class_definition":
                name_node = self.find_child_by_type(current, "identifier")
                if name_node:
                    return self.get_node_text(name_node, content)
            current = current.parent
        return None
    
    def _find_containing_entity(self, node: tree_sitter.Node, entities: List[Entity]) -> Optional[Entity]:
        """Find the entity that contains this AST node"""
        for entity in entities:
            if (entity.location.start_byte <= node.start_byte and 
                node.end_byte <= entity.location.end_byte and
                entity.entity_type in [EntityType.FUNCTION, EntityType.METHOD]):
                return entity
        return None