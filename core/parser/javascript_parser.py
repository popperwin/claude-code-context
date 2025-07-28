"""
JavaScript/TypeScript parser using Tree-sitter for comprehensive entity extraction.

Extracts functions, classes, methods, variables, imports, exports, and relationships
from JavaScript and TypeScript source code with full metadata and type information.
"""

import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Set
import re

try:
    import tree_sitter
    import tree_sitter_javascript
    import tree_sitter_typescript
except ImportError:
    tree_sitter = None
    tree_sitter_javascript = None
    tree_sitter_typescript = None

from .tree_sitter_base import TreeSitterBase
from .registry import register_parser
from ..models.entities import (
    Entity, EntityType, Relation, RelationType, 
    SourceLocation, Visibility
)

logger = logging.getLogger(__name__)


@register_parser("javascript", [".js", ".jsx", ".mjs", ".cjs"])
class JavaScriptParser(TreeSitterBase):
    """
    Comprehensive JavaScript parser with Tree-sitter.
    
    Features:
    - Functions and methods with full signatures
    - Classes with inheritance detection
    - Variables and constants (const, let, var)
    - Import/export analysis
    - Arrow functions and async patterns
    - JSX component detection
    - ES6+ module support
    """
    
    # Supported features
    SUPPORTED_FEATURES = [
        "functions", "classes", "methods", "variables", "imports", "exports",
        "arrow_functions", "async", "jsx", "modules", "destructuring"
    ]
    
    def __init__(self):
        super().__init__("javascript")
        self.__version__ = "1.0.0"
        
        # Compiled regex patterns for efficiency
        self._jsdoc_pattern = re.compile(r'/\*\*[\s\S]*?\*/', re.MULTILINE)
        self._arrow_function_pattern = re.compile(r'=>')
        
        logger.debug("JavaScript parser initialized")
    
    def get_supported_extensions(self) -> List[str]:
        return [".js", ".jsx", ".mjs", ".cjs"]
    
    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.get_supported_extensions()
    
    def extract_entities(
        self, 
        tree: tree_sitter.Tree, 
        content: str,
        file_path: Path
    ) -> List[Entity]:
        """
        Extract JavaScript entities from AST.
        
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
            entities.extend(self._extract_exports(tree, content, file_path))
            
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
        Extract relationships between JavaScript entities.
        
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
            relations.extend(self._extract_export_relations(tree, content, entities, entity_lookup))
            
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
        """Extract class declarations"""
        classes = []
        
        # Find all class declarations
        class_nodes = self.find_nodes_by_type(tree, ["class_declaration"])
        
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
        # Find class name (can be identifier or type_identifier for TypeScript)
        name_node = self.find_child_by_type(class_node, "identifier")
        if not name_node:
            name_node = self.find_child_by_type(class_node, "type_identifier")
        if not name_node:
            return None
        
        class_name = self.get_node_text(name_node, content)
        
        # Extract generic parameters (TypeScript)
        generic_parameters = []
        type_params_node = self.find_child_by_type(class_node, "type_parameters")
        if type_params_node:
            for child in type_params_node.children:
                if child.type == "type_parameter":
                    # Extract the parameter name (first identifier in the constraint)
                    param_text = self.get_node_text(child, content)
                    # Get just the parameter name (before 'extends' if present)
                    param_name = param_text.split()[0] if param_text else ""
                    if param_name:
                        generic_parameters.append(param_name)
        
        # Extract superclass
        superclasses = []
        heritage_node = self.find_child_by_type(class_node, "class_heritage")
        if heritage_node:
            # Look for extends clause (TypeScript) or direct identifier (JavaScript)
            extends_clause = self.find_child_by_type(heritage_node, "extends_clause")
            if extends_clause:
                # TypeScript extends clause
                for child in extends_clause.children:
                    if child.type == "identifier" or child.type == "type_identifier":
                        superclasses.append(self.get_node_text(child, content))
                    elif child.type == "member_expression":
                        # Handle module.Class inheritance
                        superclasses.append(self.get_node_text(child, content))
            else:
                # JavaScript direct inheritance - look for identifier after "extends"
                for child in heritage_node.children:
                    if child.type == "identifier" or child.type == "type_identifier":
                        superclasses.append(self.get_node_text(child, content))
                    elif child.type == "member_expression":
                        # Handle module.Class inheritance
                        superclasses.append(self.get_node_text(child, content))
        
        # Extract class body for methods and properties
        body_node = self.find_child_by_type(class_node, "class_body")
        methods = []
        properties = []
        
        if body_node:
            for child in body_node.children:
                if child.type == "method_definition":
                    method_name_node = self.find_child_by_type(child, "property_name")
                    if method_name_node:
                        method_name = self.get_node_text(method_name_node, content)
                        methods.append(method_name)
                elif child.type == "field_definition":
                    field_name_node = self.find_child_by_type(child, "property_name")
                    if field_name_node:
                        field_name = self.get_node_text(field_name_node, content)
                        properties.append(field_name)
        
        # Extract JSDoc comment
        jsdoc = self._extract_jsdoc_comment(class_node, content)
        
        # Build qualified name
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
        if signature.endswith('{'):
            signature = signature[:-1].strip()
        
        # Determine visibility (JavaScript doesn't have explicit visibility)
        visibility = self._determine_visibility(class_name)
        
        # Create entity
        entity = Entity(
            id=f"javascript::class::{qualified_name}::{location.start_line}",
            name=class_name,
            qualified_name=qualified_name,
            entity_type=EntityType.CLASS,
            location=location,
            signature=signature,
            docstring=jsdoc,
            source_code=self.get_node_text(class_node, content),
            visibility=visibility,
            metadata={
                "superclasses": superclasses,
                "methods": methods,
                "properties": properties,
                "generic_parameters": generic_parameters,
                "is_component": self._is_react_component(class_name, content, class_node),
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
        """Extract function declarations and arrow functions"""
        functions = []
        
        # Find function declarations
        function_nodes = self.find_nodes_by_type(tree, [
            "function_declaration", 
            "method_definition",
            "arrow_function"
        ])
        
        # Also find variable declarations with arrow functions
        variable_nodes = self.find_nodes_by_type(tree, ["variable_declarator"])
        for var_node in variable_nodes:
            value_node = self.find_child_by_type(var_node, "arrow_function")
            if value_node:
                function_nodes.append(var_node)
        
        for func_node in function_nodes:
            try:
                # Skip methods inside classes (they'll be handled separately)
                if self._is_inside_class(func_node) and func_node.type == "method_definition":
                    continue
                
                func_entity = self._extract_function_entity(func_node, content, file_path)
                if func_entity:
                    functions.append(func_entity)
            except Exception as e:
                logger.warning(f"Failed to extract function at {func_node.start_point}: {e}")
        
        return functions
    
    def _extract_methods(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract method entities from classes"""
        methods = []
        
        # Find all method definitions
        method_nodes = self.find_nodes_by_type(tree, ["method_definition"])
        
        for method_node in method_nodes:
            try:
                # Only process methods inside classes
                if not self._is_inside_class(method_node):
                    continue
                
                method_entity = self._extract_method_entity(method_node, content, file_path)
                if method_entity:
                    methods.append(method_entity)
            except Exception as e:
                logger.warning(f"Failed to extract method at {method_node.start_point}: {e}")
        
        return methods
    
    def _extract_method_entity(
        self,
        method_node: tree_sitter.Node,
        content: str,
        file_path: Path
    ) -> Optional[Entity]:
        """Extract a method entity from a method_definition node"""
        # Get method name - JavaScript methods use property_identifier
        name_node = self.find_child_by_type(method_node, "property_identifier")
        if not name_node:
            name_node = self.find_child_by_type(method_node, "property_name")
        if not name_node:
            name_node = self.find_child_by_type(method_node, "identifier")
        
        if not name_node:
            return None
        
        method_name = self.get_node_text(name_node, content)
        
        # Get containing class name
        containing_class = self._get_containing_class_name(method_node, content)
        if not containing_class:
            return None
        
        # Extract parameters
        params_node = self.find_child_by_type(method_node, "formal_parameters")
        parameters = []
        if params_node:
            for child in params_node.children:
                if child.type == "identifier":
                    parameters.append(self.get_node_text(child, content))
                elif child.type == "assignment_pattern":
                    # Default parameter
                    param_name_node = self.find_child_by_type(child, "identifier")
                    if param_name_node:
                        param_name = self.get_node_text(param_name_node, content)
                        parameters.append(param_name)
        
        # Build signature
        param_str = ', '.join(parameters)
        signature = f"{method_name}({param_str})"
        
        # Check method type
        method_text = self.get_node_text(method_node, content)
        is_async = "async " in method_text
        is_static = "static " in method_text
        is_getter = method_text.strip().startswith("get ")
        is_setter = method_text.strip().startswith("set ")
        is_constructor = method_name == "constructor"
        
        if is_async:
            signature = f"async {signature}"
        if is_static:
            signature = f"static {signature}"
        if is_getter:
            signature = f"get {signature}"
        if is_setter:
            signature = f"set {signature}"
        
        # Extract JSDoc comment
        jsdoc = self._extract_jsdoc_comment(method_node, content)
        
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
        
        # Determine visibility
        visibility = self._determine_visibility(method_name)
        
        # Create entity
        entity = Entity(
            id=f"javascript::method::{containing_class}::{method_name}::{location.start_line}",
            name=method_name,
            qualified_name=f"{containing_class}.{method_name}",
            entity_type=EntityType.METHOD,
            location=location,
            signature=signature,
            docstring=jsdoc,
            source_code=self.get_node_text(method_node, content),
            visibility=visibility,
            is_async=is_async,
            metadata={
                "containing_class": containing_class,
                "parameters": parameters,
                "is_static": is_static,
                "is_constructor": is_constructor,
                "is_getter": is_getter,
                "is_setter": is_setter,
                "ast_node_type": method_node.type
            }
        )
        
        return entity
    
    def _extract_function_entity(
        self, 
        func_node: tree_sitter.Node, 
        content: str, 
        file_path: Path
    ) -> Optional[Entity]:
        """Extract a function or method entity"""
        # Handle different function types
        if func_node.type == "variable_declarator":
            # Arrow function assigned to variable
            name_node = self.find_child_by_type(func_node, "identifier")
            arrow_node = self.find_child_by_type(func_node, "arrow_function")
            if not name_node or not arrow_node:
                return None
            
            func_name = self.get_node_text(name_node, content)
            params_node = self.find_child_by_type(arrow_node, "formal_parameters")
            body_node = None
            for child in arrow_node.children:
                if child.type in ["statement_block", "expression"]:
                    body_node = child
                    break
            is_arrow = True
            
        elif func_node.type == "function_declaration":
            # Regular function declaration
            name_node = self.find_child_by_type(func_node, "identifier")
            if not name_node:
                return None
            
            func_name = self.get_node_text(name_node, content)
            params_node = self.find_child_by_type(func_node, "formal_parameters")
            body_node = self.find_child_by_type(func_node, "statement_block")
            is_arrow = False
            
        elif func_node.type == "method_definition":
            # Method inside object or class
            name_node = self.find_child_by_type(func_node, "property_name")
            if not name_node:
                return None
            
            func_name = self.get_node_text(name_node, content)
            params_node = self.find_child_by_type(func_node, "formal_parameters")
            body_node = self.find_child_by_type(func_node, "statement_block")
            is_arrow = False
            
        else:
            return None
        
        # Extract parameters
        parameters = []
        if params_node:
            for child in params_node.children:
                if child.type == "identifier":
                    parameters.append(self.get_node_text(child, content))
                elif child.type == "rest_parameter":
                    rest_param = self.find_child_by_type(child, "identifier")
                    if rest_param:
                        parameters.append(f"...{self.get_node_text(rest_param, content)}")
                elif child.type == "assignment_pattern":
                    # Default parameter
                    param_name = self.find_child_by_type(child, "identifier")
                    if param_name:
                        param_text = self.get_node_text(param_name, content)
                        default_value = child.children[-1] if len(child.children) > 1 else None
                        if default_value:
                            default_text = self.get_node_text(default_value, content)
                            parameters.append(f"{param_text} = {default_text}")
                        else:
                            parameters.append(param_text)
        
        # Check if async
        is_async = False
        if func_node.type == "function_declaration":
            # Check if function starts with async
            func_text = content[max(0, func_node.start_byte - 10):func_node.start_byte + 20]
            is_async = "async" in func_text.lower()
        elif func_node.type == "variable_declarator":
            # Check arrow function for async
            arrow_node = self.find_child_by_type(func_node, "arrow_function")
            if arrow_node:
                arrow_text = content[max(0, arrow_node.start_byte - 10):arrow_node.start_byte + 20]
                is_async = "async" in arrow_text.lower()
        
        # Extract JSDoc comment
        jsdoc = self._extract_jsdoc_comment(func_node, content)
        
        # Build signature
        signature_parts = []
        if is_async:
            signature_parts.append("async")
        
        if is_arrow:
            if parameters:
                if len(parameters) == 1 and "=" not in parameters[0] and "..." not in parameters[0]:
                    signature_parts.append(parameters[0])
                else:
                    signature_parts.append(f"({', '.join(parameters)})")
            else:
                signature_parts.append("()")
            signature_parts.append("=>")
        else:
            signature_parts.append("function")
            signature_parts.append(func_name)
            signature_parts.append(f"({', '.join(parameters)})")
        
        signature = " ".join(signature_parts)
        
        # Build qualified name
        containing_class = self._get_containing_class_name(func_node, content)
        qualified_name = f"{containing_class}.{func_name}" if containing_class else func_name
        
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
        is_test = "test" in func_name.lower() or "describe(" in content or "it(" in content
        is_constructor = func_name == "constructor"
        is_getter = func_node.type == "method_definition" and "get " in self.get_node_text(func_node, content)
        is_setter = func_node.type == "method_definition" and "set " in self.get_node_text(func_node, content)
        
        # Determine entity type
        if containing_class:
            entity_type = EntityType.METHOD
        else:
            entity_type = EntityType.FUNCTION
        
        # Create entity
        entity = Entity(
            id=f"javascript::{entity_type.value}::{qualified_name}::{location.start_line}",
            name=func_name,
            qualified_name=qualified_name,
            entity_type=entity_type,
            location=location,
            signature=signature,
            docstring=jsdoc,
            source_code=self.get_node_text(func_node, content),
            visibility=visibility,
            is_async=is_async,
            is_test=is_test,
            metadata={
                "parameters": parameters,
                "is_arrow": is_arrow,
                "is_constructor": is_constructor,
                "is_getter": is_getter,
                "is_setter": is_setter,
                "containing_class": containing_class,
                "returns_jsx": self._returns_jsx_elements(func_node, content),
                "contains_jsx": self._contains_jsx_elements(func_node, content),
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
        """Extract variable and constant declarations"""
        variables = []
        
        # Find variable declarations (var, let, const)
        variable_nodes = self.find_nodes_by_type(tree, [
            "variable_declaration", 
            "lexical_declaration"
        ])
        
        for var_node in variable_nodes:
            try:
                # Skip if inside function (focus on module-level)
                if self._is_inside_function(var_node):
                    continue
                
                var_entities = self._extract_variable_entities(var_node, content, file_path)
                variables.extend(var_entities)
                
            except Exception as e:
                logger.warning(f"Failed to extract variable at {var_node.start_point}: {e}")
        
        return variables
    
    def _extract_variable_entities(
        self, 
        var_node: tree_sitter.Node, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract variable entities from declaration node"""
        variables = []
        
        # Get declaration type (var, let, const)
        declaration_type = "var"
        for child in var_node.children:
            if child.type in ["var", "let", "const"]:
                declaration_type = child.type
                break
        
        # Find variable declarators within this node
        declarators = self._find_child_nodes_by_type(var_node, ["variable_declarator"])
        
        for declarator in declarators:
            name_node = self.find_child_by_type(declarator, "identifier")
            if not name_node:
                continue
            
            var_name = self.get_node_text(name_node, content)
            
            # Extract TypeScript type annotation
            type_annotation = None
            type_annotation_node = self.find_child_by_type(declarator, "type_annotation")
            if type_annotation_node:
                type_text = self.get_node_text(type_annotation_node, content)
                # Remove the leading ': ' from the type annotation
                type_annotation = type_text[2:].strip() if type_text.startswith(': ') else type_text
            
            # Skip if this is a function (handled separately)
            value_node = None
            for child in declarator.children:
                if child != name_node and child.type not in ["=", "type_annotation"]:
                    value_node = child
                    break
            
            if value_node and value_node.type == "arrow_function":
                continue  # Skip arrow functions
            
            # Determine if this is a constant
            is_constant = declaration_type == "const"
            entity_type = EntityType.CONSTANT if is_constant else EntityType.VARIABLE
            
            # Create location
            location = SourceLocation(
                file_path=file_path,
                start_line=declarator.start_point[0] + 1,
                end_line=declarator.end_point[0] + 1,
                start_column=declarator.start_point[1],
                end_column=declarator.end_point[1],
                start_byte=declarator.start_byte,
                end_byte=declarator.end_byte
            )
            
            # Build signature
            signature_parts = [declaration_type, var_name]
            if type_annotation:
                signature_parts.append(f": {type_annotation}")
            if value_node:
                value_text = self.get_node_text(value_node, content)
                # Limit value length for signature
                if len(value_text) > 50:
                    value_text = value_text[:47] + "..."
                signature_parts.append(f"= {value_text}")
            
            signature = " ".join(signature_parts)
            
            # Determine visibility
            visibility = self._determine_visibility(var_name)
            
            # Create entity
            entity = Entity(
                id=f"javascript::{entity_type.value}::{var_name}::{location.start_line}",
                name=var_name,
                qualified_name=var_name,
                entity_type=entity_type,
                location=location,
                signature=signature,
                source_code=self.get_node_text(declarator, content),
                visibility=visibility,
                metadata={
                    "declaration_type": declaration_type,
                    "type_annotation": type_annotation,
                    "value": self.get_node_text(value_node, content) if value_node else None,
                    "is_constant": is_constant,
                    "ast_node_type": declarator.type
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
        import_nodes = self.find_nodes_by_type(tree, ["import_statement"])
        
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
        
        # Find source module
        source_node = self.find_child_by_type(import_node, "string")
        if not source_node:
            return imports
        
        source_text = self.get_node_text(source_node, content)
        # Remove quotes
        module_name = source_text.strip('"\'')
        
        # Find import clause
        import_clause = self.find_child_by_type(import_node, "import_clause")
        if not import_clause:
            # import "module" (side effect only)
            import_entity = self._create_import_entity(
                import_node, module_name, None, content, file_path
            )
            if import_entity:
                imports.append(import_entity)
            return imports
        
        # Handle different import types
        for child in import_clause.children:
            if child.type == "identifier":
                # Default import: import name from "module"
                imported_name = self.get_node_text(child, content)
                import_entity = self._create_import_entity(
                    import_node, module_name, imported_name, content, file_path,
                    import_type="default"
                )
                if import_entity:
                    imports.append(import_entity)
                    
            elif child.type == "namespace_import":
                # Namespace import: import * as name from "module"
                namespace_name = None
                for grandchild in child.children:
                    if grandchild.type == "identifier":
                        namespace_name = self.get_node_text(grandchild, content)
                        break
                
                if namespace_name:
                    import_entity = self._create_import_entity(
                        import_node, module_name, namespace_name, content, file_path,
                        import_type="namespace"
                    )
                    if import_entity:
                        imports.append(import_entity)
                        
            elif child.type == "named_imports":
                # Named imports: import { name1, name2 as alias } from "module"
                import_specs = self._find_child_nodes_by_type(child, ["import_specifier"])
                
                for spec in import_specs:
                    name_node = self.find_child_by_type(spec, "identifier")
                    if not name_node:
                        continue
                    
                    imported_name = self.get_node_text(name_node, content)
                    
                    # Check for alias
                    alias_name = None
                    if len(spec.children) > 1:
                        # Look for "as" alias
                        for i, spec_child in enumerate(spec.children):
                            if spec_child.type == "as" and i + 1 < len(spec.children):
                                alias_node = spec.children[i + 1]
                                if alias_node.type == "identifier":
                                    alias_name = self.get_node_text(alias_node, content)
                                break
                    
                    final_name = alias_name if alias_name else imported_name
                    import_entity = self._create_import_entity(
                        import_node, module_name, final_name, content, file_path,
                        import_type="named",
                        original_name=imported_name,
                        alias_name=alias_name
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
        import_type: str = "default",
        original_name: Optional[str] = None,
        alias_name: Optional[str] = None
    ) -> Optional[Entity]:
        """Create an import entity"""
        # Build import name and signature
        if imported_name:
            name = imported_name
            if import_type == "default":
                signature = f"import {imported_name} from '{module_name}'"
            elif import_type == "namespace":
                signature = f"import * as {imported_name} from '{module_name}'"
            elif import_type == "named":
                if alias_name and original_name != alias_name:
                    signature = f"import {{ {original_name} as {alias_name} }} from '{module_name}'"
                else:
                    signature = f"import {{ {imported_name} }} from '{module_name}'"
            else:
                signature = f"import {imported_name} from '{module_name}'"
        else:
            name = module_name
            signature = f"import '{module_name}'"
        
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
            id=f"javascript::import::{name}::{location.start_line}",
            name=name,
            qualified_name=f"{module_name}.{imported_name}" if imported_name else module_name,
            entity_type=EntityType.IMPORT,
            location=location,
            signature=signature,
            source_code=self.get_node_text(import_node, content),
            visibility=Visibility.PUBLIC,
            metadata={
                "module_name": module_name,
                "imported_name": imported_name,
                "import_type": import_type,
                "original_name": original_name,
                "alias_name": alias_name,
                "is_side_effect": imported_name is None,
                "is_default_import": import_type == "default",
                "is_namespace_import": import_type == "namespace",
                "is_named_import": import_type == "named",
                "ast_node_type": import_node.type
            }
        )
        
        return entity
    
    def _extract_exports(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract export statements"""
        exports = []
        
        # Find export statements
        export_nodes = self.find_nodes_by_type(tree, ["export_statement"])
        
        for export_node in export_nodes:
            try:
                export_entities = self._extract_export_entities(export_node, content, file_path)
                exports.extend(export_entities)
            except Exception as e:
                logger.warning(f"Failed to extract export at {export_node.start_point}: {e}")
        
        return exports
    
    def _extract_export_entities(
        self, 
        export_node: tree_sitter.Node, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract export entities from export node"""
        exports = []
        
        # Handle different export types
        declaration_node = None
        for node_type in ["function_declaration", "class_declaration", "variable_declaration", "lexical_declaration"]:
            declaration_node = self.find_child_by_type(export_node, node_type)
            if declaration_node:
                break
        
        if declaration_node:
            # export function/class/variable
            export_entity = self._create_export_entity(
                export_node, declaration_node, content, file_path, "declaration"
            )
            if export_entity:
                exports.append(export_entity)
        
        # Check for export default
        elif self.find_child_by_type(export_node, "default"):
            # export default identifier/expression
            identifier_node = self.find_child_by_type(export_node, "identifier")
            if identifier_node:
                exported_name = self.get_node_text(identifier_node, content)
                export_entity = self._create_export_entity(
                    export_node, identifier_node, content, file_path, "default",
                    exported_name=exported_name
                )
                if export_entity:
                    exports.append(export_entity)
        
        else:
            # Named exports or re-exports
            export_clause = self.find_child_by_type(export_node, "export_clause")
            if export_clause:
                export_specs = self._find_child_nodes_by_type(export_clause, ["export_specifier"])
                
                for spec in export_specs:
                    name_node = self.find_child_by_type(spec, "identifier")
                    if name_node:
                        exported_name = self.get_node_text(name_node, content)
                        
                        # Check for alias
                        alias_name = None
                        if len(spec.children) > 1:
                            for i, spec_child in enumerate(spec.children):
                                if spec_child.type == "as" and i + 1 < len(spec.children):
                                    alias_node = spec.children[i + 1]
                                    if alias_node.type == "identifier":
                                        alias_name = self.get_node_text(alias_node, content)
                                    break
                        
                        export_entity = self._create_export_entity(
                            export_node, spec, content, file_path, "named",
                            exported_name=exported_name, alias_name=alias_name
                        )
                        if export_entity:
                            exports.append(export_entity)
        
        return exports
    
    def _create_export_entity(
        self,
        export_node: tree_sitter.Node,
        target_node: tree_sitter.Node,
        content: str,
        file_path: Path,
        export_type: str,
        exported_name: Optional[str] = None,
        alias_name: Optional[str] = None
    ) -> Optional[Entity]:
        """Create an export entity"""
        if export_type == "declaration":
            # Extract name from declaration
            if target_node.type in ["function_declaration", "class_declaration"]:
                name_node = self.find_child_by_type(target_node, "identifier")
                if name_node:
                    exported_name = self.get_node_text(name_node, content)
            elif target_node.type in ["variable_declaration", "lexical_declaration"]:
                declarator = self.find_child_by_type(target_node, "variable_declarator")
                if declarator:
                    name_node = self.find_child_by_type(declarator, "identifier")
                    if name_node:
                        exported_name = self.get_node_text(name_node, content)
            
            if not exported_name:
                return None
            
            signature = f"export {self.get_node_text(target_node, content)}"
        
        elif export_type == "default":
            # Export default
            if not exported_name:
                return None
            signature = f"export default {exported_name}"
        
        else:
            # Named export
            if not exported_name:
                return None
            
            if alias_name:
                signature = f"export {{ {exported_name} as {alias_name} }}"
                name = alias_name
            else:
                signature = f"export {{ {exported_name} }}"
                name = exported_name
        
        # Create location
        location = SourceLocation(
            file_path=file_path,
            start_line=export_node.start_point[0] + 1,
            end_line=export_node.end_point[0] + 1,
            start_column=export_node.start_point[1],
            end_column=export_node.end_point[1],
            start_byte=export_node.start_byte,
            end_byte=export_node.end_byte
        )
        
        final_name = alias_name if alias_name else exported_name
        
        # Create entity
        entity = Entity(
            id=f"javascript::export::{final_name}::{location.start_line}",
            name=final_name,
            qualified_name=final_name,
            entity_type=EntityType.EXPORT,
            location=location,
            signature=signature,
            source_code=self.get_node_text(export_node, content),
            visibility=Visibility.PUBLIC,
            metadata={
                "export_type": export_type,
                "exported_name": exported_name,
                "alias_name": alias_name,
                "ast_node_type": export_node.type
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
        
        # Find all class declarations with superclasses
        class_nodes = self.find_nodes_by_type(tree, ["class_declaration"])
        
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
                
                # Find superclass
                heritage_node = self.find_child_by_type(class_node, "class_heritage")
                if not heritage_node:
                    continue
                
                for child in heritage_node.children:
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
                            f"javascript::class::{parent_name}::external",
                            context=f"class {class_name} extends {parent_name}",
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
        call_nodes = self.find_nodes_by_type(tree, ["call_expression"])
        
        for call_node in call_nodes:
            try:
                # Find the function being called
                func_node = call_node.children[0] if call_node.children else None
                if not func_node:
                    continue
                
                called_name = None
                if func_node.type == "identifier":
                    called_name = self.get_node_text(func_node, content)
                elif func_node.type == "member_expression":
                    # Method call: obj.method()
                    property_node = self.find_child_by_type(func_node, "property_name")
                    if property_node:
                        called_name = self.get_node_text(property_node, content)
                
                if not called_name:
                    continue
                
                # Find the calling context
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
                    f"javascript::function::{called_name}::external",
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
        
        # Create import relations (file imports entity)
        for import_entity in import_entities:
            relation = Relation.create_import_relation(
                f"javascript::file::{import_entity.location.file_path.stem}",
                import_entity.id,  # Point to the imported entity, not the module
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
    
    def _extract_export_relations(
        self,
        tree: tree_sitter.Tree,
        content: str,
        entities: List[Entity],
        entity_lookup: Dict[str, Entity]
    ) -> List[Relation]:
        """Extract export relations"""
        relations = []
        
        # Find export entities
        export_entities = [e for e in entities if e.metadata.get("export_type")]
        
        # Create export relations (module exports entity)
        for export_entity in export_entities:
            module_id = f"javascript::module::{export_entity.location.file_path.stem}"
            relation_id = f"export::{module_id}::{export_entity.id}"
            relation = Relation(
                id=relation_id,
                relation_type=RelationType.EXPORTS,
                source_entity_id=module_id,
                target_entity_id=export_entity.id,
                context=export_entity.signature,
                location=export_entity.location
            )
            relations.append(relation)
        
        return relations
    
    # Utility helper methods
    
    def _extract_jsdoc_comment(self, node: tree_sitter.Node, content: str) -> Optional[str]:
        """Extract JSDoc comment preceding a node"""
        # Look for JSDoc comment before this node
        start_pos = max(0, node.start_byte - 500)  # Look back up to 500 characters
        preceding_text = content[start_pos:node.start_byte]
        
        # Find JSDoc comment
        matches = list(self._jsdoc_pattern.finditer(preceding_text))
        if matches:
            # Get the last (closest) JSDoc comment
            last_match = matches[-1]
            jsdoc = last_match.group(0)
            
            # Clean up JSDoc
            lines = jsdoc.split('\n')
            cleaned_lines = []
            for line in lines:
                # Remove /** */ and leading *
                cleaned_line = line.strip()
                if cleaned_line.startswith('/**'):
                    cleaned_line = cleaned_line[3:].strip()
                elif cleaned_line.startswith('*/'):
                    continue
                elif cleaned_line.startswith('*'):
                    cleaned_line = cleaned_line[1:].strip()
                
                if cleaned_line:
                    cleaned_lines.append(cleaned_line)
            
            return '\n'.join(cleaned_lines) if cleaned_lines else None
        
        return None
    
    def _find_child_nodes_by_type(self, node: tree_sitter.Node, node_types: List[str]) -> List[tree_sitter.Node]:
        """Find all child nodes of specified types within a node"""
        matching_nodes = []
        
        def visit_node(current_node: tree_sitter.Node):
            if current_node.type in node_types:
                matching_nodes.append(current_node)
            for child in current_node.children:
                visit_node(child)
        
        visit_node(node)
        return matching_nodes
    
    def _determine_visibility(self, name: str) -> Visibility:
        """Determine visibility based on naming conventions"""
        if name.startswith('_'):
            return Visibility.PRIVATE
        elif name.startswith('#'):  # Private fields in modern JS
            return Visibility.PRIVATE
        else:
            return Visibility.PUBLIC
    
    def _is_inside_class(self, node: tree_sitter.Node) -> bool:
        """Check if node is inside a class declaration"""
        current = node.parent
        while current:
            if current.type == "class_declaration":
                return True
            current = current.parent
        return False
    
    def _is_inside_function(self, node: tree_sitter.Node) -> bool:
        """Check if node is inside a function or method"""
        current = node.parent
        while current:
            if current.type in ["function_declaration", "method_definition", "arrow_function"]:
                return True
            current = current.parent
        return False
    
    def _get_containing_class_name(self, node: tree_sitter.Node, content: str) -> Optional[str]:
        """Get the name of the class containing this node"""
        current = node.parent
        while current:
            if current.type == "class_declaration":
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
    
    def _is_react_component(self, class_name: str, content: str, class_node: tree_sitter.Node) -> bool:
        """Check if a class is a React component"""
        # Check if class extends React.Component or Component
        heritage_node = self.find_child_by_type(class_node, "class_heritage")
        if heritage_node:
            for child in heritage_node.children:
                if child.type == "identifier":
                    parent_name = self.get_node_text(child, content)
                    if "Component" in parent_name:
                        return True
                elif child.type == "member_expression":
                    parent_text = self.get_node_text(child, content)
                    if "React.Component" in parent_text or "Component" in parent_text:
                        return True
        
        # Check for render method
        body_node = self.find_child_by_type(class_node, "class_body")
        if body_node:
            for child in body_node.children:
                if child.type == "method_definition":
                    method_name_node = self.find_child_by_type(child, "property_name")
                    if method_name_node:
                        method_name = self.get_node_text(method_name_node, content)
                        if method_name == "render":
                            return True
        
        return False
    
    def _returns_jsx_elements(self, func_node: tree_sitter.Node, content: str) -> bool:
        """Check if function returns JSX elements (framework-agnostic)"""
        # Find return statements in the function
        return_nodes = []
        
        def find_returns(node):
            if node.type == "return_statement":
                return_nodes.append(node)
            for child in node.children:
                find_returns(child)
        
        find_returns(func_node)
        
        # Check if any return statement contains JSX
        for return_node in return_nodes:
            if self._contains_jsx_in_node(return_node):
                return True
        
        return False
    
    def _contains_jsx_elements(self, func_node: tree_sitter.Node, content: str) -> bool:
        """Check if function contains JSX elements anywhere (framework-agnostic)"""
        return self._contains_jsx_in_node(func_node)
    
    def _contains_jsx_in_node(self, node: tree_sitter.Node) -> bool:
        """Recursively check if node contains JSX elements"""
        # Check for JSX element nodes
        jsx_types = {"jsx_element", "jsx_self_closing_element", "jsx_fragment"}
        
        if node.type in jsx_types:
            return True
        
        # Recursively check children
        for child in node.children:
            if self._contains_jsx_in_node(child):
                return True
        
        return False


@register_parser("typescript", [".ts", ".tsx", ".d.ts"])
class TypeScriptParser(JavaScriptParser):
    """
    Comprehensive TypeScript parser extending JavaScript parser.
    
    Additional TypeScript features:
    - Interface declarations
    - Type aliases and enums
    - Generic types and constraints
    - Access modifiers (private, protected, public)
    - Decorators
    """
    
    def __init__(self):
        # Initialize with TypeScript language instead of JavaScript
        TreeSitterBase.__init__(self, "typescript")
        self.__version__ = "1.0.0"
        
        # Use TypeScript-specific patterns
        self._jsdoc_pattern = re.compile(r'/\*\*[\s\S]*?\*/', re.MULTILINE)
        self._arrow_function_pattern = re.compile(r'=>')
        
        logger.debug("TypeScript parser initialized")
    
    def get_supported_extensions(self) -> List[str]:
        return [".ts", ".tsx", ".d.ts"]
    
    def extract_entities(
        self, 
        tree: tree_sitter.Tree, 
        content: str,
        file_path: Path
    ) -> List[Entity]:
        """Extract TypeScript entities including interfaces, types, and enums"""
        entities = super().extract_entities(tree, content, file_path)
        
        try:
            # Add TypeScript-specific entities
            entities.extend(self._extract_interfaces(tree, content, file_path))
            entities.extend(self._extract_type_aliases(tree, content, file_path))
            entities.extend(self._extract_enums(tree, content, file_path))
            entities.extend(self._extract_namespaces(tree, content, file_path))
            
        except Exception as e:
            logger.error(f"TypeScript entity extraction failed for {file_path}: {e}")
        
        return entities
    
    def _extract_interfaces(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract interface declarations"""
        interfaces = []
        
        # Find interface declarations
        interface_nodes = self.find_nodes_by_type(tree, ["interface_declaration"])
        
        for interface_node in interface_nodes:
            try:
                interface_entity = self._extract_interface_entity(interface_node, content, file_path)
                if interface_entity:
                    interfaces.append(interface_entity)
            except Exception as e:
                logger.warning(f"Failed to extract interface at {interface_node.start_point}: {e}")
        
        return interfaces
    
    def _extract_interface_entity(
        self, 
        interface_node: tree_sitter.Node, 
        content: str, 
        file_path: Path
    ) -> Optional[Entity]:
        """Extract a single interface entity"""
        # Find interface name
        name_node = self.find_child_by_type(interface_node, "type_identifier")
        if not name_node:
            return None
        
        interface_name = self.get_node_text(name_node, content)
        
        # Extract generic parameters
        generic_parameters = []
        type_params_node = self.find_child_by_type(interface_node, "type_parameters")
        if type_params_node:
            for child in type_params_node.children:
                if child.type == "type_parameter":
                    # Find the parameter name
                    param_name_node = self.find_child_by_type(child, "type_identifier")
                    if param_name_node:
                        param_name = self.get_node_text(param_name_node, content)
                        generic_parameters.append(param_name)
        
        # Extract JSDoc comment
        jsdoc = self._extract_jsdoc_comment(interface_node, content)
        
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
        
        # Extract signature
        body_node = self.find_child_by_type(interface_node, "object_type")
        signature_end = body_node.start_byte if body_node else interface_node.end_byte
        signature = content[interface_node.start_byte:signature_end].strip()
        
        # Create entity
        entity = Entity(
            id=f"typescript::interface::{interface_name}::{location.start_line}",
            name=interface_name,
            qualified_name=interface_name,
            entity_type=EntityType.INTERFACE,
            location=location,
            signature=signature,
            docstring=jsdoc,
            source_code=self.get_node_text(interface_node, content),
            visibility=Visibility.PUBLIC,
            metadata={
                "is_interface": True,
                "generic_parameters": generic_parameters,
                "ast_node_type": interface_node.type
            }
        )
        
        return entity
    
    def _extract_type_aliases(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract type alias declarations"""
        type_aliases = []
        
        # Find type alias declarations
        type_nodes = self.find_nodes_by_type(tree, ["type_alias_declaration"])
        
        for type_node in type_nodes:
            try:
                type_entity = self._extract_type_alias_entity(type_node, content, file_path)
                if type_entity:
                    type_aliases.append(type_entity)
            except Exception as e:
                logger.warning(f"Failed to extract type alias at {type_node.start_point}: {e}")
        
        return type_aliases
    
    def _extract_type_alias_entity(
        self, 
        type_node: tree_sitter.Node, 
        content: str, 
        file_path: Path
    ) -> Optional[Entity]:
        """Extract a single type alias entity"""
        # Find type name
        name_node = self.find_child_by_type(type_node, "type_identifier")
        if not name_node:
            return None
        
        type_name = self.get_node_text(name_node, content)
        
        # Extract type definition (the right-hand side of the type alias)
        type_definition = ""
        for child in type_node.children:
            if child.type not in ["type", "type_identifier", "="]:
                type_definition = self.get_node_text(child, content)
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
        
        # Extract signature
        signature = self.get_node_text(type_node, content)
        
        # Create entity
        entity = Entity(
            id=f"typescript::type::{type_name}::{location.start_line}",
            name=type_name,
            qualified_name=type_name,
            entity_type=EntityType.TYPE,
            location=location,
            signature=signature,
            source_code=signature,
            visibility=Visibility.PUBLIC,
            metadata={
                "is_type_alias": True,
                "type_definition": type_definition,
                "ast_node_type": type_node.type
            }
        )
        
        return entity
    
    def _extract_enums(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract enum declarations"""
        enums = []
        
        # Find enum declarations
        enum_nodes = self.find_nodes_by_type(tree, ["enum_declaration"])
        
        for enum_node in enum_nodes:
            try:
                enum_entity = self._extract_enum_entity(enum_node, content, file_path)
                if enum_entity:
                    enums.append(enum_entity)
            except Exception as e:
                logger.warning(f"Failed to extract enum at {enum_node.start_point}: {e}")
        
        return enums
    
    def _extract_enum_entity(
        self, 
        enum_node: tree_sitter.Node, 
        content: str, 
        file_path: Path
    ) -> Optional[Entity]:
        """Extract a single enum entity"""
        # Find enum name
        name_node = self.find_child_by_type(enum_node, "identifier")
        if not name_node:
            return None
        
        enum_name = self.get_node_text(name_node, content)
        
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
        
        # Extract signature
        body_node = self.find_child_by_type(enum_node, "enum_body")
        signature_end = body_node.start_byte if body_node else enum_node.end_byte
        signature = content[enum_node.start_byte:signature_end].strip()
        
        # Extract enum members
        enum_members = []
        if body_node:
            for child in body_node.children:
                if child.type == "enum_assignment":
                    member_name_node = self.find_child_by_type(child, "property_identifier")
                    if member_name_node:
                        member_name = self.get_node_text(member_name_node, content)
                        enum_members.append(member_name)
        
        # Create entity
        entity = Entity(
            id=f"typescript::enum::{enum_name}::{location.start_line}",
            name=enum_name,
            qualified_name=enum_name,
            entity_type=EntityType.ENUM,
            location=location,
            signature=signature,
            source_code=self.get_node_text(enum_node, content),
            visibility=Visibility.PUBLIC,
            metadata={
                "is_enum": True,
                "members": enum_members,
                "ast_node_type": enum_node.type
            }
        )
        
        return entity
    
    def _extract_namespaces(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract namespace declarations"""
        namespaces = []
        
        # Find namespace declarations (TypeScript uses 'internal_module' for namespaces)
        namespace_nodes = self.find_nodes_by_type(tree, ["internal_module"])
        
        for namespace_node in namespace_nodes:
            try:
                namespace_entity = self._extract_namespace_entity(namespace_node, content, file_path)
                if namespace_entity:
                    namespaces.append(namespace_entity)
            except Exception as e:
                logger.warning(f"Failed to extract namespace at {namespace_node.start_point}: {e}")
        
        return namespaces
    
    def _extract_namespace_entity(
        self, 
        namespace_node: tree_sitter.Node, 
        content: str, 
        file_path: Path
    ) -> Optional[Entity]:
        """Extract a single namespace entity"""
        # Find namespace name
        name_node = self.find_child_by_type(namespace_node, "identifier")
        if not name_node:
            return None
        
        namespace_name = self.get_node_text(name_node, content)
        
        # Create location
        location = SourceLocation(
            file_path=file_path,
            start_line=namespace_node.start_point[0] + 1,
            end_line=namespace_node.end_point[0] + 1,
            start_column=namespace_node.start_point[1],
            end_column=namespace_node.end_point[1],
            start_byte=namespace_node.start_byte,
            end_byte=namespace_node.end_byte
        )
        
        # Extract signature
        body_node = self.find_child_by_type(namespace_node, "statement_block")
        signature_end = body_node.start_byte if body_node else namespace_node.end_byte
        signature = content[namespace_node.start_byte:signature_end].strip()
        
        # Extract namespace members (exported functions, variables, etc.)
        members = []
        if body_node:
            for child in body_node.children:
                if child.type == "function_declaration":
                    func_name_node = self.find_child_by_type(child, "identifier")
                    if func_name_node:
                        func_name = self.get_node_text(func_name_node, content)
                        members.append(func_name)
                elif child.type == "variable_declaration" or child.type == "lexical_declaration":
                    var_declarator = self.find_child_by_type(child, "variable_declarator")
                    if var_declarator:
                        var_name_node = self.find_child_by_type(var_declarator, "identifier")
                        if var_name_node:
                            var_name = self.get_node_text(var_name_node, content)
                            members.append(var_name)
        
        # Create entity
        entity = Entity(
            id=f"typescript::namespace::{namespace_name}::{location.start_line}",
            name=namespace_name,
            qualified_name=namespace_name,
            entity_type=EntityType.NAMESPACE,
            location=location,
            signature=signature,
            source_code=self.get_node_text(namespace_node, content),
            visibility=Visibility.PUBLIC,
            metadata={
                "is_namespace": True,
                "members": members,
                "ast_node_type": namespace_node.type
            }
        )
        
        return entity