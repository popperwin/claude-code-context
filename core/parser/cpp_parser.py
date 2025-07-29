"""
C++ parser for comprehensive code analysis using Tree-sitter.

Extracts C++ language constructs including classes, functions, namespaces, 
templates, inheritance, and their relationships. Follows the exact same AST 
traversal patterns as the C parser for consistency and robustness.
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


@register_parser("cpp", [".cpp", ".cxx", ".cc", ".hpp", ".hxx", ".h"])
class CppParser(TreeSitterBase):
    """
    Comprehensive C++ parser for classes, functions, and other language constructs.
    
    Features:
    - Class declarations with inheritance and templates
    - Function declarations and definitions
    - Namespace declarations
    - Template classes and functions
    - Struct and union declarations
    - Enum declarations
    - Typedef declarations
    - Global variable declarations
    - Constructor and destructor methods
    - Member function declarations
    - Friend function declarations
    - Include statement analysis
    - Function call relationships
    - Type usage relationships
    - Inheritance relationships
    """
    
    # Supported features
    SUPPORTED_FEATURES = [
        "classes", "functions", "namespaces", "templates", "inheritance",
        "structs", "unions", "enums", "typedefs", "variables", "methods",
        "constructors", "destructors", "includes", "function_calls", "type_usage"
    ]
    
    def __init__(self):
        super().__init__("cpp")
        self.__version__ = "1.0.0"
        
        logger.debug("C++ parser initialized")
    
    def get_supported_extensions(self) -> List[str]:
        return [".cpp", ".cxx", ".cc", ".hpp", ".hxx", ".h"]
    
    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.get_supported_extensions()
    
    def extract_entities(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract C++ language entities from the AST"""
        entities = []
        
        if not tree.root_node:
            return entities
        
        try:
            # Extract different types of entities following C parser pattern
            entities.extend(self._extract_classes(tree.root_node, content, file_path))
            entities.extend(self._extract_structs(tree.root_node, content, file_path))
            entities.extend(self._extract_unions(tree.root_node, content, file_path))
            entities.extend(self._extract_enums(tree.root_node, content, file_path))
            entities.extend(self._extract_functions(tree.root_node, content, file_path))
            entities.extend(self._extract_methods(tree.root_node, content, file_path))
            entities.extend(self._extract_namespaces(tree.root_node, content, file_path))
            entities.extend(self._extract_templates(tree.root_node, content, file_path))
            entities.extend(self._extract_typedefs(tree.root_node, content, file_path))
            entities.extend(self._extract_global_variables(tree.root_node, content, file_path))
            entities.extend(self._extract_includes(tree.root_node, content, file_path))
            
            logger.debug(f"Extracted {len(entities)} entities from C++ file {file_path}")
            
        except Exception as e:
            logger.error(f"Error extracting entities from C++ file {file_path}: {e}")
        
        return entities
    
    def extract_relations(
        self,
        tree: tree_sitter.Tree,
        content: str,
        entities: List[Entity],
        file_path: Path
    ) -> List[Relation]:
        """Extract relationships between C++ entities"""
        if not tree.root_node or not entities:
            return []
        
        relations = []
        
        try:
            # Build entity lookup for quick access
            entity_lookup = self._build_entity_lookup(entities)
            
            # Extract different types of relations following C parser pattern
            relations.extend(self._extract_function_calls(tree.root_node, entities, entity_lookup, content, file_path))
            relations.extend(self._extract_type_usage_relations(tree.root_node, entities, entity_lookup, content, file_path))
            relations.extend(self._extract_inheritance_relations(tree.root_node, entities, entity_lookup, content, file_path))
            relations.extend(self._extract_include_relations(tree.root_node, entities, content, file_path))
            
            logger.debug(f"Extracted {len(relations)} relations from C++ file {file_path}")
            
        except Exception as e:
            logger.error(f"Error extracting relations from C++ file {file_path}: {e}")
        
        return relations
    
    def _extract_classes(self, root_node: tree_sitter.Node, content: str, file_path: Path) -> List[Entity]:
        """Extract class declarations"""
        classes = []
        
        # Find class declarations
        class_nodes = self._find_nodes_by_type(root_node, ["class_specifier"])
        
        for node in class_nodes:
            try:
                # Only process classes with names
                name_node = self.find_child_by_type(node, "type_identifier")
                if name_node:
                    class_entity = self._create_class_entity(node, content, file_path)
                    if class_entity:
                        classes.append(class_entity)
            except Exception as e:
                logger.warning(f"Failed to extract class: {e}")
        
        return classes
    
    def _create_class_entity(self, node: tree_sitter.Node, content: str, file_path: Path) -> Optional[Entity]:
        """Create a class entity from AST node"""
        try:
            # Extract class name
            name_node = self.find_child_by_type(node, "type_identifier")
            if not name_node:
                return None
            
            class_name = self.get_node_text(name_node, content)
            
            # Extract template parameters
            template_params = []
            template_param_list = self.find_child_by_type(node, "template_parameter_list")
            if template_param_list:
                template_params = self._extract_template_parameters(template_param_list, content)
            
            # Extract base classes (inheritance)
            base_classes = []
            base_class_clause = self.find_child_by_type(node, "base_class_clause")
            if base_class_clause:
                base_classes = self._extract_base_classes(base_class_clause, content)
            
            # Extract class members
            members = {"methods": [], "fields": [], "constructors": [], "destructors": []}
            field_declaration_list = self.find_child_by_type(node, "field_declaration_list")
            if field_declaration_list:
                members = self._extract_class_members(field_declaration_list, content)
            
            # Determine visibility - C++ classes are private by default
            visibility = self._determine_visibility(node, content, default=Visibility.PRIVATE)
            
            # Create signature
            signature = f"class {class_name}"
            if template_params:
                template_str = ", ".join(template_params)
                signature = f"template<{template_str}> {signature}"
            if base_classes:
                base_str = ", ".join(base_classes)
                signature += f" : {base_str}"
            
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
                "template_parameters": template_params,
                "base_classes": base_classes,
                "methods": members["methods"],
                "fields": members["fields"],
                "constructors": members["constructors"],
                "destructors": members["destructors"],
                "method_count": len(members["methods"]),
                "field_count": len(members["fields"]),
                "constructor_count": len(members["constructors"]),
                "destructor_count": len(members["destructors"]),
                "has_inheritance": len(base_classes) > 0,
                "is_template": len(template_params) > 0,
                "ast_node_type": node.type,
                "is_class": True
            }
            
            entity = Entity(
                id=self._generate_entity_id(EntityType.CLASS, class_name, location),
                name=class_name,
                qualified_name=class_name,  # TODO: Build full qualified name with namespace
                entity_type=EntityType.CLASS,
                location=location,
                signature=signature,
                source_code=source_code,
                visibility=visibility,
                metadata=metadata
            )
            
            return entity
            
        except Exception as e:
            logger.warning(f"Failed to create class entity: {e}")
            return None
    
    def _extract_structs(self, root_node: tree_sitter.Node, content: str, file_path: Path) -> List[Entity]:
        """Extract struct declarations"""
        structs = []
        
        # Find struct declarations
        struct_nodes = self._find_nodes_by_type(root_node, ["struct_specifier"])
        
        for node in struct_nodes:
            try:
                # Only process structs with names
                name_node = self.find_child_by_type(node, "type_identifier")
                if name_node:
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
            name_node = self.find_child_by_type(node, "type_identifier")
            if not name_node:
                return None
            
            struct_name = self.get_node_text(name_node, content)
            
            # Extract template parameters
            template_params = []
            template_param_list = self.find_child_by_type(node, "template_parameter_list")
            if template_param_list:
                template_params = self._extract_template_parameters(template_param_list, content)
            
            # Extract base classes (inheritance)
            base_classes = []
            base_class_clause = self.find_child_by_type(node, "base_class_clause")
            if base_class_clause:
                base_classes = self._extract_base_classes(base_class_clause, content)
            
            # Extract fields
            fields = []
            field_declaration_list = self.find_child_by_type(node, "field_declaration_list")
            if field_declaration_list:
                fields = self._extract_struct_fields(field_declaration_list, content)
            
            # Create signature
            field_count = len(fields)
            signature = f"struct {struct_name}"
            if template_params:
                template_str = ", ".join(template_params)
                signature = f"template<{template_str}> {signature}"
            if base_classes:
                base_str = ", ".join(base_classes)
                signature += f" : {base_str}"
            signature += f" {{ {field_count} fields }}"
            
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
                "template_parameters": template_params,
                "base_classes": base_classes,
                "has_inheritance": len(base_classes) > 0,
                "is_template": len(template_params) > 0,
                "ast_node_type": node.type,
                "is_struct": True
            }
            
            entity = Entity(
                id=self._generate_entity_id(EntityType.CLASS, struct_name, location),
                name=struct_name,
                qualified_name=struct_name,
                entity_type=EntityType.CLASS,  # Struct as class in C++
                location=location,
                signature=signature,
                source_code=source_code,
                visibility=Visibility.PUBLIC,  # C++ structs are public by default
                metadata=metadata
            )
            
            return entity
            
        except Exception as e:
            logger.warning(f"Failed to create struct entity: {e}")
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
            field_declaration_list = self.find_child_by_type(node, "field_declaration_list")
            if field_declaration_list:
                fields = self._extract_struct_fields(field_declaration_list, content)  # Same as struct fields
            
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
    
    def _extract_enums(self, root_node: tree_sitter.Node, content: str, file_path: Path) -> List[Entity]:
        """Extract enum declarations"""
        enums = []
        
        # Find enum declarations (both enum and enum class)
        enum_nodes = self._find_nodes_by_type(root_node, ["enum_specifier"])
        
        for node in enum_nodes:
            try:
                # Only process enums with names
                name_node = self.find_child_by_type(node, "type_identifier")
                if name_node:
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
            name_node = self.find_child_by_type(node, "type_identifier")
            if not name_node:
                return None
            
            enum_name = self.get_node_text(name_node, content)
            
            # Check if it's an enum class
            is_enum_class = "class" in self.get_node_text(node, content)
            
            # Extract enum values
            values = []
            enumerator_list = self.find_child_by_type(node, "enumerator_list")
            if enumerator_list:
                values = self._extract_enum_values(enumerator_list, content)
            
            # Create signature
            value_count = len(values)
            enum_type = "enum class" if is_enum_class else "enum"
            signature = f"{enum_type} {enum_name} {{ {value_count} values }}"
            
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
                "is_enum_class": is_enum_class,
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
                    # For pointer functions or template functions, check nested declarators
                    pointer_declarator = self.find_child_by_type(node, "pointer_declarator")
                    if pointer_declarator:
                        function_declarator = self.find_child_by_type(pointer_declarator, "function_declarator")
                
                if function_declarator:
                    # Skip if this is a method inside a class (will be handled separately)
                    if self._is_inside_class(node):
                        continue
                        
                    is_definition = node.type == "function_definition"
                    func_entity = self._create_function_entity(node, content, file_path, is_definition)
                    if func_entity and func_entity.name not in processed_functions:
                        functions.append(func_entity)
                        processed_functions.add(func_entity.name)
            except Exception as e:
                logger.warning(f"Failed to extract function: {e}")
        
        return functions
    
    def _extract_methods(self, root_node: tree_sitter.Node, content: str, file_path: Path) -> List[Entity]:
        """Extract method declarations within classes"""
        methods = []
        
        # Find all methods inside class declarations
        class_nodes = self._find_nodes_by_type(root_node, ["class_specifier", "struct_specifier"])
        
        for class_node in class_nodes:
            try:
                field_declaration_list = self.find_child_by_type(class_node, "field_declaration_list")
                if field_declaration_list:
                    class_methods = self._extract_class_methods(field_declaration_list, content, file_path)
                    methods.extend(class_methods)
            except Exception as e:
                logger.warning(f"Failed to extract methods: {e}")
        
        return methods
    
    def _extract_namespaces(self, root_node: tree_sitter.Node, content: str, file_path: Path) -> List[Entity]:
        """Extract namespace declarations"""
        namespaces = []
        
        # Find namespace declarations
        namespace_nodes = self._find_nodes_by_type(root_node, ["namespace_definition"])
        
        for node in namespace_nodes:
            try:
                namespace_entity = self._create_namespace_entity(node, content, file_path)
                if namespace_entity:
                    namespaces.append(namespace_entity)
            except Exception as e:
                logger.warning(f"Failed to extract namespace: {e}")
        
        return namespaces
    
    def _extract_templates(self, root_node: tree_sitter.Node, content: str, file_path: Path) -> List[Entity]:
        """Extract template declarations"""
        templates = []
        
        # Find template declarations
        template_nodes = self._find_nodes_by_type(root_node, ["template_declaration"])
        
        for node in template_nodes:
            try:
                template_entity = self._create_template_entity(node, content, file_path)
                if template_entity:
                    templates.append(template_entity)
            except Exception as e:
                logger.warning(f"Failed to extract template: {e}")
        
        return templates
    
    def _extract_typedefs(self, root_node: tree_sitter.Node, content: str, file_path: Path) -> List[Entity]:
        """Extract typedef declarations"""
        typedefs = []
        
        # Find typedef declarations (traditional typedef)
        typedef_nodes = self._find_nodes_by_type(root_node, ["type_definition"])
        
        for node in typedef_nodes:
            try:
                typedef_entity = self._create_typedef_entity(node, content, file_path)
                if typedef_entity:
                    typedefs.append(typedef_entity)
            except Exception as e:
                logger.warning(f"Failed to extract typedef: {e}")
        
        # Find alias declarations (modern using syntax)
        alias_nodes = self._find_nodes_by_type(root_node, ["alias_declaration"])
        
        for node in alias_nodes:
            try:
                alias_entity = self._create_alias_entity(node, content, file_path)
                if alias_entity:
                    typedefs.append(alias_entity)
            except Exception as e:
                logger.warning(f"Failed to extract alias: {e}")
        
        return typedefs
    
    def _create_alias_entity(self, node: tree_sitter.Node, content: str, file_path: Path) -> Optional[Entity]:
        """Create an alias entity from AST node (using declaration)"""
        try:
            # Extract alias name - second child should be type_identifier
            alias_name = None
            for child in node.children:
                if child.type == "type_identifier":
                    alias_name = self.get_node_text(child, content)
                    break
            
            if not alias_name:
                return None
            
            # Extract target type - should be type_descriptor
            target_type = "unknown"
            type_descriptor = self.find_child_by_type(node, "type_descriptor")
            if type_descriptor:
                target_type = self.get_node_text(type_descriptor, content)
            
            # Create signature
            signature = f"using {alias_name} = {target_type}"
            
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
                "target_type": target_type,
                "is_alias": True,
                "is_using_declaration": True,
                "ast_node_type": node.type
            }
            
            entity = Entity(
                id=self._generate_entity_id(EntityType.TYPE_ALIAS, alias_name, location),
                name=alias_name,
                qualified_name=alias_name,
                entity_type=EntityType.TYPE_ALIAS,
                location=location,
                signature=signature,
                source_code=source_code,
                visibility=Visibility.PUBLIC,
                metadata=metadata
            )
            
            return entity
            
        except Exception as e:
            logger.warning(f"Failed to create alias entity: {e}")
            return None
    
    def _extract_global_variables(self, root_node: tree_sitter.Node, content: str, file_path: Path) -> List[Entity]:
        """Extract global variable declarations"""
        variables = []
        
        # Find variable declarations at the top level
        declaration_nodes = self._find_nodes_by_type(root_node, ["declaration"])
        
        for node in declaration_nodes:
            try:
                # Skip if this is inside a function or class (not global)
                if self._is_inside_function_or_class(node):
                    continue
                    
                # Skip if this is a function declaration
                if self.find_child_by_type(node, "function_declarator"):
                    continue
                    
                var_entity = self._create_variable_entity(node, content, file_path)
                if var_entity:
                    variables.append(var_entity)
            except Exception as e:
                logger.warning(f"Failed to extract variable: {e}")
        
        return variables
    
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
    
    # Helper methods for entity creation following C parser patterns
    
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
                # For pointer functions, check nested declarators
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
            return_type = self._extract_return_type(node, content)
            
            # Extract parameters
            params_node = self.find_child_by_type(name_node, "parameter_list")
            parameters = []
            if params_node:
                parameters = self._extract_function_parameters(params_node, content)
            
            # Check for template parameters
            template_params = []
            if node.parent and node.parent.type == "template_declaration":
                template_param_list = self.find_child_by_type(node.parent, "template_parameter_list")
                if template_param_list:
                    template_params = self._extract_template_parameters(template_param_list, content)
            
            # Create signature
            param_str = ", ".join(parameters)
            signature = f"{return_type} {func_name}({param_str})"
            if template_params:
                template_str = ", ".join(template_params)
                signature = f"template<{template_str}> {signature}"
            
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
                "template_parameters": template_params,
                "is_template": len(template_params) > 0,
                "is_definition": is_definition,
                "is_declaration": not is_definition,
                "ast_node_type": node.type,
                "has_body": is_definition
            }
            
            entity = Entity(
                id=f"file://{file_path}::cpp::{entity_type.value}::{func_name}::{location.start_line}",
                name=func_name,
                qualified_name=func_name,  # TODO: Build full qualified name with namespace
                entity_type=entity_type,
                location=location,
                signature=signature,
                source_code=source_code,
                visibility=Visibility.PUBLIC,  # C++ functions are generally public
                metadata=metadata
            )
            
            return entity
            
        except Exception as e:
            logger.warning(f"Failed to create function entity: {e}")
            return None
    
    def _create_namespace_entity(self, node: tree_sitter.Node, content: str, file_path: Path) -> Optional[Entity]:
        """Create a namespace entity from AST node"""
        try:
            # Extract namespace name
            name_node = self.find_child_by_type(node, "namespace_identifier")
            if not name_node:
                name_node = self.find_child_by_type(node, "identifier")
            if not name_node:
                return None
            
            namespace_name = self.get_node_text(name_node, content)
            
            # Extract namespace body for member count
            declaration_list = self.find_child_by_type(node, "declaration_list")
            member_count = 0
            if declaration_list:
                member_count = len([child for child in declaration_list.children 
                                 if child.type in ["class_specifier", "function_definition", "declaration"]])
            
            # Create signature
            signature = f"namespace {namespace_name} {{ {member_count} members }}"
            
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
                "member_count": member_count,
                "ast_node_type": node.type,
                "is_namespace": True
            }
            
            entity = Entity(
                id=self._generate_entity_id(EntityType.MODULE, namespace_name, location),
                name=namespace_name,
                qualified_name=namespace_name,
                entity_type=EntityType.MODULE,  # Namespace as module
                location=location,
                signature=signature,
                source_code=source_code,
                visibility=Visibility.PUBLIC,
                metadata=metadata
            )
            
            return entity
            
        except Exception as e:
            logger.warning(f"Failed to create namespace entity: {e}")
            return None
    
    def _create_template_entity(self, node: tree_sitter.Node, content: str, file_path: Path) -> Optional[Entity]:
        """Create a template entity from AST node"""
        try:
            # Extract template parameters
            template_param_list = self.find_child_by_type(node, "template_parameter_list")
            template_params = []
            if template_param_list:
                template_params = self._extract_template_parameters(template_param_list, content)
            
            # Extract the templated entity (class, function, etc.)
            templated_entity = None
            for child in node.children:
                if child.type in ["class_specifier", "function_definition", "declaration"]:
                    templated_entity = child
                    break
            
            if not templated_entity:
                return None
            
            # Extract template name from the templated entity
            template_name = "template"
            if templated_entity.type == "class_specifier":
                name_node = self.find_child_by_type(templated_entity, "type_identifier")
                if name_node:
                    template_name = f"template class {self.get_node_text(name_node, content)}"
            elif templated_entity.type in ["function_definition", "declaration"]:
                function_declarator = self.find_child_by_type(templated_entity, "function_declarator")
                if function_declarator:
                    identifier_node = self.find_child_by_type(function_declarator, "identifier")
                    if identifier_node:
                        template_name = f"template function {self.get_node_text(identifier_node, content)}"
            
            # Create signature
            template_str = ", ".join(template_params)
            signature = f"template<{template_str}> {template_name}"
            
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
                "template_parameters": template_params,
                "parameter_count": len(template_params),
                "templated_entity_type": templated_entity.type,
                "ast_node_type": node.type,
                "is_template": True
            }
            
            entity = Entity(
                id=self._generate_entity_id(EntityType.CLASS, template_name, location),
                name=template_name,
                qualified_name=template_name,
                entity_type=EntityType.CLASS,  # Template as class
                location=location,
                signature=signature,
                source_code=source_code,
                visibility=Visibility.PUBLIC,
                metadata=metadata
            )
            
            return entity
            
        except Exception as e:
            logger.warning(f"Failed to create template entity: {e}")
            return None
    
    def _create_typedef_entity(self, node: tree_sitter.Node, content: str, file_path: Path) -> Optional[Entity]:
        """Create a typedef entity from AST node"""
        try:
            # Extract typedef name - handle both simple and complex typedefs
            typedef_name = None
            
            # Look for direct type_identifier children first (simple typedefs)
            name_nodes = [child for child in node.children if child.type == "type_identifier"]
            if name_nodes:
                typedef_name = self.get_node_text(name_nodes[-1], content)  # Last identifier is the new name
            
            # Handle function pointer typedefs: typedef void (*Name)(args)
            if not typedef_name:
                function_declarator = self.find_child_by_type(node, "function_declarator")
                if function_declarator:
                    parenthesized_declarator = self.find_child_by_type(function_declarator, "parenthesized_declarator")
                    if parenthesized_declarator:
                        pointer_declarator = self.find_child_by_type(parenthesized_declarator, "pointer_declarator")
                        if pointer_declarator:
                            name_node = self.find_child_by_type(pointer_declarator, "type_identifier")
                            if name_node:
                                typedef_name = self.get_node_text(name_node, content)
            
            if not typedef_name:
                return None
            
            # Extract base type
            base_type = "unknown"
            for child in node.children:
                if child.type in ["primitive_type", "type_identifier", "class_specifier", "struct_specifier"]:
                    base_type = self.get_node_text(child, content)
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
    
    # Relation extraction methods following C parser patterns
    
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
        
        # Find function call expressions and member access calls
        call_nodes = self._find_nodes_by_type(root_node, ["call_expression"])
        
        for node in call_nodes:
            try:
                # Find the function being called
                func_name_node = self.find_child_by_type(node, "identifier")
                if not func_name_node:
                    # Check for member access (obj.method())
                    field_expression = self.find_child_by_type(node, "field_expression")
                    if field_expression:
                        func_name_node = self.find_child_by_type(field_expression, "field_identifier")
                
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
        
        # Find field declarations (class member variables) and parameter declarations
        type_usage_nodes = self._find_nodes_by_type(root_node, [
            "field_declaration",        # Class member variables like "Point center;"
            "parameter_declaration",    # Function parameters like "Circle c"
            "declaration",             # Global variable declarations
            "init_declarator"          # Variable declarators
        ])
        
        for node in type_usage_nodes:
            try:
                type_name_node = None
                var_name_node = None
                context_desc = ""
                
                if node.type == "field_declaration":
                    # Handle class member fields like "Point center;"
                    type_name_node = self.find_child_by_type(node, "type_identifier")
                    var_name_node = self.find_child_by_type(node, "field_identifier")
                    context_desc = "field"
                    
                elif node.type == "parameter_declaration":
                    # Handle function parameters like "Circle c"
                    type_name_node = self.find_child_by_type(node, "type_identifier")
                    var_name_node = self.find_child_by_type(node, "identifier")
                    context_desc = "parameter"
                    
                elif node.type == "declaration":
                    # Handle global declarations
                    type_name_node = self.find_child_by_type(node, "type_identifier")
                    init_declarator = self.find_child_by_type(node, "init_declarator")
                    if init_declarator:
                        var_name_node = self.find_child_by_type(init_declarator, "identifier") or self._find_identifier_in_declarator(init_declarator)
                    else:
                        var_name_node = self.find_child_by_type(node, "identifier")
                    context_desc = "variable"
                
                elif node.type == "init_declarator":
                    # Handle variable declarators
                    parent = node.parent
                    if parent and parent.type == "declaration":
                        type_name_node = self.find_child_by_type(parent, "type_identifier")
                        var_name_node = self.find_child_by_type(node, "identifier") or self._find_identifier_in_declarator(node)
                        context_desc = "variable"
                
                if type_name_node and var_name_node:
                    type_name = self.get_node_text(type_name_node, content)
                    var_name = self.get_node_text(var_name_node, content)
                    
                    if type_name in entity_lookup:
                        type_entity = entity_lookup[type_name]
                        
                        # Find the entity that contains this usage (class, function, etc.)
                        source_entity = self._find_containing_entity(node, entities, content)
                        if source_entity:
                            source_entity_id = source_entity.id
                        else:
                            # Fallback to a generic variable ID
                            source_entity_id = f"cpp::{context_desc}::{var_name}"
                        
                        relation_id = f"uses::{var_name}::{type_name}::{node.start_point[0]}"
                        
                        relation = Relation(
                            id=relation_id,
                            relation_type=RelationType.USES_TYPE,
                            source_entity_id=source_entity_id,
                            target_entity_id=type_entity.id,
                            context=f"{context_desc.capitalize()} {var_name} uses type {type_name}",
                            location=self._create_source_location(node, file_path)
                        )
                        relations.append(relation)
                        
            except Exception as e:
                logger.warning(f"Failed to extract type usage relation: {e}")
        
        return relations
        
    def _find_containing_entity(self, node: tree_sitter.Node, entities: List[Entity], content: str) -> Optional[Entity]:
        """Find the entity that contains the given node"""
        current = node.parent
        
        while current:
            # Check if this node corresponds to a class, function, or other entity
            if current.type in ["class_specifier", "struct_specifier"]:
                # Find class name
                class_name_node = self.find_child_by_type(current, "type_identifier")
                if class_name_node:
                    class_name = self.get_node_text(class_name_node, content)
                    # Find matching entity
                    for entity in entities:
                        if entity.name == class_name and entity.entity_type in [EntityType.CLASS, EntityType.STRUCT]:
                            return entity
                            
            elif current.type in ["function_definition", "function_declarator"]:
                # Find function name
                func_declarator = current if current.type == "function_declarator" else self.find_child_by_type(current, "function_declarator")
                if func_declarator:
                    func_name_node = self.find_child_by_type(func_declarator, "identifier")
                    if func_name_node:
                        func_name = self.get_node_text(func_name_node, content)
                        # Find matching entity
                        for entity in entities:
                            if entity.name == func_name and entity.entity_type in [EntityType.FUNCTION, EntityType.METHOD]:
                                return entity
            
            current = current.parent
            
        return None
    
    def _extract_inheritance_relations(
        self, 
        root_node: tree_sitter.Node, 
        entities: List[Entity], 
        entity_lookup: Dict[str, Entity], 
        content: str, 
        file_path: Path
    ) -> List[Relation]:
        """Extract inheritance relationships"""
        relations = []
        
        # Find class and struct declarations with base classes
        class_nodes = self._find_nodes_by_type(root_node, ["class_specifier", "struct_specifier"])
        
        for node in class_nodes:
            try:
                # Get the class name
                name_node = self.find_child_by_type(node, "type_identifier")
                if not name_node:
                    continue
                
                class_name = self.get_node_text(name_node, content)
                
                # Find base class clause
                base_class_clause = self.find_child_by_type(node, "base_class_clause")
                if base_class_clause:
                    # Extract base classes
                    base_class_nodes = self._find_nodes_by_type(base_class_clause, ["type_identifier"])
                    
                    for base_class_node in base_class_nodes:
                        base_class_name = self.get_node_text(base_class_node, content)
                        
                        if class_name in entity_lookup and base_class_name in entity_lookup:
                            derived_class = entity_lookup[class_name]
                            base_class = entity_lookup[base_class_name]
                            
                            relation = Relation.create_inheritance_relation(
                                derived_class.id,
                                base_class.id,
                                context=f"Class {class_name} inherits from {base_class_name}",
                                location=self._create_source_location(base_class_node, file_path)
                            )
                            relations.append(relation)
                            
            except Exception as e:
                logger.warning(f"Failed to extract inheritance relation: {e}")
        
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
                    f"cpp::module::{file_path.stem}",
                    include_entity.id,
                    context=f"File {file_path.name} includes {include_entity.name}",
                    location=include_entity.location
                )
                relations.append(relation)
                
            except Exception as e:
                logger.warning(f"Failed to extract include relation: {e}")
        
        return relations
    
    # Helper utility methods following C parser patterns
    
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
        return f"cpp::{entity_type.value}::{name}::{location.start_line}"
    
    # Additional helper methods for C++ specific features
    
    def _extract_template_parameters(self, template_param_list: tree_sitter.Node, content: str) -> List[str]:
        """Extract template parameter information"""
        parameters = []
        
        for child in template_param_list.children:
            if child.type == "type_parameter_declaration":
                param_text = self.get_node_text(child, content).strip()
                if param_text and param_text not in [",", "<", ">"]:
                    parameters.append(param_text)
        
        return parameters
    
    def _extract_base_classes(self, base_class_clause: tree_sitter.Node, content: str) -> List[str]:
        """Extract base class information"""
        base_classes = []
        
        # Find all type_identifier nodes in the base class clause
        base_class_nodes = self._find_nodes_by_type(base_class_clause, ["type_identifier"])
        
        for node in base_class_nodes:
            base_class_name = self.get_node_text(node, content)
            if base_class_name:
                base_classes.append(base_class_name)
        
        return base_classes
    
    def _extract_class_members(self, field_declaration_list: tree_sitter.Node, content: str) -> Dict[str, List[str]]:
        """Extract class member information"""
        members = {"methods": [], "fields": [], "constructors": [], "destructors": []}
        
        for child in field_declaration_list.children:
            if child.type == "function_definition":
                # Method definition
                func_name = self._extract_function_name(child, content)
                if func_name:
                    members["methods"].append(func_name)
            elif child.type in ["declaration", "field_declaration"]:
                # Could be method declaration or field
                if self.find_child_by_type(child, "function_declarator"):
                    func_name = self._extract_function_name(child, content)
                    if func_name:
                        members["methods"].append(func_name)
                else:
                    # Field declaration
                    field_name = self._extract_field_name(child, content)
                    if field_name:
                        members["fields"].append(field_name)
        
        return members
    
    def _extract_class_methods(self, field_declaration_list: tree_sitter.Node, content: str, file_path: Path) -> List[Entity]:
        """Extract method entities from class declaration"""
        methods = []
        current_visibility = Visibility.PRIVATE  # Default for classes
        
        # Check if we're in a struct (public by default)
        parent = field_declaration_list.parent
        if parent and parent.type == "struct_specifier":
            current_visibility = Visibility.PUBLIC
        
        for child in field_declaration_list.children:
            if child.type == "access_specifier":
                # Update current visibility based on access specifier
                specifier_text = self.get_node_text(child, content).strip()
                if specifier_text == "private":
                    current_visibility = Visibility.PRIVATE
                elif specifier_text == "protected":
                    current_visibility = Visibility.PROTECTED
                elif specifier_text == "public":
                    current_visibility = Visibility.PUBLIC
                    
            elif child.type in ["function_definition", "declaration", "field_declaration"]:
                if self.find_child_by_type(child, "function_declarator"):
                    is_definition = child.type == "function_definition"
                    method_entity = self._create_method_entity(child, content, file_path, is_definition, current_visibility)
                    if method_entity:
                        methods.append(method_entity)
        
        return methods
    
    def _create_method_entity(
        self, 
        node: tree_sitter.Node, 
        content: str, 
        file_path: Path,
        is_definition: bool,
        visibility: Visibility = Visibility.PRIVATE
    ) -> Optional[Entity]:
        """Create a method entity from AST node"""
        try:
            # Extract method name
            function_declarator = self.find_child_by_type(node, "function_declarator")
            if not function_declarator:
                return None
            
            # Try field_identifier first (for methods), then identifier (for functions)
            identifier_node = self.find_child_by_type(function_declarator, "field_identifier")
            if not identifier_node:
                identifier_node = self.find_child_by_type(function_declarator, "identifier")
            if not identifier_node:
                return None
            
            method_name = self.get_node_text(identifier_node, content)
            
            # Extract return type
            return_type = self._extract_return_type(node, content)
            
            # Extract parameters
            params_node = self.find_child_by_type(function_declarator, "parameter_list")
            parameters = []
            if params_node:
                parameters = self._extract_function_parameters(params_node, content)
            
            # Create signature
            param_str = ", ".join(parameters)
            signature = f"{return_type} {method_name}({param_str})"
            
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
            
            # Use provided visibility (already determined by caller tracking access specifiers)
            
            # Build metadata
            metadata = {
                "return_type": return_type,
                "parameters": parameters,
                "parameter_count": len(parameters),
                "is_definition": is_definition,
                "is_declaration": not is_definition,
                "is_method": True,
                "ast_node_type": node.type
            }
            
            entity = Entity(
                id=f"file://{file_path}::cpp::method::{method_name}::{location.start_line}",
                name=method_name,
                qualified_name=method_name,
                entity_type=EntityType.METHOD,
                location=location,
                signature=signature,
                source_code=source_code,
                visibility=visibility,
                metadata=metadata
            )
            
            return entity
            
        except Exception as e:
            logger.warning(f"Failed to create method entity: {e}")
            return None
    
    def _extract_struct_fields(self, field_declaration_list: tree_sitter.Node, content: str) -> List[str]:
        """Extract struct field information"""
        fields = []
        
        for child in field_declaration_list.children:
            if child.type == "field_declaration":
                field_text = self.get_node_text(child, content).strip()
                if field_text and not field_text.startswith("{") and not field_text.startswith("}"):
                    # Clean up the field text
                    field_text = field_text.replace(";", "").strip()
                    if field_text:
                        fields.append(field_text)
        
        return fields
    
    def _extract_enum_values(self, enumerator_list: tree_sitter.Node, content: str) -> List[str]:
        """Extract enum value information"""
        values = []
        
        for child in enumerator_list.children:
            if child.type == "enumerator":
                value_text = self.get_node_text(child, content).strip()
                if value_text and value_text != ",":
                    values.append(value_text)
        
        return values
    
    def _extract_function_parameters(self, params_node: tree_sitter.Node, content: str) -> List[str]:
        """Extract function parameter information"""
        parameters = []
        
        for child in params_node.children:
            if child.type == "parameter_declaration":
                param_text = self.get_node_text(child, content).strip()
                if param_text and param_text != ",":
                    parameters.append(param_text)
        
        return parameters
    
    def _extract_return_type(self, node: tree_sitter.Node, content: str) -> str:
        """Extract return type from function declaration"""
        # Look for type nodes before the function_declarator
        for child in node.children:
            if child.type in ["primitive_type", "type_identifier", "qualified_identifier"]:
                return self.get_node_text(child, content)
        
        return "void"  # Default return type
    
    def _extract_function_name(self, node: tree_sitter.Node, content: str) -> Optional[str]:
        """Extract function name from declaration/definition"""
        function_declarator = self.find_child_by_type(node, "function_declarator")
        if function_declarator:
            # Try field_identifier first (for methods), then identifier (for functions)
            identifier_node = self.find_child_by_type(function_declarator, "field_identifier")
            if not identifier_node:
                identifier_node = self.find_child_by_type(function_declarator, "identifier")
            if identifier_node:
                return self.get_node_text(identifier_node, content)
        return None
    
    def _extract_field_name(self, node: tree_sitter.Node, content: str) -> Optional[str]:
        """Extract field name from declaration"""
        init_declarator = self.find_child_by_type(node, "init_declarator")
        if init_declarator:
            identifier_node = self.find_child_by_type(init_declarator, "identifier")
            if identifier_node:
                return self.get_node_text(identifier_node, content)
        
        identifier_node = self.find_child_by_type(node, "identifier")
        if identifier_node:
            return self.get_node_text(identifier_node, content)
        
        return None
    
    def _extract_variable_type(self, node: tree_sitter.Node, content: str) -> str:
        """Extract variable type from declaration node"""
        # Look for type specifiers in the declaration
        type_parts = []
        
        # Find primitive_type, type_identifier, or storage class specifiers
        for child in node.children:
            if child.type in ["primitive_type", "type_identifier", "storage_class_specifier", "type_qualifier"]:
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
    
    def _find_identifier_in_declarator(self, node: tree_sitter.Node) -> Optional[tree_sitter.Node]:
        """Recursively find identifier node in complex declarator structures"""
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
    
    def _determine_visibility(self, node: tree_sitter.Node, content: str, default: Visibility = Visibility.PUBLIC) -> Visibility:
        """Determine the visibility of a C++ entity"""
        # Look for access specifiers in the surrounding context
        current = node.parent
        
        while current:
            if current.type == "access_specifier":
                specifier_text = self.get_node_text(current, content)
                if "private" in specifier_text:
                    return Visibility.PRIVATE
                elif "protected" in specifier_text:
                    return Visibility.PROTECTED
                elif "public" in specifier_text:
                    return Visibility.PUBLIC
            current = current.parent
        
        return default
    
    def _determine_method_visibility(self, node: tree_sitter.Node, content: str) -> Visibility:
        """Determine the visibility of a method within a class"""
        # Find the class/struct declaration list that contains this method
        declaration_list = node.parent
        while declaration_list and declaration_list.type != "field_declaration_list":
            declaration_list = declaration_list.parent
        
        if not declaration_list:
            return Visibility.PRIVATE
        
        # Find the most recent access specifier before this node
        current_visibility = None
        
        # Look through all children of the declaration list
        for child in declaration_list.children:
            if child.type == "access_specifier":
                specifier_text = self.get_node_text(child, content).strip()
                if specifier_text == "private:":
                    current_visibility = Visibility.PRIVATE
                elif specifier_text == "protected:":
                    current_visibility = Visibility.PROTECTED
                elif specifier_text == "public:":
                    current_visibility = Visibility.PUBLIC
            elif child == node:
                # We've reached our target node, use the current visibility
                break
        
        # If we found a visibility from an access specifier, use it
        if current_visibility is not None:
            return current_visibility
        
        # Check if we're in a class or struct to determine default visibility
        class_node = declaration_list.parent
        while class_node:
            if class_node.type == "class_specifier":
                return Visibility.PRIVATE  # Class methods are private by default
            elif class_node.type == "struct_specifier":
                return Visibility.PUBLIC   # Struct methods are public by default
            class_node = class_node.parent
        
        return Visibility.PRIVATE  # Default for classes
    
    def _is_inside_class(self, node: tree_sitter.Node) -> bool:
        """Check if a node is inside a class or struct declaration"""
        current = node.parent
        
        while current:
            if current.type in ["class_specifier", "struct_specifier"]:
                return True
            current = current.parent
        
        return False
    
    def _is_inside_function_or_class(self, node: tree_sitter.Node) -> bool:
        """Check if a node is inside a function or class (not global)"""
        current = node.parent
        
        while current:
            if current.type in ["function_definition", "compound_statement", "class_specifier", "struct_specifier"]:
                return True
            current = current.parent
        
        return False