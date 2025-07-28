"""
CSS parser using Tree-sitter for comprehensive stylesheet analysis.

Extracts CSS rules, selectors, properties, at-rules, and relationships
from CSS source code with full metadata and structural information.
"""

import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Set
import re

try:
    import tree_sitter
    import tree_sitter_css
except ImportError:
    tree_sitter = None
    tree_sitter_css = None

from .tree_sitter_base import TreeSitterBase
from .registry import register_parser
from ..models.entities import (
    Entity, EntityType, Relation, RelationType, 
    SourceLocation, Visibility
)

logger = logging.getLogger(__name__)


@register_parser("css", [".css", ".scss", ".sass", ".less"])
class CSSParser(TreeSitterBase):
    """
    Comprehensive CSS parser with Tree-sitter.
    
    Features:
    - CSS rules with selectors and declarations
    - At-rules (@media, @import, @keyframes, etc.)
    - CSS properties and values
    - CSS variables (custom properties)
    - Media queries and conditional logic
    - Selector specificity analysis
    - CSS imports and dependencies
    - Nested rules (SCSS/SASS support)
    """
    
    # Supported features
    SUPPORTED_FEATURES = [
        "rules", "selectors", "properties", "at_rules", "media_queries",
        "keyframes", "imports", "variables", "nested_rules", "pseudo_classes"
    ]
    
    # At-rule types
    AT_RULE_TYPES = {
        "import", "media", "keyframes", "supports", "font-face", 
        "charset", "namespace", "page", "counter-style", "document"
    }
    
    # Property categories for classification
    LAYOUT_PROPERTIES = {
        "display", "position", "top", "right", "bottom", "left", "z-index",
        "float", "clear", "width", "height", "min-width", "max-width",
        "min-height", "max-height", "margin", "padding", "border",
        "box-sizing", "overflow", "visibility"
    }
    
    TYPOGRAPHY_PROPERTIES = {
        "font-family", "font-size", "font-weight", "font-style", "font-variant",
        "line-height", "letter-spacing", "word-spacing", "text-align",
        "text-decoration", "text-transform", "text-indent", "color"
    }
    
    ANIMATION_PROPERTIES = {
        "animation", "animation-name", "animation-duration", "animation-timing-function",
        "animation-delay", "animation-iteration-count", "animation-direction",
        "animation-fill-mode", "animation-play-state", "transition",
        "transition-property", "transition-duration", "transition-timing-function",
        "transition-delay", "transform", "transform-origin"
    }
    
    def __init__(self):
        super().__init__("css")
        self.__version__ = "1.0.0"
        
        # Compiled regex patterns for efficiency
        self._selector_pattern = re.compile(r'[.#]?[a-zA-Z_][\w-]*')
        self._pseudo_pattern = re.compile(r'::?[a-zA-Z-]+(?:\([^)]*\))?')
        self._variable_pattern = re.compile(r'--[a-zA-Z][\w-]*')
        self._function_pattern = re.compile(r'[a-zA-Z-]+\([^)]*\)')
        self._media_feature_pattern = re.compile(r'\([^)]+\)')
        
        logger.debug("CSS parser initialized")
    
    def get_supported_extensions(self) -> List[str]:
        return [".css", ".scss", ".sass", ".less"]
    
    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.get_supported_extensions()
    
    def extract_entities(
        self, 
        tree: Any, 
        content: str,
        file_path: Path
    ) -> List[Entity]:
        """
        Extract CSS entities from AST.
        
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
            entities.extend(self._extract_rules(tree, content, file_path))
            entities.extend(self._extract_at_rules(tree, content, file_path))
            entities.extend(self._extract_selectors(tree, content, file_path))
            entities.extend(self._extract_properties(tree, content, file_path))
            entities.extend(self._extract_imports(tree, content, file_path))
            entities.extend(self._extract_variables(tree, content, file_path))
            
            logger.debug(f"Extracted {len(entities)} CSS entities from {file_path}")
            
        except Exception as e:
            logger.error(f"Error extracting CSS entities from {file_path}: {e}")
        
        return entities
    
    def _extract_rules(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract CSS rules (selector blocks)"""
        entities = []
        
        # Find CSS rule nodes
        rule_nodes = self.find_nodes_by_type(tree, ["rule_set"])
        
        for rule_node in rule_nodes:
            try:
                # Get selectors
                selectors_node = self.find_child_by_type(rule_node, "selectors")
                if not selectors_node:
                    continue
                
                selectors_text = self.get_node_text(selectors_node, content).strip()
                if not selectors_text:
                    continue
                
                # Create location
                location = SourceLocation(
                    file_path=file_path,
                    start_line=rule_node.start_point[0] + 1,
                    end_line=rule_node.end_point[0] + 1,
                    start_column=rule_node.start_point[1],
                    end_column=rule_node.end_point[1],
                    start_byte=rule_node.start_byte,
                    end_byte=rule_node.end_byte
                )
                
                # Extract declarations count
                block_node = self.find_child_by_type(rule_node, "block")
                declaration_count = 0
                if block_node:
                    declarations = self.find_nodes_by_type(tree, ["declaration"])
                    declaration_count = len([d for d in declarations 
                                           if d.start_byte >= block_node.start_byte 
                                           and d.end_byte <= block_node.end_byte])
                
                # Generate rule ID and metadata
                rule_id = f"file://{file_path}::css_rule::{selectors_text}::{location.start_line}"
                
                metadata = {
                    "selectors": selectors_text.split(','),
                    "selector_count": len(selectors_text.split(',')),
                    "declaration_count": declaration_count,
                    "specificity": self._calculate_specificity(selectors_text),
                    "is_nested": self._is_nested_rule(rule_node, tree),
                    "pseudo_elements": self._extract_pseudo_elements(selectors_text),
                    "pseudo_classes": self._extract_pseudo_classes(selectors_text),
                    "ast_node_type": rule_node.type
                }
                
                # Create source code snippet
                source_code = self.get_node_text(rule_node, content)
                
                # Build signature from selectors
                signature = selectors_text
                
                entity = Entity(
                    id=rule_id,
                    name=selectors_text,
                    qualified_name=selectors_text,
                    entity_type=EntityType.CSS_RULE,
                    location=location,
                    signature=signature,
                    source_code=source_code,
                    source_hash=hashlib.md5(source_code.encode()).hexdigest(),
                    visibility=Visibility.PUBLIC,
                    metadata=metadata
                )
                
                entities.append(entity)
                
            except Exception as e:
                logger.warning(f"Failed to extract CSS rule: {e}")
        
        return entities
    
    def _extract_at_rules(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract CSS at-rules (@media, @keyframes, etc.)"""
        entities = []
        
        # Find at-rule nodes
        at_rule_nodes = self.find_nodes_by_type(tree, [
            "media_statement", "keyframes_statement", "import_statement",
            "supports_statement", "charset_statement", "namespace_statement",
            "at_rule"  # Generic at-rule for @font-face, etc.
        ])
        
        for at_rule_node in at_rule_nodes:
            try:
                # Determine at-rule type
                at_rule_type = self._get_at_rule_type(at_rule_node, content)
                if not at_rule_type:
                    continue
                
                # Get at-rule name/identifier
                name = self._get_at_rule_name(at_rule_node, content, at_rule_type)
                
                # Create location
                location = SourceLocation(
                    file_path=file_path,
                    start_line=at_rule_node.start_point[0] + 1,
                    end_line=at_rule_node.end_point[0] + 1,
                    start_column=at_rule_node.start_point[1],
                    end_column=at_rule_node.end_point[1],
                    start_byte=at_rule_node.start_byte,
                    end_byte=at_rule_node.end_byte
                )
                
                # Generate entity ID
                entity_id = f"file://{file_path}::css_at_rule::{at_rule_type}::{name}::{location.start_line}"
                
                # Extract metadata based on at-rule type
                metadata = self._extract_at_rule_metadata(at_rule_node, content, at_rule_type)
                metadata["ast_node_type"] = at_rule_node.type
                
                # Create source code snippet
                source_code = self.get_node_text(at_rule_node, content)
                
                # Build signature
                signature = f"@{at_rule_type}"
                if name and name != at_rule_type:
                    signature += f" {name}"
                
                entity = Entity(
                    id=entity_id,
                    name=name or at_rule_type,
                    qualified_name=f"@{at_rule_type}#{name}" if name else f"@{at_rule_type}",
                    entity_type=EntityType.CSS_AT_RULE,
                    location=location,
                    signature=signature,
                    source_code=source_code,
                    source_hash=hashlib.md5(source_code.encode()).hexdigest(),
                    visibility=Visibility.PUBLIC,
                    metadata=metadata
                )
                
                entities.append(entity)
                
            except Exception as e:
                logger.warning(f"Failed to extract CSS at-rule: {e}")
        
        return entities
    
    def _extract_selectors(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract individual CSS selectors"""
        entities = []
        
        # Find selector nodes within selectors lists
        selectors_nodes = self.find_nodes_by_type(tree, ["selectors"])
        
        for selectors_node in selectors_nodes:
            try:
                # Get individual selectors - CSS selectors can be different node types
                selector_types = [
                    "tag_name", "class_selector", "id_selector", 
                    "universal_selector", "attribute_selector",
                    "pseudo_class_selector", "pseudo_element_selector",
                    "descendant_selector", "child_selector",
                    "adjacent_sibling_selector", "general_sibling_selector"
                ]
                
                # Get all children that are actual selectors (skip commas and whitespace)
                for child in selectors_node.children:
                    if child.type == "," or child.type.isspace():
                        continue
                    
                    selector_node = child
                    selector_text = self.get_node_text(selector_node, content).strip()
                    if not selector_text:
                        continue
                    
                    # Create location
                    location = SourceLocation(
                        file_path=file_path,
                        start_line=selector_node.start_point[0] + 1,
                        end_line=selector_node.end_point[0] + 1,
                        start_column=selector_node.start_point[1],
                        end_column=selector_node.end_point[1],
                        start_byte=selector_node.start_byte,
                        end_byte=selector_node.end_byte
                    )
                    
                    # Generate entity ID
                    entity_id = f"file://{file_path}::css_selector::{selector_text}::{location.start_line}"
                    
                    # Extract selector metadata
                    metadata = {
                        "selector_type": self._classify_selector_type_from_node(selector_node),
                        "specificity": self._calculate_specificity(selector_text),
                        "elements": self._extract_selector_elements(selector_text),
                        "classes": self._extract_selector_classes(selector_text),
                        "ids": self._extract_selector_ids(selector_text),
                        "pseudo_classes": self._extract_pseudo_classes(selector_text),
                        "pseudo_elements": self._extract_pseudo_elements(selector_text),
                        "combinators": self._extract_combinators(selector_text),
                        "ast_node_type": selector_node.type
                    }
                    
                    # Build signature
                    signature = selector_text
                    
                    entity = Entity(
                        id=entity_id,
                        name=selector_text,
                        qualified_name=selector_text,
                        entity_type=EntityType.CSS_SELECTOR,
                        location=location,
                        signature=signature,
                        source_code=selector_text,
                        source_hash=hashlib.md5(selector_text.encode()).hexdigest(),
                        visibility=Visibility.PUBLIC,
                        metadata=metadata
                    )
                    
                    entities.append(entity)
                    
            except Exception as e:
                logger.warning(f"Failed to extract CSS selector: {e}")
        
        return entities
    
    def _extract_properties(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract CSS properties and declarations"""
        entities = []
        
        # Find declaration nodes
        declaration_nodes = self.find_nodes_by_type(tree, ["declaration"])
        
        for decl_node in declaration_nodes:
            try:
                # Get property name
                property_node = self.find_child_by_type(decl_node, "property_name")
                if not property_node:
                    continue
                
                property_name = self.get_node_text(property_node, content).strip()
                if not property_name:
                    continue
                
                # Get property value - extract everything after the colon
                property_value = ""
                colon_found = False
                for child in decl_node.children:
                    if child.type == ":":
                        colon_found = True
                        continue
                    if colon_found and child.type != ";":
                        # Get text for all value nodes after colon, before semicolon
                        if property_value:
                            property_value += " "
                        property_value += self.get_node_text(child, content).strip()
                
                property_value = property_value.strip()
                
                # Create location
                location = SourceLocation(
                    file_path=file_path,
                    start_line=decl_node.start_point[0] + 1,
                    end_line=decl_node.end_point[0] + 1,
                    start_column=decl_node.start_point[1],
                    end_column=decl_node.end_point[1],
                    start_byte=decl_node.start_byte,
                    end_byte=decl_node.end_byte
                )
                
                # Generate entity ID
                entity_id = f"file://{file_path}::css_property::{property_name}::{location.start_line}"
                
                # Extract property metadata
                metadata = {
                    "property_name": property_name,
                    "property_value": property_value,
                    "property_category": self._classify_property_category(property_name),
                    "is_custom_property": property_name.startswith("--"),
                    "is_shorthand": self._is_shorthand_property(property_name),
                    "has_function": self._function_pattern.search(property_value) is not None,
                    "has_variable": self._variable_pattern.search(property_value) is not None,
                    "is_important": "!important" in property_value,
                    "ast_node_type": decl_node.type
                }
                
                # Create source code snippet
                source_code = self.get_node_text(decl_node, content)
                
                # Build signature
                signature = f"{property_name}: {property_value}"
                
                entity = Entity(
                    id=entity_id,
                    name=property_name,
                    qualified_name=f"{property_name}#{property_value}",
                    entity_type=EntityType.CSS_PROPERTY,
                    location=location,
                    signature=signature,
                    source_code=source_code,
                    source_hash=hashlib.md5(source_code.encode()).hexdigest(),
                    visibility=Visibility.PUBLIC,
                    metadata=metadata
                )
                
                entities.append(entity)
                
            except Exception as e:
                logger.warning(f"Failed to extract CSS property: {e}")
        
        return entities
    
    def _extract_imports(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract CSS import statements"""
        entities = []
        
        # Find import statement nodes
        import_nodes = self.find_nodes_by_type(tree, ["import_statement"])
        
        for import_node in import_nodes:
            try:
                # Get import URL/path
                import_text = self.get_node_text(import_node, content).strip()
                
                # Extract URL from @import statement
                url_match = re.search(r'@import\s+(?:url\()?["\']?([^"\')\s]+)["\']?\)?', import_text)
                if not url_match:
                    continue
                
                import_url = url_match.group(1)
                
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
                
                # Generate entity ID
                entity_id = f"file://{file_path}::css_import::{import_url}::{location.start_line}"
                
                # Extract import metadata
                metadata = {
                    "import_url": import_url,
                    "is_relative": not (import_url.startswith("http") or import_url.startswith("//")),
                    "media_query": self._extract_import_media_query(import_text),
                    "import_type": "external" if import_url.startswith("http") else "local",
                    "ast_node_type": import_node.type
                }
                
                # Build signature
                signature = f"@import \"{import_url}\""
                
                entity = Entity(
                    id=entity_id,
                    name=import_url,
                    qualified_name=f"@import#{import_url}",
                    entity_type=EntityType.CSS_IMPORT,
                    location=location,
                    signature=signature,
                    source_code=import_text,
                    source_hash=hashlib.md5(import_text.encode()).hexdigest(),
                    visibility=Visibility.PUBLIC,
                    metadata=metadata
                )
                
                entities.append(entity)
                
            except Exception as e:
                logger.warning(f"Failed to extract CSS import: {e}")
        
        return entities
    
    def _extract_variables(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract CSS custom properties (variables)"""
        entities = []
        
        # Find variable declarations (custom properties starting with --)
        declaration_nodes = self.find_nodes_by_type(tree, ["declaration"])
        
        for decl_node in declaration_nodes:
            try:
                # Get property name
                property_node = self.find_child_by_type(decl_node, "property_name")
                if not property_node:
                    continue
                
                property_name = self.get_node_text(property_node, content).strip()
                if not property_name.startswith("--"):
                    continue  # Not a CSS variable
                
                # Get variable value - extract everything after the colon
                variable_value = ""
                colon_found = False
                for child in decl_node.children:
                    if child.type == ":":
                        colon_found = True
                        continue
                    if colon_found and child.type != ";":
                        # Get text for all value nodes after colon, before semicolon
                        if variable_value:
                            variable_value += " "
                        variable_value += self.get_node_text(child, content).strip()
                
                variable_value = variable_value.strip()
                
                # Create location
                location = SourceLocation(
                    file_path=file_path,
                    start_line=decl_node.start_point[0] + 1,
                    end_line=decl_node.end_point[0] + 1,
                    start_column=decl_node.start_point[1],
                    end_column=decl_node.end_point[1],
                    start_byte=decl_node.start_byte,
                    end_byte=decl_node.end_byte
                )
                
                # Generate entity ID
                entity_id = f"file://{file_path}::css_variable::{property_name}::{location.start_line}"
                
                # Extract variable metadata
                metadata = {
                    "variable_name": property_name,
                    "variable_value": variable_value,
                    "value_type": self._classify_css_value_type(variable_value),
                    "is_color": self._is_color_value(variable_value),
                    "is_length": self._is_length_value(variable_value),
                    "has_fallback": "var(" in variable_value,
                    "ast_node_type": decl_node.type
                }
                
                # Create source code snippet
                source_code = self.get_node_text(decl_node, content)
                
                # Build signature
                signature = f"{property_name}: {variable_value}"
                
                entity = Entity(
                    id=entity_id,
                    name=property_name,
                    qualified_name=f"{property_name}#{variable_value}",
                    entity_type=EntityType.CSS_VARIABLE,
                    location=location,
                    signature=signature,
                    source_code=source_code,
                    source_hash=hashlib.md5(source_code.encode()).hexdigest(),
                    visibility=Visibility.PUBLIC,
                    metadata=metadata
                )
                
                entities.append(entity)
                
            except Exception as e:
                logger.warning(f"Failed to extract CSS variable: {e}")
        
        return entities
    
    def extract_relations(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        entities: List[Entity], 
        file_path: Path
    ) -> List[Relation]:
        """Extract relationships between CSS entities"""
        if not tree:
            return []
        
        relations = []
        
        try:
            # Extract import relationships
            relations.extend(self._extract_import_relations(tree, content, entities, file_path))
            
            # Extract selector-rule relationships
            relations.extend(self._extract_selector_rule_relations(entities, file_path))
            
            # Extract property-rule relationships
            relations.extend(self._extract_property_rule_relations(entities, file_path))
            
            # Extract variable usage relationships
            relations.extend(self._extract_variable_usage_relations(entities, file_path))
            
            logger.debug(f"Extracted {len(relations)} CSS relations from {file_path}")
            
        except Exception as e:
            logger.error(f"Error extracting CSS relations from {file_path}: {e}")
        
        return relations
    
    # Helper methods for CSS-specific analysis
    
    def _calculate_specificity(self, selector: str) -> Dict[str, int]:
        """Calculate CSS selector specificity"""
        id_count = len(re.findall(r'#[\w-]+', selector))
        class_count = len(re.findall(r'\.[\w-]+', selector))
        pseudo_count = len(re.findall(r'::?[\w-]+', selector))
        attr_count = len(re.findall(r'\[[^\]]+\]', selector))
        element_count = len(re.findall(r'(?<![#.\w-])(?:[a-zA-Z][\w-]*|[*])', selector))
        
        return {
            "ids": id_count,
            "classes": class_count + pseudo_count + attr_count,
            "elements": element_count,
            "total": id_count * 100 + (class_count + pseudo_count + attr_count) * 10 + element_count
        }
    
    def _classify_selector_type_from_node(self, node: tree_sitter.Node) -> str:
        """Classify the type of CSS selector from AST node"""
        node_type = node.type
        
        if node_type == "id_selector":
            return "id"
        elif node_type == "class_selector":
            return "class"
        elif node_type == "attribute_selector":
            return "attribute"
        elif node_type == "pseudo_element_selector":
            return "pseudo_element"
        elif node_type == "pseudo_class_selector":
            return "pseudo_class"
        elif node_type == "universal_selector":
            return "universal"
        elif node_type == "tag_name":
            return "element"
        elif node_type in ["descendant_selector", "child_selector", "adjacent_sibling_selector", "general_sibling_selector"]:
            return "combinator"
        else:
            return "element"  # fallback
    
    def _classify_selector_type(self, selector: str) -> str:
        """Classify the type of CSS selector (legacy text-based method)"""
        if selector.startswith("#"):
            return "id"
        elif selector.startswith("."):
            return "class"
        elif selector.startswith("["):
            return "attribute"
        elif "::" in selector:
            return "pseudo_element"
        elif ":" in selector:
            return "pseudo_class"
        elif selector == "*":
            return "universal"
        else:
            return "element"
    
    def _classify_property_category(self, property_name: str) -> str:
        """Classify CSS property into category"""
        if property_name in self.LAYOUT_PROPERTIES:
            return "layout"
        elif property_name in self.TYPOGRAPHY_PROPERTIES:
            return "typography"
        elif property_name in self.ANIMATION_PROPERTIES:
            return "animation"
        elif property_name.startswith("--"):
            return "custom"
        else:
            return "other"
    
    def _get_at_rule_type(self, node: tree_sitter.Node, content: str) -> Optional[str]:
        """Extract at-rule type from node"""
        node_text = self.get_node_text(node, content).strip()
        if node_text.startswith("@"):
            # Extract the at-rule keyword
            match = re.match(r'@([a-zA-Z-]+)', node_text)
            if match:
                return match.group(1)
        return None
    
    def _get_at_rule_name(self, node: tree_sitter.Node, content: str, rule_type: str) -> str:
        """Get name/identifier for at-rule"""
        if rule_type == "keyframes":
            # Extract keyframes name
            node_text = self.get_node_text(node, content)
            match = re.search(r'@keyframes\s+([a-zA-Z_][\w-]*)', node_text)
            return match.group(1) if match else "anonymous"
        elif rule_type == "media":
            # Extract media query conditions
            node_text = self.get_node_text(node, content)
            match = re.search(r'@media\s+([^{]+)', node_text)
            return match.group(1).strip() if match else "all"
        elif rule_type == "import":
            # Extract import URL
            node_text = self.get_node_text(node, content)
            match = re.search(r'@import\s+(?:url\()?["\']?([^"\')\s]+)', node_text)
            return match.group(1) if match else "unknown"
        elif rule_type == "charset":
            # Extract charset value
            node_text = self.get_node_text(node, content)
            match = re.search(r'@charset\s+["\']([^"\']+)["\']', node_text)
            return f'"{match.group(1)}"' if match else "UTF-8"
        elif rule_type == "font-face":
            # Extract font-family from font-face rule
            node_text = self.get_node_text(node, content)
            match = re.search(r'font-family:\s*["\']([^"\']+)["\']', node_text)
            return match.group(1) if match else "custom"
        elif rule_type == "supports":
            # Extract supports condition
            node_text = self.get_node_text(node, content)
            match = re.search(r'@supports\s+\(([^)]+)\)', node_text)
            return match.group(1) if match else "unknown"
        else:
            return rule_type
    
    def _extract_at_rule_metadata(self, node: tree_sitter.Node, content: str, rule_type: str) -> Dict[str, Any]:
        """Extract metadata specific to at-rule type"""
        metadata = {"at_rule_type": rule_type}
        
        if rule_type == "media":
            # Extract media query features
            node_text = self.get_node_text(node, content)
            features = self._media_feature_pattern.findall(node_text)
            metadata.update({
                "media_features": features,
                "media_type": self._extract_media_type(node_text),
                "is_responsive": any("width" in f or "height" in f for f in features)
            })
        elif rule_type == "keyframes":
            # Extract keyframe information
            metadata.update({
                "keyframe_selectors": self._extract_keyframe_selectors(node, content),
                "animation_properties": self._extract_animation_properties(node, content)
            })
        elif rule_type == "import":
            # Extract import details
            node_text = self.get_node_text(node, content)
            metadata.update({
                "import_url": self._get_at_rule_name(node, content, rule_type),
                "media_query": self._extract_import_media_query(node_text)
            })
        
        return metadata
    
    def _is_nested_rule(self, rule_node: tree_sitter.Node, tree: tree_sitter.Tree) -> bool:
        """Check if rule is nested inside another rule"""
        parent = rule_node.parent
        while parent:
            if parent.type == "rule_set":
                return True
            parent = parent.parent
        return False
    
    def _extract_pseudo_elements(self, selector: str) -> List[str]:
        """Extract pseudo-elements from selector"""
        return re.findall(r'::([a-zA-Z-]+)', selector)
    
    def _extract_pseudo_classes(self, selector: str) -> List[str]:
        """Extract pseudo-classes from selector"""
        # Match :pseudo-class but not ::pseudo-element
        return re.findall(r'(?<!:):([a-zA-Z-]+)(?:\([^)]*\))?', selector)
    
    def _extract_selector_elements(self, selector: str) -> List[str]:
        """Extract element names from selector"""
        # Match element names (not starting with # or .)
        return re.findall(r'(?<![#.\w-])([a-zA-Z][\w-]*)', selector)
    
    def _extract_selector_classes(self, selector: str) -> List[str]:
        """Extract class names from selector"""
        return re.findall(r'\.([a-zA-Z_][\w-]*)', selector)
    
    def _extract_selector_ids(self, selector: str) -> List[str]:
        """Extract ID names from selector"""
        return re.findall(r'#([a-zA-Z_][\w-]*)', selector)
    
    def _extract_combinators(self, selector: str) -> List[str]:
        """Extract CSS combinators from selector"""
        combinators = []
        if '>' in selector:
            combinators.append('child')
        if '+' in selector:
            combinators.append('adjacent_sibling')
        if '~' in selector:
            combinators.append('general_sibling')
        if ' ' in selector.strip():
            combinators.append('descendant')
        return combinators
    
    def _is_shorthand_property(self, property_name: str) -> bool:
        """Check if property is a shorthand property"""
        shorthand_properties = {
            "margin", "padding", "border", "font", "background", "animation",
            "transition", "flex", "grid", "outline", "list-style"
        }
        return property_name in shorthand_properties
    
    def _classify_css_value_type(self, value: str) -> str:
        """Classify CSS value type"""
        if re.match(r'^#[0-9a-fA-F]{3,8}$', value.strip()):
            return "color_hex"
        elif value.strip().startswith("rgb"):
            return "color_rgb"
        elif re.match(r'^\d+(\.\d+)?(px|em|rem|%|vh|vw|vmin|vmax)$', value.strip()):
            return "length"
        elif value.strip().isdigit():
            return "number"
        elif value.startswith("var("):
            return "variable"
        elif "(" in value and ")" in value:
            return "function"
        else:
            return "keyword"
    
    def _is_color_value(self, value: str) -> bool:
        """Check if value represents a color"""
        color_keywords = {
            "red", "green", "blue", "black", "white", "transparent",
            "inherit", "initial", "currentColor"
        }
        value = value.strip().lower()
        return (value.startswith("#") or 
                value.startswith("rgb") or 
                value.startswith("hsl") or 
                value in color_keywords)
    
    def _is_length_value(self, value: str) -> bool:
        """Check if value represents a length"""
        return bool(re.match(r'^\d+(\.\d+)?(px|em|rem|%|vh|vw|vmin|vmax|pt|pc|in|cm|mm)$', value.strip()))
    
    def _extract_import_media_query(self, import_text: str) -> Optional[str]:
        """Extract media query from import statement"""
        # Look for media query after URL
        match = re.search(r'@import\s+[^;]+?\s+([^;]+)', import_text)
        if match:
            media_part = match.group(1).strip()
            if media_part and not media_part.endswith(';'):
                return media_part
        return None
    
    def _extract_media_type(self, media_text: str) -> str:
        """Extract media type from media query"""
        match = re.search(r'@media\s+([a-zA-Z]+)', media_text)
        return match.group(1) if match else "all"
    
    def _extract_keyframe_selectors(self, node: tree_sitter.Node, content: str) -> List[str]:
        """Extract keyframe selectors (0%, 50%, 100%, etc.)"""
        # This would need actual Tree-sitter node traversal
        # For now, return empty list as placeholder
        return []
    
    def _extract_animation_properties(self, node: tree_sitter.Node, content: str) -> List[str]:
        """Extract animation-related properties from keyframes"""
        # This would need actual Tree-sitter node traversal
        # For now, return empty list as placeholder
        return []
    
    # Relation extraction methods
    
    def _extract_import_relations(
        self, 
        tree: tree_sitter.Tree, 
        content: str, 
        entities: List[Entity], 
        file_path: Path
    ) -> List[Relation]:
        """Extract import relationships"""
        relations = []
        
        # Find import entities
        import_entities = [e for e in entities if e.entity_type == EntityType.CSS_IMPORT]
        
        for import_entity in import_entities:
            try:
                # Create import relation (file imports stylesheet)
                relation_id = f"css_import::{file_path}::{import_entity.id}"
                
                relation = Relation(
                    id=relation_id,
                    relation_type=RelationType.IMPORTS,
                    source_entity_id=f"file://{file_path}",
                    target_entity_id=import_entity.id,
                    context=f"CSS import: {import_entity.name}",
                    location=import_entity.location,
                    strength=1.0
                )
                
                relations.append(relation)
                
            except Exception as e:
                logger.warning(f"Failed to extract CSS import relation: {e}")
        
        return relations
    
    def _extract_selector_rule_relations(self, entities: List[Entity], file_path: Path) -> List[Relation]:
        """Extract relationships between selectors and rules"""
        relations = []
        
        rule_entities = [e for e in entities if e.entity_type == EntityType.CSS_RULE]
        selector_entities = [e for e in entities if e.entity_type == EntityType.CSS_SELECTOR]
        
        # Map selectors to rules based on location
        for rule_entity in rule_entities:
            for selector_entity in selector_entities:
                # Check if selector is within rule bounds
                if (selector_entity.location.start_byte >= rule_entity.location.start_byte and
                    selector_entity.location.end_byte <= rule_entity.location.end_byte):
                    
                    try:
                        relation_id = f"css_contains::{rule_entity.id}::{selector_entity.id}"
                        
                        relation = Relation(
                            id=relation_id,
                            relation_type=RelationType.CONTAINS,
                            source_entity_id=rule_entity.id,
                            target_entity_id=selector_entity.id,
                            context=f"Rule contains selector: {selector_entity.name}",
                            location=rule_entity.location,
                            strength=1.0
                        )
                        
                        relations.append(relation)
                        
                    except Exception as e:
                        logger.warning(f"Failed to extract selector-rule relation: {e}")
        
        return relations
    
    def _extract_property_rule_relations(self, entities: List[Entity], file_path: Path) -> List[Relation]:
        """Extract relationships between properties and rules"""
        relations = []
        
        rule_entities = [e for e in entities if e.entity_type == EntityType.CSS_RULE]
        property_entities = [e for e in entities if e.entity_type == EntityType.CSS_PROPERTY]
        
        # Map properties to rules based on location
        for rule_entity in rule_entities:
            for property_entity in property_entities:
                # Check if property is within rule bounds
                if (property_entity.location.start_byte >= rule_entity.location.start_byte and
                    property_entity.location.end_byte <= rule_entity.location.end_byte):
                    
                    try:
                        relation_id = f"css_contains::{rule_entity.id}::{property_entity.id}"
                        
                        relation = Relation(
                            id=relation_id,
                            relation_type=RelationType.CONTAINS,
                            source_entity_id=rule_entity.id,
                            target_entity_id=property_entity.id,
                            context=f"Rule contains property: {property_entity.name}",
                            location=rule_entity.location,
                            strength=1.0
                        )
                        
                        relations.append(relation)
                        
                    except Exception as e:
                        logger.warning(f"Failed to extract property-rule relation: {e}")
        
        return relations
    
    def _extract_variable_usage_relations(self, entities: List[Entity], file_path: Path) -> List[Relation]:
        """Extract relationships for CSS variable usage"""
        relations = []
        
        variable_entities = [e for e in entities if e.entity_type == EntityType.CSS_VARIABLE]
        property_entities = [e for e in entities if e.entity_type == EntityType.CSS_PROPERTY]
        
        # Find properties that use variables
        for property_entity in property_entities:
            property_value = property_entity.metadata.get("property_value", "")
            
            # Look for var() usage
            var_matches = re.findall(r'var\(\s*(--[a-zA-Z][\w-]*)', property_value)
            
            for var_name in var_matches:
                # Find corresponding variable entity
                var_entity = next((v for v in variable_entities if v.name == var_name), None)
                
                if var_entity:
                    try:
                        relation_id = f"css_uses::{property_entity.id}::{var_entity.id}"
                        
                        relation = Relation(
                            id=relation_id,
                            relation_type=RelationType.USES_TYPE,
                            source_entity_id=property_entity.id,
                            target_entity_id=var_entity.id,
                            context=f"Property uses variable: {var_name}",
                            location=property_entity.location,
                            strength=1.0
                        )
                        
                        relations.append(relation)
                        
                    except Exception as e:
                        logger.warning(f"Failed to extract variable usage relation: {e}")
        
        return relations