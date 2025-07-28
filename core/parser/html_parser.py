"""
HTML parser using Tree-sitter for comprehensive web content extraction.

Extracts HTML elements, attributes, components, scripts, styles, and relationships
from HTML source code with full metadata and structural information.
"""

import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Set
import re

try:
    import tree_sitter
    import tree_sitter_html
except ImportError:
    tree_sitter = None
    tree_sitter_html = None

from .tree_sitter_base import TreeSitterBase
from .registry import register_parser
from ..models.entities import (
    Entity, EntityType, Relation, RelationType, 
    SourceLocation, Visibility
)

logger = logging.getLogger(__name__)


@register_parser("html", [".html", ".htm", ".xhtml"])
class HTMLParser(TreeSitterBase):
    """
    Comprehensive HTML parser with Tree-sitter.
    
    Features:
    - HTML elements with attributes
    - Web components and custom elements
    - Inline styles and scripts
    - Form elements and inputs
    - Media elements (img, video, audio)
    - Link relationships
    - Meta information
    - Accessibility attributes
    - Data attributes
    - Event handlers
    """
    
    # Supported features
    SUPPORTED_FEATURES = [
        "elements", "attributes", "components", "forms", "media",
        "links", "meta", "scripts", "styles", "accessibility"
    ]
    
    # Standard HTML elements that are commonly used as components
    COMPONENT_ELEMENTS = {
        "article", "section", "aside", "header", "footer", "nav", "main",
        "figure", "figcaption", "details", "summary", "dialog", "template"
    }
    
    # Form elements
    FORM_ELEMENTS = {
        "form", "input", "textarea", "select", "option", "optgroup",
        "button", "fieldset", "legend", "label", "datalist", "output"
    }
    
    # Media elements
    MEDIA_ELEMENTS = {
        "img", "video", "audio", "source", "track", "picture",
        "canvas", "svg", "object", "embed", "iframe"
    }
    
    # Interactive elements
    INTERACTIVE_ELEMENTS = {
        "a", "button", "input", "select", "textarea", "details", "dialog"
    }
    
    def __init__(self):
        super().__init__("html")
        self.__version__ = "1.0.0"
        
        # Compiled regex patterns for efficiency
        self._custom_element_pattern = re.compile(r'^[a-z][a-z0-9]*(-[a-z0-9]+)+$')
        self._class_pattern = re.compile(r'class\s*=\s*["\']([^"\']*)["\']')
        self._id_pattern = re.compile(r'id\s*=\s*["\']([^"\']*)["\']')
        self._data_attr_pattern = re.compile(r'data-([a-z0-9-]+)')
        self._aria_attr_pattern = re.compile(r'aria-([a-z0-9-]+)')
        
        logger.debug("HTML parser initialized")
    
    def get_supported_extensions(self) -> List[str]:
        return [".html", ".htm", ".xhtml"]
    
    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.get_supported_extensions()
    
    def extract_entities(
        self, 
        tree: Any, 
        content: str,
        file_path: Path
    ) -> List[Entity]:
        """
        Extract HTML entities from AST.
        
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
            entities.extend(self._extract_elements(tree, content, file_path))
            entities.extend(self._extract_components(tree, content, file_path))
            entities.extend(self._extract_forms(tree, content, file_path))
            entities.extend(self._extract_media(tree, content, file_path))
            entities.extend(self._extract_links(tree, content, file_path))
            entities.extend(self._extract_meta(tree, content, file_path))
            entities.extend(self._extract_scripts(tree, content, file_path))
            entities.extend(self._extract_styles(tree, content, file_path))
            
            logger.debug(f"Extracted {len(entities)} HTML entities from {file_path}")
            
        except Exception as e:
            logger.error(f"Entity extraction failed for {file_path}: {e}")
        
        return entities
    
    def _extract_elements(
        self, 
        tree: Any, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract HTML elements"""
        elements = []
        
        # Find all HTML elements
        element_nodes = self.find_nodes_by_type(tree, ["element"])
        
        for element_node in element_nodes:
            try:
                element_entity = self._extract_element_entity(element_node, content, file_path)
                if element_entity:
                    elements.append(element_entity)
            except Exception as e:
                logger.warning(f"Failed to extract element at {element_node.start_point}: {e}")
        
        return elements
    
    def _extract_element_entity(
        self, 
        element_node: Any, 
        content: str, 
        file_path: Path
    ) -> Optional[Entity]:
        """Extract a single HTML element entity"""
        
        # Get element tag name
        tag_name = self._get_element_tag_name(element_node, content)
        if not tag_name:
            return None
        
        # Create location
        location = SourceLocation(
            file_path=file_path,
            start_line=element_node.start_point[0] + 1,
            end_line=element_node.end_point[0] + 1,
            start_column=element_node.start_point[1],
            end_column=element_node.end_point[1],
            start_byte=element_node.start_byte,
            end_byte=element_node.end_byte
        )
        
        # Extract attributes
        attributes = self._extract_element_attributes(element_node, content)
        
        # Get element text content
        text_content = self._get_element_text_content(element_node, content)
        
        # Determine visibility
        visibility = self._determine_element_visibility(attributes)
        
        # Create entity ID
        element_id = f"html::element::{tag_name}::{location.start_line}"
        
        # Build metadata
        metadata = {
            "tag_name": tag_name,
            "attributes": attributes,
            "text_content": text_content[:200] if text_content else "",  # Limit length
            "is_self_closing": self._is_self_closing_element(element_node, content),
            "is_void_element": tag_name in {
                "area", "base", "br", "col", "embed", "hr", "img", "input",
                "link", "meta", "param", "source", "track", "wbr"
            },
            "element_type": self._classify_element_type(tag_name, attributes),
            "accessibility": self._extract_accessibility_info(attributes),
            "data_attributes": self._extract_data_attributes(attributes),
            "css_classes": attributes.get("class", "").split() if attributes.get("class") else [],
            "element_id": attributes.get("id", ""),
            "ast_node_type": element_node.type
        }
        
        # Create source code snippet
        source_code = self.get_node_text(element_node, content)
        
        # Build signature from key attributes
        signature_parts = []
        if attributes.get("id"):
            signature_parts.append(f"id:{attributes['id']}")
        if attributes.get("class"):
            signature_parts.append(f"class:{attributes['class']}")
        if attributes.get("type"):
            signature_parts.append(f"type:{attributes['type']}")
        if attributes.get("name"):
            signature_parts.append(f"name:{attributes['name']}")
        if "required" in attributes:
            signature_parts.append("required")
        
        # Add ARIA and accessibility attributes
        for attr_name, attr_value in attributes.items():
            if attr_name.startswith("aria-") or attr_name == "role":
                signature_parts.append(f"{attr_name}:{attr_value}")
        
        signature = " ".join(signature_parts) if signature_parts else ""

        return Entity(
            id=element_id,
            name=tag_name,
            qualified_name=f"{tag_name}#{attributes.get('id', '')}" if attributes.get('id') else tag_name,
            entity_type=EntityType.HTML_ELEMENT,  # HTML elements as structural units
            location=location,
            signature=signature,
            source_code=source_code,
            source_hash=hashlib.md5(source_code.encode()).hexdigest(),
            visibility=visibility,
            metadata=metadata
        )
    
    def _extract_components(
        self, 
        tree: Any, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract web components and custom elements"""
        components = []
        
        # Find custom elements and component-like structures
        element_nodes = self.find_nodes_by_type(tree, ["element"])
        
        for element_node in element_nodes:
            try:
                tag_name = self._get_element_tag_name(element_node, content)
                if not tag_name:
                    continue
                
                # Check if it's a web component (custom element)
                if self._is_custom_element(tag_name) or tag_name in self.COMPONENT_ELEMENTS:
                    component_entity = self._extract_component_entity(element_node, content, file_path)
                    if component_entity:
                        components.append(component_entity)
                        
            except Exception as e:
                logger.warning(f"Failed to extract component at {element_node.start_point}: {e}")
        
        return components
    
    def _extract_component_entity(
        self, 
        element_node: Any, 
        content: str, 
        file_path: Path
    ) -> Optional[Entity]:
        """Extract a web component entity"""
        
        tag_name = self._get_element_tag_name(element_node, content)
        if not tag_name:
            return None
        
        location = SourceLocation(
            file_path=file_path,
            start_line=element_node.start_point[0] + 1,
            end_line=element_node.end_point[0] + 1,
            start_column=element_node.start_point[1],
            end_column=element_node.end_point[1],
            start_byte=element_node.start_byte,
            end_byte=element_node.end_byte
        )
        attributes = self._extract_element_attributes(element_node, content)
        
        # Create entity ID
        component_id = f"html::component::{tag_name}::{location.start_line}"
        
        # Build metadata
        metadata = {
            "tag_name": tag_name,
            "attributes": attributes,
            "is_custom_element": self._is_custom_element(tag_name),
            "is_semantic_element": tag_name in self.COMPONENT_ELEMENTS,
            "component_type": self._classify_component_type(tag_name, attributes),
            "slots": self._extract_slots(element_node, content),
            "shadow_dom": self._has_shadow_dom_attributes(attributes),
            "properties": self._extract_component_properties(attributes),
            "events": self._extract_event_handlers(attributes),
            "ast_node_type": element_node.type
        }
        
        source_code = self.get_node_text(element_node, content)
        
        # Build signature from key attributes
        signature_parts = []
        for attr_name, attr_value in attributes.items():
            if attr_name.startswith("data-"):
                signature_parts.append(f"{attr_name}:{attr_value}")
        if attributes.get("id"):
            signature_parts.append(f"id:{attributes['id']}")
        if attributes.get("class"):
            signature_parts.append(f"class:{attributes['class']}")
        
        # Add ARIA and accessibility attributes (same as HTML elements)
        for attr_name, attr_value in attributes.items():
            if attr_name.startswith("aria-") or attr_name == "role":
                signature_parts.append(f"{attr_name}:{attr_value}")
        
        signature = " ".join(signature_parts) if signature_parts else ""
        
        return Entity(
            id=component_id,
            name=tag_name,
            qualified_name=f"component::{tag_name}",
            entity_type=EntityType.HTML_COMPONENT,  # Components as reusable units
            location=location,
            signature=signature,
            source_code=source_code,
            source_hash=hashlib.md5(source_code.encode()).hexdigest(),
            visibility=Visibility.PUBLIC,
            metadata=metadata
        )
    
    def _extract_forms(
        self, 
        tree: Any, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract form elements and inputs"""
        forms = []
        
        # Find form-related elements
        form_nodes = []
        for form_tag in self.FORM_ELEMENTS:
            nodes = self.find_nodes_by_type(tree, ["element"])
            for node in nodes:
                tag_name = self._get_element_tag_name(node, content)
                if tag_name == form_tag:
                    form_nodes.append(node)
        
        for form_node in form_nodes:
            try:
                form_entity = self._extract_form_entity(form_node, content, file_path)
                if form_entity:
                    forms.append(form_entity)
            except Exception as e:
                logger.warning(f"Failed to extract form element at {form_node.start_point}: {e}")
        
        return forms
    
    def _extract_form_entity(
        self, 
        form_node: Any, 
        content: str, 
        file_path: Path
    ) -> Optional[Entity]:
        """Extract a form element entity"""
        
        tag_name = self._get_element_tag_name(form_node, content)
        if not tag_name:
            return None
        
        location = SourceLocation(
            file_path=file_path,
            start_line=form_node.start_point[0] + 1,
            end_line=form_node.end_point[0] + 1,
            start_column=form_node.start_point[1],
            end_column=form_node.end_point[1],
            start_byte=form_node.start_byte,
            end_byte=form_node.end_byte
        )
        attributes = self._extract_element_attributes(form_node, content)
        
        # Create entity ID
        form_id = f"html::form::{tag_name}::{location.start_line}"
        
        # Build form-specific metadata
        metadata = {
            "tag_name": tag_name,
            "attributes": attributes,
            "form_type": tag_name,
            "input_type": attributes.get("type", "") if tag_name == "input" else "",
            "name": attributes.get("name", ""),
            "required": "required" in attributes,
            "disabled": "disabled" in attributes,
            "validation": self._extract_validation_attributes(attributes),
            "accessibility": self._extract_form_accessibility(attributes),
            "is_interactive": tag_name in self.INTERACTIVE_ELEMENTS,
            "ast_node_type": form_node.type
        }
        
        source_code = self.get_node_text(form_node, content)
        
        # Build signature from form attributes
        signature_parts = []
        if attributes.get("method"):
            signature_parts.append(f"method:{attributes['method']}")
        if attributes.get("action"):
            signature_parts.append(f"action:{attributes['action']}")
        if attributes.get("type"):
            signature_parts.append(f"type:{attributes['type']}")
        if attributes.get("name"):
            signature_parts.append(f"name:{attributes['name']}")
        if "required" in attributes:
            signature_parts.append("required")
        if attributes.get("minlength"):
            signature_parts.append(f"minlength:{attributes['minlength']}")
        
        # Add ARIA and accessibility attributes (same as HTML elements)
        for attr_name, attr_value in attributes.items():
            if attr_name.startswith("aria-") or attr_name == "role":
                signature_parts.append(f"{attr_name}:{attr_value}")
        
        signature = " ".join(signature_parts) if signature_parts else ""
        
        return Entity(
            id=form_id,
            name=tag_name,
            qualified_name=f"form::{tag_name}::{attributes.get('name', attributes.get('id', ''))}",
            entity_type=EntityType.HTML_FORM,  # Form inputs as data variables
            location=location,
            signature=signature,
            source_code=source_code,
            source_hash=hashlib.md5(source_code.encode()).hexdigest(),
            visibility=Visibility.PUBLIC,
            metadata=metadata
        )
    
    def _extract_media(
        self, 
        tree: Any, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract media elements (images, videos, etc.)"""
        media = []
        
        # Find media elements
        media_nodes = []
        for media_tag in self.MEDIA_ELEMENTS:
            nodes = self.find_nodes_by_type(tree, ["element"])
            for node in nodes:
                tag_name = self._get_element_tag_name(node, content)
                if tag_name == media_tag:
                    media_nodes.append(node)
        
        for media_node in media_nodes:
            try:
                media_entity = self._extract_media_entity(media_node, content, file_path)
                if media_entity:
                    media.append(media_entity)
            except Exception as e:
                logger.warning(f"Failed to extract media element at {media_node.start_point}: {e}")
        
        return media
    
    def _extract_media_entity(
        self, 
        media_node: Any, 
        content: str, 
        file_path: Path
    ) -> Optional[Entity]:
        """Extract a media element entity"""
        
        tag_name = self._get_element_tag_name(media_node, content)
        if not tag_name:
            return None
        
        location = SourceLocation(
            file_path=file_path,
            start_line=media_node.start_point[0] + 1,
            end_line=media_node.end_point[0] + 1,
            start_column=media_node.start_point[1],
            end_column=media_node.end_point[1],
            start_byte=media_node.start_byte,
            end_byte=media_node.end_byte
        )
        attributes = self._extract_element_attributes(media_node, content)
        
        # Create entity ID
        media_id = f"html::media::{tag_name}::{location.start_line}"
        
        # Build media-specific metadata
        metadata = {
            "tag_name": tag_name,
            "attributes": attributes,
            "media_type": tag_name,
            "src": attributes.get("src", ""),
            "alt": attributes.get("alt", ""),
            "width": attributes.get("width", ""),
            "height": attributes.get("height", ""),
            "loading": attributes.get("loading", ""),
            "lazy_loading": attributes.get("loading") == "lazy",
            "responsive": "srcset" in attributes or "sizes" in attributes,
            "accessibility": {
                "alt_text": attributes.get("alt", ""),
                "aria_label": attributes.get("aria-label", ""),
                "role": attributes.get("role", "")
            },
            "ast_node_type": media_node.type
        }
        
        source_code = self.get_node_text(media_node, content)
        
        # Build signature from media attributes
        signature_parts = []
        if attributes.get("src"):
            signature_parts.append(f"src:{attributes['src']}")
        if attributes.get("alt"):
            signature_parts.append(f"alt:{attributes['alt']}")
        if attributes.get("width"):
            signature_parts.append(f"width:{attributes['width']}")
        if attributes.get("height"):
            signature_parts.append(f"height:{attributes['height']}")
        signature = " ".join(signature_parts) if signature_parts else ""
        
        return Entity(
            id=media_id,
            name=tag_name,
            qualified_name=f"media::{tag_name}::{attributes.get('alt', attributes.get('src', ''))}",
            entity_type=EntityType.HTML_MEDIA,  # Media as resources
            location=location,
            signature=signature,
            source_code=source_code,
            source_hash=hashlib.md5(source_code.encode()).hexdigest(),
            visibility=Visibility.PUBLIC,
            metadata=metadata
        )
    
    def _extract_links(
        self, 
        tree: Any, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract link elements and anchors"""
        links = []
        
        # Find link and anchor elements
        link_nodes = []
        for link_tag in ["a", "link"]:
            nodes = self.find_nodes_by_type(tree, ["element"])
            for node in nodes:
                tag_name = self._get_element_tag_name(node, content)
                if tag_name == link_tag:
                    link_nodes.append(node)
        
        for link_node in link_nodes:
            try:
                link_entity = self._extract_link_entity(link_node, content, file_path)
                if link_entity:
                    links.append(link_entity)
            except Exception as e:
                logger.warning(f"Failed to extract link at {link_node.start_point}: {e}")
        
        return links
    
    def _extract_link_entity(
        self, 
        link_node: Any, 
        content: str, 
        file_path: Path
    ) -> Optional[Entity]:
        """Extract a link element entity"""
        
        tag_name = self._get_element_tag_name(link_node, content)
        if not tag_name:
            return None
        
        location = SourceLocation(
            file_path=file_path,
            start_line=link_node.start_point[0] + 1,
            end_line=link_node.end_point[0] + 1,
            start_column=link_node.start_point[1],
            end_column=link_node.end_point[1],
            start_byte=link_node.start_byte,
            end_byte=link_node.end_byte
        )
        attributes = self._extract_element_attributes(link_node, content)
        
        # Create entity ID
        link_id = f"html::link::{tag_name}::{location.start_line}"
        
        # Build link-specific metadata
        metadata = {
            "tag_name": tag_name,
            "attributes": attributes,
            "href": attributes.get("href", ""),
            "rel": attributes.get("rel", ""),
            "target": attributes.get("target", ""),
            "download": "download" in attributes,
            "external": attributes.get("target") == "_blank" or attributes.get("rel") == "external",
            "link_type": self._classify_link_type(attributes),
            "text_content": self._get_element_text_content(link_node, content),
            "accessibility": {
                "aria_label": attributes.get("aria-label", ""),
                "title": attributes.get("title", "")
            },
            "ast_node_type": link_node.type
        }
        
        source_code = self.get_node_text(link_node, content)
        
        # Build signature from link attributes
        signature_parts = []
        if attributes.get("href"):
            signature_parts.append(f"href:{attributes['href']}")
        if attributes.get("rel"):
            signature_parts.append(f"rel:{attributes['rel']}")
        if attributes.get("target"):
            signature_parts.append(f"target:{attributes['target']}")
        signature = " ".join(signature_parts) if signature_parts else ""
        
        return Entity(
            id=link_id,
            name=tag_name,
            qualified_name=f"link::{attributes.get('href', '')}",
            entity_type=EntityType.HTML_LINK,  # Links as references
            location=location,
            signature=signature,
            source_code=source_code,
            source_hash=hashlib.md5(source_code.encode()).hexdigest(),
            visibility=Visibility.PUBLIC,
            metadata=metadata
        )
    
    def _extract_meta(
        self, 
        tree: Any, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract meta information and head elements"""
        meta_elements = []
        
        # Find meta, title, and other head elements
        meta_tags = ["meta", "title", "base", "head"]
        meta_nodes = []
        
        for meta_tag in meta_tags:
            nodes = self.find_nodes_by_type(tree, ["element"])
            for node in nodes:
                tag_name = self._get_element_tag_name(node, content)
                if tag_name == meta_tag:
                    meta_nodes.append(node)
        
        for meta_node in meta_nodes:
            try:
                meta_entity = self._extract_meta_entity(meta_node, content, file_path)
                if meta_entity:
                    meta_elements.append(meta_entity)
            except Exception as e:
                logger.warning(f"Failed to extract meta element at {meta_node.start_point}: {e}")
        
        return meta_elements
    
    def _extract_meta_entity(
        self, 
        meta_node: Any, 
        content: str, 
        file_path: Path
    ) -> Optional[Entity]:
        """Extract a meta element entity"""
        
        tag_name = self._get_element_tag_name(meta_node, content)
        if not tag_name:
            return None
        
        location = SourceLocation(
            file_path=file_path,
            start_line=meta_node.start_point[0] + 1,
            end_line=meta_node.end_point[0] + 1,
            start_column=meta_node.start_point[1],
            end_column=meta_node.end_point[1],
            start_byte=meta_node.start_byte,
            end_byte=meta_node.end_byte
        )
        attributes = self._extract_element_attributes(meta_node, content)
        
        # Create entity ID
        meta_id = f"html::meta::{tag_name}::{location.start_line}"
        
        # Build meta-specific metadata
        metadata = {
            "tag_name": tag_name,
            "attributes": attributes,
            "name": attributes.get("name", ""),
            "property": attributes.get("property", ""),
            "content": attributes.get("content", ""),
            "charset": attributes.get("charset", ""),
            "http_equiv": attributes.get("http-equiv", ""),
            "meta_type": self._classify_meta_type(attributes),
            "seo_related": self._is_seo_related_meta(attributes),
            "social_media": self._is_social_media_meta(attributes),
            "text_content": self._get_element_text_content(meta_node, content),
            "ast_node_type": meta_node.type
        }
        
        source_code = self.get_node_text(meta_node, content)
        
        # Build signature from meta attributes
        signature_parts = []
        if attributes.get("name"):
            signature_parts.append(f"name:{attributes['name']}")
        if attributes.get("property"):
            signature_parts.append(f"property:{attributes['property']}")
        if attributes.get("content"):
            # Limit content length in signature
            content_preview = attributes['content'][:50] + "..." if len(attributes['content']) > 50 else attributes['content']
            signature_parts.append(content_preview)
        if attributes.get("charset"):
            signature_parts.append(f"charset:{attributes['charset']}")
        
        # For title elements, include the text content in signature
        if tag_name == "title":
            text_content = self._get_element_text_content(meta_node, content)
            if text_content:
                signature_parts.append(text_content)
        
        signature = " ".join(signature_parts) if signature_parts else ""
        
        return Entity(
            id=meta_id,
            name=tag_name,
            qualified_name=f"meta::{attributes.get('name', attributes.get('property', tag_name))}",
            entity_type=EntityType.HTML_META,  # Meta as configuration constants
            location=location,
            signature=signature,
            source_code=source_code,
            source_hash=hashlib.md5(source_code.encode()).hexdigest(),
            visibility=Visibility.PUBLIC,
            metadata=metadata
        )
    
    def _extract_scripts(
        self, 
        tree: Any, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract script elements"""
        scripts = []
        
        # Find script elements - they have their own node type in HTML Tree-sitter
        script_nodes = self.find_nodes_by_type(tree, ["script_element"])
        
        for script_node in script_nodes:
            try:
                script_entity = self._extract_script_entity(script_node, content, file_path)
                if script_entity:
                    scripts.append(script_entity)
            except Exception as e:
                logger.warning(f"Failed to extract script at {script_node.start_point}: {e}")
        
        return scripts
    
    def _extract_script_entity(
        self, 
        script_node: Any, 
        content: str, 
        file_path: Path
    ) -> Optional[Entity]:
        """Extract a script element entity"""
        
        location = SourceLocation(
            file_path=file_path,
            start_line=script_node.start_point[0] + 1,
            end_line=script_node.end_point[0] + 1,
            start_column=script_node.start_point[1],
            end_column=script_node.end_point[1],
            start_byte=script_node.start_byte,
            end_byte=script_node.end_byte
        )
        attributes = self._extract_element_attributes(script_node, content)
        
        # Create entity ID
        script_id = f"html::script::{location.start_line}"
        
        # Get script content
        script_content = self._get_element_text_content(script_node, content)
        
        # Build script-specific metadata
        metadata = {
            "tag_name": "script",
            "attributes": attributes,
            "src": attributes.get("src", ""),
            "type": attributes.get("type", "text/javascript"),
            "async": "async" in attributes,
            "defer": "defer" in attributes,
            "module": attributes.get("type") == "module",
            "inline": not attributes.get("src"),
            "script_content": script_content[:500] if script_content else "",  # Limit length
            "loading_strategy": self._determine_script_loading_strategy(attributes),
            "ast_node_type": script_node.type
        }
        
        source_code = self.get_node_text(script_node, content)
        
        # Build signature from script attributes
        signature_parts = []
        if attributes.get("src"):
            signature_parts.append(f"src:{attributes['src']}")
        if attributes.get("type"):
            signature_parts.append(f"type:{attributes['type']}")
        if "defer" in attributes:
            signature_parts.append("defer")
        if "async" in attributes:
            signature_parts.append("async")
        
        # If no meaningful attributes, indicate inline vs external
        if not signature_parts:
            if attributes.get("src"):
                signature_parts.append("external")
            else:
                signature_parts.append("inline")
        
        signature = " ".join(signature_parts) if signature_parts else "script"
        
        return Entity(
            id=script_id,
            name="script",
            qualified_name=f"script::{attributes.get('src', 'inline')}",
            entity_type=EntityType.HTML_SCRIPT,  # Scripts as executable code
            location=location,
            signature=signature,
            source_code=source_code,
            source_hash=hashlib.md5(source_code.encode()).hexdigest(),
            visibility=Visibility.PUBLIC,
            metadata=metadata
        )
    
    def _extract_styles(
        self, 
        tree: Any, 
        content: str, 
        file_path: Path
    ) -> List[Entity]:
        """Extract style elements and inline styles"""
        styles = []
        
        # Find style elements - they have their own node type in HTML Tree-sitter
        style_nodes = self.find_nodes_by_type(tree, ["style_element"])
        
        # Find elements with inline styles
        inline_style_nodes = []
        element_nodes = self.find_nodes_by_type(tree, ["element"])
        for node in element_nodes:
            attributes = self._extract_element_attributes(node, content)
            if attributes.get("style"):
                inline_style_nodes.append(node)
        
        # Extract style elements
        for style_node in style_nodes:
            try:
                style_entity = self._extract_style_entity(style_node, content, file_path)
                if style_entity:
                    styles.append(style_entity)
            except Exception as e:
                logger.warning(f"Failed to extract style at {style_node.start_point}: {e}")
        
        # Extract inline styles
        for inline_node in inline_style_nodes:
            try:
                inline_style_entity = self._extract_inline_style_entity(inline_node, content, file_path)
                if inline_style_entity:
                    styles.append(inline_style_entity)
            except Exception as e:
                logger.warning(f"Failed to extract inline style at {inline_node.start_point}: {e}")
        
        return styles
    
    def _extract_style_entity(
        self, 
        style_node: Any, 
        content: str, 
        file_path: Path
    ) -> Optional[Entity]:
        """Extract a style element entity"""
        
        location = SourceLocation(
            file_path=file_path,
            start_line=style_node.start_point[0] + 1,
            end_line=style_node.end_point[0] + 1,
            start_column=style_node.start_point[1],
            end_column=style_node.end_point[1],
            start_byte=style_node.start_byte,
            end_byte=style_node.end_byte
        )
        attributes = self._extract_element_attributes(style_node, content)
        
        # Create entity ID
        style_id = f"html::style::{location.start_line}"
        
        # Get style content
        style_content = self._get_element_text_content(style_node, content)
        
        # Build style-specific metadata
        metadata = {
            "tag_name": "style",
            "attributes": attributes,
            "type": attributes.get("type", "text/css"),
            "media": attributes.get("media", ""),
            "scoped": "scoped" in attributes,
            "style_content": style_content[:500] if style_content else "",  # Limit length
            "css_rules_count": style_content.count('{') if style_content else 0,
            "ast_node_type": style_node.type
        }
        
        source_code = self.get_node_text(style_node, content)
        
        # Build signature from style attributes
        signature_parts = []
        if attributes.get("type"):
            signature_parts.append(f"type:{attributes['type']}")
        if attributes.get("media"):
            signature_parts.append(f"media:{attributes['media']}")
        
        # Ensure all style elements have some signature
        if not signature_parts:
            signature_parts.append("style_block")
        
        signature = " ".join(signature_parts) if signature_parts else "style"
        
        return Entity(
            id=style_id,
            name="style",
            qualified_name=f"style::{location.start_line}",
            entity_type=EntityType.HTML_STYLE,  # Styles as styling constants
            location=location,
            signature=signature,
            source_code=source_code,
            source_hash=hashlib.md5(source_code.encode()).hexdigest(),
            visibility=Visibility.PUBLIC,
            metadata=metadata
        )
    
    def _extract_inline_style_entity(
        self, 
        element_node: Any, 
        content: str, 
        file_path: Path
    ) -> Optional[Entity]:
        """Extract inline style from element with style attribute"""
        
        location = SourceLocation(
            file_path=file_path,
            start_line=element_node.start_point[0] + 1,
            end_line=element_node.end_point[0] + 1,
            start_column=element_node.start_point[1],
            end_column=element_node.end_point[1],
            start_byte=element_node.start_byte,
            end_byte=element_node.end_byte
        )
        attributes = self._extract_element_attributes(element_node, content)
        
        inline_style = attributes.get("style", "")
        if not inline_style:
            return None
        
        # Create entity ID
        style_id = f"html::style::inline::{location.start_line}"
        
        # Build inline style metadata
        metadata = {
            "tag_name": "inline_style",
            "attributes": {"style": inline_style},
            "element_tag": self._get_element_tag_name(element_node, content),
            "inline": True,
            "style_content": inline_style,
            "css_rules_count": inline_style.count(';'),
            "ast_node_type": element_node.type
        }
        
        # Build signature from inline style content
        signature = inline_style[:100] + "..." if len(inline_style) > 100 else inline_style
        
        return Entity(
            id=style_id,
            name="inline_style",
            qualified_name=f"inline_style::{location.start_line}",
            entity_type=EntityType.HTML_STYLE,  # Inline styles as styling constants
            location=location,
            signature=signature,
            source_code=inline_style,
            source_hash=hashlib.md5(inline_style.encode()).hexdigest(),
            visibility=Visibility.PUBLIC,
            metadata=metadata
        )
    
    # Helper methods
    
    def _get_element_tag_name(self, element_node: Any, content: str) -> Optional[str]:
        """Get the tag name of an HTML element"""
        try:
            # Look for start_tag child
            start_tag = self.find_child_by_type(element_node, "start_tag")
            if start_tag:
                tag_name_node = self.find_child_by_type(start_tag, "tag_name")
                if tag_name_node:
                    return self.get_node_text(tag_name_node, content).lower()
            
            # Fallback: look for tag_name directly
            tag_name_node = self.find_child_by_type(element_node, "tag_name")
            if tag_name_node:
                return self.get_node_text(tag_name_node, content).lower()
                
            return None
        except Exception as e:
            logger.warning(f"Failed to get tag name: {e}")
            return None
    
    def _extract_element_attributes(self, element_node: Any, content: str) -> Dict[str, str]:
        """Extract attributes from an HTML element"""
        attributes = {}
        
        try:
            # Find start_tag
            start_tag = self.find_child_by_type(element_node, "start_tag")
            if not start_tag:
                return attributes
            
            # Find all attribute nodes
            attr_nodes = self.find_children_by_type(start_tag, "attribute")
            
            for attr_node in attr_nodes:
                attr_name_node = self.find_child_by_type(attr_node, "attribute_name")
                
                if attr_name_node:
                    attr_name = self.get_node_text(attr_name_node, content)
                    attr_value = ""
                    
                    # Look for quoted_attribute_value first
                    quoted_value_node = self.find_child_by_type(attr_node, "quoted_attribute_value")
                    if quoted_value_node:
                        # Look for attribute_value inside quoted_attribute_value
                        attr_value_node = self.find_child_by_type(quoted_value_node, "attribute_value")
                        if attr_value_node:
                            attr_value = self.get_node_text(attr_value_node, content)
                        else:
                            # Fallback: get the text and remove quotes
                            quoted_text = self.get_node_text(quoted_value_node, content)
                            if quoted_text.startswith(('"', "'")) and quoted_text.endswith(('"', "'")):
                                attr_value = quoted_text[1:-1]
                    else:
                        # Look for unquoted attribute_value
                        attr_value_node = self.find_child_by_type(attr_node, "attribute_value")
                        if attr_value_node:
                            attr_value = self.get_node_text(attr_value_node, content)
                        else:
                            # Handle boolean attributes (like "required", "disabled")
                            attr_value = attr_name
                    
                    attributes[attr_name] = attr_value
                    
        except Exception as e:
            logger.warning(f"Failed to extract attributes: {e}")
        
        return attributes
    
    def _get_element_text_content(self, element_node: Any, content: str) -> Optional[str]:
        """Get the text content of an HTML element"""
        try:
            # Find text nodes within the element
            text_parts = []
            for child in element_node.children:
                if child.type == "text":
                    text = self.get_node_text(child, content).strip()
                    if text:
                        text_parts.append(text)
            return " ".join(text_parts) if text_parts else None
        except Exception as e:
            logger.warning(f"Failed to get text content: {e}")
            return None
    
    def _determine_element_visibility(self, attributes: Dict[str, str]) -> Visibility:
        """Determine element visibility based on attributes"""
        if attributes.get("hidden") or attributes.get("style", "").find("display:none") != -1:
            return Visibility.PRIVATE
        return Visibility.PUBLIC
    
    def _is_self_closing_element(self, element_node: Any, content: str) -> bool:
        """Check if element is self-closing"""
        try:
            node_text = self.get_node_text(element_node, content)
            return node_text.endswith("/>")
        except:
            return False
    
    def _classify_element_type(self, tag_name: str, attributes: Dict[str, str]) -> str:
        """Classify the type of HTML element"""
        if tag_name in self.FORM_ELEMENTS:
            return "form"
        elif tag_name in self.MEDIA_ELEMENTS:
            return "media"
        elif tag_name in self.COMPONENT_ELEMENTS:
            return "component"
        elif tag_name in self.INTERACTIVE_ELEMENTS:
            return "interactive"
        elif self._is_custom_element(tag_name):
            return "custom"
        else:
            return "standard"
    
    def _extract_accessibility_info(self, attributes: Dict[str, str]) -> Dict[str, Any]:
        """Extract accessibility-related information"""
        accessibility = {}
        
        # ARIA attributes
        for attr_name, attr_value in attributes.items():
            if attr_name.startswith("aria-"):
                accessibility[attr_name] = attr_value
        
        # Other accessibility attributes
        for attr in ["alt", "title", "role", "tabindex"]:
            if attr in attributes:
                accessibility[attr] = attributes[attr]
        
        return accessibility
    
    def _extract_data_attributes(self, attributes: Dict[str, str]) -> Dict[str, str]:
        """Extract data-* attributes"""
        data_attrs = {}
        for attr_name, attr_value in attributes.items():
            if attr_name.startswith("data-"):
                data_attrs[attr_name] = attr_value
        return data_attrs
    
    def _is_custom_element(self, tag_name: str) -> bool:
        """Check if tag name represents a custom element"""
        return bool(self._custom_element_pattern.match(tag_name))
    
    def _classify_component_type(self, tag_name: str, attributes: Dict[str, str]) -> str:
        """Classify the type of component"""
        if self._is_custom_element(tag_name):
            return "custom_element"
        elif tag_name in self.COMPONENT_ELEMENTS:
            return "semantic_component"
        else:
            return "standard_component"
    
    def _extract_slots(self, element_node: Any, content: str) -> List[str]:
        """Extract slot information from component"""
        slots = []
        try:
            # Find slot elements by iterating through children
            def find_slots(node):
                tag_name = self._get_element_tag_name(node, content)
                if tag_name == "slot":
                    attributes = self._extract_element_attributes(node, content)
                    slot_name = attributes.get("name", "default")
                    slots.append(slot_name)
                
                # Recursively check children
                for child in node.children:
                    if child.type == "element":
                        find_slots(child)
            
            find_slots(element_node)
        except Exception as e:
            logger.warning(f"Failed to extract slots: {e}")
        return slots
    
    def _has_shadow_dom_attributes(self, attributes: Dict[str, str]) -> bool:
        """Check if element has Shadow DOM related attributes"""
        shadow_dom_attrs = ["slot", "part", "exportparts"]
        return any(attr in attributes for attr in shadow_dom_attrs)
    
    def _extract_component_properties(self, attributes: Dict[str, str]) -> Dict[str, str]:
        """Extract component properties from attributes"""
        # Filter out standard HTML attributes to find component-specific properties
        standard_attrs = {
            "id", "class", "style", "title", "lang", "dir", "hidden",
            "tabindex", "accesskey", "contenteditable", "draggable", "spellcheck"
        }
        
        properties = {}
        for attr_name, attr_value in attributes.items():
            if (attr_name not in standard_attrs and 
                not attr_name.startswith(("aria-", "data-", "on"))):
                properties[attr_name] = attr_value
        
        return properties
    
    def _extract_event_handlers(self, attributes: Dict[str, str]) -> Dict[str, str]:
        """Extract event handler attributes"""
        events = {}
        for attr_name, attr_value in attributes.items():
            if attr_name.startswith("on"):
                event_name = attr_name[2:]  # Remove 'on' prefix
                events[event_name] = attr_value
        return events
    
    def _extract_validation_attributes(self, attributes: Dict[str, str]) -> Dict[str, Any]:
        """Extract form validation attributes"""
        validation = {}
        validation_attrs = [
            "required", "pattern", "min", "max", "step", "minlength", "maxlength"
        ]
        
        for attr in validation_attrs:
            if attr in attributes:
                validation[attr] = attributes[attr]
        
        return validation
    
    def _extract_form_accessibility(self, attributes: Dict[str, str]) -> Dict[str, str]:
        """Extract form-specific accessibility attributes"""
        form_a11y = {}
        form_a11y_attrs = ["aria-label", "aria-describedby", "aria-required", "aria-invalid"]
        
        for attr in form_a11y_attrs:
            if attr in attributes:
                form_a11y[attr] = attributes[attr]
        
        return form_a11y
    
    def _classify_link_type(self, attributes: Dict[str, str]) -> str:
        """Classify the type of link based on attributes"""
        rel = attributes.get("rel", "")
        href = attributes.get("href", "")
        
        if rel == "stylesheet":
            return "stylesheet"
        elif rel in ["icon", "shortcut icon"]:
            return "icon"
        elif rel == "canonical":
            return "canonical"
        elif href.startswith("mailto:"):
            return "email"
        elif href.startswith("tel:"):
            return "phone"
        elif href.startswith("http"):
            return "external"
        else:
            return "internal"
    
    def _classify_meta_type(self, attributes: Dict[str, str]) -> str:
        """Classify the type of meta element"""
        if "charset" in attributes:
            return "charset"
        elif "http-equiv" in attributes:
            return "http_equiv"
        elif "name" in attributes:
            return "name"
        elif "property" in attributes:
            return "property"
        else:
            return "unknown"
    
    def _is_seo_related_meta(self, attributes: Dict[str, str]) -> bool:
        """Check if meta element is SEO-related"""
        seo_names = {
            "description", "keywords", "author", "robots", "viewport",
            "title", "canonical"
        }
        return attributes.get("name", "") in seo_names
    
    def _is_social_media_meta(self, attributes: Dict[str, str]) -> bool:
        """Check if meta element is social media-related"""
        property = attributes.get("property", "")
        return property.startswith(("og:", "twitter:", "fb:"))
    
    def _determine_script_loading_strategy(self, attributes: Dict[str, str]) -> str:
        """Determine script loading strategy"""
        if "async" in attributes:
            return "async"
        elif "defer" in attributes:
            return "defer"
        elif attributes.get("type") == "module":
            return "module"
        else:
            return "blocking"
    
    # Relation extraction methods
    
    def extract_relations(
        self,
        tree: Any,
        content: str,
        entities: List[Entity],
        file_path: Path
    ) -> List[Relation]:
        """
        Extract relations between HTML entities.
        
        Args:
            tree: Tree-sitter AST
            content: Source code content
            entities: List of extracted entities
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
            relations.extend(self._extract_containment_relations(entities))
            relations.extend(self._extract_link_relations(tree, content, entities, entity_lookup))
            relations.extend(self._extract_form_relations(tree, content, entities, entity_lookup))
            relations.extend(self._extract_style_relations(tree, content, entities, entity_lookup))
            relations.extend(self._extract_script_relations(tree, content, entities, entity_lookup))
            
            logger.debug(f"Extracted {len(relations)} HTML relations from {file_path}")
            
        except Exception as e:
            logger.error(f"Relation extraction failed for {file_path}: {e}")
        
        return relations
    
    def _build_entity_lookup(self, entities: List[Entity]) -> Dict[str, Entity]:
        """Build lookup dictionary for entities"""
        return {entity.id: entity for entity in entities}
    
    def _extract_containment_relations(self, entities: List[Entity]) -> List[Relation]:
        """Extract containment relations (element contains child elements)"""
        relations = []
        
        # HTML containment is implicit in the DOM structure
        # For now, we'll create basic containment relations based on entity hierarchy
        
        return relations
    
    def _extract_link_relations(
        self,
        tree: Any,
        content: str,
        entities: List[Entity],
        entity_lookup: Dict[str, Entity]
    ) -> List[Relation]:
        """Extract link relations"""
        relations = []
        
        # Find link entities and create reference relations
        link_entities = [e for e in entities if "link" in e.id]
        
        for link_entity in link_entities:
            href = link_entity.metadata.get("href", "")
            if href:
                # Create reference relation
                relation = Relation.create_import_relation(
                    link_entity.id,
                    f"html::resource::{href}",
                    context=f"links to {href}",
                    location=link_entity.location
                )
                relations.append(relation)
        
        return relations
    
    def _extract_form_relations(
        self,
        tree: Any,
        content: str,
        entities: List[Entity],
        entity_lookup: Dict[str, Entity]
    ) -> List[Relation]:
        """Extract form-related relations"""
        relations = []
        
        # Find form entities and create relations between forms and inputs
        form_entities = [e for e in entities if "form" in e.id and e.metadata.get("tag_name") == "form"]
        input_entities = [e for e in entities if "form" in e.id and e.metadata.get("tag_name") != "form"]
        
        for form_entity in form_entities:
            for input_entity in input_entities:
                # Create containment relation between form and its inputs
                relation = Relation(
                    id=f"form_contains::{form_entity.id}::{input_entity.id}",
                    relation_type=RelationType.CONTAINS,
                    source_entity_id=form_entity.id,
                    target_entity_id=input_entity.id,
                    context="form contains input",
                    location=form_entity.location
                )
                relations.append(relation)
        
        return relations
    
    def _extract_style_relations(
        self,
        tree: Any,
        content: str,
        entities: List[Entity],
        entity_lookup: Dict[str, Entity]
    ) -> List[Relation]:
        """Extract style-related relations"""
        relations = []
        
        # Find style entities and create relations to styled elements
        style_entities = [e for e in entities if "style" in e.id]
        
        for style_entity in style_entities:
            # Create general styling relation
            relation = Relation(
                id=f"styles::{style_entity.id}",
                relation_type=RelationType.DEFINES,
                source_entity_id=style_entity.id,
                target_entity_id="html::document",
                context="defines styles for document",
                location=style_entity.location
            )
            relations.append(relation)
        
        return relations
    
    def _extract_script_relations(
        self,
        tree: Any,
        content: str,
        entities: List[Entity],
        entity_lookup: Dict[str, Entity]
    ) -> List[Relation]:
        """Extract script-related relations"""
        relations = []
        
        # Find script entities and create relations
        script_entities = [e for e in entities if "script" in e.id]
        
        for script_entity in script_entities:
            src = script_entity.metadata.get("src", "")
            if src:
                # Create import relation for external scripts
                relation = Relation.create_import_relation(
                    script_entity.id,
                    f"html::script::{src}",
                    context=f"imports script from {src}",
                    location=script_entity.location
                )
                relations.append(relation)
        
        return relations