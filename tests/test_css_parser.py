"""
Unit tests for CSS parser.

Tests CSS entity extraction and relation detection.
"""

import pytest
from pathlib import Path
from core.parser.css_parser import CSSParser
from core.models.entities import EntityType, Visibility


class TestCSSParser:
    """Test CSS parser functionality"""
    
    def setup_method(self):
        """Setup test instance"""
        self.parser = CSSParser()
        self.test_file = Path("test.css")
    
    def test_parser_registration(self):
        """Test parser is registered correctly"""
        assert "css" in self.parser.__class__.__name__.lower()
        assert ".css" in self.parser.get_supported_extensions()
        assert ".scss" in self.parser.get_supported_extensions()
        assert ".sass" in self.parser.get_supported_extensions()
        assert ".less" in self.parser.get_supported_extensions()
    
    def test_extract_css_rules(self, tmp_path):
        """Test CSS rule extraction"""
        content = """
        /* Basic CSS rules */
        body {
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
        }
        
        .container {
            width: 100%;
            max-width: 1200px;
            margin: 0 auto;
        }
        
        #header {
            background-color: #333;
            color: white;
            padding: 20px;
        }
        
        .nav-item:hover {
            background-color: #555;
        }
        
        .card::before {
            content: "";
            display: block;
        }
        """
        
        test_file = tmp_path / "test.css"
        test_file.write_text(content)
        
        result = self.parser.parse_file(test_file)
        entities = result.entities
        
        # Filter CSS rules
        css_rules = [e for e in entities if e.entity_type == EntityType.CSS_RULE]
        
        assert len(css_rules) >= 5  # body, .container, #header, .nav-item:hover, .card::before
        
        # Check specific rules
        rule_selectors = [r.name for r in css_rules]
        assert "body" in rule_selectors
        assert ".container" in rule_selectors
        assert "#header" in rule_selectors
        assert ".nav-item:hover" in rule_selectors
        assert ".card::before" in rule_selectors
        
        # Check rule with multiple declarations
        body_rule = next((r for r in css_rules if r.name == "body"), None)
        assert body_rule is not None
        assert body_rule.metadata["declaration_count"] >= 3
        
        # Check specificity calculation
        container_rule = next((r for r in css_rules if r.name == ".container"), None)
        assert container_rule is not None
        assert container_rule.metadata["specificity"]["classes"] == 1
        assert container_rule.metadata["specificity"]["total"] == 10
        
        header_rule = next((r for r in css_rules if r.name == "#header"), None)
        assert header_rule is not None
        assert header_rule.metadata["specificity"]["ids"] == 1
        assert header_rule.metadata["specificity"]["total"] == 100
    
    def test_extract_css_selectors(self, tmp_path):
        """Test individual CSS selector extraction"""
        content = """
        h1, h2, h3 {
            font-weight: bold;
        }
        
        .btn.primary {
            background: blue;
        }
        
        article > p {
            margin-bottom: 1em;
        }
        
        input[type="text"] {
            border: 1px solid #ccc;
        }
        
        a:visited {
            color: purple;
        }
        """
        
        test_file = tmp_path / "test.css"
        test_file.write_text(content)
        
        result = self.parser.parse_file(test_file)
        entities = result.entities
        
        # Filter CSS selectors
        selectors = [e for e in entities if e.entity_type == EntityType.CSS_SELECTOR]
        
        assert len(selectors) >= 7  # h1, h2, h3, .btn.primary, article > p, input[type="text"], a:visited
        
        # Check selector types
        selector_names = [s.name for s in selectors]
        assert "h1" in selector_names
        assert "h2" in selector_names
        assert "h3" in selector_names
        assert ".btn.primary" in selector_names
        assert "article > p" in selector_names
        assert 'input[type="text"]' in selector_names
        assert "a:visited" in selector_names
        
        # Check selector classification
        h1_selector = next((s for s in selectors if s.name == "h1"), None)
        assert h1_selector is not None
        assert h1_selector.metadata["selector_type"] == "element"
        
        btn_selector = next((s for s in selectors if s.name == ".btn.primary"), None)
        assert btn_selector is not None
        assert btn_selector.metadata["selector_type"] == "class"
        assert len(btn_selector.metadata["classes"]) == 2
        
        input_selector = next((s for s in selectors if 'input[type="text"]' in s.name), None)
        assert input_selector is not None
        assert input_selector.metadata["selector_type"] == "attribute"
        
        visited_selector = next((s for s in selectors if s.name == "a:visited"), None)
        assert visited_selector is not None
        assert visited_selector.metadata["selector_type"] == "pseudo_class"
        assert "visited" in visited_selector.metadata["pseudo_classes"]
    
    def test_extract_css_properties(self, tmp_path):
        """Test CSS property extraction"""
        content = """
        .example {
            /* Layout properties */
            display: flex;
            position: relative;
            width: 100%;
            height: 50vh;
            
            /* Typography properties */
            font-family: 'Roboto', sans-serif;
            font-size: 16px;
            color: #333;
            
            /* Animation properties */
            transition: all 0.3s ease;
            transform: translateX(10px);
            
            /* Custom properties */
            --primary-color: #007bff;
            --spacing: 1rem;
            
            /* Shorthand properties */
            margin: 10px 20px;
            border: 1px solid var(--primary-color);
            
            /* Important declaration */
            z-index: 1000 !important;
        }
        """
        
        test_file = tmp_path / "test.css"
        test_file.write_text(content)
        
        result = self.parser.parse_file(test_file)
        entities = result.entities
        
        # Filter CSS properties
        properties = [e for e in entities if e.entity_type == EntityType.CSS_PROPERTY]
        
        assert len(properties) >= 12
        
        # Check property categories
        property_names = [p.name for p in properties]
        assert "display" in property_names
        assert "font-family" in property_names
        assert "transition" in property_names
        assert "--primary-color" in property_names
        
        # Check layout property classification
        display_prop = next((p for p in properties if p.name == "display"), None)
        assert display_prop is not None
        assert display_prop.metadata["property_category"] == "layout"
        
        # Check typography property classification
        font_prop = next((p for p in properties if p.name == "font-family"), None)
        assert font_prop is not None
        assert font_prop.metadata["property_category"] == "typography"
        
        # Check animation property classification
        transition_prop = next((p for p in properties if p.name == "transition"), None)
        assert transition_prop is not None
        assert transition_prop.metadata["property_category"] == "animation"
        
        # Check custom property
        custom_prop = next((p for p in properties if p.name == "--primary-color"), None)
        assert custom_prop is not None
        assert custom_prop.metadata["is_custom_property"] is True
        assert custom_prop.metadata["property_category"] == "custom"
        
        # Check shorthand property
        margin_prop = next((p for p in properties if p.name == "margin"), None)
        assert margin_prop is not None
        assert margin_prop.metadata["is_shorthand"] is True
        
        # Check important declaration
        zindex_prop = next((p for p in properties if p.name == "z-index"), None)
        assert zindex_prop is not None
        assert zindex_prop.metadata["is_important"] is True
        
        # Check variable usage
        border_prop = next((p for p in properties if p.name == "border"), None)
        assert border_prop is not None
        assert border_prop.metadata["has_variable"] is True
    
    def test_extract_css_variables(self, tmp_path):
        """Test CSS custom property (variable) extraction"""
        content = """
        :root {
            --primary-color: #007bff;
            --secondary-color: #6c757d;
            --font-size-base: 1rem;
            --spacing-unit: 8px;
            --border-radius: 4px;
            --shadow-color: rgba(0, 0, 0, 0.1);
            --gradient: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        }
        
        .theme-dark {
            --primary-color: #0d6efd;
            --background-color: #212529;
            --text-color: #fff;
        }
        """
        
        test_file = tmp_path / "test.css"
        test_file.write_text(content)
        
        result = self.parser.parse_file(test_file)
        entities = result.entities
        
        # Filter CSS variables
        variables = [e for e in entities if e.entity_type == EntityType.CSS_VARIABLE]
        
        assert len(variables) >= 10
        
        # Check specific variables
        variable_names = [v.name for v in variables]
        assert "--primary-color" in variable_names
        assert "--font-size-base" in variable_names
        assert "--spacing-unit" in variable_names
        assert "--shadow-color" in variable_names
        assert "--gradient" in variable_names
        
        # Check color variable
        primary_color = next((v for v in variables if v.name == "--primary-color"), None)
        assert primary_color is not None
        assert primary_color.metadata["is_color"] is True
        assert primary_color.metadata["value_type"] == "color_hex"
        
        # Check length variable
        spacing_var = next((v for v in variables if v.name == "--spacing-unit"), None)
        assert spacing_var is not None
        assert spacing_var.metadata["is_length"] is True
        assert spacing_var.metadata["value_type"] == "length"
        
        # Check function variable
        gradient_var = next((v for v in variables if v.name == "--gradient"), None)
        assert gradient_var is not None
        assert gradient_var.metadata["value_type"] == "function"
        assert gradient_var.metadata["has_fallback"] is True  # uses var()
    
    def test_extract_at_rules(self, tmp_path):
        """Test CSS at-rule extraction"""
        content = """
        @charset "UTF-8";
        
        @import url("normalize.css");
        @import "base.css" screen and (min-width: 768px);
        
        @media screen and (max-width: 768px) {
            .container {
                width: 100%;
                padding: 10px;
            }
        }
        
        @media print {
            .no-print {
                display: none;
            }
        }
        
        @keyframes fadeIn {
            0% {
                opacity: 0;
                transform: translateY(-10px);
            }
            100% {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @supports (display: grid) {
            .grid-container {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
            }
        }
        
        @font-face {
            font-family: 'CustomFont';
            src: url('custom-font.woff2') format('woff2');
            font-weight: normal;
        }
        """
        
        test_file = tmp_path / "test.css"
        test_file.write_text(content)
        
        result = self.parser.parse_file(test_file)
        entities = result.entities
        
        # Filter at-rules
        at_rules = [e for e in entities if e.entity_type == EntityType.CSS_AT_RULE]
        
        assert len(at_rules) >= 7
        
        # Check at-rule types
        at_rule_types = [ar.metadata["at_rule_type"] for ar in at_rules]
        assert "charset" in at_rule_types
        assert "import" in at_rule_types
        assert "media" in at_rule_types
        assert "keyframes" in at_rule_types
        assert "supports" in at_rule_types
        assert "font-face" in at_rule_types
        
        # Check specific at-rules
        charset_rule = next((ar for ar in at_rules if ar.metadata["at_rule_type"] == "charset"), None)
        assert charset_rule is not None
        assert "UTF-8" in charset_rule.signature
        
        import_rules = [ar for ar in at_rules if ar.metadata["at_rule_type"] == "import"]
        assert len(import_rules) >= 2
        
        media_rules = [ar for ar in at_rules if ar.metadata["at_rule_type"] == "media"]
        assert len(media_rules) >= 2
        assert any("max-width" in mr.name for mr in media_rules)
        assert any("print" in mr.name for mr in media_rules)
        
        keyframes_rule = next((ar for ar in at_rules if ar.metadata["at_rule_type"] == "keyframes"), None)
        assert keyframes_rule is not None
        assert keyframes_rule.name == "fadeIn"
        
        supports_rule = next((ar for ar in at_rules if ar.metadata["at_rule_type"] == "supports"), None)
        assert supports_rule is not None
        assert "display: grid" in supports_rule.name
    
    def test_extract_css_imports(self, tmp_path):
        """Test CSS import statement extraction"""
        content = """
        @import url("reset.css");
        @import url("https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700");
        @import "components/buttons.css";
        @import "print.css" print;
        @import url("mobile.css") screen and (max-width: 600px);
        """
        
        test_file = tmp_path / "test.css"
        test_file.write_text(content)
        
        result = self.parser.parse_file(test_file)
        entities = result.entities
        
        # Filter CSS imports
        imports = [e for e in entities if e.entity_type == EntityType.CSS_IMPORT]
        
        assert len(imports) >= 5
        
        # Check import URLs
        import_urls = [i.name for i in imports]
        assert "reset.css" in import_urls
        assert "https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700" in import_urls
        assert "components/buttons.css" in import_urls
        assert "print.css" in import_urls
        assert "mobile.css" in import_urls
        
        # Check import types
        external_import = next((i for i in imports if "fonts.googleapis.com" in i.name), None)
        assert external_import is not None
        assert external_import.metadata["import_type"] == "external"
        assert external_import.metadata["is_relative"] is False
        
        local_import = next((i for i in imports if i.name == "reset.css"), None)
        assert local_import is not None
        assert local_import.metadata["import_type"] == "local"
        assert local_import.metadata["is_relative"] is True
        
        # Check media queries in imports
        print_import = next((i for i in imports if i.name == "print.css"), None)
        assert print_import is not None
        assert print_import.metadata["media_query"] == "print"
        
        mobile_import = next((i for i in imports if i.name == "mobile.css"), None)
        assert mobile_import is not None
        assert "max-width: 600px" in mobile_import.metadata["media_query"]
    
    def test_extract_media_queries(self, tmp_path):
        """Test media query analysis in at-rules"""
        content = """
        @media screen and (min-width: 768px) and (max-width: 1024px) {
            .tablet-only {
                display: block;
            }
        }
        
        @media (prefers-color-scheme: dark) {
            body {
                background: #121212;
                color: #ffffff;
            }
        }
        
        @media print, (orientation: landscape) {
            .responsive-content {
                font-size: 12pt;
            }
        }
        """
        
        test_file = tmp_path / "test.css"
        test_file.write_text(content)
        
        result = self.parser.parse_file(test_file)
        entities = result.entities
        
        # Filter media query at-rules
        media_rules = [e for e in entities if e.entity_type == EntityType.CSS_AT_RULE 
                      and e.metadata.get("at_rule_type") == "media"]
        
        assert len(media_rules) >= 3
        
        # Check responsive media query
        responsive_rule = next((mr for mr in media_rules if "768px" in mr.name), None)
        assert responsive_rule is not None
        assert responsive_rule.metadata["is_responsive"] is True
        assert len(responsive_rule.metadata["media_features"]) >= 2
        
        # Check dark mode media query
        dark_mode_rule = next((mr for mr in media_rules if "prefers-color-scheme" in mr.name), None)
        assert dark_mode_rule is not None
        assert "dark" in dark_mode_rule.name
        
        # Check print media query
        print_rule = next((mr for mr in media_rules if "print" in mr.name), None)
        assert print_rule is not None
        assert print_rule.metadata["media_type"] == "print"
    
    def test_extract_keyframes(self, tmp_path):
        """Test keyframes at-rule extraction"""
        content = """
        @keyframes slideIn {
            0% {
                transform: translateX(-100%);
                opacity: 0;
            }
            50% {
                opacity: 0.5;
            }
            100% {
                transform: translateX(0);
                opacity: 1;
            }
        }
        
        @keyframes bounce {
            0%, 20%, 50%, 80%, 100% {
                transform: translateY(0);
            }
            40% {
                transform: translateY(-30px);
            }
            60% {
                transform: translateY(-15px);
            }
        }
        """
        
        test_file = tmp_path / "test.css"
        test_file.write_text(content)
        
        result = self.parser.parse_file(test_file)
        entities = result.entities
        
        # Filter keyframes at-rules
        keyframes_rules = [e for e in entities if e.entity_type == EntityType.CSS_AT_RULE 
                          and e.metadata.get("at_rule_type") == "keyframes"]
        
        assert len(keyframes_rules) >= 2
        
        # Check keyframes names
        keyframes_names = [kr.name for kr in keyframes_rules]
        assert "slideIn" in keyframes_names
        assert "bounce" in keyframes_names
        
        # Check keyframes signatures
        slide_rule = next((kr for kr in keyframes_rules if kr.name == "slideIn"), None)
        assert slide_rule is not None
        assert slide_rule.signature == "@keyframes slideIn"
        
        bounce_rule = next((kr for kr in keyframes_rules if kr.name == "bounce"), None)
        assert bounce_rule is not None
        assert bounce_rule.signature == "@keyframes bounce"
    
    def test_complex_css_document(self, tmp_path):
        """Test parsing complex CSS document with all features"""
        content = """
        /* Complex CSS with all features */
        @charset "UTF-8";
        @import url("base.css");
        
        :root {
            --primary-color: #007bff;
            --font-family: 'Roboto', sans-serif;
            --border-radius: 4px;
        }
        
        @media screen and (min-width: 768px) {
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        body {
            font-family: var(--font-family);
            line-height: 1.5;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        .btn {
            display: inline-block;
            padding: 0.375rem 0.75rem;
            margin-bottom: 0;
            font-size: 1rem;
            font-weight: 400;
            line-height: 1.5;
            text-align: center;
            text-decoration: none;
            vertical-align: middle;
            cursor: pointer;
            border: 1px solid transparent;
            border-radius: var(--border-radius);
            background-color: var(--primary-color);
            color: white;
            transition: all 0.15s ease-in-out;
        }
        
        .btn:hover,
        .btn:focus {
            background-color: darken(var(--primary-color), 10%);
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        .btn.btn-large {
            padding: 0.5rem 1rem;
            font-size: 1.125rem;
        }
        
        @supports (display: grid) {
            .grid-layout {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                grid-gap: 1rem;
            }
        }
        
        @media print {
            .btn {
                display: none;
            }
        }
        """
        
        test_file = tmp_path / "test.css"
        test_file.write_text(content)
        
        result = self.parser.parse_file(test_file)
        assert result.success is True
        
        entities = result.entities
        relations = result.relations
        
        # Check we extracted various entity types
        entity_types = {e.entity_type for e in entities}
        assert EntityType.CSS_RULE in entity_types
        assert EntityType.CSS_SELECTOR in entity_types
        assert EntityType.CSS_PROPERTY in entity_types
        assert EntityType.CSS_AT_RULE in entity_types
        assert EntityType.CSS_IMPORT in entity_types
        assert EntityType.CSS_VARIABLE in entity_types
        
        # Check we have all major components
        css_rules = [e for e in entities if e.entity_type == EntityType.CSS_RULE]
        assert len(css_rules) >= 5  # body, .btn, .btn:hover/.btn:focus, .btn.btn-large, .grid-layout, etc.
        
        at_rules = [e for e in entities if e.entity_type == EntityType.CSS_AT_RULE]
        at_rule_types = [ar.metadata["at_rule_type"] for ar in at_rules]
        assert "charset" in at_rule_types
        assert "import" in at_rule_types
        assert "media" in at_rule_types
        assert "keyframes" in at_rule_types
        assert "supports" in at_rule_types
        
        variables = [e for e in entities if e.entity_type == EntityType.CSS_VARIABLE]
        assert len(variables) >= 3
        variable_names = [v.name for v in variables]
        assert "--primary-color" in variable_names
        assert "--font-family" in variable_names
        assert "--border-radius" in variable_names
        
        # Check relations exist
        assert len(relations) > 0
        
        # Check for import relations
        import_relations = [r for r in relations if r.relation_type.value == "imports"]
        assert len(import_relations) > 0
        
        # Check for containment relations
        contains_relations = [r for r in relations if r.relation_type.value == "contains"]
        assert len(contains_relations) > 0
        
        # Check for variable usage relations
        uses_relations = [r for r in relations if r.relation_type.value == "uses_type"]
        assert len(uses_relations) > 0
        
        # Verify comprehensive parsing
        assert len(entities) >= 25  # Should have many entities in complex document
    
    def test_extract_relations(self, tmp_path):
        """Test CSS relation extraction"""
        content = """
        @import "base.css";
        
        :root {
            --primary-color: #007bff;
        }
        
        .container {
            width: 100%;
            background-color: var(--primary-color);
        }
        
        .button {
            padding: 10px;
            color: var(--primary-color);
        }
        """
        
        test_file = tmp_path / "test.css"
        test_file.write_text(content)
        
        result = self.parser.parse_file(test_file)
        assert result.success is True
        
        entities = result.entities
        relations = result.relations
        
        assert len(relations) > 0
        
        # Check for import relations
        import_relations = [r for r in relations if r.relation_type.value == "imports"]
        assert len(import_relations) > 0
        
        # Check for variable usage relations
        variable_relations = [r for r in relations if r.relation_type.value == "uses_type"]
        assert len(variable_relations) >= 2  # Two properties use --primary-color
        
        # Check for containment relations (selectors/properties within rules)
        contains_relations = [r for r in relations if r.relation_type.value == "contains"]
        assert len(contains_relations) > 0
        
        # Verify relation structure
        for relation in relations:
            assert relation.source_entity_id
            assert relation.target_entity_id
            assert relation.source_entity_id != relation.target_entity_id