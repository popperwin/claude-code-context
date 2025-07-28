"""
Unit tests for HTML parser.

Tests HTML entity extraction and relation detection.
"""

import pytest
from pathlib import Path
from core.parser.html_parser import HTMLParser
from core.models.entities import EntityType, Visibility


class TestHTMLParser:
    """Test HTML parser functionality"""
    
    def setup_method(self):
        """Setup test instance"""
        self.parser = HTMLParser()
        self.test_file = Path("test.html")
    
    def test_parser_registration(self):
        """Test parser is registered correctly"""
        assert "html" in self.parser.__class__.__name__.lower()
        assert ".html" in self.parser.get_supported_extensions()
        assert ".htm" in self.parser.get_supported_extensions()
    
    def test_extract_html_elements(self, tmp_path):
        """Test HTML element extraction"""
        content = """
        <!DOCTYPE html>
        <html lang="en">
            <head>
                <title>Test Page</title>
                <meta charset="utf-8">
            </head>
            <body>
                <div class="container" id="main">
                    <h1>Welcome</h1>
                    <p>Hello World</p>
                    <button onclick="doSomething()">Click</button>
                </div>
            </body>
        </html>
        """
        
        test_file = tmp_path / "test.html"
        test_file.write_text(content)
        
        result = self.parser.parse_file(test_file)
        entities = result.entities
        
        # Filter HTML elements
        html_elements = [e for e in entities if e.entity_type == EntityType.HTML_ELEMENT]
        
        assert len(html_elements) >= 7  # html, head, title, meta, body, div, h1, p, button
        
        # Check specific elements
        element_names = [e.name for e in html_elements]
        assert "html" in element_names
        assert "div" in element_names
        assert "h1" in element_names
        assert "button" in element_names
        
        # Check element with ID
        div_element = next((e for e in html_elements if e.name == "div"), None)
        assert div_element is not None
        assert "id:main" in div_element.signature
        assert "class:container" in div_element.signature
    
    def test_extract_web_components(self, tmp_path):
        """Test web component and custom element extraction"""
        content = """
        <html>
            <body>
                <my-custom-element data-prop="value">
                    <custom-header title="Test"></custom-header>
                    <react-component prop="data"></react-component>
                    <vue-component :bind="reactive"></vue-component>
                </my-custom-element>
            </body>
        </html>
        """
        
        test_file = tmp_path / "test.html"
        test_file.write_text(content)
        
        result = self.parser.parse_file(test_file)
        assert result.success is True
        
        entities = result.entities
        
        # Filter components
        components = [e for e in entities if e.entity_type == EntityType.HTML_COMPONENT]
        
        assert len(components) >= 4
        
        component_names = [e.name for e in components]
        assert "my-custom-element" in component_names
        assert "custom-header" in component_names
        assert "react-component" in component_names
        assert "vue-component" in component_names
        
        # Check custom element attributes
        custom_element = next((e for e in components if e.name == "my-custom-element"), None)
        assert custom_element is not None
        assert "data-prop:value" in custom_element.signature
    
    def test_extract_forms(self, tmp_path):
        """Test form element extraction"""
        content = """
        <html>
            <body>
                <form id="login-form" method="post" action="/login">
                    <input type="email" name="email" required>
                    <input type="password" name="password" minlength="8">
                    <select name="role" required>
                        <option value="user">User</option>
                        <option value="admin">Admin</option>
                    </select>
                    <textarea name="comments" maxlength="500"></textarea>
                    <button type="submit">Login</button>
                </form>
            </body>
        </html>
        """
        
        test_file = tmp_path / "test.html"
        test_file.write_text(content)
        
        result = self.parser.parse_file(test_file)
        assert result.success is True
        
        entities = result.entities
        
        # Filter form elements
        forms = [e for e in entities if e.entity_type == EntityType.HTML_FORM]
        
        assert len(forms) >= 6  # form, 2 inputs, select, textarea, button
        
        # Check form element
        form_element = next((e for e in forms if e.name == "form"), None)
        assert form_element is not None
        assert "method:post" in form_element.signature
        assert "action:/login" in form_element.signature
        
        # Check input elements
        inputs = [e for e in forms if e.name == "input"]
        assert len(inputs) >= 2
        
        email_input = next((e for e in inputs if "type:email" in e.signature), None)
        assert email_input is not None
        assert "name:email" in email_input.signature
        assert "required" in email_input.signature
    
    def test_extract_media_elements(self, tmp_path):
        """Test media element extraction"""
        content = """
        <html>
            <body>
                <img src="logo.png" alt="Company Logo" width="200">
                <video controls width="800">
                    <source src="movie.mp4" type="video/mp4">
                    <source src="movie.webm" type="video/webm">
                </video>
                <audio controls>
                    <source src="audio.mp3" type="audio/mpeg">
                </audio>
                <picture>
                    <source media="(min-width: 800px)" srcset="large.jpg">
                    <img src="small.jpg" alt="Responsive image">
                </picture>
            </body>
        </html>
        """
        
        test_file = tmp_path / "test.html"
        test_file.write_text(content)
        
        result = self.parser.parse_file(test_file)
        assert result.success is True
        
        entities = result.entities
        
        # Filter media elements
        media = [e for e in entities if e.entity_type == EntityType.HTML_MEDIA]
        
        assert len(media) >= 4  # img, video, audio, picture
        
        media_names = [e.name for e in media]
        assert "img" in media_names
        assert "video" in media_names
        assert "audio" in media_names
        assert "picture" in media_names
        
        # Check image attributes
        img_element = next((e for e in media if e.name == "img"), None)
        assert img_element is not None
        assert "src:logo.png" in img_element.signature
        assert "alt:Company Logo" in img_element.signature
        assert "width:200" in img_element.signature
    
    def test_extract_links(self, tmp_path):
        """Test link element extraction"""
        content = """
        <html>
            <head>
                <link rel="stylesheet" href="styles.css">
                <link rel="icon" href="favicon.ico">
            </head>
            <body>
                <a href="https://example.com" target="_blank" rel="noopener">External Link</a>
                <a href="/internal" class="nav-link">Internal Link</a>
                <a href="mailto:contact@example.com">Email Link</a>
                <a href="tel:+1234567890">Phone Link</a>
            </body>
        </html>
        """
        
        test_file = tmp_path / "test.html"
        test_file.write_text(content)
        
        result = self.parser.parse_file(test_file)
        assert result.success is True
        
        entities = result.entities
        
        # Filter link elements
        links = [e for e in entities if e.entity_type == EntityType.HTML_LINK]
        
        assert len(links) >= 6  # 2 link tags, 4 anchor tags
        
        # Check stylesheet link
        css_link = next((e for e in links if "href:styles.css" in e.signature), None)
        assert css_link is not None
        assert "rel:stylesheet" in css_link.signature
        
        # Check external anchor
        external_link = next((e for e in links if "https://example.com" in e.signature), None)
        assert external_link is not None
        assert "target:_blank" in external_link.signature
        assert "rel:noopener" in external_link.signature
    
    def test_extract_meta_information(self, tmp_path):
        """Test meta information extraction"""
        content = """
        <html>
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <meta name="description" content="Test page description">
                <meta name="keywords" content="html, parser, test">
                <meta property="og:title" content="Open Graph Title">
                <meta property="og:description" content="Open Graph Description">
                <meta name="twitter:card" content="summary">
                <title>Test Page Title</title>
            </head>
        </html>
        """
        
        test_file = tmp_path / "test.html"
        test_file.write_text(content)
        
        result = self.parser.parse_file(test_file)
        assert result.success is True
        
        entities = result.entities
        
        # Filter meta elements
        meta_elements = [e for e in entities if e.entity_type == EntityType.HTML_META]
        
        assert len(meta_elements) >= 7  # 6 meta tags + title
        
        # Check specific meta tags
        viewport_meta = next((e for e in meta_elements if "name:viewport" in e.signature), None)
        assert viewport_meta is not None
        assert "width=device-width" in viewport_meta.signature
        
        # Check Open Graph meta
        og_title = next((e for e in meta_elements if "property:og:title" in e.signature), None)
        assert og_title is not None
        assert "Open Graph Title" in og_title.signature
        
        # Check title
        title_element = next((e for e in meta_elements if e.name == "title"), None)
        assert title_element is not None
        assert "Test Page Title" in title_element.signature
    
    def test_extract_scripts(self, tmp_path):
        """Test script element extraction"""
        content = """
        <html>
            <head>
                <script src="jquery.min.js"></script>
                <script type="module" src="app.js"></script>
            </head>
            <body>
                <script>
                    function doSomething() {
                        console.log('Hello');
                    }
                </script>
                <script defer src="analytics.js"></script>
            </body>
        </html>
        """
        
        test_file = tmp_path / "test.html"
        test_file.write_text(content)
        
        result = self.parser.parse_file(test_file)
        assert result.success is True
        
        entities = result.entities
        
        # Filter script elements
        scripts = [e for e in entities if e.entity_type == EntityType.HTML_SCRIPT]
        
        assert len(scripts) >= 4
        
        # Check external script
        jquery_script = next((e for e in scripts if "src:jquery.min.js" in e.signature), None)
        assert jquery_script is not None
        
        # Check module script
        module_script = next((e for e in scripts if "type:module" in e.signature), None)
        assert module_script is not None
        assert "src:app.js" in module_script.signature
        
        # Check inline script
        inline_script = next((e for e in scripts if e.source_code and "doSomething" in e.source_code), None)
        assert inline_script is not None
        
        # Check deferred script
        defer_script = next((e for e in scripts if "defer" in e.signature), None)
        assert defer_script is not None
    
    def test_extract_styles(self, tmp_path):
        """Test style element extraction"""
        content = """
        <html>
            <head>
                <style>
                    body { margin: 0; }
                    .container { width: 100%; }
                </style>
                <style type="text/css">
                    h1 { color: blue; }
                </style>
            </head>
            <body>
                <div style="color: red; font-size: 16px;">Styled content</div>
            </body>
        </html>
        """
        
        test_file = tmp_path / "test.html"
        test_file.write_text(content)
        
        result = self.parser.parse_file(test_file)
        assert result.success is True
        
        entities = result.entities
        
        # Filter style elements
        styles = [e for e in entities if e.entity_type == EntityType.HTML_STYLE]
        
        assert len(styles) >= 3  # 2 style tags + 1 inline style
        
        # Check style tag with CSS content
        css_style = next((e for e in styles if e.source_code and "margin: 0" in e.source_code), None)
        assert css_style is not None
        assert "body" in css_style.source_code
        
        # Check inline style
        inline_style = next((e for e in styles if "color: red" in e.signature), None)
        assert inline_style is not None
        assert "font-size: 16px" in inline_style.signature
    
    def test_accessibility_attributes(self, tmp_path):
        """Test accessibility attribute extraction"""
        content = """
        <html>
            <body>
                <button aria-label="Close dialog" aria-pressed="false">×</button>
                <input type="text" aria-describedby="help-text" aria-required="true">
                <div role="alert" aria-live="polite">Status message</div>
                <nav aria-label="Main navigation" role="navigation">
                    <ul>
                        <li><a href="/" aria-current="page">Home</a></li>
                    </ul>
                </nav>
            </body>
        </html>
        """
        
        test_file = tmp_path / "test.html"
        test_file.write_text(content)
        
        result = self.parser.parse_file(test_file)
        assert result.success is True
        
        entities = result.entities
        
        # Check elements have accessibility attributes
        button = next((e for e in entities if e.name == "button"), None)
        assert button is not None
        assert "aria-label:Close dialog" in button.signature
        assert "aria-pressed:false" in button.signature
        
        input_elem = next((e for e in entities if e.name == "input"), None)
        assert input_elem is not None
        assert "aria-describedby:help-text" in input_elem.signature
        assert "aria-required:true" in input_elem.signature
        
        alert_div = next((e for e in entities if "role:alert" in getattr(e, 'signature', '')), None)
        assert alert_div is not None
        assert "aria-live:polite" in alert_div.signature
    
    def test_extract_relations(self, tmp_path):
        """Test HTML relation extraction"""
        content = """
        <html>
            <body>
                <div class="parent">
                    <p>Child paragraph</p>
                    <span>Child span</span>
                </div>
                <form>
                    <input type="text" name="username">
                    <button type="submit">Submit</button>
                </form>
            </body>
        </html>
        """
        
        test_file = tmp_path / "test.html"
        test_file.write_text(content)
        
        result = self.parser.parse_file(test_file)
        assert result.success is True
        
        entities = result.entities
        relations = result.relations
        
        assert len(relations) > 0
        
        # Check for containment relations
        containment_relations = [r for r in relations if r.relation_type.value == "contains"]
        assert len(containment_relations) > 0
        
        # Verify relation structure
        for relation in containment_relations:
            assert relation.source_entity_id
            assert relation.target_entity_id
            assert relation.source_entity_id != relation.target_entity_id
    
    def test_complex_html_document(self, tmp_path):
        """Test parsing complex HTML document"""
        content = """
        <!DOCTYPE html>
        <html lang="en">
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <title>Complex Test Page</title>
                <link rel="stylesheet" href="styles.css">
                <script src="app.js" defer></script>
                <style>
                    .highlight { background: yellow; }
                </style>
            </head>
            <body>
                <header class="site-header">
                    <nav role="navigation">
                        <ul class="nav-list">
                            <li><a href="/" aria-current="page">Home</a></li>
                            <li><a href="/about">About</a></li>
                        </ul>
                    </nav>
                </header>
                
                <main class="content">
                    <article class="post">
                        <h1>Article Title</h1>
                        <p>Article content with <em>emphasis</em> and <strong>strong</strong> text.</p>
                        <img src="image.jpg" alt="Article image" loading="lazy">
                    </article>
                    
                    <aside class="sidebar">
                        <my-widget data-config="{}">
                            <h2>Custom Widget</h2>
                        </my-widget>
                    </aside>
                </main>
                
                <footer class="site-footer">
                    <form class="newsletter" method="post">
                        <label for="email">Email:</label>
                        <input type="email" id="email" name="email" required>
                        <button type="submit">Subscribe</button>
                    </form>
                </footer>
                
                <script>
                    document.addEventListener('DOMContentLoaded', function() {
                        console.log('Page loaded');
                    });
                </script>
            </body>
        </html>
        """
        
        test_file = tmp_path / "test.html"
        test_file.write_text(content)
        
        result = self.parser.parse_file(test_file)
        assert result.success is True
        
        entities = result.entities
        relations = result.relations
        
        # Check we extracted various entity types
        entity_types = {e.entity_type for e in entities}
        assert EntityType.HTML_ELEMENT in entity_types
        assert EntityType.HTML_COMPONENT in entity_types  # my-widget
        assert EntityType.HTML_FORM in entity_types
        assert EntityType.HTML_MEDIA in entity_types  # img
        assert EntityType.HTML_LINK in entity_types  # nav links
        assert EntityType.HTML_META in entity_types
        assert EntityType.HTML_SCRIPT in entity_types
        assert EntityType.HTML_STYLE in entity_types
        
        # Check we have semantic HTML elements
        element_names = [e.name for e in entities if e.entity_type == EntityType.HTML_ELEMENT]
        semantic_elements = ['header', 'nav', 'main', 'article', 'aside', 'footer']
        for semantic_element in semantic_elements:
            assert semantic_element in element_names
        
        # Check custom component
        components = [e for e in entities if e.entity_type == EntityType.HTML_COMPONENT]
        assert len(components) >= 1
        widget = next((e for e in components if e.name == "my-widget"), None)
        assert widget is not None
        assert "data-config:{}" in widget.signature
        
        # Check we have relations
        assert len(relations) > 0
        
        # Verify at least some semantic structure is captured
        assert len(entities) >= 30  # Should have many entities in complex document