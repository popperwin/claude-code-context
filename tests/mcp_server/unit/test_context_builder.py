"""
Unit tests for project context builder.

Tests tree generation, documentation loading, language detection, 
and word limit enforcement.
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from claude_code_context.mcp_server.context_builder import (
    ProjectContextBuilder,
    ProjectStructure,
    DocumentationInfo,
)
from claude_code_context.mcp_server.models import MCPServerConfig


class TestProjectStructure:
    """Test ProjectStructure dataclass"""
    
    def test_project_structure_creation(self):
        """Test creating ProjectStructure"""
        structure = ProjectStructure(
            name="test-project",
            description="A Python project",
            main_language="Python",
            languages_detected=["Python", "JavaScript"],
            total_files=25,
            total_lines=1500,
            directory_tree="test-project/\n├── src/\n└── tests/",
            key_files=["README.md", "pyproject.toml"]
        )
        
        assert structure.name == "test-project"
        assert structure.main_language == "Python"
        assert structure.total_files == 25
        assert len(structure.languages_detected) == 2


class TestDocumentationInfo:
    """Test DocumentationInfo dataclass"""
    
    def test_documentation_info_defaults(self):
        """Test DocumentationInfo with defaults"""
        doc_info = DocumentationInfo()
        
        assert doc_info.readme_content is None
        assert doc_info.claude_md_content is None
        assert doc_info.other_docs is None
        assert doc_info.has_readme is False
        assert doc_info.has_claude_md is False
        assert doc_info.docs_word_count == 0
    
    def test_documentation_info_with_content(self):
        """Test DocumentationInfo with content"""
        doc_info = DocumentationInfo(
            readme_content="# Test Project\n\nThis is a test.",
            claude_md_content="# Claude Setup\n\nConfiguration here.",
            has_readme=True,
            has_claude_md=True,
            docs_word_count=150
        )
        
        assert doc_info.has_readme is True
        assert doc_info.has_claude_md is True
        assert doc_info.docs_word_count == 150


class TestProjectContextBuilder:
    """Test ProjectContextBuilder class"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return MCPServerConfig(
            project_path=Path("/test/project"),
            collection_name="test_collection",
            qdrant_url="http://localhost:6334",
            context_word_limit=1000  # Smaller limit for testing
        )
    
    @pytest.fixture
    def builder(self, config):
        """Create context builder"""
        return ProjectContextBuilder(config)
    
    def test_init(self, builder, config):
        """Test builder initialization"""
        assert builder.config == config
        assert builder.project_path == config.project_path
        assert builder.max_words == 1000
        assert builder.max_tree_depth == 5
        assert builder.max_files_in_tree == 1000
        
        # Check language patterns are loaded
        assert 'Python' in builder.language_patterns
        assert '.py' in builder.language_patterns['Python']
        
        # Check ignore patterns
        assert '.git' in builder.ignore_patterns
        assert 'node_modules' in builder.ignore_patterns
        
        # Check key files
        assert 'README.md' in builder.key_files
        assert 'CLAUDE.md' in builder.key_files
    
    def test_should_ignore(self, builder):
        """Test file/directory ignore logic"""
        # Should ignore
        assert builder._should_ignore(Path("/test/.git"))
        assert builder._should_ignore(Path("/test/node_modules"))
        assert builder._should_ignore(Path("/test/__pycache__"))
        assert builder._should_ignore(Path("/test/.hidden_file"))
        assert builder._should_ignore(Path("/test/build/output"))
        
        # Should not ignore
        assert not builder._should_ignore(Path("/test/src"))
        assert not builder._should_ignore(Path("/test/main.py"))
        assert not builder._should_ignore(Path("/test/README.md"))  # Key file
        assert not builder._should_ignore(Path("/test/.gitignore"))  # Key file
    
    def test_count_words(self, builder):
        """Test word counting"""
        assert builder._count_words("") == 0
        assert builder._count_words(None) == 0
        assert builder._count_words("hello world") == 2
        assert builder._count_words("  hello   world  test  ") == 3
        assert builder._count_words("word1\nword2\tword3") == 3
    
    def test_truncate_to_word_limit(self, builder):
        """Test text truncation to word limit"""
        text = "one two three four five six seven eight nine ten"
        
        # No truncation needed
        result = builder._truncate_to_word_limit(text, 15)
        assert result == text
        
        # Truncation needed
        result = builder._truncate_to_word_limit(text, 5)
        assert result == "one two three four five\n\n[... truncated at 5 words ...]"
        
        # Empty text
        assert builder._truncate_to_word_limit("", 10) == ""
        assert builder._truncate_to_word_limit(None, 10) is None
    
    @pytest.mark.asyncio
    async def test_detect_languages_empty_project(self, builder):
        """Test language detection with empty project"""
        with patch('os.walk', return_value=[]):
            main_lang, all_langs = await builder._detect_languages()
            
            assert main_lang == "Unknown"
            assert all_langs == ["Unknown"]
    
    @pytest.mark.asyncio
    async def test_detect_languages_python_project(self, builder):
        """Test language detection with Python project"""
        mock_walk_data = [
            ("/test/project", ["src", "tests"], ["main.py", "README.md"]),
            ("/test/project/src", [], ["app.py", "utils.py", "config.json"]),
            ("/test/project/tests", [], ["test_app.py", "conftest.py"]),
        ]
        
        with patch('os.walk', return_value=mock_walk_data):
            with patch.object(builder, '_should_ignore', return_value=False):
                main_lang, all_langs = await builder._detect_languages()
                
                assert main_lang == "Python"
                assert "Python" in all_langs
                assert "JSON" in all_langs
    
    @pytest.mark.asyncio 
    async def test_detect_languages_mixed_project(self, builder):
        """Test language detection with mixed languages"""
        mock_walk_data = [
            ("/test/project", ["frontend", "backend"], ["package.json"]),
            ("/test/project/frontend", [], ["index.js", "app.js", "style.css"]),
            ("/test/project/backend", [], ["main.py", "models.py"]),
        ]
        
        with patch('os.walk', return_value=mock_walk_data):
            with patch.object(builder, '_should_ignore', return_value=False):
                main_lang, all_langs = await builder._detect_languages()
                
                # Should detect JavaScript as main (3 files) over Python (2 files)
                assert main_lang == "JavaScript"
                assert "JavaScript" in all_langs
                assert "Python" in all_langs
                assert "CSS" in all_langs
                assert "JSON" in all_langs
    
    @pytest.mark.asyncio
    async def test_detect_languages_with_ignore(self, builder):
        """Test language detection respects ignore patterns"""
        mock_walk_data = [
            ("/test/project", ["src", "node_modules"], ["main.py"]),
            ("/test/project/src", [], ["app.py"]),
            ("/test/project/node_modules", [], ["package.js", "lib.js"]),  # Should be ignored
        ]
        
        def mock_should_ignore(path):
            return "node_modules" in str(path)
        
        with patch('os.walk', return_value=mock_walk_data):
            with patch.object(builder, '_should_ignore', side_effect=mock_should_ignore):
                main_lang, all_langs = await builder._detect_languages()
                
                assert main_lang == "Python"
                assert "JavaScript" not in all_langs  # Should be ignored
    
    @pytest.mark.asyncio
    async def test_generate_directory_tree_simple(self, builder):
        """Test directory tree generation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create test structure
            (tmpdir_path / "src").mkdir()
            (tmpdir_path / "tests").mkdir()
            (tmpdir_path / "main.py").write_text("print('hello')")
            (tmpdir_path / "README.md").write_text("# Test")
            (tmpdir_path / "src" / "app.py").write_text("def main(): pass")
            (tmpdir_path / "tests" / "test_app.py").write_text("def test(): pass")
            
            builder.project_path = tmpdir_path
            
            with patch.object(builder, '_should_ignore', return_value=False):
                tree, total_files, total_lines = await builder._generate_directory_tree()
                
                assert tmpdir_path.name in tree
                assert "src" in tree
                assert "tests" in tree
                assert "main.py" in tree
                assert "README.md" in tree
                assert "├──" in tree or "└──" in tree  # Tree structure
                assert total_files > 0
                assert total_lines > 0
    
    @pytest.mark.asyncio
    async def test_generate_directory_tree_depth_limit(self, builder):
        """Test directory tree respects depth limit"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create deep structure
            deep_path = tmpdir_path / "level1" / "level2" / "level3" / "level4" / "level5"
            deep_path.mkdir(parents=True)
            (deep_path / "deep_file.py").write_text("# deep")
            
            builder.project_path = tmpdir_path
            
            with patch.object(builder, '_should_ignore', return_value=False):
                tree, _, _ = await builder._generate_directory_tree(max_depth=3)
                
                assert "level1" in tree
                assert "level2" in tree
                # level5 should not appear due to depth limit
                assert "level5" not in tree
    
    @pytest.mark.asyncio
    async def test_generate_directory_tree_file_limit(self, builder):
        """Test directory tree respects file limit"""
        builder.max_files_in_tree = 5  # Very low limit for testing
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create many files
            for i in range(10):
                (tmpdir_path / f"file_{i}.py").write_text(f"# file {i}")
            
            builder.project_path = tmpdir_path
            
            with patch.object(builder, '_should_ignore', return_value=False):
                tree, _, _ = await builder._generate_directory_tree()
                
                assert "truncated" in tree.lower()
    
    @pytest.mark.asyncio
    async def test_load_documentation_readme_only(self, builder):
        """Test loading documentation with README only"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            readme_path = tmpdir_path / "README.md"
            readme_content = "# Test Project\n\nThis is a test project."
            readme_path.write_text(readme_content)
            
            builder.project_path = tmpdir_path
            
            doc_info = await builder._load_documentation()
            
            assert doc_info.has_readme is True
            assert doc_info.has_claude_md is False
            assert doc_info.readme_content == readme_content
            assert doc_info.claude_md_content is None
            assert doc_info.docs_word_count > 0
    
    @pytest.mark.asyncio
    async def test_load_documentation_claude_md_only(self, builder):
        """Test loading documentation with CLAUDE.md only"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            claude_path = tmpdir_path / "CLAUDE.md"
            claude_content = "# Claude Setup\n\nProject configuration for Claude."
            claude_path.write_text(claude_content)
            
            builder.project_path = tmpdir_path
            
            doc_info = await builder._load_documentation()
            
            assert doc_info.has_readme is False
            assert doc_info.has_claude_md is True
            assert doc_info.readme_content is None
            assert doc_info.claude_md_content == claude_content
            assert doc_info.docs_word_count > 0
    
    @pytest.mark.asyncio
    async def test_load_documentation_both_files(self, builder):
        """Test loading documentation with both README and CLAUDE.md"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            readme_content = "# Test Project\n\nThis is a test."
            claude_content = "# Claude Setup\n\nConfiguration here."
            
            (tmpdir_path / "README.md").write_text(readme_content)
            (tmpdir_path / "CLAUDE.md").write_text(claude_content)
            
            builder.project_path = tmpdir_path
            
            doc_info = await builder._load_documentation()
            
            assert doc_info.has_readme is True
            assert doc_info.has_claude_md is True
            assert doc_info.readme_content == readme_content
            assert doc_info.claude_md_content == claude_content
            assert doc_info.docs_word_count > 0
    
    @pytest.mark.asyncio
    async def test_load_documentation_other_docs(self, builder):
        """Test loading other documentation files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create various doc files
            (tmpdir_path / "CONTRIBUTING.md").write_text("# Contributing\n\nGuidelines here.")
            (tmpdir_path / "CHANGELOG.md").write_text("# Changelog\n\nVersion history.")
            (tmpdir_path / "docs.txt").write_text("Documentation file.")
            (tmpdir_path / "not_doc.py").write_text("# Not a doc file")
            
            builder.project_path = tmpdir_path
            
            doc_info = await builder._load_documentation()
            
            assert doc_info.other_docs is not None
            assert len(doc_info.other_docs) >= 2  # CONTRIBUTING and CHANGELOG at minimum
            
            doc_names = [name for name, _ in doc_info.other_docs]
            assert any("CONTRIBUTING" in name for name in doc_names)
            assert any("CHANGELOG" in name for name in doc_names)
    
    @pytest.mark.asyncio
    async def test_load_documentation_read_error(self, builder):
        """Test handling of documentation read errors"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            builder.project_path = tmpdir_path
            
            # Mock file existence but read failure
            with patch.object(Path, 'exists', return_value=True):
                with patch.object(Path, 'read_text', side_effect=IOError("Permission denied")):
                    doc_info = await builder._load_documentation()
                    
                    # Should handle errors gracefully
                    assert doc_info.has_readme is False
                    assert doc_info.has_claude_md is False
    
    @pytest.mark.asyncio
    async def test_identify_key_files(self, builder):
        """Test key file identification"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create key files
            (tmpdir_path / "README.md").write_text("readme")
            (tmpdir_path / "package.json").write_text("{}")
            (tmpdir_path / "pyproject.toml").write_text("[tool]")
            (tmpdir_path / "main.py").write_text("# main")
            (tmpdir_path / "random_file.txt").write_text("not key")
            
            builder.project_path = tmpdir_path
            
            key_files = await builder._identify_key_files()
            
            assert "README.md" in key_files
            assert "package.json" in key_files  
            assert "pyproject.toml" in key_files
            assert "main.py" in key_files  # Entry point
            assert "random_file.txt" not in key_files
            assert key_files == sorted(key_files)  # Should be sorted
    
    @pytest.mark.asyncio
    async def test_build_project_context_minimal(self, builder):
        """Test building project context with minimal project"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            (tmpdir_path / "main.py").write_text("print('hello')")
            
            builder.project_path = tmpdir_path
            builder.max_words = 500  # Small limit
            
            context = await builder.build_project_context()
            
            assert isinstance(context, str)
            assert len(context) > 0
            assert tmpdir_path.name in context
            assert "Python" in context  # Should detect Python
            assert "main.py" in context
            
            # Check word limit
            word_count = builder._count_words(context)
            assert word_count <= builder.max_words
    
    @pytest.mark.asyncio
    async def test_build_project_context_with_docs(self, builder):
        """Test building project context with documentation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create files
            (tmpdir_path / "main.py").write_text("print('hello')")
            (tmpdir_path / "README.md").write_text("# Test Project\n\nThis is a test project with documentation.")
            (tmpdir_path / "CLAUDE.md").write_text("# Claude Configuration\n\nSetup instructions for Claude.")
            
            src_dir = tmpdir_path / "src"
            src_dir.mkdir()
            (src_dir / "app.py").write_text("def main(): pass")
            
            builder.project_path = tmpdir_path
            builder.max_words = 1000
            
            context = await builder.build_project_context()
            
            assert isinstance(context, str)
            assert len(context) > 0
            
            # Should include project info
            assert tmpdir_path.name in context
            assert "Python" in context
            
            # Should include directory structure
            assert "Directory Structure" in context
            assert "src" in context
            
            # Should prioritize CLAUDE.md over README
            assert "CLAUDE.md" in context
            assert "Claude Configuration" in context
            
            # Check word limit
            word_count = builder._count_words(context)
            assert word_count <= builder.max_words
    
    @pytest.mark.asyncio
    async def test_build_project_context_word_limit_enforced(self, builder):
        """Test that word limit is strictly enforced"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create large documentation
            large_doc = "word " * 2000  # 2000 words
            (tmpdir_path / "README.md").write_text(large_doc)
            (tmpdir_path / "main.py").write_text("print('hello')")
            
            builder.project_path = tmpdir_path
            builder.max_words = 100  # Very small limit
            
            context = await builder.build_project_context()
            
            word_count = builder._count_words(context)
            assert word_count <= builder.max_words
            
            # Should still include basic project info even with tight limit
            assert tmpdir_path.name in context
    
    @pytest.mark.asyncio
    async def test_build_project_context_error_handling(self, builder):
        """Test error handling in context building"""
        # Non-existent project path
        builder.project_path = Path("/non/existent/path")
        
        context = await builder.build_project_context()
        
        assert isinstance(context, str)
        assert "Error building project context" in context
    
    @pytest.mark.asyncio
    async def test_get_project_summary(self, builder):
        """Test getting project summary"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create test project with more Python files than other types
            (tmpdir_path / "main.py").write_text("print('hello')\nprint('world')")
            (tmpdir_path / "utils.py").write_text("def helper(): pass")
            (tmpdir_path / "config.py").write_text("CONFIG = {}")
            (tmpdir_path / "README.md").write_text("# Test\n\nProject description.")
            (tmpdir_path / "package.json").write_text("{}")
            
            builder.project_path = tmpdir_path
            
            summary = await builder.get_project_summary()
            
            assert isinstance(summary, dict)
            assert summary["project_name"] == tmpdir_path.name
            assert summary["project_path"] == str(tmpdir_path)
            assert summary["main_language"] == "Python"  # 3 Python files vs 1 Markdown, 1 JSON
            assert "Python" in summary["languages_detected"]
            assert summary["total_files"] > 0
            assert summary["total_lines"] > 0
            assert summary["key_files_count"] > 0
            assert summary["has_readme"] is True
            assert summary["has_claude_md"] is False
            assert summary["docs_word_count"] > 0
            assert summary["max_context_words"] == builder.max_words
    
    @pytest.mark.asyncio
    async def test_get_project_summary_error(self, builder):
        """Test project summary with error"""
        builder.project_path = Path("/non/existent/path")
        
        summary = await builder.get_project_summary()
        
        assert isinstance(summary, dict)
        assert "error" in summary
        assert summary["project_name"] == "path"  # Last part of path
    
    def test_is_valid_project(self, builder):
        """Test project validation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            builder.project_path = tmpdir_path
            
            # Valid project
            assert builder.is_valid_project() is True
            
            # Non-existent path
            builder.project_path = Path("/non/existent/path")
            assert builder.is_valid_project() is False
            
            # File instead of directory
            test_file = tmpdir_path / "test.txt"
            test_file.write_text("test")
            builder.project_path = test_file
            assert builder.is_valid_project() is False
    
    def test_is_valid_project_ignored(self, builder):
        """Test project validation with ignored directories"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            git_path = tmpdir_path / ".git"
            git_path.mkdir()
            
            builder.project_path = git_path
            
            # Should be invalid because .git is ignored
            assert builder.is_valid_project() is False


class TestContextBuilderIntegration:
    """Integration tests for context builder"""
    
    @pytest.mark.asyncio
    async def test_full_context_generation_workflow(self):
        """Test complete context generation workflow"""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            
            # Create realistic project structure
            (project_path / "src").mkdir()
            (project_path / "tests").mkdir()
            (project_path / "docs").mkdir()
            
            # Python files
            (project_path / "main.py").write_text("""
#!/usr/bin/env python3
\"\"\"Main application entry point.\"\"\"

def main():
    print("Hello, World!")
    
if __name__ == "__main__":
    main()
""")
            
            (project_path / "src" / "__init__.py").write_text("")
            (project_path / "src" / "utils.py").write_text("""
\"\"\"Utility functions.\"\"\"

def helper_function():
    return "helper"
    
class UtilityClass:
    def method(self):
        pass
""")
            
            # Test files
            (project_path / "tests" / "test_main.py").write_text("""
import pytest
from main import main

def test_main():
    assert main() is None
""")
            
            # Documentation
            (project_path / "README.md").write_text("""
# Test Project

This is a comprehensive test project for demonstrating
the context builder functionality.

## Features

- Python application
- Unit tests
- Documentation
- Modern project structure

## Usage

Run with `python main.py`
""")
            
            (project_path / "CLAUDE.md").write_text("""
# Claude Configuration

This project is set up for Claude Code integration.

## Search Tips

Use specific queries for best results:
- Find main functions
- Search for utility classes
- Look for test patterns
""")
            
            # Config files
            (project_path / "pyproject.toml").write_text("""
[tool.poetry]
name = "test-project"
version = "0.1.0"
description = "Test project"

[tool.poetry.dependencies]
python = "^3.8"
""")
            
            (project_path / "requirements.txt").write_text("""
pytest>=6.0.0
black>=21.0.0
""")
            
            # Create context builder
            config = MCPServerConfig(
                project_path=project_path,
                collection_name="test_collection",
                qdrant_url="http://localhost:6334",
                context_word_limit=2000
            )
            
            builder = ProjectContextBuilder(config)
            
            # Test all functionality
            assert builder.is_valid_project() is True
            
            # Get summary
            summary = await builder.get_project_summary()
            assert summary["main_language"] == "Python"
            assert summary["has_readme"] is True
            assert summary["has_claude_md"] is True
            assert summary["total_files"] > 5
            
            # Build full context
            context = await builder.build_project_context()
            
            # Verify context content
            assert project_path.name in context  # Project directory name should be in context
            assert "Python" in context
            assert "main.py" in context
            assert "Directory Structure" in context
            assert "src" in context
            assert "tests" in context
            
            # Should prioritize CLAUDE.md
            assert "Claude Configuration" in context
            
            # Verify word limit respected
            word_count = builder._count_words(context)
            assert word_count <= config.context_word_limit
            
            # Context should be substantial but not empty
            assert word_count > 50
            assert len(context) > 500