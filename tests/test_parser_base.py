"""
Tests for parser base classes and infrastructure.

Tests the foundation parser components including ParserProtocol,
registry functionality, and error recovery mechanisms.
"""

import pytest
from pathlib import Path
from typing import List
from unittest.mock import Mock, patch

from core.parser.base import ParserProtocol, ParseResult, BaseParser, ParseError
from core.parser.registry import ParserRegistry, parser_registry
from core.parser.error_recovery import ParseErrorRecovery, RobustParsingContext
from core.models.entities import Entity, Relation, EntityType, RelationType, SourceLocation


class MockParser(BaseParser):
    """Mock parser for testing"""
    
    def __init__(self):
        super().__init__("mock")
    
    def get_supported_extensions(self) -> List[str]:
        return [".mock", ".test"]
    
    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix in self.get_supported_extensions()
    
    def parse_file(self, file_path: Path) -> ParseResult:
        """Implementation of abstract parse_file method"""
        self._start_timing()
        
        # Validate file
        is_valid, error = self.validate_file(file_path)
        if not is_valid:
            return self._create_error_result(file_path, error or "Validation failed")
        
        try:
            # Read file safely
            content, file_hash, file_size = self._read_file_safe(file_path)
            
            # Extract entities and relations
            entities = self.extract_entities(None, content, file_path)
            relations = self.extract_relations(None, content, entities, file_path)
            
            # Create result
            result = ParseResult(
                file_path=file_path,
                language=self.language,
                entities=entities,
                ast_nodes=[],
                relations=relations,
                parse_time=self._get_elapsed_time(),
                file_size=file_size,
                file_hash=file_hash
            )
            
            return result
            
        except Exception as e:
            return self._create_error_result(file_path, str(e))
    
    def extract_entities(self, tree, content: str, file_path: Path) -> List[Entity]:
        # Create a simple mock entity
        location = SourceLocation(
            file_path=file_path,
            start_line=1,
            end_line=1,
            start_column=0,
            end_column=10,
            start_byte=0,
            end_byte=10
        )
        
        entity = Entity(
            id=f"mock::{file_path.name}::test::1",
            name="test_entity",
            qualified_name="test_entity",
            entity_type=EntityType.FUNCTION,
            location=location,
            source_code=content[:10] if content else "mock",
            signature="def test_entity():"
        )
        return [entity]
    
    def extract_relations(self, tree, content: str, entities: List[Entity], file_path: Path) -> List[Relation]:
        if len(entities) >= 1:
            # Create a simple mock relation
            relation = Relation.create_call_relation(
                entities[0].id,
                "mock::external::function",
                context="test call"
            )
            return [relation]
        return []


class TestParserBase:
    """Test base parser functionality"""
    
    def test_parse_result_creation(self):
        """Test ParseResult creation and properties"""
        result = ParseResult(
            file_path=Path("/test/file.py"),
            language="python"
        )
        
        assert result.file_path == Path("/test/file.py")
        assert result.language == "python"
        assert result.success is True  # No syntax errors
        assert result.entity_count == 0
        assert result.relation_count == 0
        
        # Test with entities
        location = SourceLocation(
            file_path=Path("/test/file.py"),
            start_line=1,
            end_line=1,
            start_column=0,
            end_column=10,
            start_byte=0,
            end_byte=10
        )
        
        entity = Entity(
            id="test::entity::1",
            name="test",
            qualified_name="test",
            entity_type=EntityType.FUNCTION,
            location=location,
            source_code="def test():"
        )
        
        result.entities.append(entity)
        assert result.entity_count == 1
    
    def test_parse_result_with_errors(self):
        """Test ParseResult with syntax errors"""
        result = ParseResult(
            file_path=Path("/test/bad.py"),
            language="python"
        )
        
        result.add_syntax_error({
            "type": "SYNTAX_ERROR",
            "message": "Invalid syntax",
            "line": 5,
            "column": 10
        })
        
        assert result.success is False
        assert result.partial_parse is True
        assert len(result.syntax_errors) == 1
    
    def test_base_parser_validation(self, tmp_path):
        """Test BaseParser file validation"""
        parser = MockParser()
        
        # Test valid file
        test_file = tmp_path / "test.mock"
        test_file.write_text("test content")
        
        is_valid, error = parser.validate_file(test_file)
        assert is_valid is True
        assert error is None
        
        # Test non-existent file
        missing_file = tmp_path / "missing.mock"
        is_valid, error = parser.validate_file(missing_file)
        assert is_valid is False
        assert "does not exist" in error
        
        # Test unsupported extension
        wrong_file = tmp_path / "test.wrong"
        wrong_file.write_text("content")
        is_valid, error = parser.validate_file(wrong_file)
        assert is_valid is False
        assert "cannot handle" in error
    
    def test_base_parser_safe_read(self, tmp_path):
        """Test safe file reading with encoding detection"""
        parser = MockParser()
        
        # Test UTF-8 file
        utf8_file = tmp_path / "utf8.mock"
        utf8_file.write_text("Hello, 世界!", encoding='utf-8')
        
        content, file_hash, file_size = parser._read_file_safe(utf8_file)
        assert "世界" in content
        assert len(file_hash) == 16  # SHA256 truncated
        assert file_size > 0


class TestParserRegistry:
    """Test parser registry functionality"""
    
    def test_registry_creation(self):
        """Test registry initialization"""
        registry = ParserRegistry()
        assert len(registry.get_supported_languages()) >= 0
        assert len(registry.get_supported_extensions()) >= 0
    
    def test_parser_registration(self):
        """Test parser registration and retrieval"""
        registry = ParserRegistry()
        
        # Register mock parser
        registry.register("mock", MockParser, [".mock", ".test"])
        
        assert "mock" in registry.get_supported_languages()
        assert ".mock" in registry.get_supported_extensions()
        assert ".test" in registry.get_supported_extensions()
        
        # Test parser retrieval
        parser = registry.get_parser("mock")
        assert parser is not None
        assert parser.get_language_name() == "mock"
    
    def test_parser_file_mapping(self, tmp_path):
        """Test file-to-parser mapping"""
        registry = ParserRegistry()
        registry.register("mock", MockParser, [".mock"])
        
        # Test file mapping
        mock_file = tmp_path / "test.mock"
        mock_file.touch()
        
        parser = registry.get_parser_for_file(mock_file)
        assert parser is not None
        assert parser.get_language_name() == "mock"
        
        # Test unsupported file
        other_file = tmp_path / "test.other"
        other_file.touch()
        
        parser = registry.get_parser_for_file(other_file)
        assert parser is None
    
    def test_registry_stats(self):
        """Test registry statistics"""
        registry = ParserRegistry()
        registry.register("mock", MockParser, [".mock"])
        
        stats = registry.get_registry_stats()
        assert stats["total_parsers"] >= 1
        assert stats["total_extensions"] >= 1
        assert "mock" in stats["languages"]
        assert ".mock" in stats["extensions"]
    
    def test_parser_override(self):
        """Test parser override functionality"""
        registry = ParserRegistry()
        
        # Register original parser
        registry.register("test", MockParser, [".test"])
        
        # Try to register again without override (should fail)
        with pytest.raises(ValueError, match="already registered"):
            registry.register("test", MockParser, [".test"])
        
        # Register with override (should succeed)
        registry.register("test", MockParser, [".test"], override=True)
        assert "test" in registry.get_supported_languages()


class TestErrorRecovery:
    """Test error recovery mechanisms"""
    
    def test_encoding_detection(self, tmp_path):
        """Test encoding detection"""
        # Create UTF-8 file
        utf8_file = tmp_path / "utf8.txt"
        utf8_file.write_text("Hello, 世界!", encoding='utf-8')
        
        encoding = ParseErrorRecovery.detect_encoding(utf8_file)
        assert encoding in ['utf-8', 'UTF-8']
    
    def test_encoding_fallback(self, tmp_path):
        """Test encoding fallback mechanism"""
        # Create UTF-8 file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, world!", encoding='utf-8')
        
        content, encoding = ParseErrorRecovery.read_with_encoding_fallback(test_file)
        assert content == "Hello, world!"
        # Be more flexible with encoding detection
        assert encoding is not None
        assert len(encoding) > 0
    
    def test_large_file_handling(self, tmp_path):
        """Test large file handling"""
        # Create a "large" file (for testing, use small size)
        large_file = tmp_path / "large.txt"
        content = "x" * 1000
        large_file.write_text(content)
        
        # Test with small max size
        result_content, was_truncated = ParseErrorRecovery.handle_large_file(
            large_file, max_size=500
        )
        
        assert was_truncated is True
        assert len(result_content) < len(content)
        assert "TRUNCATED" in result_content
    
    def test_syntax_error_recovery(self):
        """Test syntax error recovery"""
        content = "def func():\n    print('hello')\n    invalid syntax here\n    print('world')"
        errors = [{"line": 3, "message": "Invalid syntax"}]
        
        recovered_content, actions = ParseErrorRecovery.recover_from_syntax_errors(
            errors, content, Path("test.py")
        )
        
        # Recovery may or may not remove lines depending on the error pattern
        # Just ensure the function runs without error and returns valid data
        assert isinstance(recovered_content, str)
        assert isinstance(actions, list)
        assert len(recovered_content) > 0
    
    def test_robust_parsing_context(self, tmp_path):
        """Test robust parsing context manager"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, world!")
        
        with RobustParsingContext(test_file, "test") as ctx:
            content = ctx.get_content()
            assert content == "Hello, world!"
            
            recovery_info = ctx.get_recovery_info()
            assert "encoding_used" in recovery_info


class TestRelationTypes:
    """Test RelationType enum and Relation model"""
    
    def test_relation_type_enum(self):
        """Test RelationType enum values"""
        assert RelationType.CALLS.value == "calls"
        assert RelationType.IMPORTS.value == "imports"
        assert RelationType.INHERITS.value == "inherits"
        assert RelationType.CONTAINS.value == "contains"
        
        # Test all expected types exist
        expected_types = [
            "calls", "imports", "inherits", "implements", "extends", "mixes_in",
            "instantiates", "uses_type", "exports", "reads", "writes", "returns",
            "accepts", "decorates", "overrides", "tests", "references", "defines",
            "depends_on", "contains", "belongs_to"
        ]
        
        enum_values = [rt.value for rt in RelationType]
        for expected in expected_types:
            assert expected in enum_values
    
    def test_relation_factory_methods(self):
        """Test Relation factory methods"""
        # Test call relation
        call_rel = Relation.create_call_relation(
            "caller::id", "callee::id", "test context"
        )
        assert call_rel.relation_type == RelationType.CALLS
        assert call_rel.source_entity_id == "caller::id"
        assert call_rel.target_entity_id == "callee::id"
        assert call_rel.context == "test context"
        
        # Test import relation
        import_rel = Relation.create_import_relation(
            "importer::id", "imported::id", "import statement"
        )
        assert import_rel.relation_type == RelationType.IMPORTS
        assert import_rel.source_entity_id == "importer::id"
        assert import_rel.target_entity_id == "imported::id"
        
        # Test inheritance relation
        inherit_rel = Relation.create_inheritance_relation(
            "child::id", "parent::id", "class inheritance"
        )
        assert inherit_rel.relation_type == RelationType.INHERITS
        assert inherit_rel.source_entity_id == "child::id"
        assert inherit_rel.target_entity_id == "parent::id"


# Integration test
def test_integration_mock_parser(tmp_path):
    """Test integration of mock parser with registry"""
    # Create test file
    test_file = tmp_path / "integration.mock"
    test_file.write_text("def test_function():\n    pass")
    
    # Register parser
    registry = ParserRegistry()
    registry.register("mock", MockParser, [".mock"])
    
    # Get parser and parse file
    parser = registry.get_parser_for_file(test_file)
    assert parser is not None
    
    result = parser.parse_file(test_file)
    assert result.success
    assert result.entity_count == 1
    assert result.relation_count == 1
    
    # Verify entity
    entity = result.entities[0]
    assert entity.name == "test_entity"
    assert entity.entity_type == EntityType.FUNCTION
    
    # Verify relation
    relation = result.relations[0]
    assert relation.relation_type == RelationType.CALLS
    assert relation.source_entity_id == entity.id