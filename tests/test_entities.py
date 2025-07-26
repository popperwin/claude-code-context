"""
Unit tests for core entity models.

Tests entity creation, validation, and serialization.
"""

import pytest
from datetime import datetime
from pathlib import Path
from core.models.entities import (
    Entity, EntityType, SourceLocation, Visibility,
    ASTNode, Relation
)


class TestSourceLocation:
    """Test SourceLocation dataclass"""
    
    def test_valid_source_location(self):
        """Test creating a valid source location"""
        location = SourceLocation(
            file_path=Path("test.py"),
            start_line=1,
            end_line=5,
            start_column=0,
            end_column=10,
            start_byte=0,
            end_byte=100
        )
        
        assert location.file_path == Path("test.py")
        assert location.start_line == 1
        assert location.end_line == 5
        assert location.location_id == "test.py:1:0"
    
    def test_invalid_line_order(self):
        """Test that start_line cannot be greater than end_line"""
        with pytest.raises(ValueError, match="start_line cannot be greater than end_line"):
            SourceLocation(
                file_path=Path("test.py"),
                start_line=5,
                end_line=1,
                start_column=0,
                end_column=10,
                start_byte=0,
                end_byte=100
            )
    
    def test_invalid_column_order(self):
        """Test that start_column cannot be greater than end_column on same line"""
        with pytest.raises(ValueError, match="start_column cannot be greater than end_column"):
            SourceLocation(
                file_path=Path("test.py"),
                start_line=1,
                end_line=1,
                start_column=10,
                end_column=5,
                start_byte=0,
                end_byte=100
            )
    
    def test_invalid_byte_order(self):
        """Test that start_byte cannot be greater than end_byte"""
        with pytest.raises(ValueError, match="start_byte cannot be greater than end_byte"):
            SourceLocation(
                file_path=Path("test.py"),
                start_line=1,
                end_line=5,
                start_column=0,
                end_column=10,
                start_byte=100,
                end_byte=50
            )


class TestEntity:
    """Test Entity model"""
    
    def test_valid_entity_creation(self):
        """Test creating a valid entity"""
        location = SourceLocation(
            file_path=Path("test.py"),
            start_line=1,
            end_line=5,
            start_column=0,
            end_column=10,
            start_byte=0,
            end_byte=100
        )
        
        entity = Entity(
            id="file://test.py::function::test_func::1",
            name="test_func",
            qualified_name="module.test_func",
            entity_type=EntityType.FUNCTION,
            location=location,
            source_code="def test_func(): pass",
            signature="def test_func() -> None"
        )
        
        assert entity.name == "test_func"
        assert entity.entity_type == EntityType.FUNCTION
        assert entity.visibility == Visibility.PUBLIC  # default
        assert entity.is_async is False  # default
        assert len(entity.source_hash) == 16  # SHA-256 truncated
        assert entity.language_hint == "python"
    
    def test_entity_id_validation(self):
        """Test entity ID validation"""
        location = SourceLocation(
            file_path=Path("test.py"),
            start_line=1,
            end_line=5,
            start_column=0,
            end_column=10,
            start_byte=0,
            end_byte=100
        )
        
        with pytest.raises(ValueError, match="Entity ID must contain"):
            Entity(
                id="invalid_id",
                name="test_func",
                qualified_name="test_func",
                entity_type=EntityType.FUNCTION,
                location=location,
                source_code="def test_func(): pass"
            )
    
    def test_empty_name_validation(self):
        """Test that entity name cannot be empty"""
        location = SourceLocation(
            file_path=Path("test.py"),
            start_line=1,
            end_line=5,
            start_column=0,
            end_column=10,
            start_byte=0,
            end_byte=100
        )
        
        with pytest.raises(ValueError, match="Entity name cannot be empty"):
            Entity(
                id="file://test.py::function::::1",
                name="",
                qualified_name="test_func",
                entity_type=EntityType.FUNCTION,
                location=location,
                source_code="def test_func(): pass"
            )
    
    def test_source_hash_generation(self):
        """Test automatic source hash generation"""
        location = SourceLocation(
            file_path=Path("test.py"),
            start_line=1,
            end_line=5,
            start_column=0,
            end_column=10,
            start_byte=0,
            end_byte=100
        )
        
        entity = Entity(
            id="file://test.py::function::test_func::1",
            name="test_func",
            qualified_name="test_func",
            entity_type=EntityType.FUNCTION,
            location=location,
            source_code="def test_func(): pass"
        )
        
        assert entity.source_hash != ""
        assert len(entity.source_hash) == 16
        
        # Test that same source produces same hash
        entity2 = Entity(
            id="file://test.py::function::test_func2::1",
            name="test_func2",
            qualified_name="test_func2",
            entity_type=EntityType.FUNCTION,
            location=location,
            source_code="def test_func(): pass"
        )
        
        assert entity.source_hash == entity2.source_hash
    
    def test_update_source(self):
        """Test updating entity source code"""
        location = SourceLocation(
            file_path=Path("test.py"),
            start_line=1,
            end_line=5,
            start_column=0,
            end_column=10,
            start_byte=0,
            end_byte=100
        )
        
        entity = Entity(
            id="file://test.py::function::test_func::1",
            name="test_func",
            qualified_name="test_func",
            entity_type=EntityType.FUNCTION,
            location=location,
            source_code="def test_func(): pass"
        )
        
        original_hash = entity.source_hash
        original_modified = entity.last_modified
        
        # Update source
        updated_entity = entity.update_source("def test_func(): return 42")
        
        assert updated_entity.source_code == "def test_func(): return 42"
        assert updated_entity.source_hash != original_hash
        assert updated_entity.last_modified > original_modified
    
    def test_is_container_property(self):
        """Test is_container property"""
        location = SourceLocation(
            file_path=Path("test.py"),
            start_line=1,
            end_line=5,
            start_column=0,
            end_column=10,
            start_byte=0,
            end_byte=100
        )
        
        # Test container types
        container_types = [
            EntityType.PROJECT,
            EntityType.DIRECTORY,
            EntityType.FILE,
            EntityType.MODULE,
            EntityType.CLASS,
            EntityType.INTERFACE
        ]
        
        for entity_type in container_types:
            entity = Entity(
                id=f"test::{entity_type.value}::name::1",
                name="test",
                qualified_name="test",
                entity_type=entity_type,
                location=location,
                source_code="test"
            )
            assert entity.is_container is True
        
        # Test non-container type
        entity = Entity(
            id="test::function::name::1",
            name="test",
            qualified_name="test",
            entity_type=EntityType.FUNCTION,
            location=location,
            source_code="test"
        )
        assert entity.is_container is False
    
    def test_language_hint_detection(self):
        """Test language hint detection from file extension"""
        test_cases = [
            (Path("test.py"), "python"),
            (Path("test.js"), "javascript"),
            (Path("test.ts"), "typescript"),
            (Path("test.go"), "go"),
            (Path("test.rs"), "rust"),
            (Path("test.java"), "java"),
            (Path("test.unknown"), None)
        ]
        
        for file_path, expected_language in test_cases:
            location = SourceLocation(
                file_path=file_path,
                start_line=1,
                end_line=5,
                start_column=0,
                end_column=10,
                start_byte=0,
                end_byte=100
            )
            
            entity = Entity(
                id="test::function::name::1",
                name="test",
                qualified_name="test",
                entity_type=EntityType.FUNCTION,
                location=location,
                source_code="test"
            )
            
            assert entity.language_hint == expected_language
    
    def test_serialization(self):
        """Test entity serialization to/from dict"""
        location = SourceLocation(
            file_path=Path("test.py"),
            start_line=1,
            end_line=5,
            start_column=0,
            end_column=10,
            start_byte=0,
            end_byte=100
        )
        
        entity = Entity(
            id="file://test.py::function::test_func::1",
            name="test_func",
            qualified_name="test_func",
            entity_type=EntityType.FUNCTION,
            location=location,
            source_code="def test_func(): pass"
        )
        
        # Test to_dict
        entity_dict = entity.to_dict()
        assert isinstance(entity_dict, dict)
        assert entity_dict["name"] == "test_func"
        assert entity_dict["location"]["file_path"] == "test.py"  # Should be string
        
        # Test from_dict
        restored_entity = Entity.from_dict(entity_dict)
        assert restored_entity.name == entity.name
        assert restored_entity.entity_type == entity.entity_type
        assert restored_entity.location.file_path == entity.location.file_path


class TestASTNode:
    """Test ASTNode model"""
    
    def test_valid_ast_node(self):
        """Test creating a valid AST node"""
        location = SourceLocation(
            file_path=Path("test.py"),
            start_line=1,
            end_line=5,
            start_column=0,
            end_column=10,
            start_byte=0,
            end_byte=100
        )
        
        node = ASTNode(
            node_id="node_1",
            node_type="function_definition",
            language="python",
            location=location,
            text="def test(): pass"
        )
        
        assert node.node_id == "node_1"
        assert node.node_type == "function_definition"
        assert node.language == "python"
        assert node.is_named is True  # default
        assert node.is_error is False  # default
    
    def test_node_id_validation(self):
        """Test node ID validation"""
        location = SourceLocation(
            file_path=Path("test.py"),
            start_line=1,
            end_line=5,
            start_column=0,
            end_column=10,
            start_byte=0,
            end_byte=100
        )
        
        with pytest.raises(ValueError, match="Node ID cannot be empty"):
            ASTNode(
                node_id="",
                node_type="function_definition",
                language="python",
                location=location,
                text="def test(): pass"
            )
    
    def test_language_validation(self):
        """Test language validation"""
        location = SourceLocation(
            file_path=Path("test.py"),
            start_line=1,
            end_line=5,
            start_column=0,
            end_column=10,
            start_byte=0,
            end_byte=100
        )
        
        with pytest.raises(ValueError, match="Unsupported language"):
            ASTNode(
                node_id="node_1",
                node_type="function_definition",
                language="unsupported",
                location=location,
                text="def test(): pass"
            )
    
    def test_is_definition(self):
        """Test is_definition method"""
        location = SourceLocation(
            file_path=Path("test.py"),
            start_line=1,
            end_line=5,
            start_column=0,
            end_column=10,
            start_byte=0,
            end_byte=100
        )
        
        # Test definition node types
        definition_types = [
            "function_definition",
            "class_definition",
            "method_definition",
            "variable_declaration"
        ]
        
        for node_type in definition_types:
            node = ASTNode(
                node_id="node_1",
                node_type=node_type,
                language="python",
                location=location,
                text="test"
            )
            assert node.is_definition() is True
        
        # Test non-definition type
        node = ASTNode(
            node_id="node_1",
            node_type="expression",
            language="python",
            location=location,
            text="test"
        )
        assert node.is_definition() is False


class TestRelation:
    """Test Relation model"""
    
    def test_valid_relation(self):
        """Test creating a valid relation"""
        relation = Relation(
            id="rel_1",
            relation_type="calls",
            source_entity_id="entity_1",
            target_entity_id="entity_2",
            context="function call in main()"
        )
        
        assert relation.id == "rel_1"
        assert relation.relation_type == "calls"
        assert relation.strength == 1.0  # default
        assert isinstance(relation.created_at, datetime)
    
    def test_relation_id_validation(self):
        """Test relation ID validation"""
        with pytest.raises(ValueError, match="Relation ID cannot be empty"):
            Relation(
                id="",
                relation_type="calls",
                source_entity_id="entity_1",
                target_entity_id="entity_2"
            )
    
    def test_relation_type_validation(self):
        """Test relation type validation"""
        with pytest.raises(ValueError, match="Invalid relation type"):
            Relation(
                id="rel_1",
                relation_type="invalid_type",
                source_entity_id="entity_1",
                target_entity_id="entity_2"
            )
    
    def test_create_call_relation(self):
        """Test creating a call relation using helper method"""
        relation = Relation.create_call_relation(
            caller_id="func_1",
            callee_id="func_2",
            context="line 10"
        )
        
        assert relation.id == "call:func_1:func_2"
        assert relation.relation_type == "calls"
        assert relation.source_entity_id == "func_1"
        assert relation.target_entity_id == "func_2"
        assert relation.context == "line 10"
    
    def test_create_import_relation(self):
        """Test creating an import relation using helper method"""
        relation = Relation.create_import_relation(
            importer_id="module_1",
            imported_id="module_2",
            context="import statement"
        )
        
        assert relation.id == "import:module_1:module_2"
        assert relation.relation_type == "imports"
        assert relation.source_entity_id == "module_1"
        assert relation.target_entity_id == "module_2"
        assert relation.context == "import statement"
    
    def test_strength_validation(self):
        """Test strength field validation"""
        # Valid strength values
        for strength in [0.0, 0.5, 1.0]:
            relation = Relation(
                id="rel_1",
                relation_type="calls",
                source_entity_id="entity_1",
                target_entity_id="entity_2",
                strength=strength
            )
            assert relation.strength == strength
        
        # Invalid strength values should be caught by Pydantic
        with pytest.raises(ValueError):
            Relation(
                id="rel_1",
                relation_type="calls",
                source_entity_id="entity_1",
                target_entity_id="entity_2",
                strength=-0.1  # Below 0
            )
        
        with pytest.raises(ValueError):
            Relation(
                id="rel_1",
                relation_type="calls",
                source_entity_id="entity_1",
                target_entity_id="entity_2",
                strength=1.1  # Above 1
            )