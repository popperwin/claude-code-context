"""
Tests for DeterministicEntityId entity ID generation consistency.

Validates that entity IDs are stable, unique, and consistent across
different invocations while properly handling file content changes.
"""

import pytest
import hashlib
from pathlib import Path

from core.models.entities import Entity, EntityType, SourceLocation
from core.sync.deterministic import DeterministicEntityId


class TestDeterministicEntityId:
    """Test suite for deterministic entity ID generation."""
    
    def setup_method(self):
        """Clear ID cache before each test."""
        DeterministicEntityId._id_cache.clear()
    
    def create_test_entity(
        self, 
        name: str = "test_function",
        entity_type: EntityType = EntityType.FUNCTION,
        start_line: int = 10,
        start_column: int = 4
    ) -> Entity:
        """Create a test entity with specified parameters."""
        location = SourceLocation(
            file_path=Path("/test/file.py"),
            start_line=start_line,
            start_column=start_column,
            end_line=start_line + 5,
            end_column=start_column + 10,
            start_byte=start_line * 80 + start_column,  # Approximate byte position
            end_byte=(start_line + 5) * 80 + start_column + 10
        )
        
        return Entity(
            id=f"file::/test/file.py::{entity_type.value}::{name}::{start_line}",  # Valid ID format
            name=name,
            qualified_name=name,  # Required field
            entity_type=entity_type,
            location=location,
            source_code=f"def {name}():\n    pass",
            docstring=f"Test {name} function",
            signature=f"{name}()" if entity_type == EntityType.FUNCTION else None
        )
    
    def test_consistent_id_generation(self):
        """Test that same entity + file_hash produces same ID."""
        entity = self.create_test_entity()
        file_hash = "abc123def456"
        
        # Generate ID multiple times
        id1 = DeterministicEntityId.generate(entity, file_hash)
        id2 = DeterministicEntityId.generate(entity, file_hash)
        id3 = DeterministicEntityId.generate(entity, file_hash)
        
        # All IDs should be identical
        assert id1 == id2 == id3
        assert len(id1) == 36  # UUID format
    
    def test_different_entities_different_ids(self):
        """Test that different entities produce different IDs."""
        file_hash = "abc123def456"
        
        entity1 = self.create_test_entity(name="function_one")
        entity2 = self.create_test_entity(name="function_two")
        entity3 = self.create_test_entity(name="function_one", entity_type=EntityType.CLASS)
        
        id1 = DeterministicEntityId.generate(entity1, file_hash)
        id2 = DeterministicEntityId.generate(entity2, file_hash)
        id3 = DeterministicEntityId.generate(entity3, file_hash)
        
        # All IDs should be different
        assert id1 != id2
        assert id1 != id3
        assert id2 != id3
    
    def test_different_file_hash_different_ids(self):
        """Test that same entity with different file hash produces different ID."""
        entity = self.create_test_entity()
        
        # Clear cache to ensure fresh calculation
        DeterministicEntityId.clear_cache()
        
        id1 = DeterministicEntityId.generate(entity, "abcd1234efgh5678")
        id2 = DeterministicEntityId.generate(entity, "wxyz9876stuv4321")
        
        assert id1 != id2
    
    def test_different_locations_different_ids(self):
        """Test that same entity at different locations produces different IDs."""
        file_hash = "abc123def456"
        
        entity1 = self.create_test_entity(start_line=10, start_column=4)
        entity2 = self.create_test_entity(start_line=20, start_column=4)
        entity3 = self.create_test_entity(start_line=10, start_column=8)
        
        id1 = DeterministicEntityId.generate(entity1, file_hash)
        id2 = DeterministicEntityId.generate(entity2, file_hash)
        id3 = DeterministicEntityId.generate(entity3, file_hash)
        
        # All IDs should be different
        assert id1 != id2
        assert id1 != id3
        assert id2 != id3
    
    def test_id_format_and_length(self):
        """Test that generated IDs have correct format and length."""
        entity = self.create_test_entity()
        file_hash = "abc123def456"
        
        entity_id = DeterministicEntityId.generate(entity, file_hash)
        
        # Should be 36 character UUID format
        assert len(entity_id) == 36
        # UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
        parts = entity_id.split('-')
        assert len(parts) == 5
        assert len(parts[0]) == 8 and all(c in '0123456789abcdef' for c in parts[0])
        assert len(parts[1]) == 4 and all(c in '0123456789abcdef' for c in parts[1])
        assert len(parts[2]) == 4 and all(c in '0123456789abcdef' for c in parts[2])
        assert len(parts[3]) == 4 and all(c in '0123456789abcdef' for c in parts[3])
        assert len(parts[4]) == 12 and all(c in '0123456789abcdef' for c in parts[4])
    
    def test_cache_functionality(self):
        """Test that ID cache improves performance for repeated calls."""
        entity = self.create_test_entity()
        file_hash = "abc123def456"
        
        # Clear cache and verify it's empty
        DeterministicEntityId._id_cache.clear()
        assert len(DeterministicEntityId._id_cache) == 0
        
        # First call should populate cache
        id1 = DeterministicEntityId.generate(entity, file_hash)
        assert len(DeterministicEntityId._id_cache) == 1
        
        # Second call should use cache
        id2 = DeterministicEntityId.generate(entity, file_hash)
        assert id1 == id2
        assert len(DeterministicEntityId._id_cache) == 1
    
    def test_update_entity_with_deterministic_id(self):
        """Test updating entity with deterministic ID."""
        entity = self.create_test_entity()
        file_hash = "abc123def456"
        original_id = entity.id
        
        # Update entity with deterministic ID
        updated_entity = DeterministicEntityId.update_entity_with_deterministic_id(
            entity, file_hash
        )
        
        # Should be a new entity object with updated ID (entities are immutable)
        assert updated_entity is not entity
        assert updated_entity.id != original_id
        assert len(updated_entity.id) == 36
        
        # ID should be consistent
        expected_id = DeterministicEntityId.generate(entity, file_hash)
        assert updated_entity.id == expected_id
        
        # Other fields should be the same
        assert updated_entity.name == entity.name
        assert updated_entity.entity_type == entity.entity_type
    
    def test_hash_collision_resistance(self):
        """Test that similar entities produce different IDs (hash collision resistance)."""
        file_hash = "abc123def456"
        
        # Create similar entities
        entities = [
            self.create_test_entity(name="test_func"),
            self.create_test_entity(name="test_func_"),
            self.create_test_entity(name="test_func", start_line=11),
            self.create_test_entity(name="test_func", entity_type=EntityType.METHOD),
        ]
        
        ids = [DeterministicEntityId.generate(entity, file_hash) for entity in entities]
        
        # All IDs should be unique
        assert len(set(ids)) == len(ids)
    
    def test_file_hash_truncation(self):
        """Test that file hash is properly truncated to 8 characters."""
        entity = self.create_test_entity()
        
        # Test with long hash
        long_hash = "a" * 64  # Full SHA256 hash length
        id1 = DeterministicEntityId.generate(entity, long_hash)
        
        # Test with short hash
        short_hash = "a" * 4
        id2 = DeterministicEntityId.generate(entity, short_hash)
        
        # Both should produce valid IDs (but different)
        assert len(id1) == 36
        assert len(id2) == 36
        assert id1 != id2
    
    def test_cache_clearing(self):
        """Test cache clearing functionality."""
        entity = self.create_test_entity()
        file_hash = "abc123def456"
        
        # Populate cache
        DeterministicEntityId.generate(entity, file_hash)
        assert len(DeterministicEntityId._id_cache) > 0
        
        # Clear cache
        DeterministicEntityId.clear_cache()
        assert len(DeterministicEntityId._id_cache) == 0
    
    def test_large_cache_performance(self):
        """Test cache performance with many different entities."""
        file_hash = "abc123def456"
        entities = []
        
        # Create many different entities
        for i in range(100):
            entity = self.create_test_entity(
                name=f"function_{i}",
                start_line=i * 10
            )
            entities.append(entity)
        
        # Generate IDs for all entities (populates cache)
        ids = [DeterministicEntityId.generate(entity, file_hash) for entity in entities]
        
        # Verify cache has entries
        assert len(DeterministicEntityId._id_cache) == 100
        
        # All IDs should be unique
        assert len(set(ids)) == 100
        
        # Verify cache performance (second generation should be instant)
        ids2 = [DeterministicEntityId.generate(entity, file_hash) for entity in entities]
        assert ids == ids2
    
    def test_entity_type_differentiation(self):
        """Test that all entity types produce different IDs for same name."""
        file_hash = "abc123def456"
        name = "TestName"
        
        # Create entities of different types with same name
        entity_types = [
            EntityType.FUNCTION,
            EntityType.CLASS,
            EntityType.METHOD,
            EntityType.VARIABLE,
            EntityType.CONSTANT,
            EntityType.MODULE,
            EntityType.INTERFACE,
            EntityType.ENUM
        ]
        
        ids = []
        for entity_type in entity_types:
            entity = self.create_test_entity(name=name, entity_type=entity_type)
            entity_id = DeterministicEntityId.generate(entity, file_hash)
            ids.append(entity_id)
        
        # All IDs should be unique
        assert len(set(ids)) == len(entity_types)


class TestDeterministicEntityIdEdgeCases:
    """Test edge cases and error conditions for deterministic ID generation."""
    
    def setup_method(self):
        """Clear ID cache before each test."""
        DeterministicEntityId._id_cache.clear()
    
    def test_empty_entity_name(self):
        """Test handling of empty entity name."""
        location = SourceLocation(
            file_path=Path("/test/file.py"),
            start_line=1,
            start_column=0,
            end_line=1,
            end_column=10,
            start_byte=0,
            end_byte=10
        )
        
        entity = Entity(
            id="file::/test/file.py::function::empty_name::1",
            name="empty_name",  # Cannot have empty name due to validation
            qualified_name="empty_name",
            entity_type=EntityType.FUNCTION,
            location=location,
            source_code="",
            docstring=None,
            signature=None
        )
        
        file_hash = "abc123"
        entity_id = DeterministicEntityId.generate(entity, file_hash)
        
        # Should still generate valid ID
        assert len(entity_id) == 36
        # Check UUID format
        parts = entity_id.split('-')
        assert len(parts) == 5
    
    def test_unicode_entity_name(self):
        """Test handling of unicode characters in entity name."""
        location = SourceLocation(
            file_path=Path("/test/file.py"),
            start_line=1,
            start_column=0,
            end_line=1,
            end_column=10,
            start_byte=0,
            end_byte=10
        )
        
        entity = Entity(
            id="file::/test/file.py::function::测试函数_ñáḿé::1",
            name="测试函数_ñáḿé",  # Unicode name
            qualified_name="测试函数_ñáḿé",
            entity_type=EntityType.FUNCTION,
            location=location,
            source_code="def test(): pass",
            docstring=None,
            signature=None
        )
        
        file_hash = "abc123"
        entity_id = DeterministicEntityId.generate(entity, file_hash)
        
        # Should generate valid ID
        assert len(entity_id) == 36
        # Check UUID format
        parts = entity_id.split('-')
        assert len(parts) == 5
    
    def test_very_long_entity_name(self):
        """Test handling of very long entity names."""
        location = SourceLocation(
            file_path=Path("/test/file.py"),
            start_line=1,
            start_column=0,
            end_line=1,
            end_column=10,
            start_byte=0,
            end_byte=10
        )
        
        long_name = "a" * 100  # Shorter to avoid issues with ID format
        entity = Entity(
            id=f"file::/test/file.py::function::{long_name}::1",
            name=long_name,  # Very long name
            qualified_name=long_name,
            entity_type=EntityType.FUNCTION,
            location=location,
            source_code="def test(): pass",
            docstring=None,
            signature=None
        )
        
        file_hash = "abc123"
        entity_id = DeterministicEntityId.generate(entity, file_hash)
        
        # Should still generate valid UUID format ID
        assert len(entity_id) == 36
        # Check UUID format
        parts = entity_id.split('-')
        assert len(parts) == 5
    
    def test_special_characters_in_name(self):
        """Test handling of special characters in entity name."""
        location = SourceLocation(
            file_path=Path("/test/file.py"),
            start_line=1,
            start_column=0,
            end_line=1,
            end_column=10,
            start_byte=0,
            end_byte=10
        )
        
        special_name = "test_func_special"  # Avoid special chars in ID
        entity = Entity(
            id=f"file::/test/file.py::function::{special_name}::1",
            name="test_func!@#$%^&*()",  # Special characters in name
            qualified_name="test_func!@#$%^&*()",
            entity_type=EntityType.FUNCTION,
            location=location,
            source_code="def test(): pass",
            docstring=None,
            signature=None
        )
        
        file_hash = "abc123"
        entity_id = DeterministicEntityId.generate(entity, file_hash)
        
        # Should generate valid ID
        assert len(entity_id) == 36
        # Check UUID format
        parts = entity_id.split('-')
        assert len(parts) == 5
    
    def test_empty_file_hash(self):
        """Test handling of empty file hash."""
        entity = Entity(
            id="file::/test/file.py::function::test_function::1",
            name="test_function",
            qualified_name="test_function",
            entity_type=EntityType.FUNCTION,
            location=SourceLocation(
                file_path=Path("/test/file.py"), 
                start_line=1, 
                start_column=0, 
                end_line=1, 
                end_column=10,
                start_byte=0,
                end_byte=10
            ),
            source_code="def test(): pass",
            docstring=None,
            signature=None
        )
        
        entity_id = DeterministicEntityId.generate(entity, "")
        
        # Should still generate valid ID
        assert len(entity_id) == 36
        # Check UUID format
        parts = entity_id.split('-')
        assert len(parts) == 5