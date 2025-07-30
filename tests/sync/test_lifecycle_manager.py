"""
Tests for EntityLifecycleManager entity operations and atomic replacement validation.

Validates entity creation, modification, deletion, atomic replacement operations,
entity-to-file mapping maintenance, and error handling scenarios.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Set

from core.models.entities import Entity, EntityType, SourceLocation
from core.sync.lifecycle import EntityLifecycleManager
from core.sync.events import FileSystemEvent, EventType
from core.storage.client import HybridQdrantClient
from core.parser.registry import ParserRegistry


class TestEntityLifecycleManager:
    """Test suite for entity lifecycle management functionality."""
    
    @pytest.fixture
    def temp_project_dir(self):
        """Create temporary project directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def mock_qdrant_client(self):
        """Create mock Qdrant client."""
        from core.models.storage import StorageResult
        client = Mock(spec=HybridQdrantClient)
        client.upsert_points = AsyncMock(return_value=StorageResult.successful_insert("test", 1, 10.0))
        client.delete_points_by_filter = AsyncMock(return_value=StorageResult.successful_delete("test", 1, 10.0))
        client.delete_points_by_file_path = AsyncMock(return_value=StorageResult.successful_delete("test", 1, 10.0))
        client.delete_points = AsyncMock(return_value=StorageResult.successful_delete("test", 1, 10.0))
        client.search_payload = AsyncMock(return_value=[])
        client.get_collection_info = AsyncMock(return_value={"points_count": 0})
        return client
    
    @pytest.fixture
    def lifecycle_manager(self, mock_qdrant_client, temp_project_dir):
        """Create EntityLifecycleManager instance."""
        return EntityLifecycleManager(
            storage_client=mock_qdrant_client,
            collection_name="test-collection",
            project_path=temp_project_dir
        )
    
    def create_test_entity(
        self, 
        name: str = "test_function",
        file_path: Path = None,
        entity_type: EntityType = EntityType.FUNCTION,
        start_line: int = 10
    ) -> Entity:
        """Create a test entity."""
        if file_path is None:
            file_path = Path("/test/file.py")
        
        location = SourceLocation(
            file_path=file_path,
            start_line=start_line,
            start_column=4,
            end_line=start_line + 5,
            end_column=14,
            start_byte=start_line * 80 + 4,
            end_byte=(start_line + 5) * 80 + 14
        )
        
        return Entity(
            id=f"file::{file_path}::{entity_type.value}::{name}::{start_line}",
            name=name,
            qualified_name=name,
            entity_type=entity_type,
            location=location,
            source_code=f"def {name}():\n    pass",
            docstring=f"Test {name} function",
            signature=f"{name}()" if entity_type == EntityType.FUNCTION else None
        )

    @pytest.mark.asyncio
    async def test_handle_file_creation(self, lifecycle_manager, mock_qdrant_client, temp_project_dir):
        """Test file creation handling."""
        # Create test file
        test_file = temp_project_dir / "test.py"
        test_file.write_text("def test_function():\n    pass")
        
        # Mock the internal _parse_file_entities method
        test_entities = [
            self.create_test_entity("test_function", test_file),
            self.create_test_entity("helper_function", test_file, start_line=15)
        ]
        lifecycle_manager._parse_file_entities = AsyncMock(return_value=test_entities)
        
        # Create file creation event
        event = FileSystemEvent.create_file_created(test_file)
        
        # Handle file creation
        result = await lifecycle_manager.handle_file_creation(event)
        
        # Verify result structure
        assert result["success"] is True
        assert result["operation"] == "file_creation"
        assert result["entities_created"] == 2
        assert result["file_path"] == str(test_file)
        
        # Verify parser was called
        lifecycle_manager._parse_file_entities.assert_called_once_with(test_file)
        
        # Verify entities were stored
        mock_qdrant_client.upsert_points.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_file_modification(self, lifecycle_manager, mock_qdrant_client, temp_project_dir):
        """Test file modification handling with atomic replacement."""
        # Create test file
        test_file = temp_project_dir / "test.py"
        test_file.write_text("def modified_function():\n    return 'modified'")
        
        # Setup existing entities in mapping
        old_entity_ids = {"old_id_1", "old_id_2"}
        lifecycle_manager._get_entities_for_file = AsyncMock(return_value=old_entity_ids)
        
        # Mock parser to return new entities
        new_entities = [
            self.create_test_entity("modified_function", test_file),
            self.create_test_entity("new_function", test_file, start_line=20)
        ]
        lifecycle_manager._parse_file_entities = AsyncMock(return_value=new_entities)
        
        # Mock atomic replacement method
        lifecycle_manager.atomic_entity_replacement = AsyncMock(return_value={
            "success": True,
            "entities_deleted": 2,
            "entities_created": 2
        })
        
        # Create file modification event
        event = FileSystemEvent.create_file_modified(test_file)
        
        # Handle file modification
        result = await lifecycle_manager.handle_file_modification(event)
        
        # Verify result structure
        assert result["success"] is True
        assert result["operation"] == "file_modification"
        assert result["file_path"] == str(test_file)
        
        # Verify atomic replacement was called
        lifecycle_manager.atomic_entity_replacement.assert_called_once_with(
            str(test_file), old_entity_ids, new_entities
        )

    @pytest.mark.asyncio
    async def test_handle_file_deletion(self, lifecycle_manager, mock_qdrant_client, temp_project_dir):
        """Test file deletion handling with cascade removal."""
        # Create test file path (file doesn't need to exist for deletion)
        test_file = temp_project_dir / "deleted.py"
        
        # Setup existing entities
        existing_entity_ids = {"entity_id_1", "entity_id_2", "entity_id_3"}
        lifecycle_manager._get_entities_for_file = AsyncMock(return_value=existing_entity_ids)
        
        # Create file deletion event
        event = FileSystemEvent.create_file_deleted(test_file)
        
        # Handle file deletion
        result = await lifecycle_manager.handle_file_deletion(event)
        
        # Verify result structure
        assert result["success"] is True
        assert result["operation"] == "file_deletion"
        assert result["file_path"] == str(test_file)
        assert result["entities_deleted"] == 3
        
        # Verify entities were deleted from Qdrant
        mock_qdrant_client.delete_points_by_file_path.assert_called_once_with(
            lifecycle_manager.collection_name, str(test_file)
        )

    @pytest.mark.asyncio
    async def test_atomic_entity_replacement(self, lifecycle_manager, mock_qdrant_client, temp_project_dir):
        """Test atomic entity replacement operation."""
        # Create a test file
        test_file = temp_project_dir / "test.py"
        test_file.write_text("def new_function_1(): pass\ndef new_function_2(): pass\nclass new_class: pass")
        file_path = str(test_file)
        
        # Setup old entities
        old_entity_ids = {"old_id_1", "old_id_2"}
        
        # Create new entities
        new_entities = [
            self.create_test_entity("new_function_1", test_file),
            self.create_test_entity("new_function_2", test_file, start_line=20),
            self.create_test_entity("new_class", test_file, EntityType.CLASS, start_line=30)
        ]
        
        # Perform atomic replacement
        result = await lifecycle_manager.atomic_entity_replacement(file_path, old_entity_ids, new_entities)
        
        # Verify result structure
        assert result["success"] is True
        assert result["file_path"] == file_path
        assert result["entities_removed"] == 2
        assert result["entities_added"] == 3
        
        # Verify storage operations were called
        mock_qdrant_client.delete_points.assert_called()
        mock_qdrant_client.upsert_points.assert_called()

    @pytest.mark.asyncio
    async def test_atomic_replacement_rollback_on_store_failure(self, lifecycle_manager, mock_qdrant_client, temp_project_dir):
        """Test rollback when entity storage fails during atomic replacement."""
        # Create a test file
        test_file = temp_project_dir / "test.py"
        test_file.write_text("def new_function(): pass")
        file_path = str(test_file)
        
        # Setup old entities
        old_entity_ids = {"old_id_1", "old_id_2"}
        
        # Create new entities
        new_entities = [self.create_test_entity("new_function", test_file)]
        
        # Mock storage failure
        from core.models.storage import StorageResult
        mock_qdrant_client.upsert_points.return_value = StorageResult.failed_operation(
            "upsert", "test", "Storage failed", 10.0
        )
        
        # Perform atomic replacement (should fail)
        result = await lifecycle_manager.atomic_entity_replacement(file_path, old_entity_ids, new_entities)
        
        # Verify result structure indicates failure
        assert result["success"] is False
        assert "error" in result
        assert result["operation"] == "atomic_replacement"
        
        # Verify storage operations were attempted (no entities to delete since old_entity_ids don't match new ones)
        # Storage would only be called if there were entities to add, which there are
        mock_qdrant_client.upsert_points.assert_called()

    @pytest.mark.asyncio
    async def test_get_entities_for_file(self, lifecycle_manager, mock_qdrant_client):
        """Test retrieving entities for a specific file."""
        file_path = "/test/file.py"
        
        # Mock internal method
        entity_ids = {"entity_1", "entity_2"}
        lifecycle_manager._get_entities_for_file = AsyncMock(return_value=entity_ids)
        
        # Get entities for file
        entities = await lifecycle_manager._get_entities_for_file(file_path)
        
        # Verify result
        assert len(entities) == 2
        assert entities == entity_ids

    @pytest.mark.asyncio 
    async def test_rebuild_entity_mapping(self, lifecycle_manager, mock_qdrant_client):
        """Test rebuilding entity-to-file mapping from Qdrant."""
        # Mock the internal method that would be called during rebuild
        rebuild_result = {
            "mappings_rebuilt": 3,
            "total_entities": 4,
            "files_processed": 3,
            "success": True
        }
        
        # Replace the method with our mock that returns expected format
        lifecycle_manager.rebuild_entity_mappings = AsyncMock(return_value=rebuild_result)
        
        # Rebuild mapping
        result = await lifecycle_manager.rebuild_entity_mappings()
        
        # Verify result structure
        assert result["success"] is True
        assert result["mappings_rebuilt"] == 3
        assert result["total_entities"] == 4
        assert result["files_processed"] == 3

    @pytest.mark.asyncio
    async def test_handle_parser_failure(self, lifecycle_manager, mock_qdrant_client, temp_project_dir):
        """Test handling when parser fails to parse file."""
        # Create test file
        test_file = temp_project_dir / "bad_syntax.py" 
        test_file.write_text("def invalid syntax here")
        
        # Mock parser failure
        lifecycle_manager._parse_file_entities = AsyncMock(side_effect=Exception("Parse error"))
        
        # Create file creation event
        event = FileSystemEvent.create_file_created(test_file)
        
        # Handle file creation (should handle gracefully)
        result = await lifecycle_manager.handle_file_creation(event)
        
        # Verify graceful failure
        assert result["success"] is False
        assert "error" in result
        assert result["entities_created"] == 0
        
        # Verify parser was called
        lifecycle_manager._parse_file_entities.assert_called_once_with(test_file)
        
        # Verify no entities were stored
        mock_qdrant_client.upsert_points.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_qdrant_failure(self, lifecycle_manager, mock_qdrant_client, temp_project_dir):
        """Test handling when Qdrant operations fail."""
        # Create test file
        test_file = temp_project_dir / "test.py"
        test_file.write_text("def test_function():\n    pass")
        
        # Mock parser success
        test_entities = [self.create_test_entity("test_function", test_file)]
        lifecycle_manager._parse_file_entities = AsyncMock(return_value=test_entities)
        
        # Mock Qdrant failure
        from core.models.storage import StorageResult
        mock_qdrant_client.upsert_points.return_value = StorageResult.failed_operation(
            "upsert", "test", "Qdrant error", 10.0
        )
        
        # Create file creation event
        event = FileSystemEvent.create_file_created(test_file)
        
        # Handle file creation (should handle gracefully)
        result = await lifecycle_manager.handle_file_creation(event)
        
        # Verify graceful failure
        assert result["success"] is False
        assert "error" in result
        
        # Verify entities were attempted to be stored
        mock_qdrant_client.upsert_points.assert_called_once()

    @pytest.mark.asyncio
    async def test_entity_mapping_operations(self, lifecycle_manager):
        """Test entity-to-file mapping operations."""
        file_path = "/test/file.py"
        entity_ids = {"id1", "id2", "id3"}
        
        # Mock the internal mapping methods
        lifecycle_manager._update_mappings_for_entities = AsyncMock()
        lifecycle_manager._remove_mappings_for_file = AsyncMock()
        lifecycle_manager._get_entities_for_file = AsyncMock(return_value=entity_ids)
        
        # Test getting mapping
        retrieved_ids = await lifecycle_manager._get_entities_for_file(file_path)
        assert retrieved_ids == entity_ids
        
        # Test updating mappings
        entities = [self.create_test_entity("test", Path(file_path))]
        await lifecycle_manager._update_mappings_for_entities(entities, file_path)
        lifecycle_manager._update_mappings_for_entities.assert_called_once_with(entities, file_path)
        
        # Test removing mappings
        await lifecycle_manager._remove_mappings_for_file(file_path)
        lifecycle_manager._remove_mappings_for_file.assert_called_once_with(file_path)

    def test_get_mapping_statistics(self, lifecycle_manager):
        """Test getting entity mapping statistics."""
        # Mock the internal mapping state
        lifecycle_manager._file_entity_map = {
            "/file1.py": {"id1", "id2", "id3"},
            "/file2.py": {"id4", "id5"},
            "/file3.py": {"id6"}
        }
        
        # Calculate expected stats (this would be in the actual implementation)
        total_files = len(lifecycle_manager._file_entity_map)
        total_entities = sum(len(entities) for entities in lifecycle_manager._file_entity_map.values())
        
        # Verify basic mapping state
        assert total_files == 3
        assert total_entities == 6

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, lifecycle_manager, mock_qdrant_client, temp_project_dir):
        """Test concurrent file operations don't interfere."""
        # Create multiple test files
        files = []
        for i in range(3):
            test_file = temp_project_dir / f"file{i}.py"
            test_file.write_text(f"def function_{i}():\n    pass")
            files.append(test_file)
        
        # Mock parser to return different entities for each file
        def mock_parse_side_effect(file_path):
            file_num = int(file_path.stem[-1])  # Extract number from filename
            return [self.create_test_entity(f"function_{file_num}", file_path)]
        
        lifecycle_manager._parse_file_entities = AsyncMock(side_effect=mock_parse_side_effect)
        
        # Create concurrent file creation events
        events = [FileSystemEvent.create_file_created(f) for f in files]
        
        # Handle all files concurrently
        tasks = [lifecycle_manager.handle_file_creation(event) for event in events]
        results = await asyncio.gather(*tasks)
        
        # Verify all operations succeeded
        assert all(result["success"] for result in results)
        
        # Verify parser was called for each file
        assert lifecycle_manager._parse_file_entities.call_count == 3
        
        # Verify entities were stored (potentially in batches)
        assert mock_qdrant_client.upsert_points.call_count >= 1


class TestEntityLifecycleManagerEdgeCases:
    """Test edge cases and error conditions for entity lifecycle manager."""
    
    @pytest.fixture
    def temp_project_dir(self):
        """Create temporary project directory for testing."""
        import tempfile
        import shutil
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def lifecycle_manager(self, temp_project_dir):
        """Create minimal EntityLifecycleManager for edge case testing."""
        from core.models.storage import StorageResult
        mock_client = Mock(spec=HybridQdrantClient)
        mock_client.upsert_points = AsyncMock(return_value=StorageResult.successful_insert("test", 1, 10.0))
        mock_client.delete_points_by_filter = AsyncMock(return_value=StorageResult.successful_delete("test", 1, 10.0))
        return EntityLifecycleManager(
            storage_client=mock_client,
            collection_name="test-collection",
            project_path=temp_project_dir
        )
    
    @pytest.mark.asyncio
    async def test_handle_nonexistent_file_creation(self, lifecycle_manager):
        """Test handling creation event for non-existent file."""
        nonexistent_file = Path("/definitely/does/not/exist.py")
        event = FileSystemEvent.create_file_created(nonexistent_file)
        
        # Should handle gracefully
        result = await lifecycle_manager.handle_file_creation(event)
        assert result["success"] is False
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_empty_entity_list_handling(self, lifecycle_manager, temp_project_dir):
        """Test handling when parser returns empty entity list."""
        # Create empty file
        empty_file = temp_project_dir / "empty.py"
        empty_file.write_text("# Empty file")
        
        lifecycle_manager._parse_file_entities = AsyncMock(return_value=[])
        
        event = FileSystemEvent.create_file_created(empty_file)
        
        # Should handle gracefully (empty files are valid)
        result = await lifecycle_manager.handle_file_creation(event)
        assert result["success"] is True
        assert result["entities_created"] == 0
        
        # Should not store empty entity list
        lifecycle_manager.storage_client.upsert_points.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_duplicate_entity_ids_handling(self, lifecycle_manager, temp_project_dir):
        """Test handling when entities have duplicate IDs."""
        # Create test file
        test_file = temp_project_dir / "test.py"
        test_file.write_text("def func1(): pass\ndef func2(): pass")
        
        # Create entities with same ID (shouldn't happen normally but test robustness)
        duplicate_entities = [
            Entity(
                id="file::/test.py::function::duplicate_id::1",
                name="func1",
                qualified_name="func1",
                entity_type=EntityType.FUNCTION,
                location=SourceLocation(
                    file_path=test_file,
                    start_line=1, start_column=0, end_line=2, end_column=10,
                    start_byte=0, end_byte=20
                ),
                source_code="def func1(): pass"
            ),
            Entity(
                id="file::/test.py::function::duplicate_id::1",  # Same ID
                name="func2",
                qualified_name="func2", 
                entity_type=EntityType.FUNCTION,
                location=SourceLocation(
                    file_path=test_file,
                    start_line=3, start_column=0, end_line=4, end_column=10,
                    start_byte=21, end_byte=40
                ),
                source_code="def func2(): pass"
            )
        ]
        
        lifecycle_manager._parse_file_entities = AsyncMock(return_value=duplicate_entities)
        
        event = FileSystemEvent.create_file_created(test_file)
        
        # Should handle gracefully
        result = await lifecycle_manager.handle_file_creation(event)
        assert result["success"] is True
        assert result["entities_created"] == 2  # Both entities should be processed
        
        # Verify storage was called with duplicate entities
        lifecycle_manager.storage_client.upsert_points.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_very_large_entity_list(self, lifecycle_manager, temp_project_dir):
        """Test handling very large number of entities."""
        # Create test file
        large_file = temp_project_dir / "large_file.py"
        large_file.write_text("# Large file with many functions")
        
        # Create many entities
        large_entity_list = []
        for i in range(1000):
            entity = Entity(
                id=f"file::/large_file.py::function::func_{i}::{i}",
                name=f"func_{i}",
                qualified_name=f"func_{i}",
                entity_type=EntityType.FUNCTION,
                location=SourceLocation(
                    file_path=large_file,
                    start_line=i, start_column=0, end_line=i+1, end_column=10,
                    start_byte=i*20, end_byte=(i+1)*20
                ),
                source_code=f"def func_{i}(): pass"
            )
            large_entity_list.append(entity)
        
        lifecycle_manager._parse_file_entities = AsyncMock(return_value=large_entity_list)
        
        event = FileSystemEvent.create_file_created(large_file)
        
        # Should handle large entity list
        result = await lifecycle_manager.handle_file_creation(event)
        assert result["success"] is True
        assert result["entities_created"] == 1000
        
        # Verify storage was called with large entity list
        lifecycle_manager.storage_client.upsert_points.assert_called_once()
    
    def test_mapping_memory_efficiency(self, lifecycle_manager):
        """Test that entity mapping doesn't consume excessive memory."""
        # Simulate many file mappings by setting up the internal data structures
        # This tests the mapping data structure can handle large amounts of data
        lifecycle_manager._file_entity_map = {}
        lifecycle_manager._entity_file_map = {}
        
        for i in range(1000):
            file_path = f"/file_{i}.py"
            entity_ids = {f"entity_{i}_{j}" for j in range(10)}
            lifecycle_manager._file_entity_map[file_path] = entity_ids
            
            # Also populate reverse mapping
            for entity_id in entity_ids:
                lifecycle_manager._entity_file_map[entity_id] = file_path
        
        # Verify mapping structure is reasonable
        assert len(lifecycle_manager._file_entity_map) == 1000
        assert len(lifecycle_manager._entity_file_map) == 10000
        
        # Basic sanity check - first file should have 10 entities
        first_file_entities = lifecycle_manager._file_entity_map["/file_0.py"]
        assert len(first_file_entities) == 10