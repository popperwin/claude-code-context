"""
Comprehensive unit tests for incremental.py

Tests FileIndexState, FileChangeDetector, and IncrementalIndexer
"""

import pytest
import asyncio
import json
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import logging

from core.indexer.incremental import (
    FileIndexState, FileChangeDetector, IncrementalIndexer
)
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse


@pytest.fixture
async def cleanup_test_collections():
    """Fixture to clean up test collections before and after tests"""
    test_collections = [
        "test-collection", "test", "test-new-files", "test-existing-state", 
        "test-deleted-files", "test-update", "test-error-update", "test-stats",
        "test-reset", "corrupted-collection"
    ]
    
    # Setup: Clean before tests
    client = QdrantClient(url="http://localhost:6334")
    for collection_name in test_collections:
        try:
            await asyncio.to_thread(client.delete_collection, collection_name)
        except (UnexpectedResponse, Exception):
            pass  # Collection doesn't exist or other error
    
    yield
    
    # Teardown: Clean after tests  
    for collection_name in test_collections:
        try:
            await asyncio.to_thread(client.delete_collection, collection_name)
        except (UnexpectedResponse, Exception):
            pass  # Collection doesn't exist or other error


class TestFileIndexState:
    """Test FileIndexState dataclass"""
    
    @pytest.mark.usefixtures("cleanup_test_collections")
    def test_file_index_state_creation(self):
        """Test FileIndexState creation with all fields"""
        state = FileIndexState(
            file_path="/test/file.py",
            file_hash="abc123",
            file_size=1024,
            last_modified=1640995200.0,  # 2022-01-01 00:00:00 UTC
            last_indexed=datetime(2022, 1, 1, 12, 0, 0),
            entity_count=5,
            relation_count=3,
            collection_name="test-collection",
            indexing_success=True,
            error_message=None
        )
        
        assert state.file_path == "/test/file.py"
        assert state.file_hash == "abc123"
        assert state.file_size == 1024
        assert state.last_modified == 1640995200.0
        assert state.entity_count == 5
        assert state.relation_count == 3
        assert state.collection_name == "test-collection"
        assert state.indexing_success is True
        assert state.error_message is None
    
    @pytest.mark.usefixtures("cleanup_test_collections")
    def test_file_index_state_with_error(self):
        """Test FileIndexState with error information"""
        state = FileIndexState(
            file_path="/test/error.py",
            file_hash="def456",
            file_size=512,
            last_modified=1640995200.0,
            last_indexed=datetime(2022, 1, 1, 12, 0, 0),
            entity_count=0,
            relation_count=0,
            collection_name="test-collection",
            indexing_success=False,
            error_message="Parse error: invalid syntax"
        )
        
        assert state.indexing_success is False
        assert state.error_message == "Parse error: invalid syntax"
        assert state.entity_count == 0
    
    @pytest.mark.usefixtures("cleanup_test_collections")
    def test_to_dict_conversion(self):
        """Test FileIndexState to dictionary conversion"""
        state = FileIndexState(
            file_path="/test/file.py",
            file_hash="abc123",
            file_size=1024,
            last_modified=1640995200.0,
            last_indexed=datetime(2022, 1, 1, 12, 0, 0),
            entity_count=5,
            relation_count=3,
            collection_name="test-collection",
            indexing_success=True
        )
        
        result_dict = state.to_dict()
        
        assert result_dict["file_path"] == "/test/file.py"
        assert result_dict["file_hash"] == "abc123"
        assert result_dict["file_size"] == 1024
        assert result_dict["last_modified"] == 1640995200.0
        assert result_dict["entity_count"] == 5
        assert result_dict["relation_count"] == 3
        assert result_dict["collection_name"] == "test-collection"
        assert result_dict["indexing_success"] is True
        assert "last_indexed" in result_dict
        # Should be ISO format string
        assert isinstance(result_dict["last_indexed"], str)
        assert "2022-01-01T12:00:00" in result_dict["last_indexed"]
    
    @pytest.mark.usefixtures("cleanup_test_collections")
    def test_from_dict_conversion(self):
        """Test FileIndexState from dictionary conversion"""
        data = {
            "file_path": "/test/file.py",
            "file_hash": "abc123",
            "file_size": 1024,
            "last_modified": 1640995200.0,
            "last_indexed": "2022-01-01T12:00:00",
            "entity_count": 5,
            "relation_count": 3,
            "collection_name": "test-collection",
            "indexing_success": True,
            "error_message": None
        }
        
        state = FileIndexState.from_dict(data)
        
        assert state.file_path == "/test/file.py"
        assert state.file_hash == "abc123"
        assert state.file_size == 1024
        assert state.last_modified == 1640995200.0
        assert isinstance(state.last_indexed, datetime)
        assert state.last_indexed == datetime(2022, 1, 1, 12, 0, 0)
        assert state.entity_count == 5
        assert state.relation_count == 3
        assert state.collection_name == "test-collection"
        assert state.indexing_success is True
        assert state.error_message is None
    
    @pytest.mark.usefixtures("cleanup_test_collections")
    def test_round_trip_serialization(self):
        """Test round-trip serialization (to_dict -> from_dict)"""
        original_state = FileIndexState(
            file_path="/test/roundtrip.py",
            file_hash="xyz789",
            file_size=2048,
            last_modified=1640995200.0,
            last_indexed=datetime(2022, 1, 2, 15, 30, 45),
            entity_count=10,
            relation_count=7,
            collection_name="roundtrip-test",
            indexing_success=True,
            error_message="Test error"
        )
        
        # Convert to dict and back
        data_dict = original_state.to_dict()
        restored_state = FileIndexState.from_dict(data_dict)
        
        # Should be identical
        assert restored_state.file_path == original_state.file_path
        assert restored_state.file_hash == original_state.file_hash
        assert restored_state.file_size == original_state.file_size
        assert restored_state.last_modified == original_state.last_modified
        assert restored_state.last_indexed == original_state.last_indexed
        assert restored_state.entity_count == original_state.entity_count
        assert restored_state.relation_count == original_state.relation_count
        assert restored_state.collection_name == original_state.collection_name
        assert restored_state.indexing_success == original_state.indexing_success
        assert restored_state.error_message == original_state.error_message


class TestFileChangeDetector:
    """Test FileChangeDetector class"""
    
    @pytest.fixture
    def detector(self):
        """Create FileChangeDetector instance"""
        return FileChangeDetector()
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_compute_file_hash(self, detector, tmp_path):
        """Test file hash computation"""
        # Create test file
        test_file = tmp_path / "test.py"
        test_content = "# Test file\nprint('hello world')\n"
        test_file.write_text(test_content)
        
        # Compute hash
        file_hash = await detector.compute_file_hash(test_file)
        
        assert isinstance(file_hash, str)
        assert len(file_hash) == 64  # SHA256 hex digest length
        assert file_hash != ""
        
        # Same content should produce same hash
        hash2 = await detector.compute_file_hash(test_file)
        assert file_hash == hash2
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_compute_file_hash_caching(self, detector, tmp_path):
        """Test file hash caching behavior"""
        # Create test file
        test_file = tmp_path / "cache_test.py"
        test_file.write_text("# Original content")
        
        # First hash computation
        hash1 = await detector.compute_file_hash(test_file)
        assert hash1 != ""
        
        # Second computation should use cache (same mtime/size)
        hash2 = await detector.compute_file_hash(test_file)
        assert hash1 == hash2
        
        # Modify file content
        time.sleep(0.1)  # Ensure different mtime
        test_file.write_text("# Modified content")
        
        # Should recompute hash
        hash3 = await detector.compute_file_hash(test_file)
        assert hash3 != hash1
        assert hash3 != ""
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_compute_file_hash_nonexistent(self, detector, tmp_path):
        """Test hash computation for nonexistent file"""
        nonexistent_file = tmp_path / "does_not_exist.py"
        
        file_hash = await detector.compute_file_hash(nonexistent_file)
        assert file_hash == ""
    
    @pytest.mark.usefixtures("cleanup_test_collections")
    def test_is_file_modified_by_mtime(self, detector, tmp_path):
        """Test file modification detection by mtime"""
        test_file = tmp_path / "mtime_test.py"
        test_file.write_text("# Test content")
        
        # Create state with earlier mtime
        stat = test_file.stat()
        old_state = FileIndexState(
            file_path=str(test_file),
            file_hash="old_hash",
            file_size=stat.st_size,
            last_modified=stat.st_mtime - 1000,  # Much earlier
            last_indexed=datetime.now() - timedelta(hours=1),
            entity_count=1,
            relation_count=0,
            collection_name="test",
            indexing_success=True
        )
        
        # Should detect as modified
        assert detector.is_file_modified(test_file, old_state) is True
    
    @pytest.mark.usefixtures("cleanup_test_collections")
    def test_is_file_modified_by_size(self, detector, tmp_path):
        """Test file modification detection by size"""
        test_file = tmp_path / "size_test.py"
        test_file.write_text("# Test content")
        
        stat = test_file.stat()
        old_state = FileIndexState(
            file_path=str(test_file),
            file_hash="old_hash",
            file_size=stat.st_size + 100,  # Different size
            last_modified=stat.st_mtime,
            last_indexed=datetime.now(),
            entity_count=1,
            relation_count=0,
            collection_name="test",
            indexing_success=True
        )
        
        # Should detect as modified
        assert detector.is_file_modified(test_file, old_state) is True
    
    @pytest.mark.usefixtures("cleanup_test_collections")
    def test_is_file_modified_unchanged(self, detector, tmp_path):
        """Test file modification detection for unchanged file"""
        test_file = tmp_path / "unchanged_test.py"
        test_file.write_text("# Test content")
        
        stat = test_file.stat()
        current_state = FileIndexState(
            file_path=str(test_file),
            file_hash="current_hash",
            file_size=stat.st_size,
            last_modified=stat.st_mtime,
            last_indexed=datetime.now(),
            entity_count=1,
            relation_count=0,
            collection_name="test",
            indexing_success=True
        )
        
        # Should not detect as modified
        assert detector.is_file_modified(test_file, current_state) is False
    
    @pytest.mark.usefixtures("cleanup_test_collections")
    def test_is_file_modified_nonexistent(self, detector, tmp_path):
        """Test file modification detection for nonexistent file"""
        nonexistent_file = tmp_path / "does_not_exist.py"
        
        old_state = FileIndexState(
            file_path=str(nonexistent_file),
            file_hash="old_hash",
            file_size=100,
            last_modified=time.time() - 1000,
            last_indexed=datetime.now() - timedelta(hours=1),
            entity_count=1,
            relation_count=0,
            collection_name="test",
            indexing_success=True
        )
        
        # Should detect as modified (file doesn't exist)
        assert detector.is_file_modified(nonexistent_file, old_state) is True
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_is_file_content_changed(self, detector, tmp_path):
        """Test definitive content change detection"""
        test_file = tmp_path / "content_test.py"
        original_content = "# Original content"
        test_file.write_text(original_content)
        
        # Compute hash for original content
        original_hash = await detector.compute_file_hash(test_file)
        
        old_state = FileIndexState(
            file_path=str(test_file),
            file_hash=original_hash,
            file_size=len(original_content),
            last_modified=test_file.stat().st_mtime,
            last_indexed=datetime.now(),
            entity_count=1,
            relation_count=0,
            collection_name="test",
            indexing_success=True
        )
        
        # Content hasn't changed
        assert await detector.is_file_content_changed(test_file, old_state) is False
        
        # Modify content
        modified_content = "# Modified content"
        test_file.write_text(modified_content)
        
        # Content has changed
        assert await detector.is_file_content_changed(test_file, old_state) is True
    
    @pytest.mark.usefixtures("cleanup_test_collections")
    def test_clear_cache(self, detector, tmp_path):
        """Test cache clearing"""
        # Add some entries to cache
        detector._hash_cache["file1"] = "hash1"
        detector._stat_cache["file1"] = Mock()
        detector._hash_cache["file2"] = "hash2"
        detector._stat_cache["file2"] = Mock()
        
        assert len(detector._hash_cache) == 2
        assert len(detector._stat_cache) == 2
        
        # Clear cache
        detector.clear_cache()
        
        assert len(detector._hash_cache) == 0
        assert len(detector._stat_cache) == 0


class TestIncrementalIndexer:
    """Test IncrementalIndexer class"""
    
    @pytest.fixture
    def temp_state_dir(self, tmp_path):
        """Create temporary state directory"""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        return state_dir
    
    @pytest.fixture
    def indexer(self, temp_state_dir):
        """Create IncrementalIndexer with temporary state directory"""
        return IncrementalIndexer(state_dir=temp_state_dir)
    
    @pytest.mark.usefixtures("cleanup_test_collections")  
    def test_indexer_initialization(self, temp_state_dir):
        """Test indexer initialization"""
        indexer = IncrementalIndexer(state_dir=temp_state_dir)
        
        assert indexer.state_dir == temp_state_dir
        assert isinstance(indexer.change_detector, FileChangeDetector)
        assert indexer._loaded_states == {}
    
    @pytest.mark.usefixtures("cleanup_test_collections")
    def test_indexer_default_state_dir(self):
        """Test indexer with default state directory"""
        indexer = IncrementalIndexer()
        
        expected_dir = Path.home() / ".claude-indexer" / "state"
        assert indexer.state_dir == expected_dir
    
    @pytest.mark.usefixtures("cleanup_test_collections")
    def test_get_state_file_path(self, indexer):
        """Test state file path generation"""
        # Normal collection name
        path1 = indexer._get_state_file_path("test-collection")
        assert path1.name == "test-collection.json"
        assert path1.parent == indexer.state_dir
        
        # Collection name with special characters
        path2 = indexer._get_state_file_path("test@collection#123")
        assert path2.name == "testcollection123.json"  # Sanitized
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_load_empty_collection_state(self, indexer):
        """Test loading state for collection with no existing state file"""
        state = await indexer._load_collection_state("new-collection")
        
        assert state == {}
        assert "new-collection" in indexer._loaded_states
        assert indexer._loaded_states["new-collection"] == {}
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_save_and_load_collection_state(self, indexer):
        """Test saving and loading collection state"""
        collection_name = "test-collection"
        
        # Create test state
        test_state = {
            "/test/file1.py": FileIndexState(
                file_path="/test/file1.py",
                file_hash="hash1",
                file_size=100,
                last_modified=time.time(),
                last_indexed=datetime.now(),
                entity_count=5,
                relation_count=2,
                collection_name=collection_name,
                indexing_success=True
            ),
            "/test/file2.py": FileIndexState(
                file_path="/test/file2.py",
                file_hash="hash2",
                file_size=200,
                last_modified=time.time(),
                last_indexed=datetime.now(),
                entity_count=3,
                relation_count=1,
                collection_name=collection_name,
                indexing_success=False,
                error_message="Parse error"
            )
        }
        
        # Save state
        await indexer._save_collection_state(collection_name, test_state)
        
        # Verify file exists
        state_file = indexer._get_state_file_path(collection_name)
        assert state_file.exists()
        
        # Clear loaded states and reload
        indexer._loaded_states.clear()
        loaded_state = await indexer._load_collection_state(collection_name)
        
        # Should match original state
        assert len(loaded_state) == 2
        assert "/test/file1.py" in loaded_state
        assert "/test/file2.py" in loaded_state
        
        file1_state = loaded_state["/test/file1.py"]
        assert file1_state.file_hash == "hash1"
        assert file1_state.entity_count == 5
        assert file1_state.indexing_success is True
        
        file2_state = loaded_state["/test/file2.py"]
        assert file2_state.file_hash == "hash2"
        assert file2_state.entity_count == 3
        assert file2_state.indexing_success is False
        assert file2_state.error_message == "Parse error"
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_load_corrupted_state_file(self, indexer):
        """Test loading corrupted state file"""
        collection_name = "corrupted-collection"
        state_file = indexer._get_state_file_path(collection_name)
        
        # Create corrupted JSON file
        state_file.write_text("{ invalid json content }")
        
        # Should handle gracefully and return empty state
        state = await indexer._load_collection_state(collection_name)
        assert state == {}
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_get_changed_files_all_new(self, indexer, tmp_path):
        """Test get_changed_files with all new files"""
        # Create test files
        file1 = tmp_path / "file1.py"
        file2 = tmp_path / "file2.py"
        file1.write_text("# File 1")
        file2.write_text("# File 2")
        
        all_files = [file1, file2]
        collection_name = "test-new-files"
        
        # All files should be considered changed (new)
        changed_files = await indexer.get_changed_files(all_files, collection_name)
        
        assert len(changed_files) == 2
        assert file1 in changed_files
        assert file2 in changed_files
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_get_changed_files_with_existing_state(self, indexer, tmp_path):
        """Test get_changed_files with existing state"""
        # Create test files
        file1 = tmp_path / "file1.py"
        file2 = tmp_path / "file2.py"
        file3 = tmp_path / "file3.py"
        
        file1.write_text("# File 1")
        file2.write_text("# File 2")
        file3.write_text("# File 3")
        
        collection_name = "test-existing-state"
        
        # Create existing state for file1 and file2
        stat1 = file1.stat()
        stat2 = file2.stat()
        
        existing_state = {
            str(file1): FileIndexState(
                file_path=str(file1),
                file_hash="hash1",
                file_size=stat1.st_size,
                last_modified=stat1.st_mtime,
                last_indexed=datetime.now(),
                entity_count=1,
                relation_count=0,
                collection_name=collection_name,
                indexing_success=True
            ),
            str(file2): FileIndexState(
                file_path=str(file2),
                file_hash="hash2",
                file_size=stat2.st_size - 10,  # Different size = modified
                last_modified=stat2.st_mtime,
                last_indexed=datetime.now(),
                entity_count=1,
                relation_count=0,
                collection_name=collection_name,
                indexing_success=True
            )
        }
        
        # Save existing state
        await indexer._save_collection_state(collection_name, existing_state)
        
        all_files = [file1, file2, file3]
        
        # Get changed files
        changed_files = await indexer.get_changed_files(all_files, collection_name)
        
        # file1: unchanged, file2: modified (size diff), file3: new
        assert len(changed_files) == 2
        assert file1 not in changed_files  # Should be unchanged
        assert file2 in changed_files      # Modified
        assert file3 in changed_files      # New
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_get_changed_files_with_deleted_files(self, indexer, tmp_path):
        """Test get_changed_files handles deleted files"""
        # Create test files
        file1 = tmp_path / "file1.py"
        file2 = tmp_path / "file2.py"
        file1.write_text("# File 1")
        file2.write_text("# File 2")
        
        collection_name = "test-deleted-files"
        
        # Create state with additional file that will be "deleted"
        deleted_file_path = str(tmp_path / "deleted.py")
        existing_state = {
            str(file1): FileIndexState(
                file_path=str(file1),
                file_hash="hash1",
                file_size=file1.stat().st_size,
                last_modified=file1.stat().st_mtime,
                last_indexed=datetime.now(),
                entity_count=1,
                relation_count=0,
                collection_name=collection_name,
                indexing_success=True
            ),
            deleted_file_path: FileIndexState(
                file_path=deleted_file_path,
                file_hash="deleted_hash",
                file_size=100,
                last_modified=time.time(),
                last_indexed=datetime.now(),
                entity_count=2,
                relation_count=1,
                collection_name=collection_name,
                indexing_success=True
            )
        }
        
        await indexer._save_collection_state(collection_name, existing_state)
        
        # Only provide existing files (deleted.py is not in current files)
        current_files = [file1, file2]
        
        changed_files = await indexer.get_changed_files(current_files, collection_name)
        
        # Should include file2 (new) but not file1 (unchanged)
        assert file2 in changed_files
        assert file1 not in changed_files
        
        # Verify deleted file was removed from state
        updated_state = await indexer._load_collection_state(collection_name)
        assert deleted_file_path not in updated_state
        assert str(file1) in updated_state  # Should still be there
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_update_file_state(self, indexer, tmp_path):
        """Test updating file state after processing"""
        test_file = tmp_path / "update_test.py"
        test_file.write_text("# Test file for update")
        
        collection_name = "test-update"
        
        # Update file state
        await indexer.update_file_state(
            file_path=test_file,
            collection_name=collection_name,
            entity_count=10,
            relation_count=5,
            success=True,
            error_message=None
        )
        
        # Verify state was saved
        state = await indexer._load_collection_state(collection_name)
        
        assert str(test_file) in state
        file_state = state[str(test_file)]
        
        assert file_state.file_path == str(test_file)
        assert file_state.entity_count == 10
        assert file_state.relation_count == 5
        assert file_state.indexing_success is True
        assert file_state.error_message is None
        assert file_state.collection_name == collection_name
        assert file_state.file_size == test_file.stat().st_size
        assert file_state.last_modified == test_file.stat().st_mtime
        assert len(file_state.file_hash) == 64  # SHA256 length
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_update_file_state_with_error(self, indexer, tmp_path):
        """Test updating file state with error information"""
        test_file = tmp_path / "error_test.py"
        test_file.write_text("# File with error")
        
        collection_name = "test-error-update"
        error_message = "Parsing failed: syntax error"
        
        await indexer.update_file_state(
            file_path=test_file,
            collection_name=collection_name,
            entity_count=0,
            relation_count=0,
            success=False,
            error_message=error_message
        )
        
        # Verify error state was saved
        state = await indexer._load_collection_state(collection_name)
        file_state = state[str(test_file)]
        
        assert file_state.indexing_success is False
        assert file_state.error_message == error_message
        assert file_state.entity_count == 0
        assert file_state.relation_count == 0
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_get_collection_stats(self, indexer):
        """Test collection statistics generation"""
        collection_name = "test-stats"
        
        # Test empty collection
        stats = await indexer.get_collection_stats(collection_name)
        
        assert stats["total_files"] == 0
        assert stats["successful_files"] == 0
        assert stats["failed_files"] == 0
        assert stats["total_entities"] == 0
        assert stats["total_relations"] == 0
        assert stats["last_update"] is None
        
        # Create test state with mixed success/failure
        now = datetime.now()
        test_state = {
            "/test/success1.py": FileIndexState(
                file_path="/test/success1.py",
                file_hash="hash1",
                file_size=100,
                last_modified=time.time(),
                last_indexed=now,
                entity_count=5,
                relation_count=2,
                collection_name=collection_name,
                indexing_success=True
            ),
            "/test/success2.py": FileIndexState(
                file_path="/test/success2.py",
                file_hash="hash2",
                file_size=200,
                last_modified=time.time(),
                last_indexed=now - timedelta(minutes=30),
                entity_count=3,
                relation_count=1,
                collection_name=collection_name,
                indexing_success=True
            ),
            "/test/failed.py": FileIndexState(
                file_path="/test/failed.py",
                file_hash="hash3",
                file_size=150,
                last_modified=time.time(),
                last_indexed=now - timedelta(minutes=10),
                entity_count=0,
                relation_count=0,
                collection_name=collection_name,
                indexing_success=False,
                error_message="Parse error"
            )
        }
        
        await indexer._save_collection_state(collection_name, test_state)
        
        # Clear the in-memory cache to force file system load
        indexer._loaded_states.clear()
        
        # Verify state was saved and loaded correctly
        loaded_state = await indexer._load_collection_state(collection_name)
        assert len(loaded_state) == 3, f"Expected 3 files, got {len(loaded_state)}"
        
        # Get stats
        stats = await indexer.get_collection_stats(collection_name)
        
        assert stats["total_files"] == 3
        assert stats["successful_files"] == 2
        assert stats["failed_files"] == 1
        assert stats["total_entities"] == 8  # 5 + 3 + 0
        assert stats["total_relations"] == 3  # 2 + 1 + 0
        assert stats["success_rate"] == 2/3  # 2 successful out of 3
        # Check that last_update is close to expected time (within 1 second)
        assert abs((stats["last_update"] - now).total_seconds()) < 1.0
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_reset_state(self, indexer):
        """Test resetting collection state"""
        collection_name = "test-reset"
        
        # Create some state
        test_state = {
            "/test/file.py": FileIndexState(
                file_path="/test/file.py",
                file_hash="hash",
                file_size=100,
                last_modified=time.time(),
                last_indexed=datetime.now(),
                entity_count=1,
                relation_count=0,
                collection_name=collection_name,
                indexing_success=True
            )
        }
        
        await indexer._save_collection_state(collection_name, test_state)
        
        # Verify state exists
        state_file = indexer._get_state_file_path(collection_name)
        assert state_file.exists()
        
        loaded_state = await indexer._load_collection_state(collection_name)
        assert len(loaded_state) == 1
        
        # Reset state
        await indexer.reset_state(collection_name)
        
        # Verify state file is gone
        assert not state_file.exists()
        
        # Verify loaded state is cleared
        assert collection_name not in indexer._loaded_states
        
        # Loading again should return empty state
        new_state = await indexer._load_collection_state(collection_name)
        assert new_state == {}
    
    @pytest.mark.usefixtures("cleanup_test_collections")
    def test_clear_caches(self, indexer):
        """Test clearing all caches"""
        # Add some data to caches
        indexer._loaded_states["collection1"] = {"file1": Mock()}
        indexer.change_detector._hash_cache["file1"] = "hash1"
        indexer.change_detector._stat_cache["file1"] = Mock()
        
        assert len(indexer._loaded_states) == 1
        assert len(indexer.change_detector._hash_cache) == 1
        assert len(indexer.change_detector._stat_cache) == 1
        
        # Clear caches
        indexer.clear_caches()
        
        assert len(indexer._loaded_states) == 0
        assert len(indexer.change_detector._hash_cache) == 0
        assert len(indexer.change_detector._stat_cache) == 0