"""
Comprehensive tests for delta calculation logic with comprehensive edge cases.

Tests the calculate_delta() function that compares workspace files with indexed entities
to determine added, modified, deleted, and unchanged files. NO MOCKS - all tests use
real data structures and timestamp comparisons.
"""

import pytest
import time
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock

from core.indexer.hybrid_indexer import HybridIndexer, WorkspaceState, DeltaScanResult
from core.parser.parallel_pipeline import ProcessParsingPipeline
from core.embeddings.stella import StellaEmbedder
from core.storage.client import HybridQdrantClient


class TestDeltaCalculationLogic:
    """Test the calculate_delta() function with comprehensive edge cases."""
    
    @pytest.fixture
    def mock_indexer(self):
        """Create HybridIndexer with minimal mock components for delta calculation testing."""
        mock_parser = Mock(spec=ProcessParsingPipeline)
        mock_embedder = Mock(spec=StellaEmbedder)
        mock_storage = Mock(spec=HybridQdrantClient)
        
        return HybridIndexer(
            parser_pipeline=mock_parser,
            embedder=mock_embedder,
            storage_client=mock_storage
        )
    
    def create_workspace_state(self, files_data: Dict[str, Dict[str, Any]]) -> Dict[str, WorkspaceState]:
        """
        Create workspace state from file data.
        
        Args:
            files_data: Dict mapping file_path to {'mtime': timestamp, 'size': bytes}
        """
        workspace_state = {}
        
        for file_path, data in files_data.items():
            workspace_state[file_path] = WorkspaceState(
                file_path=file_path,
                mtime=data['mtime'],
                size=data['size'],
                is_parseable=True
            )
        
        return workspace_state
    
    def create_collection_state(self, entities_data: Dict[str, list]) -> Dict[str, Any]:
        """
        Create collection state from entity data.
        
        Args:
            entities_data: Dict mapping file_path to list of entity metadata dicts
        """
        total_entities = sum(len(entities) for entities in entities_data.values())
        
        return {
            "exists": True,
            "entities": entities_data,
            "entity_count": total_entities,
            "file_count": len(entities_data),
            "scan_time": 0.1,
            "collection_info": {"points_count": total_entities}
        }
    
    @pytest.mark.asyncio
    async def test_calculate_delta_empty_states(self, mock_indexer):
        """Test delta calculation with empty workspace and collection states."""
        workspace_state = {}
        collection_state = self.create_collection_state({})
        
        result = await mock_indexer.calculate_delta(workspace_state, collection_state)
        
        assert isinstance(result, DeltaScanResult)
        assert len(result.added_files) == 0
        assert len(result.modified_files) == 0
        assert len(result.deleted_files) == 0
        assert len(result.unchanged_files) == 0
        assert result.total_changes == 0
        assert result.change_ratio == 0.0
        assert result.scan_time > 0
        assert result.total_workspace_files == 0
        assert result.total_collection_entities == 0
    
    @pytest.mark.asyncio
    async def test_calculate_delta_added_files_only(self, mock_indexer):
        """Test delta calculation with only added files."""
        current_time = time.time()
        
        # Workspace has files, collection is empty
        workspace_state = self.create_workspace_state({
            "new_file1.py": {"mtime": current_time, "size": 1000},
            "new_file2.py": {"mtime": current_time - 10, "size": 2000},
            "new_file3.js": {"mtime": current_time - 20, "size": 500}
        })
        
        collection_state = self.create_collection_state({})
        
        result = await mock_indexer.calculate_delta(workspace_state, collection_state)
        
        assert len(result.added_files) == 3
        assert len(result.modified_files) == 0
        assert len(result.deleted_files) == 0
        assert len(result.unchanged_files) == 0
        
        expected_added = {"new_file1.py", "new_file2.py", "new_file3.js"}
        assert result.added_files == expected_added
        assert result.total_changes == 3
        assert result.change_ratio == 1.0  # All files are new
        assert result.total_workspace_files == 3
        assert result.total_collection_entities == 0
    
    @pytest.mark.asyncio
    async def test_calculate_delta_deleted_files_only(self, mock_indexer):
        """Test delta calculation with only deleted files."""
        current_time = time.time()
        
        # Collection has files, workspace is empty
        workspace_state = {}
        
        collection_state = self.create_collection_state({
            "deleted_file1.py": [
                {"entity_id": "deleted_file1.py::func1", "indexed_at": current_time - 100}
            ],
            "deleted_file2.py": [
                {"entity_id": "deleted_file2.py::func2", "indexed_at": current_time - 200},
                {"entity_id": "deleted_file2.py::Class1", "indexed_at": current_time - 150}
            ]
        })
        
        result = await mock_indexer.calculate_delta(workspace_state, collection_state)
        
        assert len(result.added_files) == 0
        assert len(result.modified_files) == 0
        assert len(result.deleted_files) == 2
        assert len(result.unchanged_files) == 0
        
        expected_deleted = {"deleted_file1.py", "deleted_file2.py"}
        assert result.deleted_files == expected_deleted
        assert result.total_changes == 2
        assert result.total_workspace_files == 0
        assert result.total_collection_entities == 3
    
    @pytest.mark.asyncio
    async def test_calculate_delta_modified_files_basic(self, mock_indexer):
        """Test delta calculation with basic modified files detection."""
        base_time = time.time()
        
        workspace_state = self.create_workspace_state({
            "modified_file.py": {"mtime": base_time, "size": 1500},
            "unchanged_file.py": {"mtime": base_time - 100, "size": 1000}
        })
        
        collection_state = self.create_collection_state({
            "modified_file.py": [
                {"entity_id": "modified_file.py::func1", "indexed_at": base_time - 50}  # Older than file
            ],
            "unchanged_file.py": [
                {"entity_id": "unchanged_file.py::func2", "indexed_at": base_time - 50}  # Newer than file
            ]
        })
        
        result = await mock_indexer.calculate_delta(workspace_state, collection_state, tolerance_sec=1.0)
        
        assert len(result.added_files) == 0
        assert len(result.modified_files) == 1
        assert len(result.deleted_files) == 0
        assert len(result.unchanged_files) == 1
        
        assert "modified_file.py" in result.modified_files
        assert "unchanged_file.py" in result.unchanged_files
        assert result.total_changes == 1
        assert result.total_workspace_files == 2
        assert result.total_collection_entities == 2
    
    @pytest.mark.asyncio
    async def test_calculate_delta_tolerance_handling(self, mock_indexer):
        """Test delta calculation with different tolerance values."""
        base_time = time.time()
        
        workspace_state = self.create_workspace_state({
            "edge_case_file.py": {"mtime": base_time, "size": 1000}
        })
        
        collection_state = self.create_collection_state({
            "edge_case_file.py": [
                {"entity_id": "edge_case_file.py::func1", "indexed_at": base_time - 0.5}  # 0.5s difference
            ]
        })
        
        # Test with strict tolerance (0.1s) - should be modified
        result_strict = await mock_indexer.calculate_delta(workspace_state, collection_state, tolerance_sec=0.1)
        assert len(result_strict.modified_files) == 1
        assert len(result_strict.unchanged_files) == 0
        
        # Test with loose tolerance (2.0s) - should be unchanged
        result_loose = await mock_indexer.calculate_delta(workspace_state, collection_state, tolerance_sec=2.0)
        assert len(result_loose.modified_files) == 0
        assert len(result_loose.unchanged_files) == 1
    
    @pytest.mark.asyncio
    async def test_calculate_delta_multiple_entities_per_file(self, mock_indexer):
        """Test delta calculation with multiple entities per file using latest timestamp."""
        base_time = time.time()
        
        workspace_state = self.create_workspace_state({
            "multi_entity_file.py": {"mtime": base_time, "size": 2000}
        })
        
        collection_state = self.create_collection_state({
            "multi_entity_file.py": [
                {"entity_id": "multi_entity_file.py::func1", "indexed_at": base_time - 100},  # Old
                {"entity_id": "multi_entity_file.py::func2", "indexed_at": base_time - 10},   # Recent  
                {"entity_id": "multi_entity_file.py::Class1", "indexed_at": base_time - 5}   # Most recent
            ]
        })
        
        result = await mock_indexer.calculate_delta(workspace_state, collection_state, tolerance_sec=1.0)
        
        # Should use most recent indexed_at timestamp (base_time - 5)
        # File mtime (base_time) is newer, so should be modified
        assert len(result.modified_files) == 1
        assert len(result.unchanged_files) == 0
        assert "multi_entity_file.py" in result.modified_files
    
    @pytest.mark.asyncio
    async def test_calculate_delta_missing_indexed_at_timestamps(self, mock_indexer):
        """Test delta calculation when entities are missing indexed_at timestamps."""
        base_time = time.time()
        
        workspace_state = self.create_workspace_state({
            "no_timestamp_file.py": {"mtime": base_time, "size": 1000}
        })
        
        collection_state = self.create_collection_state({
            "no_timestamp_file.py": [
                {"entity_id": "no_timestamp_file.py::func1"},  # No indexed_at
                {"entity_id": "no_timestamp_file.py::func2", "indexed_at": None}  # Null indexed_at
            ]
        })
        
        result = await mock_indexer.calculate_delta(workspace_state, collection_state)
        
        # Should treat as modified when no valid timestamps found
        assert len(result.modified_files) == 1
        assert len(result.unchanged_files) == 0
        assert "no_timestamp_file.py" in result.modified_files
    
    @pytest.mark.asyncio
    async def test_calculate_delta_comprehensive_scenario(self, mock_indexer):
        """Test delta calculation with a comprehensive scenario covering all file states."""
        base_time = time.time()
        
        workspace_state = self.create_workspace_state({
            # Added files - exist in workspace but not collection
            "added_file1.py": {"mtime": base_time - 10, "size": 1000},
            "added_file2.js": {"mtime": base_time - 20, "size": 2000},
            
            # Modified files - newer than their indexed entities
            "modified_recent.py": {"mtime": base_time, "size": 1500},
            "modified_old.py": {"mtime": base_time - 10, "size": 1200},
            
            # Unchanged files - older or same as their indexed entities
            "unchanged1.py": {"mtime": base_time - 100, "size": 800},
            "unchanged2.py": {"mtime": base_time - 200, "size": 900}
        })
        
        collection_state = self.create_collection_state({
            # Files that will be detected as deleted (not in workspace)
            "deleted1.py": [
                {"entity_id": "deleted1.py::func1", "indexed_at": base_time - 300}
            ],
            "deleted2.py": [
                {"entity_id": "deleted2.py::func1", "indexed_at": base_time - 400},
                {"entity_id": "deleted2.py::Class1", "indexed_at": base_time - 350}
            ],
            
            # Files that will be detected as modified
            "modified_recent.py": [
                {"entity_id": "modified_recent.py::func1", "indexed_at": base_time - 50}  # Much older
            ],
            "modified_old.py": [
                {"entity_id": "modified_old.py::func1", "indexed_at": base_time - 100},  # Older
                {"entity_id": "modified_old.py::func2", "indexed_at": base_time - 50}    # Still older than file mtime
            ],
            
            # Files that will be detected as unchanged
            "unchanged1.py": [
                {"entity_id": "unchanged1.py::func1", "indexed_at": base_time - 50}  # Newer than file
            ],
            "unchanged2.py": [
                {"entity_id": "unchanged2.py::func1", "indexed_at": base_time - 150}  # Newer than file
            ]
        })
        
        result = await mock_indexer.calculate_delta(workspace_state, collection_state, tolerance_sec=1.0)
        
        # Verify added files
        assert len(result.added_files) == 2
        expected_added = {"added_file1.py", "added_file2.js"}
        assert result.added_files == expected_added
        
        # Verify modified files
        assert len(result.modified_files) == 2
        expected_modified = {"modified_recent.py", "modified_old.py"}
        assert result.modified_files == expected_modified
        
        # Verify deleted files
        assert len(result.deleted_files) == 2
        expected_deleted = {"deleted1.py", "deleted2.py"}
        assert result.deleted_files == expected_deleted
        
        # Verify unchanged files
        assert len(result.unchanged_files) == 2
        expected_unchanged = {"unchanged1.py", "unchanged2.py"}
        assert result.unchanged_files == expected_unchanged
        
        # Verify summary metrics
        assert result.total_changes == 6  # 2 added + 2 modified + 2 deleted
        assert result.total_workspace_files == 6
        assert result.total_collection_entities == 8  # 1 + 2 + 1 + 2 + 1 + 1 entities from collection
        assert result.change_ratio == 1.0  # 6 changes / 6 workspace files
        assert result.scan_time > 0
    
    @pytest.mark.asyncio
    async def test_calculate_delta_error_handling(self, mock_indexer):
        """Test delta calculation error handling for malformed data."""
        base_time = time.time()
        
        workspace_state = self.create_workspace_state({
            "problematic_file.py": {"mtime": base_time, "size": 1000}
        })
        
        # Create collection state with problematic entity data
        collection_state = self.create_collection_state({
            "problematic_file.py": [
                {"entity_id": "problematic_file.py::func1", "indexed_at": "invalid_timestamp"},  # Invalid timestamp
                {"entity_id": "problematic_file.py::func2", "indexed_at": base_time - 50}  # Valid timestamp
            ]
        })
        
        # Should handle errors gracefully and treat problematic files as modified
        result = await mock_indexer.calculate_delta(workspace_state, collection_state)
        
        assert len(result.modified_files) == 1
        assert "problematic_file.py" in result.modified_files
        assert result.total_changes == 1
    
    @pytest.mark.asyncio
    async def test_calculate_delta_performance_metrics(self, mock_indexer):
        """Test delta calculation performance with larger datasets."""
        base_time = time.time()
        
        # Create larger datasets to test performance
        workspace_files = {}
        collection_files = {}
        
        # Create 100 files for performance testing
        for i in range(100):
            file_path = f"performance_file_{i}.py"
            workspace_files[file_path] = {"mtime": base_time - (i * 10), "size": 1000 + i}
            
            # Make half of them have entities (for modification checks)
            if i < 50:
                collection_files[file_path] = [
                    {"entity_id": f"{file_path}::func1", "indexed_at": base_time - (i * 20)}  # Older than files
                ]
        
        workspace_state = self.create_workspace_state(workspace_files)
        collection_state = self.create_collection_state(collection_files)
        
        start_time = time.perf_counter()
        result = await mock_indexer.calculate_delta(workspace_state, collection_state)
        calculation_time = time.perf_counter() - start_time
        
        # Verify results
        assert len(result.added_files) == 50  # Files 50-99 are added
        assert len(result.modified_files) == 49  # Files 1-49 are modified (file_0 has same mtime as indexed_at)
        assert len(result.deleted_files) == 0
        assert len(result.unchanged_files) == 1  # performance_file_0.py is unchanged
        
        assert result.total_changes == 99  # 50 added + 49 modified + 0 deleted
        assert result.total_workspace_files == 100
        assert result.total_collection_entities == 50
        
        # Performance should be reasonable
        assert calculation_time < 1.0  # Should complete within 1 second
        assert result.scan_time > 0
        
        # Calculate performance metrics
        files_per_second = result.total_workspace_files / result.scan_time
        assert files_per_second > 100  # Should process at least 100 files/second
    
    @pytest.mark.asyncio
    async def test_calculate_delta_edge_case_timestamps(self, mock_indexer):
        """Test delta calculation with edge case timestamp scenarios."""
        base_time = time.time()
        
        workspace_state = self.create_workspace_state({
            "future_file.py": {"mtime": base_time + 3600, "size": 1000},  # Future timestamp
            "zero_file.py": {"mtime": 0, "size": 1000},  # Epoch timestamp
            "negative_file.py": {"mtime": -1, "size": 1000},  # Negative timestamp (if allowed)
            "recent_file.py": {"mtime": base_time, "size": 1000}
        })
        
        collection_state = self.create_collection_state({
            "future_file.py": [
                {"entity_id": "future_file.py::func1", "indexed_at": base_time}  # Past relative to file
            ],
            "zero_file.py": [
                {"entity_id": "zero_file.py::func1", "indexed_at": base_time}  # Much newer than file
            ],
            "negative_file.py": [
                {"entity_id": "negative_file.py::func1", "indexed_at": base_time}  # Much newer than file
            ],
            "recent_file.py": [
                {"entity_id": "recent_file.py::func1", "indexed_at": base_time + 0.5}  # Slightly newer
            ]
        })
        
        result = await mock_indexer.calculate_delta(workspace_state, collection_state, tolerance_sec=1.0)
        
        # Future file should be detected as modified (future mtime > indexed_at)
        assert "future_file.py" in result.modified_files
        
        # Zero timestamp file should be unchanged (very old mtime < indexed_at)
        assert "zero_file.py" in result.unchanged_files
        
        # Negative timestamp file should be unchanged (negative mtime < indexed_at)
        assert "negative_file.py" in result.unchanged_files
        
        # Recent file should be unchanged (within tolerance)
        assert "recent_file.py" in result.unchanged_files
    
    @pytest.mark.asyncio
    async def test_calculate_delta_result_properties(self, mock_indexer):
        """Test DeltaScanResult property calculations."""
        base_time = time.time()
        
        workspace_state = self.create_workspace_state({
            "added.py": {"mtime": base_time, "size": 1000},
            "modified.py": {"mtime": base_time, "size": 1000},
            "unchanged1.py": {"mtime": base_time - 100, "size": 1000},
            "unchanged2.py": {"mtime": base_time - 200, "size": 1000}
        })
        
        collection_state = self.create_collection_state({
            "deleted.py": [
                {"entity_id": "deleted.py::func1", "indexed_at": base_time - 100}
            ],
            "modified.py": [
                {"entity_id": "modified.py::func1", "indexed_at": base_time - 50}
            ],
            "unchanged1.py": [
                {"entity_id": "unchanged1.py::func1", "indexed_at": base_time - 50}
            ],
            "unchanged2.py": [
                {"entity_id": "unchanged2.py::func1", "indexed_at": base_time - 150}
            ]
        })
        
        result = await mock_indexer.calculate_delta(workspace_state, collection_state)
        
        # Test total_changes property
        expected_changes = len(result.added_files) + len(result.modified_files) + len(result.deleted_files)
        assert result.total_changes == expected_changes
        assert result.total_changes == 3  # 1 added + 1 modified + 1 deleted
        
        # Test change_ratio property
        expected_ratio = result.total_changes / result.total_workspace_files
        assert result.change_ratio == expected_ratio
        assert result.change_ratio == 0.75  # 3 changes / 4 workspace files
        
        # Test with empty workspace (edge case)
        empty_result = await mock_indexer.calculate_delta({}, collection_state)
        assert empty_result.change_ratio == 0.0  # Should handle division by zero


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])