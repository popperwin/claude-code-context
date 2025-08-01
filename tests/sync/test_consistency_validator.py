"""
Tests for CollectionConsistencyValidator validation and repair mechanism testing.

Validates consistency checking, orphaned entity detection, missing entity identification,
automatic reconciliation, and auto-repair mechanisms for collection integrity.

NOTE: CollectionConsistencyValidator is NOT currently used by any production code.
Only these test files create instances of it. The validator was implemented as
infrastructure but never integrated into the main application workflows.
This entire test file and the validator module may be deleted in the future
if the functionality remains unused.
"""
import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Set

from core.models.entities import Entity, EntityType, SourceLocation
from core.sync.validator import CollectionConsistencyValidator, ValidationResult, ConsistencyIssue, ValidationStatus
from core.storage.client import HybridQdrantClient
from core.storage.utils import entity_id_to_qdrant_id
from core.models.storage import QdrantPoint, SearchResult


class TestCollectionConsistencyValidator:
    """Test suite for collection consistency validation functionality."""
    
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
        # Return empty list of SearchResult objects (proper API contract)
        client.search_hybrid = AsyncMock(return_value=[])
        client.get_collection_info = AsyncMock(return_value={"points_count": 0})
        client.delete_points_by_filter = AsyncMock(return_value=StorageResult.successful_delete("test", 0, 10.0))
        client.upsert_points = AsyncMock(return_value=StorageResult.successful_insert("test", 0, 10.0))
        return client
    
    @pytest.fixture
    def consistency_validator(self, mock_qdrant_client, temp_project_dir):
        """Create CollectionConsistencyValidator instance."""
        return CollectionConsistencyValidator(
            storage_client=mock_qdrant_client,
            collection_name="test-collection",
            project_path=temp_project_dir,
            validation_interval_seconds=60
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
            source_code=f"def {name}():\\n    pass",
            docstring=f"Test {name} function",
            signature=f"{name}()" if entity_type == EntityType.FUNCTION else None
        )
    
    def create_mock_search_result(
        self,
        entity_id: str,
        file_path: str,
        entity_name: str,
        entity_type: str = "function",
        score: float = 0.8
    ) -> SearchResult:
        """Create a mock SearchResult object for testing."""
        point = QdrantPoint(
            id=entity_id_to_qdrant_id(entity_id),
            vector=[0.1] * 1024,  # Mock vector
            payload={
                'file_path': file_path,
                'entity_name': entity_name,
                'entity_type': entity_type,
                'entity_id': entity_id
            }
        )
        
        return SearchResult(
            point=point,
            score=score,
            query="*",
            search_type="hybrid",
            rank=1,
            total_results=1
        )

    @pytest.mark.asyncio
    async def test_validate_consistency_no_issues(self, consistency_validator, mock_qdrant_client, temp_project_dir):
        """Test consistency validation when no issues are found."""
        # Create test files that exist
        test_files = ["file1.py", "file2.py", "file3.py"]
        for filename in test_files:
            (temp_project_dir / filename).write_text(f"def function_in_{filename.replace('.', '_')}(): pass")
        
        # Mock collection entities that have corresponding files
        mock_search_results = []
        for i, filename in enumerate(test_files):
            file_path = temp_project_dir / filename
            search_result = self.create_mock_search_result(
                entity_id=f"entity_{i}",
                file_path=str(file_path),
                entity_name=f"function_in_{filename.replace('.', '_')}",
                entity_type="function"
            )
            mock_search_results.append(search_result)
        
        mock_qdrant_client.search_hybrid.return_value = mock_search_results
        
        # Validate consistency
        result = await consistency_validator.validate_consistency()
        
        # Should find no issues
        assert result.status == ValidationStatus.HEALTHY
        assert len(result.issues) == 0
        assert result.orphaned_entities == 0
        assert result.missing_entities == 0
        assert result.total_entities_checked == len(test_files)
    
    @pytest.mark.asyncio
    async def test_detect_orphaned_entities(self, consistency_validator, mock_qdrant_client, temp_project_dir):
        """Test detection of orphaned entities (entities without corresponding files)."""
        # Create some test files
        existing_files = ["existing1.py", "existing2.py"]
        for filename in existing_files:
            (temp_project_dir / filename).write_text(f"def function(): pass")
        
        # Mock collection entities - some have missing files
        mock_search_results = [
            self.create_mock_search_result(
                entity_id='entity_1',
                file_path=str(temp_project_dir / "existing1.py"),
                entity_name='existing_function',
                entity_type='function'
            ),
            self.create_mock_search_result(
                entity_id='entity_2',
                file_path=str(temp_project_dir / "missing1.py"),  # File doesn't exist
                entity_name='orphaned_function',
                entity_type='function'
            ),
            self.create_mock_search_result(
                entity_id='entity_3',
                file_path=str(temp_project_dir / "missing2.py"),  # File doesn't exist
                entity_name='another_orphaned',
                entity_type='class'
            )
        ]
        
        mock_qdrant_client.search_hybrid.return_value = mock_search_results
        
        # Validate consistency
        result = await consistency_validator.validate_consistency()
        
        # Should detect orphaned entities
        assert result.status == ValidationStatus.ISSUES_FOUND
        assert result.orphaned_entities == 2
        assert result.missing_entities == 1  # existing2.py has no entity in collection (correct behavior)
        assert result.total_entities_checked == 3
        
        # Should have orphaned entity issues
        orphaned_issues = [issue for issue in result.issues if issue.issue_type == "orphaned_entity"]
        assert len(orphaned_issues) == 2
        
        # Check issue details
        orphaned_files = {issue.file_path for issue in orphaned_issues}
        expected_orphaned = {str(temp_project_dir / "missing1.py"), str(temp_project_dir / "missing2.py")}
        assert orphaned_files == expected_orphaned
    
    @pytest.mark.asyncio
    async def test_detect_missing_entities(self, consistency_validator, mock_qdrant_client, temp_project_dir):
        """Test detection of missing entities (files without corresponding entities)."""
        # Create test files that should have entities
        test_files = ["file1.py", "file2.py", "file3.py", "file4.py"]
        for filename in test_files:
            (temp_project_dir / filename).write_text(f"def function_in_{filename.replace('.', '_')}(): pass")
        
        # Mock collection entities - missing some files
        mock_search_results = [
            self.create_mock_search_result(
                entity_id='entity_1',
                file_path=str(temp_project_dir / "file1.py"),
                entity_name='function_in_file1_py',
                entity_type='function'
            ),
            self.create_mock_search_result(
                entity_id='entity_2',
                file_path=str(temp_project_dir / "file2.py"),
                entity_name='function_in_file2_py',
                entity_type='function'
            )
            # file3.py and file4.py are missing from collection
        ]
        
        mock_qdrant_client.search_hybrid.return_value = mock_search_results
        
        # Validate consistency
        result = await consistency_validator.validate_consistency()
        
        # Should detect missing entities
        assert result.status == ValidationStatus.ISSUES_FOUND
        assert result.orphaned_entities == 0
        assert result.missing_entities == 2  # file3.py and file4.py
        assert result.total_entities_checked == 2  # Only entities in collection
        
        # Should have missing entity issues
        missing_issues = [issue for issue in result.issues if issue.issue_type == "missing_entity"]
        assert len(missing_issues) == 2
        
        # Check issue details
        missing_files = {issue.file_path for issue in missing_issues}
        expected_missing = {str((temp_project_dir / "file3.py").resolve()), str((temp_project_dir / "file4.py").resolve())}
        assert missing_files == expected_missing
    
    @pytest.mark.asyncio
    async def test_auto_repair_orphaned_entities(self, consistency_validator, mock_qdrant_client, temp_project_dir):
        """Test automatic repair of orphaned entities by deletion."""
        # Create one existing file
        (temp_project_dir / "existing.py").write_text("def function(): pass")
        
        # Mock collection with orphaned entities
        mock_search_results = [
            self.create_mock_search_result(
                entity_id='good_entity',
                file_path=str(temp_project_dir / "existing.py"),
                entity_name='good_function',
                entity_type='function'
            ),
            self.create_mock_search_result(
                entity_id='orphaned_1',
                file_path=str(temp_project_dir / "deleted.py"),
                entity_name='orphaned_function',
                entity_type='function'
            ),
            self.create_mock_search_result(
                entity_id='orphaned_2',
                file_path=str(temp_project_dir / "removed.py"),
                entity_name='another_orphaned',
                entity_type='class'
            )
        ]
        
        mock_qdrant_client.search_hybrid.return_value = mock_search_results
        
        # Validate with auto-repair enabled
        result = await consistency_validator.validate_consistency(auto_repair=True)
        
        # Should detect and repair orphaned entities
        assert result.status == ValidationStatus.REPAIRED
        assert result.orphaned_entities == 2
        assert result.repairs_attempted == 2
        assert result.repairs_successful == 2
        
        # Should have called delete for orphaned entities
        assert mock_qdrant_client.delete_points_by_filter.call_count == 2
    
    @pytest.mark.asyncio
    async def test_auto_repair_missing_entities(self, consistency_validator, mock_qdrant_client, temp_project_dir):
        """Test automatic repair of missing entities by re-indexing files."""
        # Create test files
        test_files = ["indexed.py", "missing1.py", "missing2.py"]
        for filename in test_files:
            (temp_project_dir / filename).write_text(f"def function_in_{filename.replace('.', '_')}(): pass")
        
        # Mock collection with only one entity
        mock_search_results = [
            self.create_mock_search_result(
                entity_id='existing_entity',
                file_path=str(temp_project_dir / "indexed.py"),
                entity_name='function_in_indexed_py',
                entity_type='function'
            )
        ]
        
        mock_qdrant_client.search_hybrid.return_value = mock_search_results
        
        # Mock the parser to return entities for missing files
        with patch('core.sync.validator.ParserRegistry') as mock_parser_registry:
            mock_registry = Mock()
            mock_parser = Mock()
            
            def mock_parse_side_effect(file_path):
                # Return appropriate entity based on file path
                if "missing1.py" in str(file_path):
                    return [self.create_test_entity("function_in_missing1_py", file_path)]
                elif "missing2.py" in str(file_path):
                    return [self.create_test_entity("function_in_missing2_py", file_path)]
                else:
                    return [self.create_test_entity("parsed_function", file_path)]
            
            mock_parser.parse_file = AsyncMock(side_effect=mock_parse_side_effect)
            mock_registry.can_parse.return_value = True
            mock_registry.get_parser.return_value = mock_parser
            mock_parser_registry.return_value = mock_registry
            
            # Validate with auto-repair enabled
            result = await consistency_validator.validate_consistency(auto_repair=True)
        
        # Should detect missing entities and attempt repair
        assert result.status == ValidationStatus.REPAIRED
        assert result.missing_entities == 2
        assert result.repairs_attempted == 2
        
        # Should have called upsert for missing entities
        assert mock_qdrant_client.upsert_points.call_count == 2
    
    @pytest.mark.asyncio
    async def test_consistency_scan_with_mixed_issues(self, consistency_validator, mock_qdrant_client, temp_project_dir):
        """Test consistency scan with both orphaned and missing entities."""
        # Create some files
        (temp_project_dir / "good_file.py").write_text("def good_function(): pass")
        (temp_project_dir / "missing_from_collection.py").write_text("def missing_function(): pass")
        
        # Mock collection with mixed entities
        mock_search_results = [
            self.create_mock_search_result(
                entity_id='good_entity',
                file_path=str(temp_project_dir / "good_file.py"),
                entity_name='good_function',
                entity_type='function'
            ),
            self.create_mock_search_result(
                entity_id='orphaned_entity',
                file_path=str(temp_project_dir / "deleted_file.py"),  # Doesn't exist
                entity_name='orphaned_function',
                entity_type='function'
            )
        ]
        
        mock_qdrant_client.search_hybrid.return_value = mock_search_results
        
        # Validate consistency
        result = await consistency_validator.validate_consistency()
        
        # Should detect both types of issues
        assert result.status == ValidationStatus.ISSUES_FOUND
        assert result.orphaned_entities == 1
        assert result.missing_entities == 1
        assert result.total_entities_checked == 2
        assert len(result.issues) == 2
        
        # Check issue types
        issue_types = {issue.issue_type for issue in result.issues}
        assert issue_types == {"orphaned_entity", "missing_entity"}
    
    @pytest.mark.asyncio
    async def test_periodic_validation_scheduling(self, consistency_validator, mock_qdrant_client):
        """Test periodic validation scheduling and execution."""
        # Mock empty collection for quick validation
        mock_qdrant_client.search_hybrid.return_value = []
        
        # Start periodic validation with short interval
        consistency_validator.validation_interval_seconds = 0.1
        await consistency_validator.start_periodic_validation()
        
        # Wait for at least one validation cycle
        await asyncio.sleep(0.2)
        
        # Stop periodic validation
        await consistency_validator.stop_periodic_validation()
        
        # Should have performed at least one validation
        assert mock_qdrant_client.search_hybrid.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_validation_error_handling(self, consistency_validator, mock_qdrant_client, temp_project_dir):
        """Test error handling during validation operations."""
        # Mock Qdrant client to raise an exception
        mock_qdrant_client.search_hybrid.side_effect = Exception("Qdrant connection failed")
        
        # Validate consistency (should handle error gracefully)
        result = await consistency_validator.validate_consistency()
        
        # Should return error status
        assert result.status == ValidationStatus.ERROR
        assert "failed" in result.summary.lower()
        assert result.total_entities_checked == 0
    
    @pytest.mark.asyncio
    async def test_repair_failure_handling(self, consistency_validator, mock_qdrant_client, temp_project_dir):
        """Test handling of repair operation failures."""
        # Mock collection with orphaned entity
        mock_search_results = [
            self.create_mock_search_result(
                entity_id='orphaned',
                file_path=str(temp_project_dir / "missing.py"),
                entity_name='orphaned_function',
                entity_type='function'
            )
        ]
        
        mock_qdrant_client.search_hybrid.return_value = mock_search_results
        
        # Mock delete operation to fail
        from core.models.storage import StorageResult
        mock_qdrant_client.delete_points_by_filter.return_value = StorageResult.failed_operation(
            "delete", "test-collection", "Delete failed", 10.0
        )
        
        # Validate with auto-repair
        result = await consistency_validator.validate_consistency(auto_repair=True)
        
        # Should detect repair failure
        assert result.status == ValidationStatus.REPAIR_FAILED
        assert result.repairs_attempted == 1
        assert result.repairs_successful == 0
        assert result.orphaned_entities == 1
    
    @pytest.mark.asyncio
    async def test_validation_performance_metrics(self, consistency_validator, mock_qdrant_client, temp_project_dir):
        """Test validation performance metrics tracking."""
        # Create test files
        for i in range(5):
            (temp_project_dir / f"file{i}.py").write_text(f"def function{i}(): pass")
        
        # Mock collection entities
        mock_search_results = [
            self.create_mock_search_result(
                entity_id=f'entity_{i}',
                file_path=str(temp_project_dir / f"file{i}.py"),
                entity_name=f'function{i}',
                entity_type='function'
            )
            for i in range(5)
        ]
        
        mock_qdrant_client.search_hybrid.return_value = mock_search_results
        
        # Validate consistency
        result = await consistency_validator.validate_consistency()
        
        # Should have performance metrics
        assert result.validation_duration_ms > 0
        assert result.total_entities_checked == 5
        assert result.files_scanned == 5
    
    def test_get_validation_status(self, consistency_validator):
        """Test getting validation status information."""
        status = consistency_validator.get_status()
        
        # Should have basic status information
        assert "collection_name" in status
        assert "project_path" in status
        assert "validation_interval_seconds" in status
        assert "is_running_periodic" in status
        assert "last_validation_time" in status
        
        assert status["collection_name"] == "test-collection"
        assert status["is_running_periodic"] is False
    
    @pytest.mark.asyncio
    async def test_manual_repair_operation(self, consistency_validator, mock_qdrant_client, temp_project_dir):
        """Test manual repair operation for specific issues."""
        # Create test issues
        issues = [
            ConsistencyIssue(
                issue_type="orphaned_entity",
                entity_id="orphaned_1",
                file_path=str(temp_project_dir / "missing.py"),
                description="Entity references non-existent file"
            ),
            ConsistencyIssue(
                issue_type="missing_entity", 
                entity_id=None,
                file_path=str(temp_project_dir / "unindexed.py"),
                description="File exists but has no entities in collection"
            )
        ]
        
        # Create the unindexed file
        (temp_project_dir / "unindexed.py").write_text("def unindexed_function(): pass")
        
        # Mock parser for missing entity repair
        with patch('core.sync.validator.ParserRegistry') as mock_parser_registry:
            mock_registry = Mock()
            mock_parser = Mock()
            
            def mock_parse_side_effect(file_path):
                # Return appropriate entity based on file path
                if "unindexed.py" in str(file_path):
                    return [self.create_test_entity("unindexed_function", file_path)]
                else:
                    return [self.create_test_entity("test_function", file_path)]
            
            mock_parser.parse_file = AsyncMock(side_effect=mock_parse_side_effect)
            mock_registry.can_parse.return_value = True
            mock_registry.get_parser.return_value = mock_parser
            mock_parser_registry.return_value = mock_registry
            
            # Perform manual repair
            repair_result = await consistency_validator.repair_issues(issues)
        
        # Should attempt repairs for both issues
        assert repair_result["repairs_attempted"] == 2
        assert repair_result["success"] is True
        
        # Should have called appropriate repair operations
        assert mock_qdrant_client.delete_points_by_filter.call_count == 1  # For orphaned entity
        assert mock_qdrant_client.upsert_points.call_count == 1  # For missing entity


class TestCollectionConsistencyValidatorEdgeCases:
    """Test edge cases and error conditions for consistency validator."""
    
    @pytest.fixture
    def temp_project_dir(self):
        """Create temporary project directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def create_mock_search_result(
        self,
        entity_id: str,
        file_path: str,
        entity_name: str,
        entity_type: str = "function",
        score: float = 0.8
    ) -> SearchResult:
        """Create a mock SearchResult object for testing."""
        point = QdrantPoint(
            id=entity_id_to_qdrant_id(entity_id),
            vector=[0.1] * 1024,  # Mock vector
            payload={
                'file_path': file_path,
                'entity_name': entity_name,
                'entity_type': entity_type,
                'entity_id': entity_id
            }
        )
        
        return SearchResult(
            point=point,
            score=score,
            query="*",
            search_type="hybrid",
            rank=1,
            total_results=1
        )
    
    @pytest.fixture
    def consistency_validator(self, temp_project_dir):
        """Create minimal ConsistencyValidator for edge case testing."""
        from core.models.storage import StorageResult
        mock_client = Mock(spec=HybridQdrantClient)
        mock_client.search_hybrid = AsyncMock(return_value=[])
        mock_client.get_collection_info = AsyncMock(return_value={"points_count": 0})
        
        return CollectionConsistencyValidator(
            storage_client=mock_client,
            collection_name="test-collection",
            project_path=temp_project_dir
        )
    
    @pytest.mark.asyncio
    async def test_validation_with_empty_collection(self, consistency_validator, temp_project_dir):
        """Test validation when collection is empty."""
        # Create some files that should be indexed
        (temp_project_dir / "file1.py").write_text("def function1(): pass")
        (temp_project_dir / "file2.py").write_text("def function2(): pass")
        
        # Collection is empty (default mock behavior)
        result = await consistency_validator.validate_consistency()
        
        # Should detect missing entities
        assert result.status == ValidationStatus.ISSUES_FOUND
        assert result.missing_entities == 2
        assert result.orphaned_entities == 0
        assert result.total_entities_checked == 0
    
    @pytest.mark.asyncio
    async def test_validation_with_empty_project(self, consistency_validator, temp_project_dir):
        """Test validation when project directory is empty."""
        # Mock collection with entities
        mock_search_results = [
            self.create_mock_search_result(
                entity_id='entity_1',
                file_path=str(temp_project_dir / "nonexistent.py"),
                entity_name='some_function',
                entity_type='function'
            )
        ]
        
        consistency_validator.storage_client.search_hybrid.return_value = mock_search_results
        
        # Validate consistency
        result = await consistency_validator.validate_consistency()
        
        # Should detect all entities as orphaned
        assert result.status == ValidationStatus.ISSUES_FOUND
        assert result.orphaned_entities == 1
        assert result.missing_entities == 0
        assert result.total_entities_checked == 1
    
    @pytest.mark.asyncio
    async def test_validation_with_invalid_entity_data(self, consistency_validator, temp_project_dir):
        """Test validation with malformed entity data."""
        # Create the valid file
        (temp_project_dir / "valid.py").write_text("def valid_function(): pass")
        
        # Mock collection with mixed valid and invalid entities 
        # Note: In practice, SearchResult objects should always be valid, but we test robustness
        valid_search_result = self.create_mock_search_result(
            entity_id='valid_entity',
            file_path=str(temp_project_dir / "valid.py"), 
            entity_name='valid_function',
            entity_type='function'
        )
        
        # Create a malformed SearchResult for robustness testing
        # Use minimal required fields but with empty file_path to test robustness
        invalid_point = QdrantPoint(
            id=entity_id_to_qdrant_id('invalid_entity_1'),
            vector=[0.1] * 1024,
            payload={
                'entity_id': 'invalid_entity_1',
                'entity_type': 'function',
                'file_path': ''  # Empty file_path to test robustness
            }
        )
        invalid_search_result = SearchResult(
            point=invalid_point,
            score=0.5,
            query="*",
            search_type="hybrid",
            rank=1,
            total_results=1
        )
        
        mock_search_results = [valid_search_result, invalid_search_result]
        consistency_validator.storage_client.search_hybrid.return_value = mock_search_results
        
        # Validate consistency (should handle invalid data gracefully)
        result = await consistency_validator.validate_consistency()
        
        # Should still complete validation
        assert result.status in [ValidationStatus.HEALTHY, ValidationStatus.ISSUES_FOUND]
        # Should process at least the valid entity
        assert result.total_entities_checked >= 1
    
    @pytest.mark.asyncio
    async def test_double_start_periodic_validation(self, consistency_validator):
        """Test starting periodic validation when already running."""
        # Start periodic validation
        await consistency_validator.start_periodic_validation()
        assert consistency_validator.is_running_periodic
        
        # Start again (should handle gracefully)
        await consistency_validator.start_periodic_validation()
        assert consistency_validator.is_running_periodic
        
        # Stop validation
        await consistency_validator.stop_periodic_validation()
        assert not consistency_validator.is_running_periodic
    
    @pytest.mark.asyncio
    async def test_stop_periodic_validation_when_not_running(self, consistency_validator):
        """Test stopping periodic validation when not running."""
        assert not consistency_validator.is_running_periodic
        
        # Stop when not running (should handle gracefully)
        await consistency_validator.stop_periodic_validation()
        assert not consistency_validator.is_running_periodic
    
    @pytest.mark.asyncio
    async def test_repair_with_empty_issues_list(self, consistency_validator):
        """Test repair operation with empty issues list."""
        # Repair empty issues list
        result = await consistency_validator.repair_issues([])
        
        # Should handle gracefully
        assert result["success"] is True
        assert result["repairs_attempted"] == 0
        assert result["repairs_successful"] == 0
    
    @pytest.mark.asyncio
    async def test_validation_with_huge_collection(self, consistency_validator, temp_project_dir):
        """Test validation performance with large collection."""
        # Mock large collection (simulate, don't create actual entities)
        large_entity_count = 10000
        
        # Mock search to return count but not actual entities (for performance)
        consistency_validator.storage_client.get_collection_info.return_value = {
            "points_count": large_entity_count
        }
        
        # Mock search to return empty results (simulating pagination handling)
        consistency_validator.storage_client.search_hybrid.return_value = []
        
        # Validate consistency
        result = await consistency_validator.validate_consistency()
        
        # Should complete without issues
        assert result.status in [ValidationStatus.HEALTHY, ValidationStatus.ISSUES_FOUND]
        assert result.validation_duration_ms > 0
    
    def test_validation_result_serialization(self, consistency_validator):
        """Test ValidationResult can be properly serialized."""
        # Create a validation result with various data
        issues = [
            ConsistencyIssue(
                issue_type="orphaned_entity",
                entity_id="test_entity",
                file_path="/test/path.py", 
                description="Test issue"
            )
        ]
        
        result = ValidationResult(
            status=ValidationStatus.ISSUES_FOUND,
            total_entities_checked=100,
            orphaned_entities=1,
            missing_entities=0,
            files_scanned=50,
            issues=issues,
            validation_duration_ms=250.5,
            repairs_attempted=0,
            repairs_successful=0,
            summary="Found 1 issue"
        )
        
        # Should be able to convert to dict (for JSON serialization)
        result_dict = result.__dict__
        assert "status" in result_dict
        assert "issues" in result_dict
        assert result_dict["total_entities_checked"] == 100
        assert result_dict["validation_duration_ms"] == 250.5