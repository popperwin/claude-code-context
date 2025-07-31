"""
Tests for EntityScanner with Tree-sitter integration and parallel processing.

Validates entity scanning with comprehensive coverage of parallel processing,
entity batching, streaming capabilities, and integration with EntityLifecycleManager.
"""

import pytest
import asyncio
import tempfile
import shutil
import time
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any

from core.indexer.entity_scanner import (
    EntityScanner, EntityScanRequest, EntityScanResult, EntityBatch
)
from core.models.entities import Entity, EntityType, SourceLocation
from core.storage.client import HybridQdrantClient
from core.models.storage import StorageResult, QdrantPoint
from core.sync.lifecycle import EntityLifecycleManager
from core.parser.base import ParseResult
from core.parser.parallel_pipeline import ProcessParsingPipeline, PipelineStats
from core.parser.registry import ParserRegistry, parser_registry


class TestEntityScanner:
    """Test suite for EntityScanner functionality."""
    
    @pytest.fixture
    def temp_project_dir(self):
        """Create temporary project directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def mock_storage_client(self):
        """Create mock storage client."""
        client = Mock(spec=HybridQdrantClient)
        client.upsert_points = AsyncMock(return_value=StorageResult.successful_insert("test", 5, 10.0))
        client.delete_points = AsyncMock(return_value=StorageResult.successful_delete("test", 3, 5.0))
        return client
    
    @pytest.fixture
    def mock_lifecycle_manager(self, mock_storage_client, temp_project_dir):
        """Create mock lifecycle manager."""
        manager = Mock(spec=EntityLifecycleManager)
        manager.storage_client = mock_storage_client
        manager.collection_name = "test-collection"
        manager.project_path = temp_project_dir
        return manager
    
    @pytest.fixture
    def entity_scanner(self, mock_storage_client):
        """Create EntityScanner instance."""
        return EntityScanner(
            storage_client=mock_storage_client,
            enable_parallel=True,
            default_batch_size=10,
            max_workers=2
        )
    
    @pytest.fixture
    def entity_scanner_with_lifecycle(self, mock_storage_client, mock_lifecycle_manager):
        """Create EntityScanner with lifecycle manager."""
        return EntityScanner(
            storage_client=mock_storage_client,
            lifecycle_manager=mock_lifecycle_manager,
            enable_parallel=True,
            default_batch_size=10,
            max_workers=2
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
    
    def create_mock_parse_result(
        self, 
        file_path: Path, 
        entities: List[Entity] = None,
        success: bool = True
    ) -> ParseResult:
        """Create a mock parse result."""
        if entities is None:
            entities = [self.create_test_entity("test_func", file_path)]
        
        result = ParseResult(
            file_path=file_path,
            language="python",
            entities=entities,
            relations=[],
            ast_nodes=[],
            parse_time=0.1,
            file_size=100,
            file_hash="test_hash"
        )
        
        if not success:
            result.add_syntax_error({
                "type": "SYNTAX_ERROR",
                "message": "Test syntax error",
                "line": 1,
                "column": 1
            })
        
        return result

    @pytest.mark.asyncio
    async def test_scanner_initialization(self, mock_storage_client):
        """Test EntityScanner initialization."""
        scanner = EntityScanner(
            storage_client=mock_storage_client,
            enable_parallel=True,
            default_batch_size=25,
            max_workers=4
        )
        
        assert scanner.storage_client == mock_storage_client
        assert scanner.enable_parallel is True
        assert scanner.default_batch_size == 25
        assert scanner.max_workers == 4
        assert scanner.lifecycle_manager is None
        assert isinstance(scanner.parser_pipeline, ProcessParsingPipeline)
        
        # Check initial state
        assert scanner._scan_count == 0
        assert scanner._total_entities_processed == 0
        assert scanner._total_scan_time == 0.0
        assert scanner._last_scan_time is None

    @pytest.mark.asyncio
    async def test_scan_files_parallel_success(self, entity_scanner, temp_project_dir):
        """Test successful parallel file scanning."""
        # Create test files
        test_files = []
        for i in range(3):
            test_file = temp_project_dir / f"test_{i}.py"
            test_file.write_text(f"def function_{i}():\n    pass")
            test_files.append(test_file)
        
        # Mock parallel pipeline
        mock_entities = []
        mock_results = []
        for i, file_path in enumerate(test_files):
            entities = [self.create_test_entity(f"function_{i}", file_path)]
            mock_entities.extend(entities)
            mock_results.append(self.create_mock_parse_result(file_path, entities))
        
        mock_stats = PipelineStats(
            total_files=3,
            processed_files=3,
            successful_files=3,
            failed_files=0,
            total_entities=3,
            total_relations=0,
            total_time=0.5
        )
        
        entity_scanner.parser_pipeline.parse_files = Mock(return_value=(mock_results, mock_stats))
        
        # Create scan request
        request = EntityScanRequest(
            file_paths=test_files,
            collection_name="test-collection",
            project_path=temp_project_dir,
            scan_mode="full_scan",
            batch_size=10,
            enable_parallel=True
        )
        
        # Execute scan
        result = await entity_scanner.scan_files(request)
        
        # Verify result
        assert result.total_files == 3
        assert result.processed_files == 3
        assert result.successful_files == 3
        assert result.failed_files == 0
        assert result.total_entities == 3
        assert result.total_relations == 0
        assert result.success_rate == 1.0
        assert result.entities_per_second > 0
        assert result.scan_time > 0
        
        # Verify parallel processing was used
        assert result.performance_metrics["parallel_processing"] is True
        
        # Verify scanner state updated
        assert entity_scanner._scan_count == 1
        assert entity_scanner._total_entities_processed == 3
        assert entity_scanner._last_scan_time is not None

    @pytest.mark.asyncio
    async def test_scan_files_sequential_mode(self, entity_scanner, temp_project_dir):
        """Test sequential file scanning mode."""
        # Create test file
        test_file = temp_project_dir / "test.py"
        test_file.write_text("def test_function():\n    pass")
        
        # Mock parser registry
        mock_parser = Mock()
        mock_result = self.create_mock_parse_result(test_file)
        mock_parser.parse_file.return_value = mock_result
        
        with patch.object(parser_registry, 'get_parser_for_file', return_value=mock_parser):
            # Create scan request with sequential mode
            request = EntityScanRequest(
                file_paths=[test_file],
                collection_name="test-collection",
                project_path=temp_project_dir,
                scan_mode="full_scan",
                enable_parallel=False
            )
            
            # Execute scan
            result = await entity_scanner.scan_files(request)
            
            # Verify result
            assert result.total_files == 1
            assert result.successful_files == 1
            assert result.failed_files == 0
            assert result.total_entities == 1
            assert result.performance_metrics["parallel_processing"] is False
            
            # Verify parser was called
            mock_parser.parse_file.assert_called_once_with(test_file)

    @pytest.mark.asyncio
    async def test_scan_files_with_failures(self, entity_scanner, temp_project_dir):
        """Test scanning with parse failures."""
        # Create test files
        good_file = temp_project_dir / "good.py"
        good_file.write_text("def good_function():\n    pass")
        
        bad_file = temp_project_dir / "bad.py"
        bad_file.write_text("def invalid syntax")
        
        # Mock results with one failure
        good_result = self.create_mock_parse_result(good_file, success=True)
        bad_result = self.create_mock_parse_result(bad_file, success=False)
        bad_result.entities = []  # No entities from failed parse
        
        mock_stats = PipelineStats(
            total_files=2,
            processed_files=2,
            successful_files=1,
            failed_files=1,
            total_entities=1,
            total_relations=0,
            total_time=0.3
        )
        
        entity_scanner.parser_pipeline.parse_files = Mock(return_value=([good_result, bad_result], mock_stats))
        
        # Create scan request
        request = EntityScanRequest(
            file_paths=[good_file, bad_file],
            collection_name="test-collection",
            project_path=temp_project_dir
        )
        
        # Execute scan
        result = await entity_scanner.scan_files(request)
        
        # Verify result
        assert result.total_files == 2
        assert result.successful_files == 1
        assert result.failed_files == 1
        assert result.total_entities == 1
        assert result.success_rate == 0.5
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_entity_batch_processing(self, entity_scanner_with_lifecycle, temp_project_dir):
        """Test entity batch processing functionality."""
        # Create test entities
        test_file = temp_project_dir / "test.py"
        test_file.write_text("def func1():\n    pass\ndef func2():\n    pass")
        
        entities = [
            self.create_test_entity("func1", test_file, start_line=1),
            self.create_test_entity("func2", test_file, start_line=3)
        ]
        
        # Mock file content reading
        with patch.object(Path, 'read_text', return_value="def func1():\n    pass\ndef func2():\n    pass"):
            # Process entities in batches
            await entity_scanner_with_lifecycle._process_entities_in_batches(
                entities,
                "test-collection",
                batch_size=1
            )
        
        # Verify storage was called for both batches
        assert entity_scanner_with_lifecycle.storage_client.upsert_points.call_count == 2

    @pytest.mark.asyncio
    async def test_entity_batch_creation(self, entity_scanner):
        """Test EntityBatch creation and properties."""
        test_entities = [
            self.create_test_entity("func1"),
            self.create_test_entity("func2"),
            self.create_test_entity("func3")
        ]
        
        batch = EntityBatch(
            batch_id=1,
            entities=test_entities,
            file_path="/test/file.py",
            batch_size=3
        )
        
        assert batch.batch_id == 1
        assert batch.entity_count == 3
        assert batch.file_path == "/test/file.py"
        assert batch.batch_size == 3
        assert batch.success is True
        assert batch.error_message is None

    @pytest.mark.asyncio
    async def test_stream_entities(self, entity_scanner, temp_project_dir):
        """Test entity streaming functionality."""
        # Create test files
        test_files = []
        for i in range(3):
            test_file = temp_project_dir / f"stream_{i}.py"
            test_file.write_text(f"def stream_func_{i}():\n    pass")
            test_files.append(test_file)
        
        # Mock parser registry
        mock_parser = Mock()
        
        def mock_parse_side_effect(file_path):
            return self.create_mock_parse_result(
                file_path, 
                [self.create_test_entity(f"stream_func_{file_path.stem[-1]}", file_path)]
            )
        
        mock_parser.parse_file.side_effect = mock_parse_side_effect
        
        with patch.object(parser_registry, 'get_parser_for_file', return_value=mock_parser):
            # Stream entities
            batches = []
            async for batch in entity_scanner.stream_entities(test_files, batch_size=2):
                batches.append(batch)
            
            # Verify streaming results
            assert len(batches) >= 1  # At least one batch
            
            total_entities = sum(batch.entity_count for batch in batches)
            assert total_entities == 3
            
            # Verify batch properties
            for batch in batches:
                assert isinstance(batch, EntityBatch)
                assert batch.entity_count > 0
                assert batch.success is True

    @pytest.mark.asyncio
    async def test_scan_directory(self, entity_scanner, temp_project_dir):
        """Test directory scanning functionality."""
        # Create test directory structure
        (temp_project_dir / "subdir").mkdir()
        
        # Create test files
        files = [
            temp_project_dir / "main.py",
            temp_project_dir / "utils.py",
            temp_project_dir / "subdir" / "module.py"
        ]
        
        for i, file_path in enumerate(files):
            file_path.write_text(f"def function_{i}():\n    pass")
        
        # Mock parser registry file discovery
        with patch.object(parser_registry, 'discover_files', return_value=files):
            # Mock scan_files method
            mock_result = EntityScanResult(
                request_id="test_scan",
                total_files=3,
                processed_files=3,
                successful_files=3,
                failed_files=0,
                total_entities=3,
                total_relations=0,
                scan_time=0.5,
                entities_per_second=6.0,
                success_rate=1.0
            )
            
            entity_scanner.scan_files = AsyncMock(return_value=mock_result)
            
            # Execute directory scan
            result = await entity_scanner.scan_directory(
                temp_project_dir,
                "test-collection",
                recursive=True,
                scan_mode="full_scan"
            )
            
            # Verify result
            assert result.total_files == 3
            assert result.successful_files == 3
            assert result.total_entities == 3
            assert result.success_rate == 1.0
            
            # Verify discover_files was called correctly
            parser_registry.discover_files.assert_called_once_with(temp_project_dir, recursive=True)

    @pytest.mark.asyncio
    async def test_scan_directory_no_files(self, entity_scanner, temp_project_dir):
        """Test directory scanning with no parseable files."""
        # Mock empty file discovery
        with patch.object(parser_registry, 'discover_files', return_value=[]):
            result = await entity_scanner.scan_directory(
                temp_project_dir,
                "test-collection"
            )
            
            # Verify empty result
            assert result.total_files == 0
            assert result.processed_files == 0
            assert result.successful_files == 0
            assert result.failed_files == 0
            assert result.total_entities == 0
            assert result.success_rate == 0

    @pytest.mark.asyncio
    async def test_progress_callback(self, entity_scanner, temp_project_dir):
        """Test progress callback functionality."""
        # Create test file
        test_file = temp_project_dir / "progress_test.py"
        test_file.write_text("def progress_func():\n    pass")
        
        # Create progress callback mock
        progress_callback = Mock()
        
        # Mock sequential scanning since it's easier to control
        entity_scanner.enable_parallel = False
        
        mock_parser = Mock()
        mock_result = self.create_mock_parse_result(test_file)
        mock_parser.parse_file.return_value = mock_result
        
        with patch.object(parser_registry, 'get_parser_for_file', return_value=mock_parser):
            request = EntityScanRequest(
                file_paths=[test_file],
                collection_name="test-collection",
                project_path=temp_project_dir,
                progress_callback=progress_callback,
                enable_parallel=False
            )
            
            await entity_scanner.scan_files(request)
            
            # Verify progress callback was called
            progress_callback.assert_called()
            
            # Check callback arguments
            call_args = progress_callback.call_args[0]
            assert len(call_args) == 3  # processed, total, progress_data
            assert call_args[0] > 0  # processed
            assert call_args[1] > 0  # total
            assert isinstance(call_args[2], dict)  # progress_data

    @pytest.mark.asyncio
    async def test_scanner_statistics(self, entity_scanner, temp_project_dir):
        """Test scanner statistics collection."""
        # Initial stats
        initial_stats = entity_scanner.get_scanner_stats()
        assert initial_stats["performance"]["total_scans"] == 0
        assert initial_stats["performance"]["total_entities_processed"] == 0
        
        # Create and run a scan
        test_file = temp_project_dir / "stats_test.py"
        test_file.write_text("def stats_func():\n    pass")
        
        mock_result = self.create_mock_parse_result(test_file)
        mock_stats = PipelineStats(
            total_files=1,
            processed_files=1,
            successful_files=1,
            failed_files=0,
            total_entities=1,
            total_relations=0,
            total_time=0.1
        )
        
        entity_scanner.parser_pipeline.parse_files = Mock(return_value=([mock_result], mock_stats))
        
        request = EntityScanRequest(
            file_paths=[test_file],
            collection_name="test-collection",
            project_path=temp_project_dir,
            scan_mode="parse_only"  # Skip entity processing for speed
        )
        
        await entity_scanner.scan_files(request)
        
        # Check updated stats
        updated_stats = entity_scanner.get_scanner_stats()
        assert updated_stats["performance"]["total_scans"] == 1
        assert updated_stats["performance"]["total_entities_processed"] == 1
        assert updated_stats["performance"]["last_scan_time"] is not None
        
        # Check scanner info
        assert updated_stats["scanner_info"]["enable_parallel"] is True
        assert updated_stats["scanner_info"]["default_batch_size"] == 10
        assert updated_stats["scanner_info"]["max_workers"] == 2

    @pytest.mark.asyncio
    async def test_scan_request_validation(self, entity_scanner):
        """Test scan request validation and error handling."""
        # Test with empty file list
        empty_request = EntityScanRequest(
            file_paths=[],
            collection_name="test-collection",
            project_path=Path("/tmp")
        )
        
        result = await entity_scanner.scan_files(empty_request)
        assert result.total_files == 0
        assert result.success_rate == 0

    @pytest.mark.asyncio
    async def test_entity_processing_error_handling(self, entity_scanner, temp_project_dir):
        """Test error handling during entity processing."""
        # Create test file
        test_file = temp_project_dir / "error_test.py"
        test_file.write_text("def error_func():\n    pass")
        
        entities = [self.create_test_entity("error_func", test_file)]
        
        # Mock storage failure
        entity_scanner.storage_client.upsert_points.return_value = StorageResult.failed_operation(
            "upsert", "test-collection", "Storage error", 0.1
        )
        
        # Mock file content reading
        with patch.object(Path, 'read_text', return_value="def error_func():\n    pass"):
            # This should handle the error gracefully
            await entity_scanner._process_entities_in_batches(
                entities,
                "test-collection",
                batch_size=1
            )
        
        # Verify storage was attempted
        entity_scanner.storage_client.upsert_points.assert_called()

    @pytest.mark.asyncio
    async def test_performance_metrics_storage(self, entity_scanner):
        """Test performance metrics storage and history."""
        # Check initial empty history
        stats = entity_scanner.get_scanner_stats()
        assert len(stats["recent_performance"]) == 0
        
        # Create a mock scan result
        result = EntityScanResult(
            request_id="test_perf",
            total_files=5,
            processed_files=5,
            successful_files=5,
            failed_files=0,
            total_entities=10,
            total_relations=5,
            scan_time=1.0,
            entities_per_second=10.0,
            success_rate=1.0,
            performance_metrics={"parallel_processing": True}
        )
        
        # Store metrics
        entity_scanner._store_performance_metrics(result)
        
        # Check history updated
        updated_stats = entity_scanner.get_scanner_stats()
        assert len(updated_stats["recent_performance"]) == 1
        
        recent_perf = updated_stats["recent_performance"][0]
        assert recent_perf["request_id"] == "test_perf"
        assert recent_perf["files_processed"] == 5
        assert recent_perf["entities_processed"] == 10
        assert recent_perf["entities_per_second"] == 10.0
        assert recent_perf["success_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_deterministic_id_integration(self, entity_scanner, temp_project_dir):
        """Test integration with deterministic entity ID system."""
        # Create test file
        test_file = temp_project_dir / "det_id_test.py"
        test_content = "def deterministic_func():\n    pass"
        test_file.write_text(test_content)
        
        entity = self.create_test_entity("deterministic_func", test_file)
        
        # Mock file content reading for deterministic ID generation
        with patch.object(Path, 'read_text', return_value=test_content):
            # Process single entity batch
            batch = EntityBatch(
                batch_id=0,
                entities=[entity],
                file_path=str(test_file),
                batch_size=1
            )
            
            await entity_scanner._process_entity_batch(batch, "test-collection")
            
            # Verify batch succeeded
            assert batch.success is True
            assert batch.processing_time > 0
            
            # Verify storage was called
            entity_scanner.storage_client.upsert_points.assert_called_once()

    def test_entity_scan_result_serialization(self):
        """Test EntityScanResult serialization."""
        result = EntityScanResult(
            request_id="test_serialize",
            total_files=10,
            processed_files=10,
            successful_files=8,
            failed_files=2,
            total_entities=25,
            total_relations=15,
            scan_time=2.5,
            entities_per_second=10.0,
            success_rate=0.8,
            errors=[{"file": "test.py", "error": "syntax error"}],
            performance_metrics={"parallel_processing": True, "batch_size": 10}
        )
        
        # Test serialization
        result_dict = result.to_dict()
        
        # Verify structure
        assert result_dict["request_id"] == "test_serialize"
        assert result_dict["summary"]["total_files"] == 10
        assert result_dict["summary"]["success_rate"] == 0.8
        assert result_dict["entities"]["total_entities"] == 25
        assert result_dict["entities"]["entities_per_second"] == 10.0
        assert result_dict["performance"]["scan_time"] == 2.5
        assert result_dict["performance"]["parallel_processing"] is True
        assert len(result_dict["errors"]) == 1


class TestEntityScannerEdgeCases:
    """Test edge cases and error conditions for EntityScanner."""
    
    @pytest.fixture
    def entity_scanner(self):
        """Create minimal EntityScanner for edge case testing."""
        mock_client = Mock(spec=HybridQdrantClient)
        mock_client.upsert_points = AsyncMock(return_value=StorageResult.successful_insert("test", 1, 10.0))
        return EntityScanner(storage_client=mock_client, enable_parallel=False)
    
    @pytest.mark.asyncio
    async def test_scan_with_exception(self, entity_scanner):
        """Test handling of unexpected exceptions during scanning."""
        # For sequential processing (which this scanner uses), mock the parser registry
        mock_parser = Mock()
        mock_parser.parse_file.side_effect = Exception("Pipeline error")
        
        with patch.object(parser_registry, 'get_parser_for_file', return_value=mock_parser):
            request = EntityScanRequest(
                file_paths=[Path("/test/file.py")],  # Use a valid-looking path
                collection_name="test-collection",
                project_path=Path("/tmp")
            )
            
            result = await entity_scanner.scan_files(request)
            
            # Should handle exception gracefully
            assert result.total_files == 1
            assert result.successful_files == 0
            assert result.failed_files == 1
            assert result.success_rate == 0
            # Sequential processing creates empty parse results on exception
            # so no errors are added to the error list in this case
    
    @pytest.mark.asyncio 
    async def test_empty_entity_batch_processing(self, entity_scanner):
        """Test processing empty entity batches."""
        # Should handle empty list gracefully
        await entity_scanner._process_entities_in_batches(
            [],
            "test-collection",
            batch_size=10
        )
        
        # Storage should not be called for empty batch
        entity_scanner.storage_client.upsert_points.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_very_large_entity_batch(self, entity_scanner):
        """Test processing very large entity batches."""
        # Create large number of entities
        large_entity_list = []
        for i in range(1000):
            entity = Entity(
                id=f"file::/large_batch.py::function::func_{i}::{i}",
                name=f"func_{i}",
                qualified_name=f"func_{i}",
                entity_type=EntityType.FUNCTION,
                location=SourceLocation(
                    file_path=Path("/large_batch.py"),
                    start_line=i, start_column=0, end_line=i+1, end_column=10,
                    start_byte=i*20, end_byte=(i+1)*20
                ),
                source_code=f"def func_{i}(): pass"
            )
            large_entity_list.append(entity)
        
        # Mock file content reading
        with patch.object(Path, 'read_text', return_value="# Large batch file"):
            # Process large batch
            await entity_scanner._process_entities_in_batches(
                large_entity_list,
                "test-collection",
                batch_size=100  # 10 batches total
            )
        
        # Should process in multiple batches
        assert entity_scanner.storage_client.upsert_points.call_count == 10
    
    @pytest.mark.asyncio
    async def test_concurrent_scanning(self, entity_scanner):
        """Test concurrent scanning operations."""
        # Create multiple scan requests
        requests = []
        for i in range(3):
            request = EntityScanRequest(
                file_paths=[Path(f"/test/concurrent_{i}.py")],
                collection_name=f"collection_{i}",
                project_path=Path("/tmp")
            )
            requests.append(request)
        
        # Mock successful results
        entity_scanner.parser_pipeline.parse_files = Mock(return_value=([], PipelineStats()))
        
        # Run concurrent scans
        tasks = [entity_scanner.scan_files(req) for req in requests]
        results = await asyncio.gather(*tasks)
        
        # All should complete successfully
        assert len(results) == 3
        assert all(isinstance(r, EntityScanResult) for r in results)
    
    def test_batch_properties_edge_cases(self):
        """Test EntityBatch edge cases."""
        # Empty batch
        empty_batch = EntityBatch(
            batch_id=0,
            entities=[],
            file_path="empty.py",
            batch_size=0
        )
        assert empty_batch.entity_count == 0
        
        # Batch with error
        error_batch = EntityBatch(
            batch_id=1,
            entities=[],
            file_path="error.py",
            batch_size=0,
            success=False,
            error_message="Processing failed"
        )
        assert error_batch.success is False
        assert error_batch.error_message == "Processing failed"
    
    @pytest.mark.asyncio
    async def test_streaming_with_parse_failures(self, entity_scanner):
        """Test entity streaming with parse failures."""
        test_files = [Path("/test/fail1.py"), Path("/test/fail2.py")]
        
        # Mock parser to return None (no parser available)
        with patch.object(parser_registry, 'get_parser_for_file', return_value=None):
            batches = []
            async for batch in entity_scanner.stream_entities(test_files, batch_size=1):
                batches.append(batch)
            
            # Should handle gracefully with no entities yielded
            assert len(batches) == 0
    
    @pytest.mark.asyncio
    async def test_streaming_parser_exceptions(self, entity_scanner):
        """Test streaming with parser exceptions."""
        test_files = [Path("/test/exception.py")]
        
        # Mock parser that raises exception
        mock_parser = Mock()
        mock_parser.parse_file.side_effect = Exception("Parse error")
        
        with patch.object(parser_registry, 'get_parser_for_file', return_value=mock_parser):
            batches = []
            async for batch in entity_scanner.stream_entities(test_files, batch_size=1):
                batches.append(batch)
            
            # Should handle exception and yield no batches
            assert len(batches) == 0
    
    def test_performance_history_limit(self, entity_scanner):
        """Test performance history size limiting."""
        # Fill history beyond max size
        for i in range(150):  # More than max_history_size (100)
            result = EntityScanResult(
                request_id=f"test_{i}",
                total_files=1,
                processed_files=1,
                successful_files=1,
                failed_files=0,
                total_entities=1,
                total_relations=0,
                scan_time=0.1,
                entities_per_second=10.0,
                success_rate=1.0
            )
            entity_scanner._store_performance_metrics(result)
        
        # Should maintain max size
        assert len(entity_scanner._performance_history) <= entity_scanner._max_history_size
        
        # Should keep most recent entries
        recent_metrics = entity_scanner._performance_history[-1]
        assert recent_metrics["request_id"] == "test_149"