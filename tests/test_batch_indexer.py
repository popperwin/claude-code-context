"""
Unit tests for BatchIndexer functionality.

Tests batch processing, progress tracking, error handling, and performance metrics.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, AsyncGenerator

from core.storage.indexing import BatchIndexer, IndexingProgress, IndexingResult
from core.storage.client import HybridQdrantClient
from core.models.entities import Entity, EntityType, SourceLocation, Visibility
from core.models.storage import QdrantPoint, StorageResult
from core.embeddings.base import BaseEmbedder


class MockEmbedder(BaseEmbedder):
    """Mock embedder for testing"""
    
    def __init__(self):
        super().__init__()
        self._is_loaded = True
    
    @property
    def model_name(self) -> str:
        return "mock-model"
    
    @property
    def dimensions(self) -> int:
        return 1024
    
    @property
    def max_sequence_length(self) -> int:
        return 512
    
    async def load_model(self) -> bool:
        self._is_loaded = True
        return True
    
    async def unload_model(self) -> None:
        self._is_loaded = False
    
    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings"""
        return [[0.1, 0.2, 0.3, 0.4] * 256 for _ in texts]


@pytest.fixture
def mock_embedder():
    """Create mock embedder"""
    return MockEmbedder()


@pytest.fixture
def mock_client():
    """Create mock HybridQdrantClient"""
    client = Mock(spec=HybridQdrantClient)
    client.embedder = None
    return client


@pytest.fixture
def sample_entities():
    """Create sample entities for testing"""
    from pathlib import Path
    entities = []
    
    for i in range(5):
        entity = Entity(
            id=f"test_{i}.py::test_function_{i}",
            name=f"test_function_{i}",
            qualified_name=f"module.test_function_{i}",
            entity_type=EntityType.FUNCTION,
            signature=f"def test_function_{i}():",
            docstring=f"Test function {i} description",
            source_code=f"def test_function_{i}():\n    return {i}",
            location=SourceLocation(
                file_path=Path(f"test_{i}.py"),
                start_line=i * 10,
                end_line=i * 10 + 2,
                start_column=0,
                end_column=20,
                start_byte=i * 100,
                end_byte=i * 100 + 50
            ),
            visibility=Visibility.PUBLIC,
            is_async=False,
            source_hash=f"hash_{i}"
        )
        entities.append(entity)
    
    return entities


class TestIndexingProgress:
    """Test IndexingProgress data class"""
    
    def test_progress_initialization(self):
        """Test progress object initialization"""
        progress = IndexingProgress(
            total_entities=100,
            processed_entities=25,
            successful_entities=20,
            failed_entities=5,
            current_batch=3,
            total_batches=10,
            elapsed_time=30.0,
            estimated_remaining_time=90.0,
            entities_per_second=0.83
        )
        
        assert progress.total_entities == 100
        assert progress.processed_entities == 25
        assert progress.successful_entities == 20
        assert progress.failed_entities == 5
        assert progress.current_batch == 3
        assert progress.total_batches == 10
        assert progress.elapsed_time == 30.0
        assert progress.estimated_remaining_time == 90.0
        assert progress.entities_per_second == 0.83
    
    def test_progress_percentage(self):
        """Test progress percentage calculation"""
        progress = IndexingProgress(
            total_entities=100,
            processed_entities=25,
            successful_entities=20,
            failed_entities=5,
            current_batch=3,
            total_batches=10,
            elapsed_time=30.0,
            estimated_remaining_time=90.0,
            entities_per_second=0.83
        )
        
        assert progress.progress_percentage == 25.0
        
        # Test edge case - no entities
        progress_empty = IndexingProgress(
            total_entities=0,
            processed_entities=0,
            successful_entities=0,
            failed_entities=0,
            current_batch=0,
            total_batches=0,
            elapsed_time=0.0,
            estimated_remaining_time=0.0,
            entities_per_second=0.0
        )
        
        assert progress_empty.progress_percentage == 100.0
    
    def test_success_rate(self):
        """Test success rate calculation"""
        progress = IndexingProgress(
            total_entities=100,
            processed_entities=25,
            successful_entities=20,
            failed_entities=5,
            current_batch=3,
            total_batches=10,
            elapsed_time=30.0,
            estimated_remaining_time=90.0,
            entities_per_second=0.83
        )
        
        assert progress.success_rate == 80.0
        
        # Test edge case - no processed entities
        progress_empty = IndexingProgress(
            total_entities=100,
            processed_entities=0,
            successful_entities=0,
            failed_entities=0,
            current_batch=0,
            total_batches=10,
            elapsed_time=0.0,
            estimated_remaining_time=0.0,
            entities_per_second=0.0
        )
        
        assert progress_empty.success_rate == 100.0


class TestIndexingResult:
    """Test IndexingResult data class"""
    
    def test_result_initialization(self):
        """Test result object initialization"""
        result = IndexingResult(
            total_entities=100,
            successful_entities=95,
            failed_entities=5,
            total_time=120.0,
            average_time_per_entity=1.2,
            entities_per_second=0.83,
            errors=["Error 1", "Error 2"],
            collection_name="test-collection"
        )
        
        assert result.total_entities == 100
        assert result.successful_entities == 95
        assert result.failed_entities == 5
        assert result.total_time == 120.0
        assert result.average_time_per_entity == 1.2
        assert result.entities_per_second == 0.83
        assert result.errors == ["Error 1", "Error 2"]
        assert result.collection_name == "test-collection"
    
    def test_success_rate_calculation(self):
        """Test success rate calculation"""
        result = IndexingResult(
            total_entities=100,
            successful_entities=95,
            failed_entities=5,
            total_time=120.0,
            average_time_per_entity=1.2,
            entities_per_second=0.83,
            errors=[],
            collection_name="test-collection"
        )
        
        assert result.success_rate == 95.0
        
        # Test edge case - no entities
        result_empty = IndexingResult(
            total_entities=0,
            successful_entities=0,
            failed_entities=0,
            total_time=0.0,
            average_time_per_entity=0.0,
            entities_per_second=0.0,
            errors=[],
            collection_name="test-collection"
        )
        
        assert result_empty.success_rate == 100.0
    
    def test_to_dict_conversion(self):
        """Test conversion to dictionary"""
        result = IndexingResult(
            total_entities=100,
            successful_entities=95,
            failed_entities=5,
            total_time=120.5,
            average_time_per_entity=1.205,
            entities_per_second=0.83,
            errors=["Error 1"],
            collection_name="test-collection"
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["total_entities"] == 100
        assert result_dict["successful_entities"] == 95
        assert result_dict["failed_entities"] == 5
        assert result_dict["success_rate"] == 95.0
        assert result_dict["total_time_s"] == 120.5
        assert result_dict["average_time_per_entity_ms"] == 1205.0
        assert result_dict["entities_per_second"] == 0.83
        assert result_dict["collection_name"] == "test-collection"
        assert result_dict["errors"] == ["Error 1"]


class TestBatchIndexer:
    """Test BatchIndexer functionality"""
    
    def test_indexer_initialization(self, mock_client, mock_embedder):
        """Test indexer initialization"""
        # Default initialization
        indexer = BatchIndexer(mock_client)
        assert indexer.client == mock_client
        assert indexer.embedder is None  # Uses client's embedder
        assert indexer.batch_size == 100
        assert indexer.max_retries == 3
        assert indexer.retry_delay == 1.0
        
        # Custom initialization
        indexer = BatchIndexer(
            mock_client,
            embedder=mock_embedder,
            batch_size=50,
            max_retries=5,
            retry_delay=2.0
        )
        assert indexer.embedder == mock_embedder
        assert indexer.batch_size == 50
        assert indexer.max_retries == 5
        assert indexer.retry_delay == 2.0
    
    def test_progress_callback_management(self, mock_client):
        """Test progress callback management"""
        indexer = BatchIndexer(mock_client)
        
        callback1 = Mock()
        callback2 = Mock()
        
        # Add callbacks
        indexer.add_progress_callback(callback1)
        indexer.add_progress_callback(callback2)
        
        assert len(indexer._progress_callbacks) == 2
        assert callback1 in indexer._progress_callbacks
        assert callback2 in indexer._progress_callbacks
        
        # Remove callback
        indexer.remove_progress_callback(callback1)
        
        assert len(indexer._progress_callbacks) == 1
        assert callback1 not in indexer._progress_callbacks
        assert callback2 in indexer._progress_callbacks
    
    @pytest.mark.asyncio
    async def test_index_entities_empty(self, mock_client):
        """Test indexing empty entity list"""
        indexer = BatchIndexer(mock_client)
        
        result = await indexer.index_entities([], "test-collection")
        
        assert result.total_entities == 0
        assert result.successful_entities == 0
        assert result.failed_entities == 0
        assert result.total_time == 0.0
        assert result.average_time_per_entity == 0.0
        assert result.entities_per_second == 0.0
        assert result.errors == []
        assert result.collection_name == "test-collection"
    
    @pytest.mark.asyncio
    async def test_index_entities_success(self, mock_client, mock_embedder, sample_entities):
        """Test successful entity indexing"""
        indexer = BatchIndexer(mock_client, embedder=mock_embedder, batch_size=2)
        
        # Mock successful upsert - should match actual batch sizes: 2, 2, 1
        batch_results = [
            StorageResult.successful_insert("test-collection", 2, 100.0),  # First batch: 2 entities
            StorageResult.successful_insert("test-collection", 2, 100.0),  # Second batch: 2 entities 
            StorageResult.successful_insert("test-collection", 1, 100.0),  # Third batch: 1 entity
        ]
        mock_client.upsert_points = AsyncMock(side_effect=batch_results)
        
        result = await indexer.index_entities(
            sample_entities, "test-collection", show_progress=False
        )
        
        assert result.total_entities == 5
        assert result.successful_entities == 5  # All 5 entities should succeed
        assert result.failed_entities == 0
        assert result.total_time > 0
        assert result.entities_per_second > 0
        assert result.collection_name == "test-collection"
        assert len(result.errors) == 0
    
    @pytest.mark.asyncio
    async def test_index_entities_with_failures(self, mock_client, mock_embedder, sample_entities):
        """Test entity indexing with some failures"""
        indexer = BatchIndexer(mock_client, embedder=mock_embedder, batch_size=2, max_retries=1)
        
        # Mock mixed success/failure - account for retry logic
        # First batch: succeeds on first try
        # Second batch: fails on first try, fails on retry too  
        # Third batch: succeeds on first try
        batch_results = [
            StorageResult.successful_insert("test-collection", 2, 100.0),  # First batch: 2 entities succeed
            StorageResult.failed_operation("upsert", "test-collection", "Test failure", 100.0),  # Second batch first attempt: fails
            StorageResult.failed_operation("upsert", "test-collection", "Test failure retry", 100.0),  # Second batch retry: fails again
            StorageResult.successful_insert("test-collection", 1, 100.0),  # Third batch: 1 entity succeeds
        ]
        
        mock_client.upsert_points = AsyncMock(side_effect=batch_results)
        
        result = await indexer.index_entities(
            sample_entities, "test-collection", show_progress=False
        )
        
        assert result.total_entities == 5
        assert result.successful_entities == 3  # First batch (2) + Third batch (1) = 3
        assert result.failed_entities == 2  # Second batch fails with 2 entities
        assert len(result.errors) > 0
        assert any("Test failure" in str(error) for error in result.errors)
    
    @pytest.mark.asyncio
    async def test_entity_to_text_conversion(self, mock_client, sample_entities):
        """Test entity to text conversion"""
        from pathlib import Path
        indexer = BatchIndexer(mock_client)
        
        entity = sample_entities[0]
        text = indexer._entity_to_text(entity)
        
        assert f"Type: {entity.entity_type.value}" in text
        assert f"Name: {entity.name}" in text
        assert f"Qualified: {entity.qualified_name}" in text
        assert f"Signature: {entity.signature}" in text
        assert f"Description: {entity.docstring}" in text
        assert f"Code: {entity.source_code}" in text
        assert f"File: test_0.py" in text
        assert f"Visibility: {entity.visibility.value}" in text
        
        # Test with minimal entity
        minimal_entity = Entity(
            id="minimal.py::minimal_func",
            name="minimal_func",
            qualified_name="minimal_func",  # Required field
            entity_type=EntityType.FUNCTION,
            source_code="def minimal_func(): pass",  # Required field
            location=SourceLocation(
                file_path=Path("minimal.py"),  # Use Path object
                start_line=1,
                end_line=1,
                start_column=0,
                end_column=10,
                start_byte=0,
                end_byte=10
            )
        )
        
        minimal_text = indexer._entity_to_text(minimal_entity)
        assert "Name: minimal_func" in minimal_text
        assert "File: minimal.py" in minimal_text
    
    @pytest.mark.asyncio
    async def test_process_batch_no_embedder(self, mock_client, sample_entities):
        """Test batch processing without embedder"""
        indexer = BatchIndexer(mock_client, embedder=None)
        
        # Mock successful upsert
        mock_client.upsert_points = AsyncMock(return_value=StorageResult.successful_insert(
            "test-collection", 2, 100.0
        ))
        
        # Process a small batch
        batch_result = await indexer._process_batch(
            sample_entities[:2], "test-collection", 0
        )
        
        assert batch_result.successful_count == 2
        assert batch_result.failed_count == 0
        assert len(batch_result.errors) == 0
        
        # Verify upsert was called with zero embeddings
        mock_client.upsert_points.assert_called_once()
        call_args = mock_client.upsert_points.call_args[0]
        points = call_args[1]  # Second argument is points list
        
        # Check that points have zero embeddings
        for point in points:
            assert point.vector == [0.0] * 1024
    
    @pytest.mark.asyncio
    async def test_process_batch_with_retries(self, mock_client, mock_embedder, sample_entities):
        """Test batch processing with retry logic"""
        indexer = BatchIndexer(mock_client, embedder=mock_embedder, max_retries=2, retry_delay=0.1)
        
        # Mock failure then success
        failed_result = StorageResult.failed_operation(
            "upsert", "test-collection", "Temporary failure", 100.0
        )
        successful_result = StorageResult.successful_insert("test-collection", 2, 100.0)
        
        mock_client.upsert_points = AsyncMock(side_effect=[failed_result, successful_result])
        
        batch_result = await indexer._process_batch_with_retries(
            sample_entities[:2], "test-collection", 0
        )
        
        assert batch_result.successful_count == 2
        assert batch_result.failed_count == 0
        assert len(batch_result.errors) == 0
        
        # Should have been called twice (failure then success)
        assert mock_client.upsert_points.call_count == 2
    
    @pytest.mark.asyncio
    async def test_process_batch_all_retries_fail(self, mock_client, mock_embedder, sample_entities):
        """Test batch processing when all retries fail"""
        indexer = BatchIndexer(mock_client, embedder=mock_embedder, max_retries=2, retry_delay=0.1)
        
        # Mock persistent failure
        failed_result = StorageResult.failed_operation(
            "upsert", "test-collection", "Persistent failure", 100.0
        )
        mock_client.upsert_points = AsyncMock(side_effect=Exception("Connection error"))
        
        batch_result = await indexer._process_batch_with_retries(
            sample_entities[:2], "test-collection", 0
        )
        
        assert batch_result.successful_count == 0
        assert batch_result.failed_count == 2
        assert len(batch_result.errors) == 1
        assert "Connection error" in batch_result.errors[0]
        
        # Should have been called 3 times (initial + 2 retries)
        assert mock_client.upsert_points.call_count == 3
    
    @pytest.mark.asyncio
    async def test_index_entities_stream(self, mock_client, mock_embedder, sample_entities):
        """Test streaming entity indexing"""
        indexer = BatchIndexer(mock_client, embedder=mock_embedder, batch_size=2)
        
        # Mock successful upsert
        mock_client.upsert_points = AsyncMock(return_value=StorageResult.successful_insert(
            "test-collection", 2, 100.0
        ))
        
        # Create async generator
        async def entity_generator():
            for entity in sample_entities:
                yield entity
        
        result = await indexer.index_entities_stream(
            entity_generator(), "test-collection", show_progress=False
        )
        
        assert result.total_entities == 5
        assert result.successful_entities == 5  # All entities should succeed
        assert result.failed_entities == 0
        assert result.total_time > 0
        assert result.entities_per_second > 0
        assert result.collection_name == "test-collection"
    
    @pytest.mark.asyncio
    async def test_progress_callback_execution(self, mock_client, mock_embedder, sample_entities):
        """Test that progress callbacks are executed"""
        indexer = BatchIndexer(mock_client, embedder=mock_embedder, batch_size=2)
        
        # Mock successful upsert
        mock_client.upsert_points = AsyncMock(return_value=StorageResult.successful_insert(
            "test-collection", 2, 100.0
        ))
        
        # Add progress callback
        progress_updates = []
        def progress_callback(progress: IndexingProgress):
            progress_updates.append(progress)
        
        indexer.add_progress_callback(progress_callback)
        
        await indexer.index_entities(
            sample_entities, "test-collection", show_progress=False
        )
        
        # Should have received progress updates
        assert len(progress_updates) > 0
        
        # Verify progress updates have correct structure
        for progress in progress_updates:
            assert isinstance(progress, IndexingProgress)
            assert progress.total_entities == 5
            assert progress.current_batch > 0
            assert progress.total_batches == 3
    
    @pytest.mark.asyncio
    async def test_progress_callback_error_handling(self, mock_client, mock_embedder, sample_entities):
        """Test that progress callback errors don't break indexing"""
        indexer = BatchIndexer(mock_client, embedder=mock_embedder, batch_size=2)
        
        # Mock successful upsert
        mock_client.upsert_points = AsyncMock(return_value=StorageResult.successful_insert(
            "test-collection", 2, 100.0
        ))
        
        # Add faulty progress callback
        def faulty_callback(progress: IndexingProgress):
            raise Exception("Callback error")
        
        indexer.add_progress_callback(faulty_callback)
        
        # Indexing should still complete successfully
        result = await indexer.index_entities(
            sample_entities, "test-collection", show_progress=False
        )
        
        assert result.total_entities == 5
        assert result.successful_entities == 5
    
    def test_performance_metrics(self, mock_client):
        """Test performance metrics tracking"""
        indexer = BatchIndexer(mock_client)
        
        metrics = indexer.get_performance_metrics()
        
        assert "total_indexed" in metrics
        assert "total_indexing_time_s" in metrics
        assert "failed_batches" in metrics
        assert "average_indexing_speed" in metrics
        assert "batch_size" in metrics
        assert "max_retries" in metrics
        assert "retry_delay_s" in metrics
        
        assert metrics["total_indexed"] == 0
        assert metrics["failed_batches"] == 0
        assert metrics["batch_size"] == 100
        assert metrics["max_retries"] == 3
        assert metrics["retry_delay_s"] == 1.0