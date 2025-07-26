"""
Unit tests for storage models.

Tests storage operations, results, and Qdrant integration models.
"""

import pytest
from datetime import datetime
from core.models.storage import (
    OperationStatus, OperationResult, QdrantPoint, SearchResult,
    StorageResult, CollectionInfo
)


class TestOperationResult:
    """Test OperationResult model"""
    
    def test_success_result_creation(self):
        """Test creating a successful operation result"""
        data = {"test": "value"}
        result = OperationResult.success_result(
            data=data,
            operation_type="test_operation",
            processing_time_ms=150.5,
            items_processed=5
        )
        
        assert result.status == OperationStatus.SUCCESS
        assert result.success is True
        assert result.data == data
        assert result.operation_type == "test_operation"
        assert result.processing_time_ms == 150.5
        assert result.items_processed == 5
        assert result.items_succeeded == 5
        assert result.success_rate == 1.0
    
    def test_error_result_creation(self):
        """Test creating an error operation result"""
        result = OperationResult.error_result(
            error="Test error",
            operation_type="test_operation",
            error_code="TEST_ERROR",
            error_details={"detail": "test"},
            processing_time_ms=50.0
        )
        
        assert result.status == OperationStatus.FAILED
        assert result.success is False
        assert result.error == "Test error"
        assert result.error_code == "TEST_ERROR"
        assert result.error_details == {"detail": "test"}
        assert result.operation_type == "test_operation"
        assert result.processing_time_ms == 50.0
    
    def test_success_rate_calculation(self):
        """Test success rate calculation"""
        # 100% success
        result = OperationResult(
            status=OperationStatus.SUCCESS,
            operation_type="test",
            items_processed=10,
            items_succeeded=10
        )
        assert result.success_rate == 1.0
        
        # 50% success
        result = OperationResult(
            status=OperationStatus.PARTIAL,
            operation_type="test", 
            items_processed=10,
            items_succeeded=5
        )
        assert result.success_rate == 0.5
        
        # No items processed
        result = OperationResult(
            status=OperationStatus.SUCCESS,
            operation_type="test",
            items_processed=0,
            items_succeeded=0
        )
        assert result.success_rate == 1.0  # Success case


class TestQdrantPoint:
    """Test QdrantPoint model"""
    
    def test_valid_qdrant_point(self):
        """Test creating a valid Qdrant point"""
        vector = [0.1] * 1024  # Stella dimensions
        payload = {
            "entity_id": "test_entity",
            "entity_type": "function",
            "file_path": "test.py"
        }
        
        point = QdrantPoint(
            id="point_1",
            vector=vector,
            payload=payload
        )
        
        assert point.id == "point_1"
        assert len(point.vector) == 1024
        assert point.entity_id == "test_entity"
        assert point.entity_type == "function"
        assert point.file_path == "test.py"
        assert point.version == 1  # default
    
    def test_point_id_validation(self):
        """Test point ID validation"""
        vector = [0.1] * 1024
        payload = {
            "entity_id": "test_entity",
            "entity_type": "function",
            "file_path": "test.py"
        }
        
        with pytest.raises(ValueError, match="Point ID cannot be empty"):
            QdrantPoint(
                id="",
                vector=vector,
                payload=payload
            )
    
    def test_vector_validation(self):
        """Test vector validation"""
        payload = {
            "entity_id": "test_entity",
            "entity_type": "function",
            "file_path": "test.py"
        }
        
        # Empty vector
        with pytest.raises(ValueError, match="Vector cannot be empty"):
            QdrantPoint(
                id="point_1",
                vector=[],
                payload=payload
            )
        
        # Wrong dimensions
        with pytest.raises(ValueError, match="Vector must have 1024 dimensions"):
            QdrantPoint(
                id="point_1",
                vector=[0.1] * 512,  # Wrong size
                payload=payload
            )
    
    def test_payload_validation(self):
        """Test payload validation"""
        vector = [0.1] * 1024
        
        # Missing required fields
        incomplete_payloads = [
            {"entity_id": "test"},  # Missing entity_type, file_path
            {"entity_type": "function"},  # Missing entity_id, file_path
            {"file_path": "test.py"},  # Missing entity_id, entity_type
        ]
        
        for payload in incomplete_payloads:
            with pytest.raises(ValueError, match="Payload missing required fields"):
                QdrantPoint(
                    id="point_1",
                    vector=vector,
                    payload=payload
                )
    
    def test_qdrant_format_conversion(self):
        """Test conversion to/from Qdrant format"""
        vector = [0.1] * 1024
        payload = {
            "entity_id": "test_entity",
            "entity_type": "function",
            "file_path": "test.py"
        }
        
        point = QdrantPoint(
            id="point_1",
            vector=vector,
            payload=payload
        )
        
        # Test to_qdrant_format
        qdrant_format = point.to_qdrant_format()
        expected_format = {
            "id": "point_1",
            "vector": vector,
            "payload": payload
        }
        assert qdrant_format == expected_format
        
        # Test from_qdrant_format
        restored_point = QdrantPoint.from_qdrant_format(qdrant_format)
        assert restored_point.id == point.id
        assert restored_point.vector == point.vector
        assert restored_point.payload == point.payload


class TestSearchResult:
    """Test SearchResult model"""
    
    def test_valid_search_result(self):
        """Test creating a valid search result"""
        vector = [0.1] * 1024
        payload = {
            "entity_id": "test_entity",
            "entity_type": "function",
            "file_path": "test.py"
        }
        
        point = QdrantPoint(
            id="point_1",
            vector=vector,
            payload=payload
        )
        
        result = SearchResult(
            point=point,
            score=0.85,
            query="test query",
            search_type="semantic",
            rank=1,
            total_results=10
        )
        
        assert result.score == 0.85
        assert result.query == "test query"
        assert result.search_type == "semantic"
        assert result.rank == 1
        assert result.total_results == 10
        assert result.entity_id == "test_entity"
        assert result.entity_type == "function"
        assert result.file_path == "test.py"
    
    def test_score_validation(self):
        """Test score validation and rounding"""
        vector = [0.1] * 1024
        payload = {
            "entity_id": "test_entity",
            "entity_type": "function",
            "file_path": "test.py"
        }
        
        point = QdrantPoint(
            id="point_1",
            vector=vector,
            payload=payload
        )
        
        # Valid scores
        for score in [0.0, 0.5, 1.0]:
            result = SearchResult(
                point=point,
                score=score,
                query="test",
                search_type="semantic",
                rank=1,
                total_results=1
            )
            assert result.score == score
        
        # Score precision rounding
        result = SearchResult(
            point=point,
            score=0.123456789,
            query="test",
            search_type="semantic",
            rank=1,
            total_results=1
        )
        assert result.score == 0.123457  # Rounded to 6 decimal places
        
        # Invalid scores
        with pytest.raises(ValueError):
            SearchResult(
                point=point,
                score=-0.1,  # Below 0
                query="test",
                search_type="semantic",
                rank=1,
                total_results=1
            )
        
        with pytest.raises(ValueError):
            SearchResult(
                point=point,
                score=1.1,  # Above 1
                query="test",
                search_type="semantic",
                rank=1,
                total_results=1
            )
    
    def test_relevance_properties(self):
        """Test relevance assessment properties"""
        vector = [0.1] * 1024
        payload = {
            "entity_id": "test_entity",
            "entity_type": "function",
            "file_path": "test.py"
        }
        
        point = QdrantPoint(
            id="point_1",
            vector=vector,
            payload=payload
        )
        
        # Highly relevant (>0.8)
        result = SearchResult(
            point=point,
            score=0.9,
            query="test",
            search_type="semantic",
            rank=1,
            total_results=1
        )
        assert result.is_highly_relevant is True
        assert result.is_moderately_relevant is True
        
        # Moderately relevant (>0.6)
        result = SearchResult(
            point=point,
            score=0.7,
            query="test",
            search_type="semantic",
            rank=1,
            total_results=1
        )
        assert result.is_highly_relevant is False
        assert result.is_moderately_relevant is True
        
        # Low relevance
        result = SearchResult(
            point=point,
            score=0.3,
            query="test",
            search_type="semantic",
            rank=1,
            total_results=1
        )
        assert result.is_highly_relevant is False
        assert result.is_moderately_relevant is False


class TestStorageResult:
    """Test StorageResult model"""
    
    def test_valid_storage_result(self):
        """Test creating a valid storage result"""
        result = StorageResult(
            operation="insert",
            collection_name="test-collection",
            success=True,
            processing_time_ms=250.5,
            affected_count=5,
            total_count=5
        )
        
        assert result.operation == "insert"
        assert result.collection_name == "test-collection"
        assert result.success is True
        assert result.processing_time_ms == 250.5
        assert result.affected_count == 5
        assert result.total_count == 5
        assert result.success_rate == 1.0
        assert result.throughput_per_second > 0
    
    def test_operation_validation(self):
        """Test operation type validation"""
        valid_operations = [
            "insert", "update", "delete", "search", "bulk_insert",
            "bulk_update", "bulk_delete", "upsert", "create_collection"
        ]
        
        for operation in valid_operations:
            result = StorageResult(
                operation=operation,
                collection_name="test-collection",
                success=True,
                processing_time_ms=100.0
            )
            assert result.operation == operation.lower()
        
        # Invalid operation
        with pytest.raises(ValueError, match="Invalid operation"):
            StorageResult(
                operation="invalid_operation",
                collection_name="test-collection",
                success=True,
                processing_time_ms=100.0
            )
    
    def test_collection_name_validation(self):
        """Test collection name validation"""
        valid_names = ["test-collection", "test_collection", "test123"]
        
        for name in valid_names:
            result = StorageResult(
                operation="insert",
                collection_name=name,
                success=True,
                processing_time_ms=100.0
            )
            assert result.collection_name == name.lower()
        
        # Invalid collection names
        invalid_names = ["", "test collection", "test-collection!"]
        
        for name in invalid_names:
            with pytest.raises(ValueError, match="Collection name must be alphanumeric"):
                StorageResult(
                    operation="insert",
                    collection_name=name,
                    success=True,
                    processing_time_ms=100.0
                )
    
    def test_success_rate_calculation(self):
        """Test success rate calculation"""
        # Perfect success
        result = StorageResult(
            operation="insert",
            collection_name="test-collection",
            success=True,
            processing_time_ms=100.0,
            affected_count=10,
            total_count=10
        )
        assert result.success_rate == 1.0
        
        # Partial success
        result = StorageResult(
            operation="insert",
            collection_name="test-collection",
            success=False,
            processing_time_ms=100.0,
            affected_count=7,
            total_count=10
        )
        assert result.success_rate == 0.7
        
        # No operations
        result = StorageResult(
            operation="insert",
            collection_name="test-collection",
            success=True,
            processing_time_ms=100.0,
            total_count=0
        )
        assert result.success_rate == 1.0  # Success with no operations
    
    def test_throughput_calculation(self):
        """Test throughput calculation"""
        result = StorageResult(
            operation="insert",
            collection_name="test-collection",
            success=True,
            processing_time_ms=1000.0,  # 1 second
            total_count=100
        )
        
        assert result.throughput_per_second == 100.0  # 100 items/second
        
        # Zero time case
        result = StorageResult(
            operation="insert",
            collection_name="test-collection",
            success=True,
            processing_time_ms=0.0,
            total_count=100
        )
        
        assert result.throughput_per_second == 0.0
    
    def test_helper_methods(self):
        """Test helper class methods"""
        # Test successful_insert
        result = StorageResult.successful_insert(
            collection_name="test-collection",
            count=10,
            processing_time_ms=500.0
        )
        
        assert result.operation == "insert"
        assert result.collection_name == "test-collection"
        assert result.success is True
        assert result.affected_count == 10
        assert result.total_count == 10
        assert result.processing_time_ms == 500.0
        
        # Test failed_operation
        result = StorageResult.failed_operation(
            operation="update",
            collection_name="test-collection",
            error="Test error",
            processing_time_ms=200.0,
            error_details={"detail": "test"}
        )
        
        assert result.operation == "update"
        assert result.collection_name == "test-collection"
        assert result.success is False
        assert result.error == "Test error"
        assert result.error_details == {"detail": "test"}
        assert result.processing_time_ms == 200.0
    
    def test_mark_completed(self):
        """Test marking operation as completed"""
        result = StorageResult(
            operation="insert",
            collection_name="test-collection",
            success=True,
            processing_time_ms=100.0
        )
        
        assert result.completed_at is None
        
        result.mark_completed()
        
        assert result.completed_at is not None
        assert isinstance(result.completed_at, datetime)


class TestCollectionInfo:
    """Test CollectionInfo model"""
    
    def test_valid_collection_info(self):
        """Test creating valid collection info"""
        info = CollectionInfo(
            name="test-collection",
            vectors_count=1000,
            indexed_vectors_count=950,
            points_count=1000,
            vector_size=1024,
            distance_metric="cosine",
            status="green"
        )
        
        assert info.name == "test-collection"
        assert info.vectors_count == 1000
        assert info.indexed_vectors_count == 950
        assert info.points_count == 1000
        assert info.vector_size == 1024
        assert info.distance_metric == "cosine"
        assert info.status == "green"
    
    def test_indexing_progress(self):
        """Test indexing progress calculation"""
        # Complete indexing
        info = CollectionInfo(
            name="test-collection",
            vectors_count=1000,
            indexed_vectors_count=1000,
            points_count=1000,
            vector_size=1024,
            distance_metric="cosine",
            status="green"
        )
        assert info.indexing_progress == 1.0
        
        # Partial indexing
        info = CollectionInfo(
            name="test-collection",
            vectors_count=1000,
            indexed_vectors_count=750,
            points_count=1000,
            vector_size=1024,
            distance_metric="cosine",
            status="yellow"
        )
        assert info.indexing_progress == 0.75
        
        # No vectors
        info = CollectionInfo(
            name="test-collection",
            vectors_count=0,
            indexed_vectors_count=0,
            points_count=0,
            vector_size=1024,
            distance_metric="cosine",
            status="green"
        )
        assert info.indexing_progress == 1.0  # Complete if no vectors
    
    def test_is_ready_property(self):
        """Test is_ready property"""
        ready_statuses = ["green", "ready", "active"]
        
        for status in ready_statuses:
            info = CollectionInfo(
                name="test-collection",
                vectors_count=1000,
                indexed_vectors_count=1000,
                points_count=1000,
                vector_size=1024,
                distance_metric="cosine",
                status=status
            )
            assert info.is_ready is True
        
        # Not ready status
        info = CollectionInfo(
            name="test-collection",
            vectors_count=1000,
            indexed_vectors_count=1000,
            points_count=1000,
            vector_size=1024,
            distance_metric="cosine",
            status="red"
        )
        assert info.is_ready is False
    
    def test_size_conversion(self):
        """Test disk and RAM usage conversion"""
        info = CollectionInfo(
            name="test-collection",
            vectors_count=1000,
            indexed_vectors_count=1000,
            points_count=1000,
            vector_size=1024,
            distance_metric="cosine",
            status="green",
            disk_usage_bytes=1048576,  # 1 MB
            ram_usage_bytes=2097152   # 2 MB
        )
        
        assert info.disk_usage_mb == 1.0
        assert info.ram_usage_mb == 2.0
        
        # Test None values
        info = CollectionInfo(
            name="test-collection",
            vectors_count=1000,
            indexed_vectors_count=1000,
            points_count=1000,
            vector_size=1024,
            distance_metric="cosine",
            status="green"
        )
        
        assert info.disk_usage_mb is None
        assert info.ram_usage_mb is None