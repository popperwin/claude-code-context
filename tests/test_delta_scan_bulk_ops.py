"""
FEAT4 Tests: Chunked delete operations for delta-scan implementation.

Tests bulk delete functionality with real Qdrant instances - NO MOCKS.
Comprehensive validation of chunked deletion with race condition prevention.
"""

import pytest
import asyncio
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from core.indexer.hybrid_indexer import HybridIndexer
from core.storage.client import HybridQdrantClient
from core.storage.schemas import QdrantSchema, CollectionType
from core.storage.utils import entity_id_to_qdrant_id
from core.models.storage import QdrantPoint, StorageResult
from core.models.entities import Entity, EntityType, SourceLocation, Visibility
from core.embeddings.stella import StellaEmbedder


@pytest.fixture
async def stella_embedder():
    """Create real Stella embedder for testing"""
    embedder = StellaEmbedder()
    await embedder.load_model()
    yield embedder
    await embedder.unload_model()


@pytest.fixture 
async def qdrant_client():
    """Create real Qdrant client for testing"""
    client = HybridQdrantClient(url="http://localhost:6334")
    connection_result = await client.connect()
    if not connection_result:
        pytest.skip("Qdrant not available for testing")
    yield client
    await client.disconnect()


@pytest.fixture
async def test_collection(qdrant_client):
    """Create isolated test collection"""
    collection_name = f"test-delta-bulk-{uuid.uuid4().hex[:8]}"
    config = QdrantSchema.get_code_collection_config(collection_name)
    
    # Create collection
    create_result = await qdrant_client.create_collection(config)
    assert create_result.success, f"Failed to create test collection: {create_result.error}"
    
    yield collection_name
    
    # Cleanup
    try:
        await qdrant_client._client.delete_collection(collection_name)
    except Exception:
        pass  # Collection might not exist if test failed


@pytest.fixture
async def hybrid_indexer(qdrant_client, stella_embedder, test_collection):
    """Create HybridIndexer with real components"""
    from core.parser.parallel_pipeline import ProcessParsingPipeline
    
    # Initialize parser pipeline
    parser_pipeline = ProcessParsingPipeline(max_workers=2)
    
    # Create HybridIndexer with correct parameters
    indexer = HybridIndexer(
        parser_pipeline=parser_pipeline,
        embedder=stella_embedder,
        storage_client=qdrant_client
    )
    
    yield indexer


def create_test_entities(count: int, base_timestamp: float = None) -> List[Entity]:
    """Create test entities with specific timestamps"""
    if base_timestamp is None:
        base_timestamp = time.time()
    
    entities = []
    for i in range(count):
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


async def create_test_points_with_timestamps(
    qdrant_client: HybridQdrantClient,
    stella_embedder: StellaEmbedder,
    entities: List[Entity],
    indexed_at_timestamps: List[float]
) -> List[QdrantPoint]:
    """Create test points with specific indexed_at timestamps"""
    assert len(entities) == len(indexed_at_timestamps)
    
    # Generate embeddings for entities
    texts = [f"{entity.name} {entity.docstring}" for entity in entities]
    embedding_response = await stella_embedder.embed_texts(texts)
    embeddings = embedding_response.embeddings
    
    points = []
    for entity, embedding, timestamp in zip(entities, embeddings, indexed_at_timestamps):
        # Create payload with indexed_at timestamp
        payload = entity.to_qdrant_payload()
        payload["indexed_at"] = timestamp
        payload["entity_id"] = entity.id
        
        # Convert entity ID to consistent point ID using centralized function
        point_id = entity_id_to_qdrant_id(entity.id)
        
        point = QdrantPoint(
            id=point_id,  # Use integer directly - clean contract
            vector=embedding,
            payload=payload
        )
        points.append(point)
    
    return points


class TestChunkedEntityDelete:
    """Test chunked entity deletion with race condition prevention"""
    
    @pytest.mark.asyncio
    async def test_empty_entity_list(self, hybrid_indexer, test_collection):
        """Test chunked delete with empty entity list"""
        result = await hybrid_indexer._chunked_entity_delete(
            collection_name=test_collection,
            stale_entity_ids=[],
            cutoff_timestamp=time.time()
        )
        
        assert result["success"] is True
        assert result["total_entities"] == 0
        assert result["deleted_entities"] == 0
        assert result["validated_chunks"] == 0
        assert result["skipped_entities"] == 0
        assert len(result["errors"]) == 0
    
    @pytest.mark.asyncio
    async def test_basic_chunked_deletion(self, hybrid_indexer, qdrant_client, stella_embedder, test_collection):
        """Test basic chunked deletion with small dataset"""
        # Create test entities with old timestamps (stale)
        old_timestamp = time.time() - 3600  # 1 hour ago
        current_timestamp = time.time()
        
        entities = create_test_entities(5)
        timestamps = [old_timestamp] * 5  # All stale
        
        # Create and insert test points
        points = await create_test_points_with_timestamps(
            qdrant_client, stella_embedder, entities, timestamps
        )
        
        # Insert points into collection
        insert_result = await qdrant_client.upsert_points(test_collection, points)
        assert insert_result.success, f"Failed to insert test points: {insert_result.error}"
        
        # Wait for indexing
        await asyncio.sleep(1)
        
        # Perform chunked deletion
        entity_ids = [entity.id for entity in entities]
        result = await hybrid_indexer._chunked_entity_delete(
            collection_name=test_collection,
            stale_entity_ids=entity_ids,
            cutoff_timestamp=current_timestamp,  # All entities should be stale
            chunk_size=3  # Test chunking with 3 entities per chunk
        )
        
        # Verify results
        assert result["success"] is True
        assert result["total_entities"] == 5
        assert result["deleted_entities"] == 5  # All should be deleted
        assert result["validated_chunks"] == 2  # 2 chunks: [3, 2]
        assert result["skipped_entities"] == 0
        assert len(result["errors"]) == 0
        assert result["processing_time_ms"] > 0
        
        # Verify entities were actually deleted from Qdrant
        # Use direct scroll since get_points_by_filter doesn't work with empty conditions
        remaining_points = []
        async for point in hybrid_indexer._scroll_collection_points(test_collection, chunk_size=10):
            remaining_points.append(point)
        assert len(remaining_points) == 0  # All points should be deleted
    
    @pytest.mark.asyncio
    async def test_race_condition_prevention(self, hybrid_indexer, qdrant_client, stella_embedder, test_collection):
        """Test that validation prevents deletion of recently updated entities"""
        # Create test entities with mixed timestamps
        old_timestamp = time.time() - 3600  # 1 hour ago (stale)
        recent_timestamp = time.time() - 10   # 10 seconds ago (fresh)
        cutoff_timestamp = time.time() - 300  # 5 minutes ago
        
        entities = create_test_entities(4)
        timestamps = [old_timestamp, old_timestamp, recent_timestamp, recent_timestamp]
        
        # Create and insert test points
        points = await create_test_points_with_timestamps(
            qdrant_client, stella_embedder, entities, timestamps
        )
        
        # Insert points
        insert_result = await qdrant_client.upsert_points(test_collection, points)
        assert insert_result.success
        
        # Wait for indexing
        await asyncio.sleep(1)
        
        # Attempt to delete all entities (but validation should protect recent ones)
        entity_ids = [entity.id for entity in entities]
        result = await hybrid_indexer._chunked_entity_delete(
            collection_name=test_collection,
            stale_entity_ids=entity_ids,
            cutoff_timestamp=cutoff_timestamp,
            chunk_size=2  # 2 chunks: [2, 2]
        )
        
        # Verify results - only stale entities should be deleted
        assert result["success"] is True
        assert result["total_entities"] == 4
        assert result["deleted_entities"] == 2  # Only 2 stale entities deleted
        assert result["validated_chunks"] == 1  # Only 1 chunk had stale entities
        assert result["skipped_entities"] == 2  # 2 entities skipped due to recent timestamps
        assert len(result["errors"]) == 0
        
        # Verify correct entities remain in Qdrant
        # Use direct scroll since get_points_by_filter doesn't work with empty conditions
        remaining_points = []
        async for point in hybrid_indexer._scroll_collection_points(test_collection, chunk_size=10):
            remaining_points.append(point)
        assert len(remaining_points) == 2  # 2 recent entities should remain
        
        # Verify remaining entities have recent timestamps
        for point in remaining_points:
            indexed_at = point.payload.get("indexed_at", 0)
            assert indexed_at >= cutoff_timestamp  # Should be recent
    
    @pytest.mark.asyncio
    async def test_large_chunk_processing(self, hybrid_indexer, qdrant_client, stella_embedder, test_collection):
        """Test chunked deletion with 10k chunk size limit"""
        # Create test entities - use moderate size to avoid test timeout
        entity_count = 25
        old_timestamp = time.time() - 3600
        current_timestamp = time.time()
        
        entities = create_test_entities(entity_count)
        timestamps = [old_timestamp] * entity_count
        
        # Create and insert test points
        points = await create_test_points_with_timestamps(
            qdrant_client, stella_embedder, entities, timestamps
        )
        
        # Insert in batches to avoid timeouts
        batch_size = 10
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            insert_result = await qdrant_client.upsert_points(test_collection, batch)
            assert insert_result.success
        
        # Wait for indexing
        await asyncio.sleep(2)
        
        # Test with default 10k chunk size (should be single chunk)
        entity_ids = [entity.id for entity in entities]
        result = await hybrid_indexer._chunked_entity_delete(
            collection_name=test_collection,
            stale_entity_ids=entity_ids,
            cutoff_timestamp=current_timestamp
            # chunk_size defaults to 10000
        )
        
        # Verify results
        assert result["success"] is True
        assert result["total_entities"] == entity_count
        assert result["deleted_entities"] == entity_count
        assert result["validated_chunks"] == 1  # Single chunk for 25 entities with 10k limit
        assert result["skipped_entities"] == 0
        assert len(result["errors"]) == 0
        
        # Verify all entities deleted
        # Use direct scroll since get_points_by_filter doesn't work with empty conditions
        remaining_points = []
        async for point in hybrid_indexer._scroll_collection_points(test_collection, chunk_size=100):
            remaining_points.append(point)
        assert len(remaining_points) == 0
    
    @pytest.mark.asyncio
    async def test_validation_batch_failure_handling(self, hybrid_indexer, qdrant_client, test_collection):
        """Test handling when validation fails for a batch"""
        # Create entity IDs that don't exist in collection
        fake_entity_ids = [f"fake_{i}.py::fake_function_{i}" for i in range(5)]
        
        result = await hybrid_indexer._chunked_entity_delete(
            collection_name=test_collection,
            stale_entity_ids=fake_entity_ids,
            cutoff_timestamp=time.time(),
            chunk_size=3
        )
        
        # Should succeed but delete nothing (entities don't exist)
        assert result["success"] is True
        assert result["total_entities"] == 5
        assert result["deleted_entities"] == 0
        assert result["validated_chunks"] == 0  # No chunks validated because no entities found
        assert result["skipped_entities"] == 5  # All skipped because validation found nothing
        assert len(result["errors"]) == 0
    
    @pytest.mark.asyncio
    async def test_chunk_list_utility(self, hybrid_indexer):
        """Test the chunk list utility function"""
        # Test normal chunking  
        items = list(range(10))
        chunks = hybrid_indexer._chunk_list(items, 3)
        
        expected_chunks = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        assert chunks == expected_chunks
        
        # Test empty list
        empty_chunks = hybrid_indexer._chunk_list([], 3)
        assert empty_chunks == []
        
        # Test chunk size larger than list
        small_chunks = hybrid_indexer._chunk_list([1, 2], 5)
        assert small_chunks == [[1, 2]]
    
    @pytest.mark.asyncio
    async def test_entity_id_to_point_id_conversion(self, hybrid_indexer, qdrant_client):
        """Test consistent entity ID to point ID conversion using client normalization"""
        entity_id = "test.py::test_function"
        
        # Should produce consistent results using client's public method
        point_id_1 = qdrant_client.normalize_point_id(entity_id)
        point_id_2 = qdrant_client.normalize_point_id(entity_id)
        
        assert point_id_1 == point_id_2
        assert isinstance(point_id_1, int)  # Should be integer
        
        # Different entity IDs should produce different point IDs
        different_point_id = qdrant_client.normalize_point_id("different.py::different_function")
        assert different_point_id != point_id_1
    
    @pytest.mark.asyncio 
    async def test_performance_metrics(self, hybrid_indexer, qdrant_client, stella_embedder, test_collection):
        """Test that chunked deletion includes performance metrics"""
        # Create small test dataset
        entities = create_test_entities(3)
        timestamps = [time.time() - 3600] * 3  # All stale
        
        points = await create_test_points_with_timestamps(
            qdrant_client, stella_embedder, entities, timestamps
        )
        
        insert_result = await qdrant_client.upsert_points(test_collection, points)
        assert insert_result.success
        
        await asyncio.sleep(1)
        
        # Measure deletion performance
        start_time = time.perf_counter()
        
        entity_ids = [entity.id for entity in entities]
        result = await hybrid_indexer._chunked_entity_delete(
            collection_name=test_collection,
            stale_entity_ids=entity_ids,
            cutoff_timestamp=time.time(),
            chunk_size=2
        )
        
        end_time = time.perf_counter()
        actual_time_ms = (end_time - start_time) * 1000
        
        # Verify performance metrics are reasonable
        assert result["success"] is True
        assert result["processing_time_ms"] > 0
        assert result["processing_time_ms"] < actual_time_ms * 2  # Should be within reasonable bounds
        
        # Performance should be reasonable for small dataset
        assert result["processing_time_ms"] < 5000  # Should complete within 5 seconds


class TestValidateStaleBatch:
    """Test stale entity validation functionality"""
    
    @pytest.mark.asyncio
    async def test_validate_empty_batch(self, hybrid_indexer, test_collection):
        """Test validation with empty entity list"""
        result = await hybrid_indexer._validate_stale_entities_batch(
            collection_name=test_collection,
            entity_ids=[],
            cutoff_timestamp=time.time()
        )
        
        assert result == []
    
    @pytest.mark.asyncio
    async def test_validate_nonexistent_entities(self, hybrid_indexer, test_collection):
        """Test validation with entities that don't exist"""
        fake_ids = ["fake1.py::fake_func", "fake2.py::fake_func"]
        
        result = await hybrid_indexer._validate_stale_entities_batch(
            collection_name=test_collection,
            entity_ids=fake_ids,
            cutoff_timestamp=time.time()
        )
        
        assert result == []  # No entities found, so none validated as stale
    
    @pytest.mark.asyncio
    async def test_validate_mixed_timestamps(self, hybrid_indexer, qdrant_client, stella_embedder, test_collection):
        """Test validation with mix of stale and fresh entities"""
        cutoff_timestamp = time.time() - 300  # 5 minutes ago
        old_timestamp = time.time() - 3600     # 1 hour ago (stale)
        recent_timestamp = time.time() - 60    # 1 minute ago (fresh)
        
        entities = create_test_entities(4)
        timestamps = [old_timestamp, recent_timestamp, old_timestamp, recent_timestamp]
        
        # Insert test data
        points = await create_test_points_with_timestamps(
            qdrant_client, stella_embedder, entities, timestamps
        )
        
        insert_result = await qdrant_client.upsert_points(test_collection, points)
        assert insert_result.success
        
        await asyncio.sleep(1)
        
        # Validate all entity IDs
        entity_ids = [entity.id for entity in entities]
        validated_ids = await hybrid_indexer._validate_stale_entities_batch(
            collection_name=test_collection,
            entity_ids=entity_ids,
            cutoff_timestamp=cutoff_timestamp
        )
        
        # Only stale entities (indices 0, 2) should be validated
        expected_stale_ids = [entities[0].id, entities[2].id]
        assert len(validated_ids) == 2
        assert set(validated_ids) == set(expected_stale_ids)


class TestIntegrationWithExistingInfrastructure:
    """Test integration with existing HybridQdrantClient infrastructure"""
    
    @pytest.mark.asyncio
    async def test_leverages_existing_delete_points(self, hybrid_indexer, qdrant_client, stella_embedder, test_collection):
        """Test that chunked delete uses existing HybridQdrantClient.delete_points()"""
        # Create test data
        entities = create_test_entities(3)
        timestamps = [time.time() - 3600] * 3
        
        points = await create_test_points_with_timestamps(
            qdrant_client, stella_embedder, entities, timestamps
        )
        
        insert_result = await qdrant_client.upsert_points(test_collection, points)
        assert insert_result.success
        
        await asyncio.sleep(1)
        
        # Track calls to delete_points (we can't mock, but we can verify behavior)
        entity_ids = [entity.id for entity in entities]
        
        # Perform chunked deletion
        result = await hybrid_indexer._chunked_entity_delete(
            collection_name=test_collection,
            stale_entity_ids=entity_ids,
            cutoff_timestamp=time.time(),
            chunk_size=2  # Force multiple chunks
        )
        
        # Verify successful integration with existing infrastructure
        assert result["success"] is True
        assert result["deleted_entities"] == 3
        
        # Verify actual deletion occurred (proves integration works)
        # Use direct scroll since get_points_by_filter doesn't work with empty conditions
        remaining_points = []
        async for point in hybrid_indexer._scroll_collection_points(test_collection, chunk_size=10):
            remaining_points.append(point)
        assert len(remaining_points) == 0
    
    @pytest.mark.asyncio
    async def test_consistent_point_id_conversion(self, hybrid_indexer, qdrant_client, stella_embedder, test_collection):
        """Test that point ID conversion matches indexing pipeline"""
        # Create entity and get its point ID using both methods
        entity = create_test_entities(1)[0]
        
        # Method 1: Client normalization (uses centralized function)
        client_point_id = qdrant_client.normalize_point_id(entity.id)
        
        # Method 2: Direct centralized function call (should match client method)
        direct_point_id = entity_id_to_qdrant_id(entity.id)
        
        # Should be identical
        assert client_point_id == direct_point_id
        
        # Verify by actually inserting and then deleting using converted ID
        points = await create_test_points_with_timestamps(
            qdrant_client, stella_embedder, [entity], [time.time() - 3600]
        )
        
        insert_result = await qdrant_client.upsert_points(test_collection, points)
        assert insert_result.success
        
        await asyncio.sleep(1)
        
        # Delete using converted point ID (pass integer to match stored format)
        delete_result = await qdrant_client.delete_points(
            collection_name=test_collection,
            point_ids=[client_point_id]
        )
        
        assert delete_result.success
        assert delete_result.affected_count == 1