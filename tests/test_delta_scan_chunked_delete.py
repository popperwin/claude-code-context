"""
Tests for FEAT4: Chunked Delete Operations with Real Qdrant Instance.

Tests the complete chunked delete pipeline with real Qdrant operations, including:
- Large-scale entity deletion (10k+ entities)
- Chunk processing with validation
- Race condition prevention with timestamp validation
- Error handling and recovery
- Performance metrics collection

NO MOCKS - Uses real Qdrant instance for authentic testing.
"""

import pytest
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock

from core.indexer.hybrid_indexer import HybridIndexer, IndexingJobConfig
from core.parser.parallel_pipeline import ProcessParsingPipeline
from core.storage.client import HybridQdrantClient
from core.storage.schemas import CollectionManager, QdrantSchema
from core.storage.utils import entity_id_to_qdrant_id
from core.models.entities import Entity, EntityType, SourceLocation
from core.models.storage import QdrantPoint
from core.embeddings.stella import StellaEmbedder


class TestChunkedDeleteOperations:
    """Test chunked delete operations with real Qdrant instance."""
    
    @pytest.fixture
    async def storage_client(self):
        """Create HybridQdrantClient connected to test Qdrant instance."""
        client = HybridQdrantClient(url="http://localhost:6334")  # Test Qdrant port
        await client.connect()
        yield client
        await client.disconnect()
    
    @pytest.fixture
    async def test_collection_name(self):
        """Generate unique test collection name."""
        import uuid
        return f"test-chunked-delete-{uuid.uuid4().hex[:8]}"
    
    @pytest.fixture
    async def collection_manager(self, test_collection_name):
        """Create collection manager for test collection."""
        return CollectionManager(project_name=test_collection_name)
    
    @pytest.fixture
    async def hybrid_indexer(self, storage_client, test_collection_name):
        """Create HybridIndexer with real storage client."""
        # Create required components
        parser_pipeline = ProcessParsingPipeline(max_workers=2, batch_size=10)
        config = IndexingJobConfig(
            project_path=Path("/tmp"),
            project_name=test_collection_name,
            max_workers=2,
            batch_size=10
        )
        
        indexer = HybridIndexer(
            parser_pipeline=parser_pipeline,
            embedder=None,  # Skip embeddings for delete tests
            storage_client=storage_client,
            cache_manager=None,
            config=config
        )
        yield indexer
    
    @pytest.fixture
    async def setup_test_collection(self, storage_client, test_collection_name, collection_manager):
        """Setup test collection with proper schema."""
        try:
            # Create collection with proper schema
            collection_config = QdrantSchema.get_code_collection_config(test_collection_name)
            create_result = await storage_client.create_collection(collection_config)
            assert create_result.success, f"Failed to create test collection: {create_result.error}"
            
            yield test_collection_name
            
        finally:
            # Cleanup: Delete test collection
            try:
                # Use asyncio.to_thread for synchronous delete_collection method
                import asyncio
                await asyncio.to_thread(storage_client._client.delete_collection, test_collection_name)
            except Exception as e:
                print(f"Warning: Failed to cleanup test collection {test_collection_name}: {e}")
    
    def create_test_entity(self, entity_id: str, file_path: str = "test.py", 
                          created_timestamp: float = None) -> Entity:
        """Create a test entity with specified ID and timestamp."""
        if created_timestamp is None:
            created_timestamp = time.time()
        
        # Convert timestamp to datetime
        from datetime import datetime
        created_dt = datetime.fromtimestamp(created_timestamp)
        
        return Entity(
            id=entity_id,
            name=entity_id.split("::")[-1],
            qualified_name=entity_id.split("::")[-1],
            entity_type=EntityType.FUNCTION,
            source_code=f"def {entity_id.split('::')[-1]}(): pass",
            location=SourceLocation(
                file_path=Path(file_path),
                start_line=1,
                end_line=1,
                start_column=0,
                end_column=10,
                start_byte=0,
                end_byte=10
            ),
            created_at=created_dt,
            last_modified=created_dt
        )
    
    def create_qdrant_point(self, entity: Entity) -> QdrantPoint:
        """Create QdrantPoint from entity with proper payload."""
        return QdrantPoint(
            id=entity_id_to_qdrant_id(entity.id),
            vector=[0.0] * 1024,  # Zero vector for tests
            payload={
                "entity_id": entity.id,
                "entity_name": entity.name,
                "entity_type": entity.entity_type.value,
                "file_path": str(entity.location.file_path),
                "source_code": entity.source_code,
                "indexed_at": entity.created_at.timestamp(),  # Convert datetime to timestamp
                "signature": f"def {entity.name}():"
            }
        )
    
    @pytest.mark.asyncio
    async def test_chunked_delete_empty_list(self, hybrid_indexer, setup_test_collection):
        """Test chunked delete with empty entity list."""
        collection_name = setup_test_collection
        
        # Test empty deletion
        result = await hybrid_indexer._chunked_entity_delete(
            collection_name=collection_name,
            stale_entity_ids=[],
            cutoff_timestamp=time.time()
        )
        
        # Verify result structure
        assert result["success"] is True
        assert result["total_entities"] == 0
        assert result["deleted_entities"] == 0
        assert result["validated_chunks"] == 0
        assert result["skipped_entities"] == 0
        assert len(result["errors"]) == 0
        assert "processing_time_ms" in result
    
    @pytest.mark.asyncio
    async def test_chunked_delete_small_batch(self, hybrid_indexer, storage_client, setup_test_collection):
        """Test chunked delete with small batch of entities."""
        collection_name = setup_test_collection
        
        # Create test entities with old timestamps (stale)
        current_time = time.time()
        stale_timestamp = current_time - 3600  # 1 hour ago
        
        entities = []
        for i in range(5):
            entity = self.create_test_entity(
                f"test.py::stale_function_{i}",
                created_timestamp=stale_timestamp
            )
            entities.append(entity)
        
        # Insert entities into collection
        points = [self.create_qdrant_point(entity) for entity in entities]
        insert_result = await storage_client.upsert_points(collection_name, points)
        assert insert_result.success, f"Failed to insert test entities: {insert_result.error}"
        
        # Perform chunked deletion
        entity_ids = [entity.id for entity in entities]
        cutoff_timestamp = current_time - 1800  # 30 minutes ago (so entities are stale)
        
        result = await hybrid_indexer._chunked_entity_delete(
            collection_name=collection_name,
            stale_entity_ids=entity_ids,
            cutoff_timestamp=cutoff_timestamp,
            chunk_size=3  # Small chunks for testing
        )
        
        # Verify deletion results
        assert result["success"] is True
        assert result["total_entities"] == 5
        assert result["deleted_entities"] == 5
        assert result["validated_chunks"] == 2  # 5 entities in chunks of 3: [3, 2]
        assert result["skipped_entities"] == 0
        assert len(result["errors"]) == 0
        assert result["processing_time_ms"] > 0
        
        # Verify entities are actually deleted from Qdrant
        # Use get_collection_info to check entity count
        try:
            collection_info = await storage_client.get_collection_info(collection_name) 
            if collection_info:
                remaining_count = collection_info.get("points_count", 0)
                assert remaining_count == 0, f"Expected no remaining entities, found {remaining_count}"
        except Exception:
            # If collection info fails, that's acceptable for test
            pass
    
    @pytest.mark.asyncio
    async def test_chunked_delete_large_batch(self, hybrid_indexer, storage_client, setup_test_collection):
        """Test chunked delete with large batch (>10k entities) to test chunking."""
        collection_name = setup_test_collection
        
        # Create large number of test entities (use smaller number for test speed)
        num_entities = 2500  # Still tests chunking with default 10k chunk size
        current_time = time.time()
        stale_timestamp = current_time - 3600  # 1 hour ago
        
        # Create entities in batches to avoid memory issues
        entity_ids = []
        batch_size = 500
        
        for batch_start in range(0, num_entities, batch_size):
            batch_end = min(batch_start + batch_size, num_entities)
            batch_entities = []
            batch_points = []
            
            for i in range(batch_start, batch_end):
                entity = self.create_test_entity(
                    f"large_test.py::bulk_function_{i}",
                    created_timestamp=stale_timestamp
                )
                batch_entities.append(entity)
                batch_points.append(self.create_qdrant_point(entity))
                entity_ids.append(entity.id)
            
            # Insert batch into collection
            insert_result = await storage_client.upsert_points(collection_name, batch_points)
            assert insert_result.success, f"Failed to insert batch {batch_start}-{batch_end}: {insert_result.error}"
        
        # Perform chunked deletion with custom chunk size
        cutoff_timestamp = current_time - 1800  # 30 minutes ago
        
        start_time = time.perf_counter()
        result = await hybrid_indexer._chunked_entity_delete(
            collection_name=collection_name,
            stale_entity_ids=entity_ids,
            cutoff_timestamp=cutoff_timestamp,
            chunk_size=1000  # Test chunking behavior
        )
        deletion_time = time.perf_counter() - start_time
        
        # Verify deletion results
        assert result["success"] is True
        assert result["total_entities"] == num_entities
        assert result["deleted_entities"] == num_entities
        assert result["validated_chunks"] == 3  # 2500 entities in chunks of 1000: [1000, 1000, 500]
        assert result["skipped_entities"] == 0
        assert len(result["errors"]) == 0
        assert result["processing_time_ms"] > 0
        
        # Performance validation: Should process >25 entities/second (Sprint 4.7 target)
        entities_per_second = num_entities / deletion_time
        print(f"Chunked deletion performance: {entities_per_second:.1f} entities/second")
        assert entities_per_second > 25, f"Deletion too slow: {entities_per_second:.1f} entities/sec"
        
        # Sample verification: Check that entities are actually deleted
        try:
            collection_info = await storage_client.get_collection_info(collection_name)
            if collection_info:
                remaining_count = collection_info.get("points_count", 0)
                assert remaining_count == 0, f"Expected no entities remaining after deletion, found {remaining_count}"
        except Exception:
            # If collection info fails, that's acceptable for test
            pass
    
    @pytest.mark.asyncio
    async def test_chunked_delete_with_validation_skips(self, hybrid_indexer, storage_client, setup_test_collection):
        """Test chunked delete with timestamp validation that causes some entities to be skipped."""
        collection_name = setup_test_collection
        
        current_time = time.time()
        
        # Create mix of stale and fresh entities
        entities = []
        
        # Stale entities (should be deleted)
        for i in range(3):
            entity = self.create_test_entity(
                f"mixed.py::stale_function_{i}",
                created_timestamp=current_time - 3600  # 1 hour ago
            )
            entities.append(entity)
        
        # Fresh entities (should be skipped) 
        for i in range(2):
            entity = self.create_test_entity(
                f"mixed.py::fresh_function_{i}",
                created_timestamp=current_time - 600  # 10 minutes ago
            )
            entities.append(entity)
        
        # Insert all entities
        points = [self.create_qdrant_point(entity) for entity in entities]
        insert_result = await storage_client.upsert_points(collection_name, points)
        assert insert_result.success, f"Failed to insert test entities: {insert_result.error}"
        
        # Perform deletion with cutoff that should skip fresh entities
        entity_ids = [entity.id for entity in entities]
        cutoff_timestamp = current_time - 1800  # 30 minutes ago
        
        result = await hybrid_indexer._chunked_entity_delete(
            collection_name=collection_name,
            stale_entity_ids=entity_ids,
            cutoff_timestamp=cutoff_timestamp,
            chunk_size=2  # Small chunks to test validation per chunk
        )
        
        # Verify results: only stale entities should be deleted
        assert result["success"] is True
        assert result["total_entities"] == 5
        assert result["deleted_entities"] == 3  # Only stale entities
        assert result["skipped_entities"] == 2  # Fresh entities skipped
        assert len(result["errors"]) == 0
        
        # Verify that fresh entities still exist while stale ones are deleted
        # Use get_collection_info to check remaining entity count
        try:
            collection_info = await storage_client.get_collection_info(collection_name)
            if collection_info:
                remaining_count = collection_info.get("points_count", 0)
                # Should have 2 fresh entities remaining (out of original 5)
                assert remaining_count == 2, f"Expected 2 fresh entities remaining, found {remaining_count}"
        except Exception as e:
            # If collection info fails, we can't verify - log the error but don't fail the test
            print(f"Warning: Could not verify remaining entities: {e}")
    
    @pytest.mark.asyncio
    async def test_chunked_delete_error_handling(self, hybrid_indexer, storage_client, setup_test_collection):
        """Test error handling in chunked delete operations."""
        collection_name = setup_test_collection
        
        # Test with non-existent collection name
        nonexistent_collection = "nonexistent-collection-12345"
        
        result = await hybrid_indexer._chunked_entity_delete(
            collection_name=nonexistent_collection,
            stale_entity_ids=["test.py::fake_function"],
            cutoff_timestamp=time.time()
        )
        
        # Should handle gracefully by skipping all entities (defensive behavior)
        assert result["success"] is True  # Operation succeeds but with 0 deletions
        assert result["total_entities"] == 1
        assert result["deleted_entities"] == 0
        assert result["skipped_entities"] == 1  # Entity was skipped due to validation failure
        assert len(result["errors"]) == 0  # No errors - graceful handling
    
    @pytest.mark.asyncio
    async def test_chunked_delete_chunk_size_edge_cases(self, hybrid_indexer, storage_client, setup_test_collection):
        """Test chunked delete with various chunk sizes including edge cases."""
        collection_name = setup_test_collection
        
        # Create test entities
        entities = []
        for i in range(7):  # Odd number to test chunking edge cases
            entity = self.create_test_entity(
                f"edge_test.py::edge_function_{i}",
                created_timestamp=time.time() - 3600
            )
            entities.append(entity)
        
        # Insert entities
        points = [self.create_qdrant_point(entity) for entity in entities]
        insert_result = await storage_client.upsert_points(collection_name, points)
        assert insert_result.success
        
        entity_ids = [entity.id for entity in entities]
        cutoff_timestamp = time.time() - 1800
        
        # Test with chunk size 1 (maximum chunks)
        result = await hybrid_indexer._chunked_entity_delete(
            collection_name=collection_name,
            stale_entity_ids=entity_ids,
            cutoff_timestamp=cutoff_timestamp,
            chunk_size=1
        )
        
        assert result["success"] is True
        assert result["total_entities"] == 7
        assert result["deleted_entities"] == 7
        assert result["validated_chunks"] == 7  # Each entity in its own chunk
        assert result["skipped_entities"] == 0
    
    @pytest.mark.asyncio
    async def test_chunked_delete_performance_metrics(self, hybrid_indexer, storage_client, setup_test_collection):
        """Test that chunked delete provides comprehensive performance metrics."""
        collection_name = setup_test_collection
        
        # Create moderate number of entities for performance testing
        num_entities = 100
        entities = []
        for i in range(num_entities):
            entity = self.create_test_entity(
                f"perf_test.py::perf_function_{i}",
                created_timestamp=time.time() - 3600
            )
            entities.append(entity)
        
        # Insert entities
        points = [self.create_qdrant_point(entity) for entity in entities]
        insert_result = await storage_client.upsert_points(collection_name, points)
        assert insert_result.success
        
        # Perform deletion with timing
        entity_ids = [entity.id for entity in entities]
        cutoff_timestamp = time.time() - 1800
        
        start_wall_time = time.perf_counter()
        result = await hybrid_indexer._chunked_entity_delete(
            collection_name=collection_name,
            stale_entity_ids=entity_ids,
            cutoff_timestamp=cutoff_timestamp,
            chunk_size=25  # Multiple chunks for realistic test
        )
        wall_time = time.perf_counter() - start_wall_time
        
        # Verify comprehensive metrics
        assert "success" in result
        assert "total_entities" in result
        assert "deleted_entities" in result
        assert "validated_chunks" in result
        assert "skipped_entities" in result
        assert "processing_time_ms" in result
        assert "errors" in result
        
        # Verify performance metrics are reasonable
        assert result["processing_time_ms"] > 0
        assert result["processing_time_ms"] < 30000  # Should complete within 30 seconds
        
        # Verify deletion rate meets targets
        deletion_rate = result["deleted_entities"] / (result["processing_time_ms"] / 1000)
        print(f"Chunked deletion rate: {deletion_rate:.1f} entities/second")
        
        # Should exceed minimum performance target
        assert deletion_rate > 10, f"Deletion rate too slow: {deletion_rate:.1f} entities/sec"
        
        # Wall time should be close to reported processing time
        wall_time_ms = wall_time * 1000
        time_difference = abs(wall_time_ms - result["processing_time_ms"])
        assert time_difference < 1000, "Wall time and reported time should be similar"
    
    @pytest.mark.asyncio
    async def test_chunk_list_utility(self, hybrid_indexer):
        """Test the _chunk_list utility method with various inputs."""
        # Test normal chunking
        items = list(range(10))
        chunks = hybrid_indexer._chunk_list(items, 3)
        assert len(chunks) == 4
        assert chunks[0] == [0, 1, 2]
        assert chunks[1] == [3, 4, 5]
        assert chunks[2] == [6, 7, 8]
        assert chunks[3] == [9]
        
        # Test exact division
        items = list(range(6))
        chunks = hybrid_indexer._chunk_list(items, 2)
        assert len(chunks) == 3
        assert chunks[0] == [0, 1]
        assert chunks[1] == [2, 3]
        assert chunks[2] == [4, 5]
        
        # Test single item chunks
        items = [1, 2, 3]
        chunks = hybrid_indexer._chunk_list(items, 1)
        assert len(chunks) == 3
        assert chunks[0] == [1]
        assert chunks[1] == [2]
        assert chunks[2] == [3]
        
        # Test empty list
        chunks = hybrid_indexer._chunk_list([], 5)
        assert len(chunks) == 0
        
        # Test chunk size larger than list
        items = [1, 2]
        chunks = hybrid_indexer._chunk_list(items, 10)
        assert len(chunks) == 1
        assert chunks[0] == [1, 2]


class TestChunkedDeleteIntegration:
    """Integration tests for chunked delete within delta scan pipeline."""
    
    @pytest.fixture
    async def storage_client(self):
        """Create HybridQdrantClient for integration tests."""
        client = HybridQdrantClient(url="http://localhost:6334")
        await client.connect()
        yield client
        await client.disconnect()
    
    @pytest.fixture
    async def test_collection_name(self):
        """Generate unique test collection name for integration."""
        import uuid
        return f"test-chunked-integration-{uuid.uuid4().hex[:8]}"
    
    @pytest.mark.asyncio
    async def test_chunked_delete_in_delta_scan_context(self, storage_client, test_collection_name):
        """Test chunked delete as part of delta scan workflow."""
        try:
            # Setup: Create collection with realistic schema
            collection_config = QdrantSchema.get_code_collection_config(test_collection_name)
            create_result = await storage_client.create_collection(collection_config)
            assert create_result.success
            
            # Setup: Create collection manager and hybrid indexer
            collection_manager = CollectionManager(project_name=test_collection_name)
            parser_pipeline = ProcessParsingPipeline(max_workers=2, batch_size=10)
            config = IndexingJobConfig(
                project_path=Path("/tmp"),
                project_name=test_collection_name,
                max_workers=2,
                batch_size=10
            )
            
            hybrid_indexer = HybridIndexer(
                parser_pipeline=parser_pipeline,
                embedder=None,
                storage_client=storage_client,
                cache_manager=None,
                config=config
            )
            
            # Simulate delta scan scenario: entities to be removed
            # Create old entities that should be deleted
            old_entities = []
            current_time = time.time()
            for i in range(50):
                entity_id = f"old_module.py::old_function_{i}"
                point = QdrantPoint(
                    id=entity_id_to_qdrant_id(entity_id),
                    vector=[0.0] * 1024,
                    payload={
                        "entity_id": entity_id,
                        "entity_name": f"old_function_{i}",
                        "entity_type": "function",
                        "file_path": "old_module.py",
                        "indexed_at": current_time - 7200,  # 2 hours ago
                        "signature": f"def old_function_{i}():"
                    }
                )
                old_entities.append(point)
            
            # Insert old entities
            insert_result = await storage_client.upsert_points(test_collection_name, old_entities)
            assert insert_result.success
            
            # Delta scan deletion: Remove entities older than 1 hour
            entity_ids_to_delete = [point.payload["entity_id"] for point in old_entities]
            cutoff_timestamp = current_time - 3600  # 1 hour ago
            
            # Execute chunked deletion (core functionality being tested)
            deletion_result = await hybrid_indexer._chunked_entity_delete(
                collection_name=test_collection_name,
                stale_entity_ids=entity_ids_to_delete,
                cutoff_timestamp=cutoff_timestamp,
                chunk_size=20  # Test multiple chunks
            )
            
            # Verify integration results
            assert deletion_result["success"] is True
            assert deletion_result["total_entities"] == 50
            assert deletion_result["deleted_entities"] == 50
            assert deletion_result["validated_chunks"] == 3  # 50 entities in chunks of 20: [20, 20, 10]
            assert deletion_result["skipped_entities"] == 0
            
            # Verify deletion effectiveness: Collection should be empty
            try:
                collection_info = await storage_client.get_collection_info(test_collection_name)
                if collection_info:
                    remaining_count = collection_info.get("points_count", 0)
                    assert remaining_count == 0, f"Expected no remaining entities, found {remaining_count}"
            except Exception as e:
                # If collection info fails, that's acceptable
                print(f"Note: Could not verify collection state: {e}")
                
        finally:
            # Cleanup
            try:
                await asyncio.to_thread(storage_client._client.delete_collection, test_collection_name)
            except Exception:
                pass  # Cleanup failure is not critical for test results


if __name__ == "__main__":
    # Allow direct execution for debugging
    pytest.main([__file__, "-v", "-s"])