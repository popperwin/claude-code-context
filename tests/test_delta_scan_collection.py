"""
Comprehensive tests for get_collection_state() function using real Qdrant operations.

Tests the high-performance Qdrant scroll-based entity metadata retrieval implementation
for delta-scan operations. NO MOCKS - all tests use real Qdrant operations.
"""

import asyncio
import pytest
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock

from core.indexer.hybrid_indexer import HybridIndexer
from core.parser.parallel_pipeline import ProcessParsingPipeline
from core.embeddings.stella import StellaEmbedder
from core.storage.client import HybridQdrantClient
from core.storage.schemas import QdrantSchema
from core.models.entities import Entity, EntityType, SourceLocation, Visibility
from core.models.storage import QdrantPoint


class TestGetCollectionStateFunction:
    """Test the get_collection_state() function with real Qdrant operations."""
    
    @pytest.fixture
    def mock_components(self):
        """Create minimal mock components for HybridIndexer."""
        mock_parser = Mock(spec=ProcessParsingPipeline)
        mock_parser.max_workers = 4
        mock_parser.batch_size = 10
        
        mock_embedder = Mock(spec=StellaEmbedder)
        
        # Use real HybridQdrantClient - no mocking for storage
        storage_client = HybridQdrantClient(url="http://localhost:6334")
        
        return {
            "parser": mock_parser,
            "embedder": mock_embedder,
            "storage": storage_client
        }
    
    @pytest.fixture
    async def indexer_with_real_storage(self, mock_components):
        """Create HybridIndexer with real Qdrant storage client."""
        indexer = HybridIndexer(
            parser_pipeline=mock_components["parser"],
            embedder=mock_components["embedder"],
            storage_client=mock_components["storage"]
        )
        
        # Ensure connection
        connection_result = await indexer.storage_client.connect()
        if not connection_result:
            pytest.skip("Qdrant not available for testing")
        
        yield indexer
        
        # Cleanup: disconnect
        await indexer.storage_client.disconnect()
    
    def create_test_entities(self, count: int, file_prefix: str = "test") -> List[Entity]:
        """Create test entities for populating collections."""
        entities = []
        
        for i in range(count):
            file_path = f"{file_prefix}_{i}.py"
            entity = Entity(
                id=f"{file_path}::test_function_{i}",
                name=f"test_function_{i}",
                qualified_name=f"module.test_function_{i}",
                entity_type=EntityType.FUNCTION,
                signature=f"def test_function_{i}():",
                docstring=f"Test function {i} for delta-scan testing",
                source_code=f"def test_function_{i}():\n    return {i}",
                location=SourceLocation(
                    file_path=Path(file_path),
                    start_line=i * 5 + 1,
                    end_line=i * 5 + 3,
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
    
    def entities_to_qdrant_points(self, entities: List[Entity]) -> List[QdrantPoint]:
        """Convert entities to QdrantPoints for storage."""
        from core.storage.utils import entity_id_to_qdrant_id
        points = []
        
        for entity in entities:
            # Convert entity ID to consistent point ID using centralized function
            point_id = entity_id_to_qdrant_id(entity.id)
            
            point = QdrantPoint(
                id=point_id,  # Use integer directly - clean contract
                vector=[0.1] * 1024,   # Mock embedding vector
                payload={
                    "entity_id": entity.id,
                    "entity_name": entity.name,
                    "entity_type": entity.entity_type.value,
                    "file_path": str(entity.location.file_path),
                    "signature": entity.signature,
                    "docstring": entity.docstring or "",
                    "qualified_name": entity.qualified_name,
                    "visibility": entity.visibility.value,
                    "source_hash": entity.source_hash,
                    "created_at": entity.created_at,
                    "indexed_at": time.time(),  # Mock indexed_at timestamp
                    "start_line": entity.location.start_line,
                    "end_line": entity.location.end_line
                }
            )
            points.append(point)
        
        return points
    
    @pytest.mark.asyncio
    async def test_get_collection_state_empty_collection(self, indexer_with_real_storage):
        """Test get_collection_state on empty collection."""
        indexer = indexer_with_real_storage
        collection_name = "test-empty-collection"
        
        # Create empty collection
        collection_config = QdrantSchema.get_code_collection_config(collection_name)
        create_result = await indexer.storage_client.create_collection(collection_config, recreate=True)
        assert create_result.success
        
        try:
            # Test get_collection_state on empty collection
            start_time = time.perf_counter()
            collection_state = await indexer.get_collection_state(collection_name)
            scan_time = time.perf_counter() - start_time
            
            # Verify results
            assert isinstance(collection_state, dict)
            assert collection_state["entity_count"] == 0
            assert collection_state["file_count"] == 0
            assert collection_state["scan_time"] > 0
            assert abs(collection_state["scan_time"] - scan_time) < 0.01  # Allow small timing difference
            assert collection_state["entities"] == {}
            
            # Performance check - should be very fast for empty collection
            assert scan_time < 1.0
            
        finally:
            # Cleanup
            try:
                await indexer.storage_client._client.delete_collection(collection_name)
            except Exception:
                pass
    
    @pytest.mark.asyncio
    async def test_get_collection_state_small_collection(self, indexer_with_real_storage):
        """Test get_collection_state with small collection."""
        indexer = indexer_with_real_storage
        collection_name = "test-small-collection"
        
        # Create test entities
        test_entities = self.create_test_entities(5, "small")
        test_points = self.entities_to_qdrant_points(test_entities)
        
        # Create collection and add data
        collection_config = QdrantSchema.get_code_collection_config(collection_name)
        create_result = await indexer.storage_client.create_collection(collection_config, recreate=True)
        assert create_result.success
        
        # Insert test points
        upsert_result = await indexer.storage_client.upsert_points(collection_name, test_points)
        assert upsert_result.success
        assert upsert_result.affected_count == len(test_points)
        
        try:
            # Test get_collection_state
            start_time = time.perf_counter()
            collection_state = await indexer.get_collection_state(collection_name)
            scan_time = time.perf_counter() - start_time
            
            # Verify basic metrics
            assert isinstance(collection_state, dict)
            assert collection_state["entity_count"] == 5
            assert collection_state["file_count"] == 5  # 5 different files
            assert collection_state["scan_time"] > 0
            
            # Verify entities structure
            entities_by_file = collection_state["entities"]
            assert isinstance(entities_by_file, dict)
            assert len(entities_by_file) == 5
            
            # Check that each file has one entity
            for file_path, entities in entities_by_file.items():
                assert file_path.startswith("small_")
                assert file_path.endswith(".py")
                assert len(entities) == 1
                
                # Verify entity structure
                entity = entities[0]
                assert "entity_id" in entity
                assert "name" in entity
                assert "entity_type" in entity
                assert "indexed_at" in entity
                
                # Verify required fields
                assert entity["name"].startswith("test_function_")
                assert entity["entity_type"] == "function"
            
            # Performance check
            assert scan_time < 5.0
            
        finally:
            # Cleanup
            try:
                await indexer.storage_client._client.delete_collection(collection_name)
            except Exception:
                pass
    
    @pytest.mark.asyncio
    async def test_get_collection_state_multiple_entities_per_file(self, indexer_with_real_storage):
        """Test get_collection_state with multiple entities per file."""
        indexer = indexer_with_real_storage
        collection_name = "test-multi-entities-collection"
        
        # Create entities with multiple entities per file
        entities = []
        files = ["module1.py", "module2.py", "module3.py"]
        
        for file_idx, file_name in enumerate(files):
            for entity_idx in range(3):  # 3 entities per file
                entity_id = f"{file_name}::entity_{file_idx}_{entity_idx}"
                entity = Entity(
                    id=entity_id,
                    name=f"entity_{file_idx}_{entity_idx}",
                    qualified_name=f"module{file_idx}.entity_{file_idx}_{entity_idx}",
                    entity_type=EntityType.FUNCTION if entity_idx % 2 == 0 else EntityType.CLASS,
                    signature=f"def entity_{file_idx}_{entity_idx}():" if entity_idx % 2 == 0 else f"class entity_{file_idx}_{entity_idx}:",
                    docstring=f"Entity {entity_idx} in file {file_name}",
                    source_code=f"# Entity {entity_idx} in {file_name}",
                    location=SourceLocation(
                        file_path=Path(file_name),
                        start_line=entity_idx * 10 + 1,
                        end_line=entity_idx * 10 + 5,
                        start_column=0,
                        end_column=20,
                        start_byte=entity_idx * 200,
                        end_byte=entity_idx * 200 + 100
                    ),
                    visibility=Visibility.PUBLIC,
                    is_async=False,
                    source_hash=f"hash_{file_idx}_{entity_idx}",
                )
                entities.append(entity)
        
        test_points = self.entities_to_qdrant_points(entities)
        
        # Create collection and add data
        collection_config = QdrantSchema.get_code_collection_config(collection_name)
        create_result = await indexer.storage_client.create_collection(collection_config, recreate=True)
        assert create_result.success
        
        # Insert test points
        upsert_result = await indexer.storage_client.upsert_points(collection_name, test_points)
        assert upsert_result.success
        assert upsert_result.affected_count == len(test_points)
        
        try:
            # Test get_collection_state
            collection_state = await indexer.get_collection_state(collection_name)
            
            # Verify metrics
            assert collection_state["entity_count"] == 9  # 3 files * 3 entities each
            assert collection_state["file_count"] == 3
            
            # Verify entities grouping
            entities_by_file = collection_state["entities"]
            assert len(entities_by_file) == 3
            
            for file_name in files:
                assert file_name in entities_by_file
                file_entities = entities_by_file[file_name]
                assert len(file_entities) == 3
                
                # Verify entities are properly grouped by file
                for entity in file_entities:
                    assert entity["name"].startswith(f"entity_{files.index(file_name)}_")
                    assert entity["entity_type"] in ["function", "class"]
            
        finally:
            # Cleanup
            try:
                await indexer.storage_client._client.delete_collection(collection_name)
            except Exception:
                pass
    
    @pytest.mark.asyncio
    async def test_get_collection_state_chunked_processing(self, indexer_with_real_storage):
        """Test get_collection_state with chunked processing for larger collections."""
        indexer = indexer_with_real_storage
        collection_name = "test-chunked-collection"
        
        # Create larger number of entities to test chunking
        test_entities = self.create_test_entities(25, "chunked")  # 25 entities
        test_points = self.entities_to_qdrant_points(test_entities)
        
        # Create collection and add data
        collection_config = QdrantSchema.get_code_collection_config(collection_name)
        create_result = await indexer.storage_client.create_collection(collection_config, recreate=True)
        assert create_result.success
        
        # Insert test points
        upsert_result = await indexer.storage_client.upsert_points(collection_name, test_points)
        assert upsert_result.success
        assert upsert_result.affected_count == len(test_points)
        
        try:
            # Test get_collection_state with smaller chunk size
            start_time = time.perf_counter()
            collection_state = await indexer.get_collection_state(collection_name, chunk_size=10)
            scan_time = time.perf_counter() - start_time
            
            # Verify all entities are found despite chunking
            assert collection_state["entity_count"] == 25
            assert collection_state["file_count"] == 25  # Each entity in different file
            
            # Verify all entities are captured
            entities_by_file = collection_state["entities"]
            assert len(entities_by_file) == 25
            
            total_entities_found = sum(len(entities) for entities in entities_by_file.values())
            assert total_entities_found == 25
            
            # Performance check - should handle chunking efficiently
            assert scan_time < 10.0
            
        finally:
            # Cleanup
            try:
                await indexer.storage_client._client.delete_collection(collection_name)
            except Exception:
                pass
    
    @pytest.mark.asyncio
    async def test_get_collection_state_nonexistent_collection(self, indexer_with_real_storage):
        """Test get_collection_state with nonexistent collection."""
        indexer = indexer_with_real_storage
        collection_name = "nonexistent-collection"
        
        # Test on nonexistent collection - should handle gracefully
        collection_state = await indexer.get_collection_state(collection_name)
        
        # Should return empty state for nonexistent collection
        assert isinstance(collection_state, dict)
        assert collection_state["entity_count"] == 0
        assert collection_state["file_count"] == 0
        assert collection_state["entities"] == {}
        assert collection_state["scan_time"] == 0.0  # Returns 0.0 for nonexistent collections
    
    @pytest.mark.asyncio
    async def test_get_collection_state_entity_metadata_accuracy(self, indexer_with_real_storage):
        """Test that get_collection_state returns accurate entity metadata."""
        indexer = indexer_with_real_storage
        collection_name = "test-metadata-accuracy"
        
        # Create entity with specific metadata
        specific_entity = Entity(
            id="accuracy_test.py::specific_function",
            name="specific_function",
            qualified_name="accuracy_test.specific_function",
            entity_type=EntityType.FUNCTION,
            signature="def specific_function(param1: str, param2: int) -> bool:",
            docstring="Specific function for metadata accuracy testing",
            source_code="def specific_function(param1: str, param2: int) -> bool:\n    return True",
            location=SourceLocation(
                file_path=Path("accuracy_test.py"),
                start_line=42,
                end_line=44,
                start_column=0,
                end_column=15,
                start_byte=1000,
                end_byte=1150
            ),
            visibility=Visibility.PRIVATE,
            is_async=True,
            source_hash="specific_hash_12345"
        )
        
        test_points = self.entities_to_qdrant_points([specific_entity])
        
        # Create collection and add data
        collection_config = QdrantSchema.get_code_collection_config(collection_name)
        create_result = await indexer.storage_client.create_collection(collection_config, recreate=True)
        assert create_result.success
        
        # Insert test point
        upsert_result = await indexer.storage_client.upsert_points(collection_name, test_points)
        assert upsert_result.success
        
        try:
            # Test get_collection_state
            collection_state = await indexer.get_collection_state(collection_name)
            
            # Verify entity found
            assert collection_state["entity_count"] == 1
            assert collection_state["file_count"] == 1
            
            # Get the entity
            entities_by_file = collection_state["entities"]
            assert "accuracy_test.py" in entities_by_file
            
            retrieved_entity = entities_by_file["accuracy_test.py"][0]
            
            # Verify metadata fields are preserved (based on the implementation structure)
            assert retrieved_entity["entity_id"] == "accuracy_test.py::specific_function"
            assert retrieved_entity["name"] == "specific_function"
            assert retrieved_entity["entity_type"] == "function"
            assert retrieved_entity["qualified_name"] == "accuracy_test.specific_function"
            assert "indexed_at" in retrieved_entity  # Should have indexed_at timestamp
            
        finally:
            # Cleanup
            try:
                await indexer.storage_client._client.delete_collection(collection_name)
            except Exception:
                pass
    
    @pytest.mark.asyncio
    async def test_get_collection_state_performance_metrics(self, indexer_with_real_storage):
        """Test get_collection_state performance characteristics."""
        indexer = indexer_with_real_storage
        collection_name = "test-performance-metrics"
        
        # Create moderate-sized collection for performance testing
        test_entities = self.create_test_entities(50, "perf")
        test_points = self.entities_to_qdrant_points(test_entities)
        
        # Create collection and add data
        collection_config = QdrantSchema.get_code_collection_config(collection_name)
        create_result = await indexer.storage_client.create_collection(collection_config, recreate=True)
        assert create_result.success
        
        # Insert test points
        upsert_result = await indexer.storage_client.upsert_points(collection_name, test_points)
        assert upsert_result.success
        
        try:
            # Test get_collection_state performance
            start_time = time.perf_counter()
            collection_state = await indexer.get_collection_state(collection_name, chunk_size=20)
            scan_time = time.perf_counter() - start_time
            
            # Verify correctness
            assert collection_state["entity_count"] == 50
            assert collection_state["file_count"] == 50
            
            # Performance metrics
            entities_per_second = collection_state["entity_count"] / scan_time
            
            print(f"Collection state scan: {collection_state['entity_count']} entities in {scan_time:.3f}s "
                  f"({entities_per_second:.1f} entities/sec)")
            
            # Should achieve reasonable performance
            assert entities_per_second > 10  # Should process at least 10 entities/second
            assert scan_time < 15.0  # Should complete within reasonable time
            
            # Verify scan_time matches actual measurement
            assert abs(collection_state["scan_time"] - scan_time) < 0.1
            
        finally:
            # Cleanup
            try:
                await indexer.storage_client._client.delete_collection(collection_name)
            except Exception:
                pass
    
    @pytest.mark.asyncio
    async def test_get_collection_state_error_handling(self, indexer_with_real_storage):
        """Test get_collection_state error handling during Qdrant operations."""
        indexer = indexer_with_real_storage
        collection_name = "test-error-handling"
        
        # Test with invalid chunk size
        collection_state = await indexer.get_collection_state(collection_name, chunk_size=0)
        
        # Should handle gracefully with minimum chunk size
        assert isinstance(collection_state, dict)
        assert collection_state["entity_count"] == 0
        assert collection_state["file_count"] == 0
        
        # Test with extremely large chunk size
        collection_state = await indexer.get_collection_state(collection_name, chunk_size=100000)
        
        # Should handle gracefully
        assert isinstance(collection_state, dict)
        assert collection_state["entity_count"] == 0
        assert collection_state["file_count"] == 0
    
    @pytest.mark.asyncio
    async def test_get_collection_state_mixed_entity_types(self, indexer_with_real_storage):
        """Test get_collection_state with mixed entity types."""
        indexer = indexer_with_real_storage
        collection_name = "test-mixed-types"
        
        # Create entities of different types
        mixed_entities = []
        entity_types = [EntityType.FUNCTION, EntityType.CLASS, EntityType.VARIABLE, EntityType.INTERFACE]
        
        for i, entity_type in enumerate(entity_types):
            entity = Entity(
                id=f"mixed_{i}.py::entity_{i}",
                name=f"entity_{i}",
                qualified_name=f"mixed.entity_{i}",
                entity_type=entity_type,
                signature=f"signature for {entity_type.value} entity_{i}",
                docstring=f"Mixed type entity {i} of type {entity_type.value}",
                source_code=f"# {entity_type.value} entity_{i}",
                location=SourceLocation(
                    file_path=Path(f"mixed_{i}.py"),
                    start_line=i * 5 + 1,
                    end_line=i * 5 + 3,
                    start_column=0,
                    end_column=15,
                    start_byte=i * 100,
                    end_byte=i * 100 + 50
                ),
                visibility=Visibility.PUBLIC,
                is_async=False,
                source_hash=f"mixed_hash_{i}",
            )
            mixed_entities.append(entity)
        
        test_points = self.entities_to_qdrant_points(mixed_entities)
        
        # Create collection and add data
        collection_config = QdrantSchema.get_code_collection_config(collection_name)
        create_result = await indexer.storage_client.create_collection(collection_config, recreate=True)
        assert create_result.success
        
        # Insert test points
        upsert_result = await indexer.storage_client.upsert_points(collection_name, test_points)
        assert upsert_result.success
        
        try:
            # Test get_collection_state
            collection_state = await indexer.get_collection_state(collection_name)
            
            # Verify all entity types are captured
            assert collection_state["entity_count"] == 4
            assert collection_state["file_count"] == 4
            
            # Verify entity types are preserved
            entities_by_file = collection_state["entities"]
            found_types = set()
            
            for file_path, entities in entities_by_file.items():
                for entity in entities:
                    found_types.add(entity["entity_type"])
            
            expected_types = {"function", "class", "variable", "interface"}
            assert found_types == expected_types
            
        finally:
            # Cleanup
            try:
                await indexer.storage_client._client.delete_collection(collection_name)
            except Exception:
                pass


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])