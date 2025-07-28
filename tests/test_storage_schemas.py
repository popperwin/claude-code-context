"""
Unit tests for storage schemas and collection management.

Tests collection configuration, schema generation, and validation.
"""

import pytest
from typing import Dict, Any

from core.storage.schemas import (
    CollectionType, DistanceMetric, IndexConfig, CollectionConfig,
    QdrantSchema, CollectionManager
)


class TestIndexConfig:
    """Test IndexConfig functionality"""
    
    def test_index_config_creation(self):
        """Test creating index configurations"""
        config = IndexConfig("entity_name", "text")
        assert config.field_name == "entity_name"
        assert config.field_type == "text"
        assert config.index is True
        assert config.store is True
        
        # Custom settings
        config = IndexConfig("score", "float", index=False, store=True)
        assert config.field_name == "score"
        assert config.field_type == "float"
        assert config.index is False
        assert config.store is True
    
    def test_to_qdrant_schema(self):
        """Test conversion to Qdrant schema format (returns type string)"""
        config = IndexConfig("entity_name", "text", index=True, store=True)
        schema = config.to_qdrant_schema()
        
        # Should return just the type string as needed by Qdrant workflow
        assert schema == "text"
        
        # Test different field types
        test_cases = [
            ("keyword_field", "keyword", "keyword"),
            ("int_field", "integer", "integer"),
            ("float_field", "float", "float"),
            ("bool_field", "bool", "bool"),
            ("geo_field", "geo", "geo"),
            ("unknown_field", "unknown", "keyword")  # Fallback
        ]
        
        for field_name, field_type, expected_type in test_cases:
            config = IndexConfig(field_name, field_type)
            schema = config.to_qdrant_schema()
            assert schema == expected_type


class TestCollectionConfig:
    """Test CollectionConfig functionality"""
    
    def test_collection_config_creation(self):
        """Test creating collection configurations"""
        config = CollectionConfig(
            name="test-collection",
            collection_type=CollectionType.CODE
        )
        
        assert config.name == "test-collection"
        assert config.collection_type == CollectionType.CODE
        assert config.vector_size == 1024
        assert config.distance_metric == DistanceMetric.COSINE
        assert config.payload_indexes == []
        
        # Custom settings
        indexes = [IndexConfig("field1", "text")]
        config = CollectionConfig(
            name="custom-collection",
            collection_type=CollectionType.RELATIONS,
            vector_size=512,
            distance_metric=DistanceMetric.EUCLIDEAN,
            payload_indexes=indexes
        )
        
        assert config.vector_size == 512
        assert config.distance_metric == DistanceMetric.EUCLIDEAN
        assert len(config.payload_indexes) == 1


class TestQdrantSchema:
    """Test QdrantSchema functionality"""
    
    def test_get_code_collection_config(self):
        """Test code collection configuration generation"""
        config = QdrantSchema.get_code_collection_config("test-code")
        
        assert config.name == "test-code"
        assert config.collection_type == CollectionType.CODE
        assert config.vector_size == 1024
        assert config.distance_metric == DistanceMetric.COSINE
        
        # Check payload indexes
        index_names = [idx.field_name for idx in config.payload_indexes]
        expected_fields = [
            "entity_name", "qualified_name", "signature", "docstring",
            "source_code", "entity_type", "file_path", "visibility",
            "language", "start_line", "end_line", "start_column",
            "end_column", "start_byte", "end_byte", "is_async",
            "is_test", "is_deprecated", "source_hash"
        ]
        
        for field in expected_fields:
            assert field in index_names
        
        # Check specific field configurations
        entity_name_idx = next(
            idx for idx in config.payload_indexes 
            if idx.field_name == "entity_name"
        )
        assert entity_name_idx.field_type == "text"
        assert entity_name_idx.index is True
        
        source_code_idx = next(
            idx for idx in config.payload_indexes 
            if idx.field_name == "source_code"
        )
        assert source_code_idx.field_type == "text"
        assert source_code_idx.index is False  # Store but don't index
        assert source_code_idx.store is True
    
    def test_get_relations_collection_config(self):
        """Test relations collection configuration generation"""
        config = QdrantSchema.get_relations_collection_config("test-relations")
        
        assert config.name == "test-relations"
        assert config.collection_type == CollectionType.RELATIONS
        assert config.on_disk_payload is False  # Keep in memory for fast queries
        
        # Check payload indexes
        index_names = [idx.field_name for idx in config.payload_indexes]
        expected_fields = [
            "relation_type", "source_entity_id", "target_entity_id",
            "context", "strength", "source_file_path", "target_file_path"
        ]
        
        for field in expected_fields:
            assert field in index_names
    
    def test_get_embeddings_collection_config(self):
        """Test embeddings collection configuration generation"""
        config = QdrantSchema.get_embeddings_collection_config("test-embeddings")
        
        assert config.name == "test-embeddings"
        assert config.collection_type == CollectionType.EMBEDDINGS
        assert config.segment_number == 8  # Optimized for vector search
        
        # Check minimal payload indexes
        index_names = [idx.field_name for idx in config.payload_indexes]
        expected_fields = [
            "content_type", "content_hash", "file_path", "chunk_index"
        ]
        
        for field in expected_fields:
            assert field in index_names
    
    def test_get_collection_config_by_type(self):
        """Test getting collection config by type"""
        # Test all collection types
        for collection_type in CollectionType:
            config = QdrantSchema.get_collection_config(
                "test-collection", collection_type
            )
            assert config.collection_type == collection_type
            assert config.name == "test-collection"
    
    def test_get_qdrant_vectors_config(self):
        """Test Qdrant vectors configuration generation"""
        config = CollectionConfig(
            name="test",
            collection_type=CollectionType.CODE,
            vector_size=512,
            distance_metric=DistanceMetric.EUCLIDEAN
        )
        
        vectors_config = QdrantSchema.get_qdrant_vectors_config(config)
        
        expected = {
            "size": 512,
            "distance": "Euclid",
            "on_disk": True
        }
        assert vectors_config == expected
        
        # Test different distance metrics
        distance_mappings = [
            (DistanceMetric.COSINE, "Cosine"),
            (DistanceMetric.EUCLIDEAN, "Euclid"),
            (DistanceMetric.DOT_PRODUCT, "Dot")
        ]
        
        for metric, expected_name in distance_mappings:
            config.distance_metric = metric
            vectors_config = QdrantSchema.get_qdrant_vectors_config(config)
            assert vectors_config["distance"] == expected_name
    
    def test_get_qdrant_config_params(self):
        """Test Qdrant configuration parameters generation"""
        config = CollectionConfig(
            name="test",
            collection_type=CollectionType.CODE,
            replication_factor=2,
            write_consistency_factor=1,
            segment_number=4,
            memmap_threshold=50000,
            indexing_threshold=30000
        )
        
        params = QdrantSchema.get_qdrant_config_params(config)
        
        assert params["replication_factor"] == 2
        assert params["write_consistency_factor"] == 1
        
        # Check optimizers config
        opt_config = params["optimizers_config"]
        assert opt_config["default_segment_number"] == 4
        assert opt_config["memmap_threshold"] == 50000
        assert opt_config["indexing_threshold"] == 30000
        assert opt_config["deleted_threshold"] == 0.2
        
        # Check HNSW config
        hnsw_config = params["hnsw_config"]
        assert hnsw_config["m"] == 16
        assert hnsw_config["ef_construct"] == 100
        assert "on_disk" in hnsw_config
        
        # Check WAL config
        wal_config = params["wal_config"]
        assert wal_config["wal_capacity_mb"] == 32
    
    def test_get_payload_schema(self):
        """Test payload schema generation"""
        indexes = [
            IndexConfig("entity_name", "text"),
            IndexConfig("entity_type", "keyword"),
            IndexConfig("start_line", "integer"),
            IndexConfig("is_test", "bool")
        ]
        
        config = CollectionConfig(
            name="test",
            collection_type=CollectionType.CODE,
            payload_indexes=indexes
        )
        
        schema = QdrantSchema.get_payload_schema(config)
        
        # Schema should be Dict[str, str] mapping field names to type strings
        assert len(schema) == 4
        assert schema["entity_name"] == "text"
        assert schema["entity_type"] == "keyword"
        assert schema["start_line"] == "integer"
        assert schema["is_test"] == "bool"
    
    def test_validate_collection_config(self):
        """Test collection configuration validation"""
        # Valid configuration
        config = CollectionConfig(
            name="valid-collection",
            collection_type=CollectionType.CODE,
            vector_size=1024,
            replication_factor=1,
            write_consistency_factor=1,
            segment_number=2
        )
        
        errors = QdrantSchema.validate_collection_config(config)
        assert len(errors) == 0
        
        # Invalid configurations
        invalid_configs = [
            # Empty name
            CollectionConfig("", CollectionType.CODE),
            # Invalid vector size
            CollectionConfig("test", CollectionType.CODE, vector_size=0),
            # Invalid replication factor
            CollectionConfig("test", CollectionType.CODE, replication_factor=0),
            # Invalid segment number
            CollectionConfig("test", CollectionType.CODE, segment_number=0)
        ]
        
        for invalid_config in invalid_configs:
            errors = QdrantSchema.validate_collection_config(invalid_config)
            assert len(errors) > 0
        
        # Test duplicate field names
        duplicate_indexes = [
            IndexConfig("field1", "text"),
            IndexConfig("field1", "keyword")  # Duplicate name
        ]
        config = CollectionConfig(
            name="test",
            collection_type=CollectionType.CODE,
            payload_indexes=duplicate_indexes
        )
        
        errors = QdrantSchema.validate_collection_config(config)
        assert any("Duplicate field name" in error for error in errors)
        
        # Test invalid field type
        invalid_indexes = [IndexConfig("field1", "invalid_type")]
        config = CollectionConfig(
            name="test",
            collection_type=CollectionType.CODE,
            payload_indexes=invalid_indexes
        )
        
        errors = QdrantSchema.validate_collection_config(config)
        assert any("Invalid field type" in error for error in errors)


class TestCollectionManager:
    """Test CollectionManager functionality"""
    
    def test_initialization(self):
        """Test collection manager initialization"""
        manager = CollectionManager("test-project")
        assert manager.project_name == "test-project"
        assert len(manager._configs) == 0
    
    def test_get_collection_name(self):
        """Test collection name generation"""
        manager = CollectionManager("My Test Project")
        
        # Test name normalization
        code_name = manager.get_collection_name(CollectionType.CODE)
        assert code_name == "my-test-project-code"
        
        relations_name = manager.get_collection_name(CollectionType.RELATIONS)
        assert relations_name == "my-test-project-relations"
        
        embeddings_name = manager.get_collection_name(CollectionType.EMBEDDINGS)
        assert embeddings_name == "my-test-project-embeddings"
    
    def test_get_all_collection_names(self):
        """Test getting all collection names"""
        manager = CollectionManager("test-project")
        names = manager.get_all_collection_names()
        
        assert len(names) == 3
        assert names[CollectionType.CODE] == "test-project-code"
        assert names[CollectionType.RELATIONS] == "test-project-relations"
        assert names[CollectionType.EMBEDDINGS] == "test-project-embeddings"
    
    def test_create_collection_config(self):
        """Test collection configuration creation"""
        manager = CollectionManager("test-project")
        
        # Create code collection config
        config = manager.create_collection_config(CollectionType.CODE)
        
        assert config.name == "test-project-code"
        assert config.collection_type == CollectionType.CODE
        assert config.vector_size == 1024
        assert len(config.payload_indexes) > 0
        
        # Verify it's cached
        cached_config = manager.get_collection_config(CollectionType.CODE)
        assert cached_config is config
    
    def test_create_collection_config_with_custom_settings(self):
        """Test collection configuration with custom settings"""
        manager = CollectionManager("test-project")
        
        custom_config = {
            "vector_size": 512,
            "distance_metric": DistanceMetric.EUCLIDEAN,
            "segment_number": 8
        }
        
        config = manager.create_collection_config(
            CollectionType.CODE,
            custom_config=custom_config
        )
        
        assert config.vector_size == 512
        assert config.distance_metric == DistanceMetric.EUCLIDEAN
        assert config.segment_number == 8
    
    def test_create_collection_config_validation_error(self):
        """Test collection configuration validation error"""
        manager = CollectionManager("test-project")
        
        # Invalid custom configuration
        custom_config = {"vector_size": -1}
        
        with pytest.raises(ValueError, match="Invalid collection config"):
            manager.create_collection_config(
                CollectionType.CODE,
                custom_config=custom_config
            )
    
    def test_list_collection_configs(self):
        """Test listing cached collection configurations"""
        manager = CollectionManager("test-project")
        
        # Initially empty
        configs = manager.list_collection_configs()
        assert len(configs) == 0
        
        # Create some configs
        manager.create_collection_config(CollectionType.CODE)
        manager.create_collection_config(CollectionType.RELATIONS)
        
        configs = manager.list_collection_configs()
        assert len(configs) == 2
        assert "test-project-code" in configs
        assert "test-project-relations" in configs
    
    def test_clear_configs(self):
        """Test clearing cached configurations"""
        manager = CollectionManager("test-project")
        
        # Create some configs
        manager.create_collection_config(CollectionType.CODE)
        manager.create_collection_config(CollectionType.RELATIONS)
        
        assert len(manager.list_collection_configs()) == 2
        
        # Clear configs
        manager.clear_configs()
        assert len(manager.list_collection_configs()) == 0
        
        # Verify they're actually gone
        assert manager.get_collection_config(CollectionType.CODE) is None
    
    def test_get_nonexistent_config(self):
        """Test getting non-existent configuration"""
        manager = CollectionManager("test-project")
        
        config = manager.get_collection_config(CollectionType.CODE)
        assert config is None


class TestCollectionTypeEnum:
    """Test CollectionType enum"""
    
    def test_collection_type_values(self):
        """Test collection type enum values"""
        assert CollectionType.CODE.value == "code"
        assert CollectionType.RELATIONS.value == "relations"
        assert CollectionType.EMBEDDINGS.value == "embeddings"
    
    def test_collection_type_iteration(self):
        """Test iterating over collection types"""
        types = list(CollectionType)
        assert len(types) == 3
        assert CollectionType.CODE in types
        assert CollectionType.RELATIONS in types
        assert CollectionType.EMBEDDINGS in types


class TestDistanceMetricEnum:
    """Test DistanceMetric enum"""
    
    def test_distance_metric_values(self):
        """Test distance metric enum values"""
        assert DistanceMetric.COSINE.value == "cosine"
        assert DistanceMetric.EUCLIDEAN.value == "euclidean"
        assert DistanceMetric.DOT_PRODUCT.value == "dot"
    
    def test_distance_metric_iteration(self):
        """Test iterating over distance metrics"""
        metrics = list(DistanceMetric)
        assert len(metrics) == 3
        assert DistanceMetric.COSINE in metrics
        assert DistanceMetric.EUCLIDEAN in metrics
        assert DistanceMetric.DOT_PRODUCT in metrics