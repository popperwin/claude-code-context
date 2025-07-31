"""
Qdrant schema definitions and collection management for claude-code-context.

Defines collection schemas, indexing strategies, and optimization parameters.
"""

from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging
import asyncio

logger = logging.getLogger(__name__)


class CollectionType(Enum):
    """Types of collections in the system"""
    CODE = "code"                # Code entities (functions, classes, etc.)
    RELATIONS = "relations"      # Entity relationships  
    EMBEDDINGS = "embeddings"    # Pure embeddings for semantic search


class DistanceMetric(Enum):
    """Distance metrics for vector similarity"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean" 
    DOT_PRODUCT = "dot"


@dataclass
class IndexConfig:
    """Configuration for payload field indexing"""
    field_name: str
    field_type: str  # text, keyword, integer, float, bool, geo
    index: bool = True
    store: bool = True
    
    def to_qdrant_schema(self) -> str:
        """Convert to Qdrant payload schema format"""
        schema_type_map = {
            'text': 'text',
            'keyword': 'keyword', 
            'integer': 'integer',
            'float': 'float',
            'bool': 'bool',
            'geo': 'geo'
        }
        
        # Qdrant expects just the type string, not a dictionary
        return schema_type_map.get(self.field_type, 'keyword')


@dataclass
class CollectionConfig:
    """Complete collection configuration"""
    name: str
    collection_type: CollectionType
    vector_size: int = 1024
    distance_metric: DistanceMetric = DistanceMetric.COSINE
    
    # Performance settings
    replication_factor: int = 1
    write_consistency_factor: int = 1
    on_disk_payload: bool = True
    
    # Optimization settings
    segment_number: int = 2
    memmap_threshold: int = 20000
    indexing_threshold: int = 20000
    
    # Payload schema
    payload_indexes: List[IndexConfig] = None
    
    def __post_init__(self):
        if self.payload_indexes is None:
            self.payload_indexes = []


class QdrantSchema:
    """Schema definitions for different collection types"""
    
    @staticmethod
    def get_code_collection_config(
        collection_name: str,
        vector_size: int = 1024,
        distance_metric: DistanceMetric = DistanceMetric.COSINE
    ) -> CollectionConfig:
        """
        Get configuration for code entities collection.
        
        Args:
            collection_name: Name of the collection
            vector_size: Embedding vector dimensions
            distance_metric: Vector similarity metric
            
        Returns:
            Complete collection configuration
        """
        # Define payload indexes for code entities
        payload_indexes = [
            # Text fields for semantic search
            IndexConfig("entity_name", "text", index=True, store=True),
            IndexConfig("qualified_name", "text", index=True, store=True),
            IndexConfig("signature", "text", index=True, store=True),
            IndexConfig("docstring", "text", index=True, store=True),
            IndexConfig("source_code", "text", index=False, store=True),  # Store but don't index
            
            # Keyword fields for filtering
            IndexConfig("entity_type", "keyword", index=True, store=True),
            IndexConfig("file_path", "keyword", index=True, store=True),
            IndexConfig("visibility", "keyword", index=True, store=True),
            IndexConfig("language", "keyword", index=True, store=True),
            
            # Numeric fields
            IndexConfig("start_line", "integer", index=True, store=True),
            IndexConfig("end_line", "integer", index=True, store=True),
            IndexConfig("start_column", "integer", index=True, store=True),
            IndexConfig("end_column", "integer", index=True, store=True),
            IndexConfig("start_byte", "integer", index=True, store=True),
            IndexConfig("end_byte", "integer", index=True, store=True),
            
            # Boolean fields
            IndexConfig("is_async", "bool", index=True, store=True),
            IndexConfig("is_test", "bool", index=True, store=True),
            IndexConfig("is_deprecated", "bool", index=True, store=True),
            
            # Hash for deduplication
            IndexConfig("source_hash", "keyword", index=True, store=True),
        ]
        
        return CollectionConfig(
            name=collection_name,
            collection_type=CollectionType.CODE,
            vector_size=vector_size,
            distance_metric=distance_metric,
            payload_indexes=payload_indexes,
            
            # Optimized for code search
            on_disk_payload=True,  # Store payload on disk for large codebases
            segment_number=4,      # More segments for better parallelization
            memmap_threshold=10000,  # Memory map for large collections
            indexing_threshold=10000
        )
    
    @staticmethod
    def get_relations_collection_config(
        collection_name: str,
        vector_size: int = 1024,
        distance_metric: DistanceMetric = DistanceMetric.COSINE
    ) -> CollectionConfig:
        """
        Get configuration for entity relations collection.
        
        Args:
            collection_name: Name of the collection
            vector_size: Embedding vector dimensions  
            distance_metric: Vector similarity metric
            
        Returns:
            Complete collection configuration
        """
        payload_indexes = [
            # Relation identification
            IndexConfig("relation_type", "keyword", index=True, store=True),
            IndexConfig("source_entity_id", "keyword", index=True, store=True),
            IndexConfig("target_entity_id", "keyword", index=True, store=True),
            
            # Context information
            IndexConfig("context", "text", index=True, store=True),
            IndexConfig("strength", "float", index=True, store=True),
            
            # File and location info
            IndexConfig("source_file_path", "keyword", index=True, store=True),
            IndexConfig("target_file_path", "keyword", index=True, store=True),
        ]
        
        return CollectionConfig(
            name=collection_name,
            collection_type=CollectionType.RELATIONS,
            vector_size=vector_size,
            distance_metric=distance_metric,
            payload_indexes=payload_indexes,
            
            # Optimized for relationship queries
            on_disk_payload=False,  # Keep in memory for fast relation queries
            segment_number=2,
            memmap_threshold=50000,
            indexing_threshold=20000
        )
    
    @staticmethod
    def get_embeddings_collection_config(
        collection_name: str,
        vector_size: int = 1024,
        distance_metric: DistanceMetric = DistanceMetric.COSINE
    ) -> CollectionConfig:
        """
        Get configuration for pure embeddings collection.
        
        Args:
            collection_name: Name of the collection
            vector_size: Embedding vector dimensions
            distance_metric: Vector similarity metric
            
        Returns:
            Complete collection configuration
        """
        payload_indexes = [
            # Minimal payload for pure semantic search
            IndexConfig("content_type", "keyword", index=True, store=True),
            IndexConfig("content_hash", "keyword", index=True, store=True),
            IndexConfig("file_path", "keyword", index=True, store=True),
            IndexConfig("chunk_index", "integer", index=True, store=True),
        ]
        
        return CollectionConfig(
            name=collection_name,
            collection_type=CollectionType.EMBEDDINGS,
            vector_size=vector_size,
            distance_metric=distance_metric,
            payload_indexes=payload_indexes,
            
            # Optimized for pure vector search
            on_disk_payload=True,  # Minimal payload can be on disk
            segment_number=8,      # More segments for vector search optimization
            memmap_threshold=100000,
            indexing_threshold=50000
        )
    
    @staticmethod
    def get_collection_config(
        collection_name: str,
        collection_type: CollectionType,
        vector_size: int = 1024,
        distance_metric: DistanceMetric = DistanceMetric.COSINE
    ) -> CollectionConfig:
        """
        Get collection configuration by type.
        
        Args:
            collection_name: Name of the collection
            collection_type: Type of collection
            vector_size: Embedding vector dimensions
            distance_metric: Vector similarity metric
            
        Returns:
            Complete collection configuration
        """
        if collection_type == CollectionType.CODE:
            return QdrantSchema.get_code_collection_config(
                collection_name, vector_size, distance_metric
            )
        elif collection_type == CollectionType.RELATIONS:
            return QdrantSchema.get_relations_collection_config(
                collection_name, vector_size, distance_metric
            )
        elif collection_type == CollectionType.EMBEDDINGS:
            return QdrantSchema.get_embeddings_collection_config(
                collection_name, vector_size, distance_metric
            )
        else:
            raise ValueError(f"Unknown collection type: {collection_type}")
    
    @staticmethod
    def get_qdrant_vectors_config(config: CollectionConfig) -> Dict[str, Any]:
        """
        Convert collection config to Qdrant vectors configuration.
        
        Args:
            config: Collection configuration
            
        Returns:
            Qdrant vectors config dictionary
        """
        # Map distance metrics
        distance_map = {
            DistanceMetric.COSINE: "Cosine",
            DistanceMetric.EUCLIDEAN: "Euclid", 
            DistanceMetric.DOT_PRODUCT: "Dot"
        }
        
        return {
            "size": config.vector_size,
            "distance": distance_map[config.distance_metric],
            "on_disk": config.on_disk_payload
        }
    
    @staticmethod
    def get_qdrant_config_params(config: CollectionConfig) -> Dict[str, Any]:
        """
        Convert collection config to Qdrant configuration parameters.
        
        Args:
            config: Collection configuration
            
        Returns:
            Qdrant config parameters dictionary
        """
        return {
            "replication_factor": config.replication_factor,
            "write_consistency_factor": config.write_consistency_factor,
            "optimizers_config": {
                "deleted_threshold": 0.2,
                "vacuum_min_vector_number": 1000,
                "default_segment_number": config.segment_number,
                "max_segment_size": None,
                "memmap_threshold": config.memmap_threshold,
                "indexing_threshold": config.indexing_threshold,
                "flush_interval_sec": 5,
                "max_optimization_threads": None
            },
            "hnsw_config": {
                "m": 16,
                "ef_construct": 100,
                "full_scan_threshold": 10000,
                "max_indexing_threads": 0,
                "on_disk": config.on_disk_payload
            },
            "wal_config": {
                "wal_capacity_mb": 32,
                "wal_segments_ahead": 0
            }
        }
    
    @staticmethod
    def get_payload_schema(config: CollectionConfig) -> Dict[str, str]:
        """
        Convert collection config to Qdrant payload schema.
        
        Args:
            config: Collection configuration
            
        Returns:
            Qdrant payload schema dictionary mapping field names to type strings
        """
        schema = {}
        
        for index_config in config.payload_indexes:
            field_name = index_config.field_name
            schema[field_name] = index_config.to_qdrant_schema()
        
        return schema
    
    @staticmethod
    def validate_collection_config(config: CollectionConfig) -> List[str]:
        """
        Validate collection configuration.
        
        Args:
            config: Collection configuration to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate basic parameters
        if not config.name or not isinstance(config.name, str):
            errors.append("Collection name must be a non-empty string")
        
        if config.vector_size <= 0:
            errors.append("Vector size must be positive")
        
        if config.replication_factor < 1:
            errors.append("Replication factor must be at least 1")
        
        if config.write_consistency_factor < 1:
            errors.append("Write consistency factor must be at least 1")
        
        if config.segment_number < 1:
            errors.append("Segment number must be at least 1")
        
        # Validate payload indexes
        field_names = set()
        for index_config in config.payload_indexes:
            if not index_config.field_name:
                errors.append("Index field name cannot be empty")
            
            if index_config.field_name in field_names:
                errors.append(f"Duplicate field name: {index_config.field_name}")
            
            field_names.add(index_config.field_name)
            
            if index_config.field_type not in ['text', 'keyword', 'integer', 'float', 'bool', 'geo']:
                errors.append(f"Invalid field type: {index_config.field_type}")
        
        return errors


class CollectionManager:
    """Manager for collection lifecycle and schema operations"""
    
    def __init__(self, project_name: str):
        """
        Initialize collection manager.
        
        Args:
            project_name: Project name for collection naming
        """
        self.project_name = project_name
        self._configs: Dict[str, CollectionConfig] = {}
        
        logger.info(f"Initialized collection manager for project: {project_name}")
    
    def get_collection_name(self, collection_type: CollectionType) -> str:
        """
        Generate collection name for project and type.
        
        Args:
            collection_type: Type of collection
            
        Returns:
            Generated collection name
        """
        safe_project = self.project_name.lower().replace(' ', '-').replace('_', '-')
        return f"{safe_project}-{collection_type.value}"
    
    def get_all_collection_names(self) -> Dict[CollectionType, str]:
        """Get all collection names for the project"""
        return {
            collection_type: self.get_collection_name(collection_type)
            for collection_type in CollectionType
        }
    
    def create_collection_config(
        self,
        collection_type: CollectionType,
        vector_size: int = 1024,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> CollectionConfig:
        """
        Create collection configuration.
        
        Args:
            collection_type: Type of collection
            vector_size: Embedding vector dimensions
            distance_metric: Vector similarity metric
            custom_config: Optional custom configuration overrides
            
        Returns:
            Collection configuration
        """
        collection_name = self.get_collection_name(collection_type)
        
        # Get base configuration
        config = QdrantSchema.get_collection_config(
            collection_name,
            collection_type,
            vector_size,
            distance_metric
        )
        
        # Apply custom configuration
        if custom_config:
            for key, value in custom_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # Validate configuration
        errors = QdrantSchema.validate_collection_config(config)
        if errors:
            raise ValueError(f"Invalid collection config: {'; '.join(errors)}")
        
        # Cache configuration
        self._configs[collection_name] = config
        
        logger.debug(f"Created collection config: {collection_name}")
        return config
    
    def get_collection_config(self, collection_type: CollectionType) -> Optional[CollectionConfig]:
        """Get cached collection configuration"""
        collection_name = self.get_collection_name(collection_type)
        return self._configs.get(collection_name)
    
    def list_collection_configs(self) -> Dict[str, CollectionConfig]:
        """List all cached collection configurations"""
        return self._configs.copy()
    
    def clear_configs(self) -> None:
        """Clear all cached configurations"""
        self._configs.clear()
        logger.debug("Cleared all collection configurations")
    
    async def ensure_collection_exists(
        self,
        collection_type: CollectionType,
        storage_client,  # HybridQdrantClient - avoiding circular import
        vector_size: int = 1024,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Ensure collection exists, creating it if necessary.
        
        Args:
            collection_type: Type of collection to ensure exists
            storage_client: HybridQdrantClient for storage operations
            vector_size: Embedding vector dimensions
            distance_metric: Vector similarity metric
            custom_config: Optional custom configuration overrides
            
        Returns:
            The actual collection name created/verified
            
        Raises:
            RuntimeError: If collection creation fails
        """
        collection_name = self.get_collection_name(collection_type)
        
        # Check if collection already exists
        existing_info = await storage_client.get_collection_info(collection_name)
        
        if existing_info:
            logger.debug(f"Collection '{collection_name}' already exists")
            return collection_name
        
        # Create collection with proper schema
        config = self.create_collection_config(
            collection_type=collection_type,
            vector_size=vector_size,
            distance_metric=distance_metric,
            custom_config=custom_config
        )
        
        result = await storage_client.create_collection(config, recreate=False)
        
        if result.success:
            logger.info(f"Created collection '{collection_name}' for project '{self.project_name}'")
        else:
            error_msg = f"Failed to create collection '{collection_name}': {result.error}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        return collection_name
    
    async def delete_collection(
        self,
        collection_type: CollectionType,
        storage_client  # HybridQdrantClient - avoiding circular import
    ) -> bool:
        """
        Delete a collection.
        
        Args:
            collection_type: Type of collection to delete
            storage_client: HybridQdrantClient for storage operations
            
        Returns:
            True if deletion was successful, False otherwise
        """
        collection_name = self.get_collection_name(collection_type)
        
        try:
            result = await storage_client.delete_collection(collection_name)
            
            if result.success:
                # Clear from cache
                self._configs.pop(collection_name, None)
                logger.info(f"Deleted collection '{collection_name}' for project '{self.project_name}'")
                return True
            else:
                logger.error(f"Failed to delete collection '{collection_name}': {result.error}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting collection '{collection_name}': {e}")
            return False
    
    async def get_collection_info(
        self,
        collection_type: CollectionType,
        storage_client  # HybridQdrantClient - avoiding circular import
    ) -> Optional[Dict[str, Any]]:
        """
        Get information about a collection.
        
        Args:
            collection_type: Type of collection to get info for
            storage_client: HybridQdrantClient for storage operations
            
        Returns:
            Collection information dictionary, or None if collection doesn't exist
        """
        collection_name = self.get_collection_name(collection_type)
        
        try:
            return await storage_client.get_collection_info(collection_name)
        except Exception as e:
            logger.error(f"Error getting collection info for '{collection_name}': {e}")
            return None
    
    async def collection_exists(
        self,
        collection_type: CollectionType,
        storage_client  # HybridQdrantClient - avoiding circular import
    ) -> bool:
        """
        Check if a collection exists.
        
        Args:
            collection_type: Type of collection to check
            storage_client: HybridQdrantClient for storage operations
            
        Returns:
            True if collection exists, False otherwise
        """
        info = await self.get_collection_info(collection_type, storage_client)
        return info is not None
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get collection manager status.
        
        Returns:
            Dictionary with collection manager status and metrics
        """
        return {
            "project_name": self.project_name,
            "cached_configs": len(self._configs),
            "collection_names": self.get_all_collection_names()
        }