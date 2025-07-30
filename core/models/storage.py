"""
Storage models for Qdrant integration and operation results.

Handles vector storage, search results, and operation tracking.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Generic, TypeVar
from pydantic import BaseModel, Field, ConfigDict, field_validator, computed_field


T = TypeVar('T')


class OperationStatus(Enum):
    """Status of storage operations"""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class OperationResult(BaseModel, Generic[T]):
    """Standard operation result wrapper for all operations"""
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )
    
    # Result status
    status: OperationStatus
    
    # Data
    data: Optional[T] = None
    
    # Error information
    error: Optional[str] = None
    error_code: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    
    # Operation metadata
    operation_type: str
    timestamp: datetime = Field(default_factory=datetime.now)
    processing_time_ms: Optional[float] = None
    
    # Performance metrics
    items_processed: int = 0
    items_succeeded: int = 0
    items_failed: int = 0
    
    @computed_field
    @property
    def success(self) -> bool:
        """Computed property for success status"""
        return self.status == OperationStatus.SUCCESS
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate for batch operations"""
        if self.items_processed == 0:
            return 1.0 if self.success else 0.0
        return self.items_succeeded / self.items_processed
    
    @classmethod
    def success_result(
        cls,
        data: T,
        operation_type: str,
        processing_time_ms: Optional[float] = None,
        items_processed: int = 1
    ) -> 'OperationResult[T]':
        """Create successful result"""
        return cls(
            status=OperationStatus.SUCCESS,
            data=data,
            operation_type=operation_type,
            processing_time_ms=processing_time_ms,
            items_processed=items_processed,
            items_succeeded=items_processed
        )
    
    @classmethod
    def error_result(
        cls,
        error: str,
        operation_type: str,
        error_code: Optional[str] = None,
        error_details: Optional[Dict[str, Any]] = None,
        processing_time_ms: Optional[float] = None
    ) -> 'OperationResult[T]':
        """Create error result"""
        return cls(
            status=OperationStatus.FAILED,
            error=error,
            error_code=error_code,
            error_details=error_details,
            operation_type=operation_type,
            processing_time_ms=processing_time_ms
        )


class QdrantPoint(BaseModel):
    """Qdrant vector storage point representation"""
    model_config = ConfigDict(frozen=True)
    
    # Point identification
    id: str  # Usually entity ID or content hash
    
    # Vector data
    vector: List[float]
    
    # Payload data for filtering and retrieval
    payload: Dict[str, Any]
    
    # Metadata
    collection_name: Optional[str] = None
    version: int = 1
    
    @field_validator('id')
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Validate point ID is not empty"""
        if not v.strip():
            raise ValueError('Point ID cannot be empty')
        return v
    
    @field_validator('vector')
    @classmethod
    def validate_vector(cls, v: List[float]) -> List[float]:
        """Validate vector dimensions"""
        if not v:
            raise ValueError('Vector cannot be empty')
        if len(v) != 1024:  # Stella embedding dimension
            raise ValueError(f'Vector must have 1024 dimensions, got {len(v)}')
        return v
    
    @field_validator('payload')
    @classmethod
    def validate_payload(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate payload contains required fields"""
        required_fields = {'entity_id', 'entity_type', 'file_path'}
        missing = required_fields - set(v.keys())
        if missing:
            raise ValueError(f'Payload missing required fields: {missing}')
        return v
    
    @property
    def entity_id(self) -> str:
        """Get entity ID from payload"""
        return self.payload.get('entity_id', self.id)
    
    @property
    def entity_type(self) -> str:
        """Get entity type from payload"""
        return self.payload.get('entity_type', 'unknown')
    
    @property
    def file_path(self) -> str:
        """Get file path from payload"""
        return self.payload.get('file_path', '')
    
    def to_qdrant_format(self) -> Dict[str, Any]:
        """Convert to Qdrant client format"""
        return {
            'id': self.id,
            'vector': self.vector,
            'payload': self.payload
        }
    
    @classmethod
    def from_qdrant_format(cls, data: Dict[str, Any]) -> 'QdrantPoint':
        """Create from Qdrant client response"""
        return cls(
            id=str(data['id']),
            vector=data['vector'],
            payload=data['payload']
        )


class SearchResult(BaseModel):
    """Search result with score and metadata"""
    model_config = ConfigDict(frozen=True)
    
    # Result identification
    point: QdrantPoint
    score: float = Field(ge=0.0, le=1.0)
    
    # Search context
    query: str
    search_type: str  # semantic, hybrid, payload
    
    # Ranking metadata
    rank: int = Field(ge=1)
    total_results: int = Field(ge=1)
    
    # Additional scoring
    relevance_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    semantic_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    keyword_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    
    @field_validator('score')
    @classmethod
    def validate_score(cls, v: float) -> float:
        """Ensure score is reasonable"""
        if v < 0.0 or v > 1.0:
            raise ValueError('Score must be between 0.0 and 1.0')
        return round(v, 6)  # Limit precision
    
    @property
    def entity_id(self) -> str:
        """Get entity ID from point"""
        return self.point.entity_id
    
    @property
    def entity_type(self) -> str:
        """Get entity type from point"""
        return self.point.entity_type
    
    @property
    def file_path(self) -> str:
        """Get file path from point"""
        return self.point.file_path
    
    @property
    def is_highly_relevant(self) -> bool:
        """Check if result is highly relevant (>0.8 score)"""
        return self.score > 0.8
    
    @property
    def is_moderately_relevant(self) -> bool:
        """Check if result is moderately relevant (>0.6 score)"""
        return self.score > 0.6


class StorageResult(BaseModel):
    """Result of storage operations with detailed metrics"""
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )
    
    # Operation details
    operation: str  # insert, update, delete, search, bulk_insert
    collection_name: str
    success: bool
    
    # Performance metrics
    processing_time_ms: float
    affected_count: int = 0
    total_count: int = 0
    
    # Error handling
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    warnings: List[str] = Field(default_factory=list)
    
    # Timestamps
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    # Operation-specific data
    details: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('operation')
    @classmethod
    def validate_operation(cls, v: str) -> str:
        """Validate operation type"""
        valid_ops = {
            'insert', 'update', 'delete', 'search', 'bulk_insert',
            'bulk_update', 'bulk_delete', 'upsert', 'create_collection',
            'delete_collection', 'get_collection_info'
        }
        if v.lower() not in valid_ops:
            raise ValueError(f'Invalid operation: {v}')
        return v.lower()
    
    @field_validator('collection_name')
    @classmethod
    def validate_collection_name(cls, v: str) -> str:
        """Validate collection name format"""
        if not v or not v.replace('-', '').replace('_', '').isalnum():
            raise ValueError('Collection name must be alphanumeric with dashes/underscores')
        return v.lower()
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate for operations"""
        if self.total_count == 0:
            return 1.0 if self.success else 0.0
        return self.affected_count / self.total_count
    
    @property
    def throughput_per_second(self) -> float:
        """Calculate processing throughput"""
        if self.processing_time_ms <= 0:
            return 0.0
        return (self.total_count * 1000) / self.processing_time_ms
    
    def mark_completed(self) -> None:
        """Mark operation as completed"""
        self.completed_at = datetime.now()
    
    @classmethod
    def successful_insert(
        cls,
        collection_name: str,
        count: int,
        processing_time_ms: float
    ) -> 'StorageResult':
        """Create successful insert result"""
        return cls(
            operation='insert',
            collection_name=collection_name,
            success=True,
            processing_time_ms=processing_time_ms,
            affected_count=count,
            total_count=count
        )
    
    @classmethod
    def successful_delete(
        cls,
        collection_name: str,
        count: int,
        processing_time_ms: float
    ) -> 'StorageResult':
        """Create successful delete result"""
        return cls(
            operation='delete',
            collection_name=collection_name,
            success=True,
            processing_time_ms=processing_time_ms,
            affected_count=count,
            total_count=count
        )
    
    @classmethod
    def failed_operation(
        cls,
        operation: str,
        collection_name: str,
        error: str,
        processing_time_ms: float,
        error_details: Optional[Dict[str, Any]] = None
    ) -> 'StorageResult':
        """Create failed operation result"""
        return cls(
            operation=operation,
            collection_name=collection_name,
            success=False,
            processing_time_ms=processing_time_ms,
            error=error,
            error_details=error_details
        )


class CollectionInfo(BaseModel):
    """Information about a Qdrant collection"""
    model_config = ConfigDict(frozen=True)
    
    # Collection details
    name: str
    vectors_count: int
    indexed_vectors_count: int
    points_count: int
    
    # Configuration
    vector_size: int
    distance_metric: str
    
    # Status
    status: str
    optimizer_status: Optional[str] = None
    
    # Timestamps
    created_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    
    # Storage metrics
    disk_usage_bytes: Optional[int] = None
    ram_usage_bytes: Optional[int] = None
    
    @property
    def indexing_progress(self) -> float:
        """Calculate indexing progress percentage"""
        if self.vectors_count == 0:
            return 1.0
        return self.indexed_vectors_count / self.vectors_count
    
    @property
    def is_ready(self) -> bool:
        """Check if collection is ready for operations"""
        return self.status.lower() in {'green', 'ready', 'active'}
    
    @property
    def disk_usage_mb(self) -> Optional[float]:
        """Get disk usage in MB"""
        if self.disk_usage_bytes is None:
            return None
        return self.disk_usage_bytes / (1024 * 1024)
    
    @property
    def ram_usage_mb(self) -> Optional[float]:
        """Get RAM usage in MB"""
        if self.ram_usage_bytes is None:
            return None
        return self.ram_usage_bytes / (1024 * 1024)