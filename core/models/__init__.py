"""
Core data models for claude-code-context

All Pydantic models for entities, storage, configuration, and hooks.
"""

from .entities import Entity, EntityType, SourceLocation, Visibility
from .storage import QdrantPoint, StorageResult, OperationResult
from .config import ProjectConfig, QdrantConfig, StellaConfig
from .hooks import HookRequest, HookResponse, CCCQuery
from .embeddings import EmbeddingResponse, EmbeddingRequest, EmbeddingConfig, EmbeddingStats

__all__ = [
    # Entities
    "Entity",
    "EntityType",
    "SourceLocation", 
    "Visibility",
    
    # Storage
    "QdrantPoint",
    "StorageResult",
    "OperationResult",
    
    # Configuration
    "ProjectConfig",
    "QdrantConfig",
    "StellaConfig",
    
    # Hooks
    "HookRequest",
    "HookResponse", 
    "CCCQuery",
    
    # Embeddings
    "EmbeddingResponse",
    "EmbeddingRequest",
    "EmbeddingConfig",
    "EmbeddingStats"
]