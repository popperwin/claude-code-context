"""
Embedding data models for claude-code-context.

Defines embedding request/response models and configuration.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pydantic import BaseModel, Field


@dataclass
class EmbeddingResponse:
    """Response from embedding generation"""
    embeddings: List[List[float]]
    model: str
    total_tokens: int
    processing_time_ms: Optional[float] = None
    cache_hits: int = 0
    cache_misses: int = 0
    
    @property
    def embedding_count(self) -> int:
        """Number of embeddings generated"""
        return len(self.embeddings)
    
    @property
    def cache_hit_rate(self) -> float:
        """Cache hit rate (0.0 to 1.0)"""
        total_requests = self.cache_hits + self.cache_misses
        if total_requests == 0:
            return 0.0
        return self.cache_hits / total_requests


@dataclass
class EmbeddingRequest:
    """Request for embedding generation"""
    texts: List[str]
    model: Optional[str] = None
    normalize: bool = True
    batch_size: Optional[int] = None
    
    @property
    def text_count(self) -> int:
        """Number of texts to embed"""
        return len(self.texts)


class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation"""
    model_name: str = Field(default="stella_en_400M_v5", description="Embedding model name")
    vector_size: int = Field(default=1024, description="Embedding vector dimensions")
    normalize_embeddings: bool = Field(default=True, description="Whether to normalize embeddings")
    batch_size: int = Field(default=32, description="Batch size for embedding generation")
    cache_embeddings: bool = Field(default=True, description="Whether to cache embeddings")
    device: Optional[str] = Field(default=None, description="Device to use (auto-detected if None)")
    
    class Config:
        """Pydantic configuration"""
        frozen = True
        extra = "forbid"


@dataclass
class EmbeddingStats:
    """Statistics for embedding operations"""
    total_embeddings_generated: int = 0
    total_processing_time_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    average_embedding_time_ms: float = 0.0
    
    @property
    def cache_hit_rate(self) -> float:
        """Cache hit rate (0.0 to 1.0)"""
        total_requests = self.cache_hits + self.cache_misses
        if total_requests == 0:
            return 0.0
        return self.cache_hits / total_requests
    
    @property
    def embeddings_per_second(self) -> float:
        """Embeddings generated per second"""
        if self.total_processing_time_ms == 0:
            return 0.0
        return (self.total_embeddings_generated * 1000) / self.total_processing_time_ms
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "total_embeddings_generated": self.total_embeddings_generated,
            "total_processing_time_ms": self.total_processing_time_ms,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hit_rate,
            "average_embedding_time_ms": self.average_embedding_time_ms,
            "embeddings_per_second": self.embeddings_per_second
        }