"""
Embedding providers for claude-code-context.

Supports local embedding generation with multiple providers.
"""

from .base import (
    BaseEmbedder, 
    EmbedderProtocol, 
    EmbeddingRequest, 
    EmbeddingResponse,
    EmbeddingManager
)
from .stella import StellaEmbedder
from .cache import EmbeddingCache, get_default_cache, clear_default_cache
from .registry import (
    EmbedderRegistry, 
    get_registry, 
    create_embedder, 
    list_available_embedders,
    get_default_embedder,
    EmbedderManager as RegistryEmbedderManager
)

__all__ = [
    "BaseEmbedder",
    "EmbedderProtocol", 
    "EmbeddingRequest",
    "EmbeddingResponse",
    "EmbeddingManager",
    "StellaEmbedder",
    "EmbeddingCache",
    "get_default_cache",
    "clear_default_cache",
    "EmbedderRegistry",
    "get_registry",
    "create_embedder",
    "list_available_embedders",
    "get_default_embedder",
    "RegistryEmbedderManager"
]