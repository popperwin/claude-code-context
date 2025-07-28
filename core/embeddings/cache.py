"""
Embedding cache implementation with LRU eviction and TTL expiration.

Provides thread-safe caching for embeddings to avoid redundant computations.
"""

import hashlib
import threading
import time
from collections import OrderedDict
from typing import List, Optional, Tuple, Dict, Any
import logging
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with embedding and metadata"""
    embedding: List[float]
    timestamp: float
    access_count: int = 0
    last_access: float = 0.0
    
    def __post_init__(self):
        if self.last_access == 0.0:
            self.last_access = self.timestamp


class EmbeddingCache:
    """Thread-safe LRU cache with TTL for embeddings"""
    
    def __init__(
        self, 
        max_size: int = 10000, 
        ttl_seconds: int = 3600,
        cleanup_interval: int = 300
    ):
        """
        Initialize embedding cache.
        
        Args:
            max_size: Maximum number of entries to cache
            ttl_seconds: Time-to-live for cache entries
            cleanup_interval: Interval between cleanup operations (seconds)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cleanup_interval = cleanup_interval
        
        # Thread-safe storage
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._last_cleanup = time.time()
        
        logger.info(
            f"Initialized embedding cache: max_size={max_size}, "
            f"ttl={ttl_seconds}s, cleanup_interval={cleanup_interval}s"
        )
    
    def _generate_cache_key(self, text: str, model: str = "default") -> str:
        """Generate cache key from text and model"""
        # Use SHA-256 for consistent, collision-resistant keys
        content = f"{model}:{text}".encode('utf-8')
        return hashlib.sha256(content).hexdigest()[:32]  # 32 chars for performance
    
    def get(self, text: str, model: str = "default") -> Optional[List[float]]:
        """
        Get embedding from cache.
        
        Args:
            text: Input text
            model: Model identifier
            
        Returns:
            Cached embedding or None if not found/expired
        """
        if not text.strip():
            return None
        
        key = self._generate_cache_key(text, model)
        current_time = time.time()
        
        with self._lock:
            # Check if entry exists
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check TTL
            if current_time - entry.timestamp > self.ttl_seconds:
                # Expired - remove entry
                del self._cache[key]
                self._misses += 1
                logger.debug(f"Cache entry expired: {key[:8]}...")
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            
            # Update access statistics
            entry.access_count += 1
            entry.last_access = current_time
            
            self._hits += 1
            logger.debug(f"Cache hit: {key[:8]}... (access #{entry.access_count})")
            return entry.embedding.copy()  # Return copy to prevent modification
    
    def put(
        self, 
        text: str, 
        embedding: List[float], 
        model: str = "default"
    ) -> None:
        """
        Store embedding in cache.
        
        Args:
            text: Input text
            embedding: Generated embedding
            model: Model identifier
        """
        if not text.strip() or not embedding:
            return
        
        key = self._generate_cache_key(text, model)
        current_time = time.time()
        
        with self._lock:
            # Create new entry
            entry = CacheEntry(
                embedding=embedding.copy(),  # Store copy to prevent external modification
                timestamp=current_time,
                last_access=current_time
            )
            
            # Add to cache
            self._cache[key] = entry
            self._cache.move_to_end(key)  # Mark as most recently used
            
            # Enforce size limit
            while len(self._cache) > self.max_size:
                # Remove least recently used entry
                oldest_key, oldest_entry = self._cache.popitem(last=False)
                self._evictions += 1
                logger.debug(
                    f"Evicted LRU entry: {oldest_key[:8]}... "
                    f"(age: {current_time - oldest_entry.timestamp:.1f}s, "
                    f"accesses: {oldest_entry.access_count})"
                )
            
            # Periodic cleanup
            if current_time - self._last_cleanup > self.cleanup_interval:
                self._cleanup_expired()
                self._last_cleanup = current_time
            
            logger.debug(f"Cached embedding: {key[:8]}... (size: {len(self._cache)})")
    
    def _cleanup_expired(self) -> int:
        """Remove expired entries from cache"""
        current_time = time.time()
        expired_keys = []
        
        # Identify expired entries
        for key, entry in self._cache.items():
            if current_time - entry.timestamp > self.ttl_seconds:
                expired_keys.append(key)
        
        # Remove expired entries
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    def get_batch(
        self, 
        texts: List[str], 
        model: str = "default"
    ) -> Tuple[List[Optional[List[float]]], List[int]]:
        """
        Get batch of embeddings from cache.
        
        Args:
            texts: List of input texts
            model: Model identifier
            
        Returns:
            Tuple of (embeddings_list, cache_miss_indices)
            embeddings_list contains cached embeddings or None for misses
            cache_miss_indices contains indices of texts that need computation
        """
        embeddings = []
        cache_miss_indices = []
        
        for i, text in enumerate(texts):
            embedding = self.get(text, model)
            embeddings.append(embedding)
            
            if embedding is None:
                cache_miss_indices.append(i)
        
        logger.debug(
            f"Batch cache lookup: {len(texts)} texts, "
            f"{len(cache_miss_indices)} misses ({len(cache_miss_indices)/len(texts)*100:.1f}%)"
        )
        
        return embeddings, cache_miss_indices
    
    def put_batch(
        self, 
        texts: List[str], 
        embeddings: List[List[float]], 
        model: str = "default"
    ) -> None:
        """
        Store batch of embeddings in cache.
        
        Args:
            texts: List of input texts
            embeddings: List of generated embeddings
            model: Model identifier
        """
        if len(texts) != len(embeddings):
            raise ValueError("texts and embeddings must have same length")
        
        for text, embedding in zip(texts, embeddings):
            self.put(text, embedding, model)
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            
            # Calculate memory usage estimate
            memory_bytes = 0
            if self._cache:
                # Estimate: 32 bytes per key + 4 bytes per float + overhead
                avg_embedding_size = len(next(iter(self._cache.values())).embedding)
                memory_bytes = len(self._cache) * (32 + avg_embedding_size * 4 + 64)
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": hit_rate,
                "memory_usage_mb": memory_bytes / (1024 * 1024),
                "ttl_seconds": self.ttl_seconds,
                "cleanup_interval": self.cleanup_interval
            }
    
    def get_info(self) -> str:
        """Get human-readable cache information"""
        stats = self.get_stats()
        return (
            f"EmbeddingCache(size={stats['size']}/{stats['max_size']}, "
            f"hit_rate={stats['hit_rate']:.1%}, "
            f"memory={stats['memory_usage_mb']:.1f}MB)"
        )
    
    def force_cleanup(self) -> int:
        """Force cleanup of expired entries"""
        with self._lock:
            return self._cleanup_expired()
    
    def resize(self, new_max_size: int) -> None:
        """
        Resize cache to new maximum size.
        
        Args:
            new_max_size: New maximum cache size
        """
        if new_max_size < 1:
            raise ValueError("max_size must be at least 1")
        
        with self._lock:
            self.max_size = new_max_size
            
            # Evict entries if necessary
            while len(self._cache) > self.max_size:
                oldest_key, oldest_entry = self._cache.popitem(last=False)
                self._evictions += 1
            
            logger.info(f"Cache resized to max_size={new_max_size}")


class ThreadSafeEmbeddingCache(EmbeddingCache):
    """
    Thread-safe embedding cache with additional safety features.
    
    This is an alias for EmbeddingCache since it's already thread-safe,
    but provides explicit naming for clarity.
    """
    pass


# Global cache instance for default usage
_default_cache: Optional[EmbeddingCache] = None


def get_default_cache(
    max_size: int = 10000, 
    ttl_seconds: int = 3600
) -> EmbeddingCache:
    """Get or create default cache instance"""
    global _default_cache
    
    if _default_cache is None:
        _default_cache = EmbeddingCache(max_size, ttl_seconds)
    
    return _default_cache


def clear_default_cache() -> None:
    """Clear the default cache"""
    global _default_cache
    
    if _default_cache is not None:
        _default_cache.clear()