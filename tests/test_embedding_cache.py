"""
Unit tests for embedding cache functionality.

Tests LRU eviction, TTL expiration, thread safety, and performance.
"""

import pytest
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List

from core.embeddings.cache import EmbeddingCache, CacheEntry


class TestEmbeddingCache:
    """Test embedding cache functionality"""
    
    def test_cache_initialization(self):
        """Test cache initialization with various parameters"""
        # Default initialization
        cache = EmbeddingCache()
        assert cache.max_size == 10000
        assert cache.ttl_seconds == 3600
        
        # Custom initialization
        cache = EmbeddingCache(max_size=100, ttl_seconds=60)
        assert cache.max_size == 100
        assert cache.ttl_seconds == 60
        
        stats = cache.get_stats()
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
    
    def test_basic_get_put_operations(self):
        """Test basic cache operations"""
        cache = EmbeddingCache(max_size=10)
        
        # Test miss
        result = cache.get("test_text")
        assert result is None
        
        # Test put and hit
        embedding = [1.0, 2.0, 3.0]
        cache.put("test_text", embedding)
        
        result = cache.get("test_text")
        assert result == embedding
        assert result is not embedding  # Should be a copy
        
        # Test stats
        stats = cache.get_stats()
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
    
    def test_cache_key_generation(self):
        """Test cache key generation for different inputs"""
        cache = EmbeddingCache()
        
        # Same text, different models
        embedding1 = [1.0, 2.0]
        embedding2 = [3.0, 4.0]
        
        cache.put("test", embedding1, "model1")
        cache.put("test", embedding2, "model2")
        
        result1 = cache.get("test", "model1")
        result2 = cache.get("test", "model2")
        
        assert result1 == embedding1
        assert result2 == embedding2
        assert result1 != result2
    
    def test_lru_eviction(self):
        """Test LRU eviction when cache is full"""
        cache = EmbeddingCache(max_size=3)
        
        # Fill cache
        cache.put("key1", [1.0])
        cache.put("key2", [2.0])
        cache.put("key3", [3.0])
        
        assert cache.get_stats()["size"] == 3
        
        # Access key1 to make it recently used
        cache.get("key1")
        
        # Add new item, should evict key2 (least recently used)
        cache.put("key4", [4.0])
        
        assert cache.get_stats()["size"] == 3
        assert cache.get("key1") == [1.0]  # Still there
        assert cache.get("key2") is None   # Evicted
        assert cache.get("key3") == [3.0]  # Still there
        assert cache.get("key4") == [4.0]  # New item
    
    def test_ttl_expiration(self):
        """Test TTL-based expiration"""
        cache = EmbeddingCache(max_size=10, ttl_seconds=1)
        
        # Add item
        cache.put("test", [1.0, 2.0])
        assert cache.get("test") == [1.0, 2.0]
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired
        result = cache.get("test")
        assert result is None
        
        # Stats should show miss
        stats = cache.get_stats()
        assert stats["misses"] >= 1
    
    def test_batch_operations(self):
        """Test batch get and put operations"""
        cache = EmbeddingCache(max_size=10)
        
        texts = ["text1", "text2", "text3"]
        embeddings = [[1.0], [2.0], [3.0]]
        
        # Put batch
        cache.put_batch(texts, embeddings)
        
        # Get batch
        results, miss_indices = cache.get_batch(texts)
        
        assert len(results) == 3
        assert len(miss_indices) == 0
        assert results[0] == [1.0]
        assert results[1] == [2.0]
        assert results[2] == [3.0]
        
        # Test partial miss
        all_texts = texts + ["text4", "text5"]
        results, miss_indices = cache.get_batch(all_texts)
        
        assert len(results) == 5
        assert len(miss_indices) == 2
        assert miss_indices == [3, 4]
        assert results[0] == [1.0]
        assert results[3] is None
        assert results[4] is None
    
    def test_cache_clear(self):
        """Test cache clearing"""
        cache = EmbeddingCache()
        
        # Add items
        cache.put("key1", [1.0])
        cache.put("key2", [2.0])
        
        assert cache.get_stats()["size"] == 2
        
        # Clear cache
        cache.clear()
        
        stats = cache.get_stats()
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        
        # Verify items are gone
        assert cache.get("key1") is None
        assert cache.get("key2") is None
    
    def test_cache_resize(self):
        """Test cache resizing"""
        cache = EmbeddingCache(max_size=5)
        
        # Fill cache
        for i in range(5):
            cache.put(f"key{i}", [float(i)])
        
        assert cache.get_stats()["size"] == 5
        
        # Resize down - should evict items
        cache.resize(3)
        assert cache.max_size == 3
        assert cache.get_stats()["size"] == 3
        
        # Resize up
        cache.resize(10)
        assert cache.max_size == 10
        assert cache.get_stats()["size"] == 3
    
    def test_force_cleanup(self):
        """Test forced cleanup of expired entries"""
        cache = EmbeddingCache(max_size=10, ttl_seconds=1)
        
        # Add items
        cache.put("key1", [1.0])
        cache.put("key2", [2.0])
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Force cleanup
        expired_count = cache.force_cleanup()
        
        assert expired_count == 2
        assert cache.get_stats()["size"] == 0
    
    def test_thread_safety(self):
        """Test thread safety of cache operations"""
        cache = EmbeddingCache(max_size=100)
        
        def worker(thread_id: int, operations: int):
            """Worker function for threaded testing"""
            for i in range(operations):
                key = f"thread{thread_id}_item{i}"
                embedding = [float(thread_id), float(i)]
                
                # Put and get
                cache.put(key, embedding)
                result = cache.get(key)
                assert result == embedding
        
        # Run multiple threads
        num_threads = 5
        operations_per_thread = 20
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for thread_id in range(num_threads):
                future = executor.submit(worker, thread_id, operations_per_thread)
                futures.append(future)
            
            # Wait for all threads to complete
            for future in futures:
                future.result()
        
        # Verify cache state
        stats = cache.get_stats()
        expected_size = num_threads * operations_per_thread
        assert stats["size"] == expected_size
        assert stats["hits"] >= expected_size
    
    def test_empty_input_handling(self):
        """Test handling of empty and invalid inputs"""
        cache = EmbeddingCache()
        
        # Empty text
        result = cache.get("")
        assert result is None
        
        cache.put("", [1.0])
        result = cache.get("")
        assert result is None  # Should not cache empty strings
        
        # Whitespace-only text
        result = cache.get("   ")
        assert result is None
        
        # Empty embedding
        cache.put("test", [])
        result = cache.get("test")
        assert result is None  # Should not cache empty embeddings
    
    def test_cache_info_and_stats(self):
        """Test cache information and statistics"""
        cache = EmbeddingCache(max_size=5, ttl_seconds=60)
        
        # Add some items
        cache.put("key1", [1.0])
        cache.put("key2", [2.0])
        
        # Generate some hits and misses
        cache.get("key1")  # Hit
        cache.get("key3")  # Miss
        
        stats = cache.get_stats()
        
        assert stats["size"] == 2
        assert stats["max_size"] == 5
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["ttl_seconds"] == 60
        assert "memory_usage_mb" in stats
        
        # Test info string
        info = cache.get_info()
        assert "EmbeddingCache" in info
        assert "2/5" in info  # size/max_size
        assert "50.0%" in info  # hit rate
    
    def test_memory_usage_estimation(self):
        """Test memory usage estimation"""
        cache = EmbeddingCache(max_size=10)
        
        # Empty cache
        stats = cache.get_stats()
        assert stats["memory_usage_mb"] == 0
        
        # Add items with known size
        embedding_size = 1024  # Standard Stella embedding size
        cache.put("test", [1.0] * embedding_size)
        
        stats = cache.get_stats()
        assert stats["memory_usage_mb"] > 0
    
    def test_access_count_tracking(self):
        """Test access count tracking in cache entries"""
        cache = EmbeddingCache()
        
        # Add item
        cache.put("test", [1.0])
        
        # Access multiple times
        for _ in range(5):
            cache.get("test")
        
        # Access count should be tracked internally
        # (This is implementation detail, but useful for debugging)
        cache_key = cache._generate_cache_key("test", "default")
        assert cache._cache[cache_key].access_count == 5
    
    def test_default_cache_functions(self):
        """Test default cache utility functions"""
        from core.embeddings.cache import get_default_cache, clear_default_cache
        
        # Get default cache
        cache1 = get_default_cache()
        cache2 = get_default_cache()
        
        # Should be same instance
        assert cache1 is cache2
        
        # Add item
        cache1.put("test", [1.0])
        assert cache2.get("test") == [1.0]
        
        # Clear default cache
        clear_default_cache()
        assert cache1.get_stats()["size"] == 0


class TestCacheEntry:
    """Test CacheEntry data class"""
    
    def test_cache_entry_creation(self):
        """Test cache entry creation and properties"""
        embedding = [1.0, 2.0, 3.0]
        timestamp = time.time()
        
        entry = CacheEntry(
            embedding=embedding,
            timestamp=timestamp
        )
        
        assert entry.embedding == embedding
        assert entry.timestamp == timestamp
        assert entry.access_count == 0
        assert entry.last_access == timestamp
    
    def test_cache_entry_post_init(self):
        """Test cache entry post-initialization"""
        embedding = [1.0, 2.0]
        timestamp = time.time()
        
        # Without last_access
        entry = CacheEntry(embedding=embedding, timestamp=timestamp)
        assert entry.last_access == timestamp
        
        # With explicit last_access
        last_access = timestamp + 10
        entry = CacheEntry(
            embedding=embedding,
            timestamp=timestamp,
            last_access=last_access
        )
        assert entry.last_access == last_access