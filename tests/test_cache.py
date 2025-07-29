"""
Comprehensive unit tests for cache.py

Tests all caching components: CacheEntry, FileCacheEntry, LRUCache, CacheManager
"""

import pytest
import asyncio
import json
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import logging
import aiofiles

from core.indexer.cache import (
    CacheEntry, FileCacheEntry, LRUCache, CacheManager
)
from core.parser.base import ParseResult
from core.models.storage import SearchResult
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse


@pytest.fixture
async def cleanup_test_collections():
    """Fixture to clean up test collections before and after tests"""
    test_collections = [
        "test-collection", "test-cache", "test-query-cache"
    ]
    
    # Setup: Clean before tests
    client = QdrantClient(url="http://localhost:6334")
    for collection_name in test_collections:
        try:
            await asyncio.to_thread(client.delete_collection, collection_name)
        except (UnexpectedResponse, Exception):
            pass  # Collection doesn't exist or other error
    
    yield
    
    # Teardown: Clean after tests  
    for collection_name in test_collections:
        try:
            await asyncio.to_thread(client.delete_collection, collection_name)
        except (UnexpectedResponse, Exception):
            pass  # Collection doesn't exist or other error


class TestCacheEntry:
    """Test CacheEntry generic dataclass"""
    
    @pytest.mark.usefixtures("cleanup_test_collections")
    def test_cache_entry_creation(self):
        """Test CacheEntry creation with all fields"""
        now = datetime.now()
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=now,
            last_accessed=now,
            access_count=5,
            ttl_seconds=300
        )
        
        assert entry.key == "test_key"
        assert entry.value == "test_value" 
        assert entry.created_at == now
        assert entry.last_accessed == now
        assert entry.access_count == 5
        assert entry.ttl_seconds == 300
    
    @pytest.mark.usefixtures("cleanup_test_collections")
    def test_cache_entry_defaults(self):
        """Test CacheEntry with default values"""
        now = datetime.now()
        entry = CacheEntry(
            key="test_key",
            value={"data": "test"},
            created_at=now,
            last_accessed=now
        )
        
        assert entry.access_count == 0
        assert entry.ttl_seconds is None
    
    @pytest.mark.usefixtures("cleanup_test_collections")
    def test_is_expired_no_ttl(self):
        """Test expiration check with no TTL"""
        now = datetime.now()
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=now - timedelta(hours=1),
            last_accessed=now,
            ttl_seconds=None
        )
        
        assert entry.is_expired is False
    
    @pytest.mark.usefixtures("cleanup_test_collections")
    def test_is_expired_with_ttl_not_expired(self):
        """Test expiration check with TTL - not expired"""
        now = datetime.now()
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=now - timedelta(seconds=30),
            last_accessed=now,
            ttl_seconds=60  # 1 minute TTL
        )
        
        assert entry.is_expired is False
    
    @pytest.mark.usefixtures("cleanup_test_collections")
    def test_is_expired_with_ttl_expired(self):
        """Test expiration check with TTL - expired"""
        now = datetime.now()
        entry = CacheEntry(
            key="test_key", 
            value="test_value",
            created_at=now - timedelta(seconds=120),  # 2 minutes ago
            last_accessed=now,
            ttl_seconds=60  # 1 minute TTL
        )
        
        assert entry.is_expired is True
    
    @pytest.mark.usefixtures("cleanup_test_collections")
    def test_age_seconds(self):
        """Test age calculation"""
        past_time = datetime.now() - timedelta(seconds=45)
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=past_time,
            last_accessed=past_time
        )
        
        age = entry.age_seconds
        assert 44 <= age <= 46  # Allow for small timing variations


class TestFileCacheEntry:
    """Test FileCacheEntry dataclass"""
    
    @pytest.mark.usefixtures("cleanup_test_collections")
    def test_file_cache_entry_creation(self):
        """Test FileCacheEntry creation with all fields"""
        now = datetime.now()
        entry = FileCacheEntry(
            file_path="/test/file.py",
            file_hash="abc123",
            file_size=1024,
            last_modified=1640995200.0,
            parse_result_hash="def456",
            cached_at=now
        )
        
        assert entry.file_path == "/test/file.py"
        assert entry.file_hash == "abc123"
        assert entry.file_size == 1024
        assert entry.last_modified == 1640995200.0
        assert entry.parse_result_hash == "def456"
        assert entry.cached_at == now
    
    @pytest.mark.usefixtures("cleanup_test_collections")
    def test_file_cache_entry_defaults(self):
        """Test FileCacheEntry with default values"""
        entry = FileCacheEntry(
            file_path="/test/file.py",
            file_hash="abc123",
            file_size=1024,
            last_modified=1640995200.0
        )
        
        assert entry.parse_result_hash is None
        assert isinstance(entry.cached_at, datetime)
    
    @pytest.mark.usefixtures("cleanup_test_collections")
    def test_to_dict_conversion(self):
        """Test FileCacheEntry to dictionary conversion"""
        now = datetime.now()
        entry = FileCacheEntry(
            file_path="/test/file.py",
            file_hash="abc123",
            file_size=1024,
            last_modified=1640995200.0,
            parse_result_hash="def456",
            cached_at=now
        )
        
        result_dict = entry.to_dict()
        
        assert result_dict["file_path"] == "/test/file.py"
        assert result_dict["file_hash"] == "abc123"
        assert result_dict["file_size"] == 1024
        assert result_dict["last_modified"] == 1640995200.0
        assert result_dict["parse_result_hash"] == "def456"
        assert isinstance(result_dict["cached_at"], str)
        assert now.isoformat() in result_dict["cached_at"]
    
    @pytest.mark.usefixtures("cleanup_test_collections")
    def test_from_dict_conversion(self):
        """Test FileCacheEntry from dictionary conversion"""
        now = datetime.now()
        data = {
            "file_path": "/test/file.py",
            "file_hash": "abc123",
            "file_size": 1024,
            "last_modified": 1640995200.0,
            "parse_result_hash": "def456",
            "cached_at": now.isoformat()
        }
        
        entry = FileCacheEntry.from_dict(data)
        
        assert entry.file_path == "/test/file.py"
        assert entry.file_hash == "abc123"
        assert entry.file_size == 1024
        assert entry.last_modified == 1640995200.0
        assert entry.parse_result_hash == "def456"
        assert isinstance(entry.cached_at, datetime)
        assert abs((entry.cached_at - now).total_seconds()) < 1.0
    
    @pytest.mark.usefixtures("cleanup_test_collections")
    def test_round_trip_serialization(self):
        """Test round-trip serialization (to_dict -> from_dict)"""
        original_entry = FileCacheEntry(
            file_path="/test/roundtrip.py",
            file_hash="xyz789",
            file_size=2048,
            last_modified=1640995200.0,
            parse_result_hash="uvw123"
        )
        
        # Convert to dict and back
        data_dict = original_entry.to_dict()
        restored_entry = FileCacheEntry.from_dict(data_dict)
        
        # Should be identical
        assert restored_entry.file_path == original_entry.file_path
        assert restored_entry.file_hash == original_entry.file_hash
        assert restored_entry.file_size == original_entry.file_size
        assert restored_entry.last_modified == original_entry.last_modified
        assert restored_entry.parse_result_hash == original_entry.parse_result_hash
        assert abs((restored_entry.cached_at - original_entry.cached_at).total_seconds()) < 1.0


class TestLRUCache:
    """Test LRUCache implementation"""
    
    @pytest.fixture
    def cache(self):
        """Create LRUCache instance for testing"""
        return LRUCache[str](max_size=3, default_ttl=60)
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_cache_initialization(self, cache):
        """Test cache initialization"""
        assert cache.max_size == 3
        assert cache.default_ttl == 60
        assert len(cache._cache) == 0
        assert cache._hits == 0
        assert cache._misses == 0
        assert cache._evictions == 0
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_put_and_get(self, cache):
        """Test basic put and get operations"""
        await cache.put("key1", "value1")
        
        result = await cache.get("key1")
        assert result == "value1"
        
        # Check miss
        result = await cache.get("nonexistent")
        assert result is None
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_lru_eviction(self, cache):
        """Test LRU eviction when cache is full"""
        # Fill cache to capacity
        await cache.put("key1", "value1")
        await cache.put("key2", "value2")
        await cache.put("key3", "value3")
        
        # Access key1 to make it recently used
        await cache.get("key1")
        
        # Add one more item - should evict key2 (least recently used)
        await cache.put("key4", "value4")
        
        assert await cache.get("key1") == "value1"  # Should still exist
        assert await cache.get("key2") is None      # Should be evicted
        assert await cache.get("key3") == "value3"  # Should still exist
        assert await cache.get("key4") == "value4"  # Should exist
        
        assert cache._evictions == 1
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_ttl_expiration(self):
        """Test TTL-based expiration"""
        cache = LRUCache[str](max_size=10, default_ttl=1)  # 1 second TTL
        
        await cache.put("key1", "value1")
        
        # Should be available immediately
        result = await cache.get("key1")
        assert result == "value1"
        
        # Wait for expiration
        await asyncio.sleep(1.1)
        
        # Should be expired
        result = await cache.get("key1")
        assert result is None
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_custom_ttl_override(self, cache):
        """Test custom TTL override"""
        # Put with custom TTL
        await cache.put("key1", "value1", ttl=1)  # 1 second
        await cache.put("key2", "value2")         # Default TTL (60 seconds)
        
        # Wait for key1 to expire
        await asyncio.sleep(1.1)
        
        assert await cache.get("key1") is None      # Expired
        assert await cache.get("key2") == "value2"  # Not expired
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_update_existing_entry(self, cache):
        """Test updating existing cache entry"""
        await cache.put("key1", "value1") 
        await cache.put("key1", "value1_updated")
        
        result = await cache.get("key1")
        assert result == "value1_updated"
        
        # Should not have caused eviction
        assert cache._evictions == 0
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_remove_entry(self, cache):
        """Test removing cache entry"""
        await cache.put("key1", "value1")
        await cache.put("key2", "value2")
        
        # Remove existing key
        removed = await cache.remove("key1")
        assert removed is True
        
        # Try to remove non-existent key
        removed = await cache.remove("nonexistent")
        assert removed is False
        
        # Verify removal
        assert await cache.get("key1") is None
        assert await cache.get("key2") == "value2"
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_clear_cache(self, cache):
        """Test clearing entire cache"""
        await cache.put("key1", "value1")
        await cache.put("key2", "value2")
        
        await cache.clear()
        
        assert len(cache._cache) == 0
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_cleanup_expired(self):
        """Test cleanup of expired entries"""
        cache = LRUCache[str](max_size=10, default_ttl=None)  # No default TTL
        
        await cache.put("key1", "value1", ttl=1)  # Will expire
        await cache.put("key2", "value2", ttl=60) # Won't expire
        await cache.put("key3", "value3", ttl=None)  # Won't expire (no TTL)
        
        # Wait for key1 to expire
        await asyncio.sleep(1.1)
        
        # Cleanup expired entries
        removed_count = await cache.cleanup_expired()
        
        assert removed_count == 1
        assert await cache.get("key1") is None
        assert await cache.get("key2") == "value2"
        assert await cache.get("key3") == "value3"
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_access_statistics(self, cache):
        """Test access statistics tracking"""
        await cache.put("key1", "value1")
        
        # Generate hits and misses
        await cache.get("key1")    # Hit
        await cache.get("key1")    # Hit
        await cache.get("missing") # Miss
        await cache.get("missing") # Miss
        
        stats = cache.get_stats()
        
        assert stats["hits"] == 2
        assert stats["misses"] == 2
        assert stats["hit_rate"] == 0.5
        assert stats["total_requests"] == 4
        assert stats["size"] == 1
        assert stats["max_size"] == 3
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_access_count_tracking(self, cache):
        """Test that entries track access count"""
        await cache.put("key1", "value1")
        
        # Access multiple times
        await cache.get("key1")
        await cache.get("key1")
        await cache.get("key1")
        
        # Check internal entry
        entry = cache._cache["key1"]
        assert entry.access_count == 3


class TestCacheManager:
    """Test CacheManager integration"""
    
    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create temporary cache directory"""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        return cache_dir
    
    @pytest.fixture
    async def cache_manager(self, temp_cache_dir):
        """Create CacheManager with temporary directory"""
        manager = CacheManager(
            cache_dir=temp_cache_dir,
            file_cache_size=10,
            parse_cache_size=5,
            query_cache_size=3,
            query_cache_ttl=60
        )
        yield manager
        # Cleanup
        if manager._cleanup_task:
            manager._cleanup_task.cancel()
            try:
                await manager._cleanup_task
            except asyncio.CancelledError:
                pass
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_cache_manager_initialization(self, cache_manager, temp_cache_dir):
        """Test cache manager initialization"""
        assert cache_manager.cache_dir == temp_cache_dir
        assert isinstance(cache_manager.file_cache, LRUCache)
        assert isinstance(cache_manager.parse_cache, LRUCache)  
        assert isinstance(cache_manager.query_cache, LRUCache)
        assert cache_manager._file_cache_loaded is False
        assert len(cache_manager._file_cache_data) == 0
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_cache_manager_default_directory(self):
        """Test cache manager with default directory"""
        manager = CacheManager()
        expected_dir = Path.home() / ".claude-indexer" / "cache"
        assert manager.cache_dir == expected_dir
        
        # Cleanup
        if manager._cleanup_task:
            manager._cleanup_task.cancel()
            try:
                await manager._cleanup_task
            except asyncio.CancelledError:
                pass
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_file_cache_operations(self, cache_manager, tmp_path):
        """Test file cache get/update operations"""
        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("# Test file content")
        
        # Initially no cache entry
        entry = await cache_manager.get_file_cache_entry(test_file)
        assert entry is None
        
        # Update cache
        await cache_manager.update_file_cache(test_file, "test-collection")
        
        # Should now have cache entry
        entry = await cache_manager.get_file_cache_entry(test_file)
        assert entry is not None
        assert entry.file_path == str(test_file)
        assert len(entry.file_hash) == 64  # SHA256 length
        assert entry.file_size == test_file.stat().st_size
        assert entry.last_modified == test_file.stat().st_mtime
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_parse_cache_operations(self, cache_manager):
        """Test parse result caching"""
        # Create mock parse result
        mock_result = Mock(spec=ParseResult)
        mock_result.success = True
        mock_result.entities = []
        mock_result.relations = []
        
        file_hash = "abc123"
        
        # Initially no cached result
        result = await cache_manager.get_parse_result(file_hash)
        assert result is None
        
        # Cache the result
        await cache_manager.cache_parse_result(file_hash, mock_result)
        
        # Should now have cached result
        result = await cache_manager.get_parse_result(file_hash)
        assert result == mock_result
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_query_cache_operations(self, cache_manager):
        """Test query result caching"""
        # Create mock search results
        mock_results = [
            Mock(spec=SearchResult),
            Mock(spec=SearchResult)
        ]
        
        query_key = cache_manager.create_query_key(
            collection_name="test",
            query="test query",
            filters={"type": "function"},
            limit=10
        )
        
        # Initially no cached results
        results = await cache_manager.get_query_results(query_key)
        assert results is None
        
        # Cache the results
        await cache_manager.cache_query_results(query_key, mock_results)
        
        # Should now have cached results
        results = await cache_manager.get_query_results(query_key)
        assert results == mock_results
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_query_key_generation(self, cache_manager):
        """Test query key generation consistency"""
        # Same parameters should generate same key
        key1 = cache_manager.create_query_key(
            collection_name="test",
            query="search term",
            filters={"type": "function"},
            limit=50
        )
        
        key2 = cache_manager.create_query_key(
            collection_name="test",
            query="search term", 
            filters={"type": "function"},
            limit=50
        )
        
        assert key1 == key2
        assert len(key1) == 16  # SHA256 truncated to 16 chars
        
        # Different parameters should generate different keys
        key3 = cache_manager.create_query_key(
            collection_name="test",
            query="different term",
            filters={"type": "function"},
            limit=50
        )
        
        assert key1 != key3
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_query_cache_with_custom_ttl(self, cache_manager):
        """Test query caching with custom TTL"""
        mock_results = [Mock(spec=SearchResult)]
        query_key = "test_key"
        
        # Cache with custom TTL
        await cache_manager.cache_query_results(
            query_key, mock_results, ttl=1  # 1 second
        )
        
        # Should be available immediately
        results = await cache_manager.get_query_results(query_key)
        assert results == mock_results
        
        # Wait for expiration
        await asyncio.sleep(1.1)
        
        # Should be expired
        results = await cache_manager.get_query_results(query_key)
        assert results is None
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_persistent_file_cache_save_load(self, cache_manager, tmp_path):
        """Test persistent file cache save and load"""
        # Create test file and update cache
        test_file = tmp_path / "test.py"
        test_file.write_text("# Test content")
        
        await cache_manager.update_file_cache(test_file, "test-collection")
        
        # Wait a moment for async save to complete
        await asyncio.sleep(0.1)
        
        # Verify cache file was created
        cache_file = cache_manager.cache_dir / "file_cache.json"
        assert cache_file.exists()
        
        # Create new cache manager with same directory
        new_manager = CacheManager(cache_dir=cache_manager.cache_dir)
        
        try:
            # Should load the cached entry
            entry = await new_manager.get_file_cache_entry(test_file)
            assert entry is not None
            assert entry.file_path == str(test_file)
        finally:
            # Cleanup new manager
            if new_manager._cleanup_task:
                new_manager._cleanup_task.cancel()
                try:
                    await new_manager._cleanup_task
                except asyncio.CancelledError:
                    pass
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_clear_all_caches(self, cache_manager, tmp_path):
        """Test clearing all cache layers"""
        # Populate caches
        test_file = tmp_path / "test.py"
        test_file.write_text("# Test content")
        
        await cache_manager.update_file_cache(test_file, "test-collection")
        await cache_manager.cache_parse_result("hash1", Mock(spec=ParseResult))
        await cache_manager.cache_query_results("query1", [Mock(spec=SearchResult)])
        
        # Clear all caches
        await cache_manager.clear_all_caches()
        
        # All caches should be empty
        assert await cache_manager.get_file_cache_entry(test_file) is None
        assert await cache_manager.get_parse_result("hash1") is None
        assert await cache_manager.get_query_results("query1") is None
        
        # Persistent cache file should be removed
        cache_file = cache_manager.cache_dir / "file_cache.json"
        assert not cache_file.exists()
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_cache_statistics(self, cache_manager):
        """Test comprehensive cache statistics"""
        # Populate caches with some data
        await cache_manager.file_cache.put("key1", Mock())
        await cache_manager.parse_cache.put("key2", Mock())
        await cache_manager.query_cache.put("key3", Mock())
        
        stats = cache_manager.get_stats()
        
        assert "file_cache" in stats
        assert "parse_cache" in stats
        assert "query_cache" in stats
        assert "persistent_file_entries" in stats
        
        assert stats["file_cache"]["size"] == 1
        assert stats["parse_cache"]["size"] == 1  
        assert stats["query_cache"]["size"] == 1
        assert stats["persistent_file_entries"] == 0  # No persistent entries yet
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_async_context_manager(self, temp_cache_dir):
        """Test cache manager as async context manager"""
        async with CacheManager(cache_dir=temp_cache_dir) as manager:
            assert isinstance(manager, CacheManager)
            
            # Use the manager
            await manager.file_cache.put("key1", "value1")
            result = await manager.file_cache.get("key1")
            assert result == "value1"
        
        # Context manager should have handled cleanup
        # (In this case, just saving persistent cache)
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_file_cache_error_handling(self, cache_manager, tmp_path):
        """Test file cache error handling"""
        # Try to update cache for non-existent file
        nonexistent_file = tmp_path / "does_not_exist.py"
        
        # Should handle gracefully without throwing
        await cache_manager.update_file_cache(nonexistent_file, "test-collection")
        
        # Should not have created a cache entry
        entry = await cache_manager.get_file_cache_entry(nonexistent_file)
        assert entry is None
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections") 
    async def test_corrupted_persistent_cache_handling(self, cache_manager):
        """Test handling of corrupted persistent cache file"""
        # Create corrupted cache file
        cache_file = cache_manager.cache_dir / "file_cache.json"
        cache_file.write_text("{ invalid json content }")
        
        # Should handle gracefully and start with empty cache
        entry = await cache_manager.get_file_cache_entry(Path("/some/file"))
        assert entry is None
        
        # File should still exist (not overwritten yet)
        assert cache_file.exists()