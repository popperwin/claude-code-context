"""
Multi-level caching system for performance optimization.

This module provides comprehensive caching at multiple levels:
1. File-level caching (file hash → changed/unchanged)
2. Parse result caching (file content → parsed entities)
3. Query result caching (query → search results with TTL)
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, TypeVar, Generic
from dataclasses import dataclass, asdict
from collections import OrderedDict
import aiofiles

from ..parser.base import ParseResult
from ..models.storage import SearchResult

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class CacheEntry(Generic[T]):
    """Generic cache entry with metadata"""
    key: str
    value: T
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl_seconds is None:
            return False
        
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl_seconds
    
    @property
    def age_seconds(self) -> float:
        """Get entry age in seconds"""
        return (datetime.now() - self.created_at).total_seconds()


@dataclass
class FileCacheEntry:
    """Cache entry for file metadata"""
    file_path: str
    file_hash: str
    file_size: int
    last_modified: float
    parse_result_hash: Optional[str] = None
    cached_at: datetime = None
    
    def __post_init__(self):
        if self.cached_at is None:
            self.cached_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['cached_at'] = self.cached_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileCacheEntry':
        """Create from dictionary"""
        data['cached_at'] = datetime.fromisoformat(data['cached_at'])
        return cls(**data)


@dataclass
class ParseCacheEntry:
    """Cache entry for parse results"""
    file_hash: str
    parse_result: ParseResult
    cached_at: datetime = None
    
    def __post_init__(self):
        if self.cached_at is None:
            self.cached_at = datetime.now()


class LRUCache(Generic[T]):
    """
    LRU (Least Recently Used) cache with TTL support.
    
    Features:
    - Size-based eviction (LRU policy)
    - Time-based expiration (TTL)
    - Thread-safe operations
    - Access statistics
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: Optional[int] = None):
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._lock = asyncio.Lock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    async def get(self, key: str) -> Optional[T]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        async with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check expiration
            if entry.is_expired:
                del self._cache[key]
                self._misses += 1
                return None
            
            # Update access info
            entry.last_accessed = datetime.now()
            entry.access_count += 1
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            
            self._hits += 1
            return entry.value
    
    async def put(
        self, 
        key: str, 
        value: T, 
        ttl: Optional[int] = None
    ) -> None:
        """
        Put value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds (overrides default)
        """
        async with self._lock:
            now = datetime.now()
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                last_accessed=now,
                ttl_seconds=ttl or self.default_ttl
            )
            
            # Add/update entry
            if key in self._cache:
                # Update existing entry
                self._cache[key] = entry
                self._cache.move_to_end(key)
            else:
                # Add new entry
                self._cache[key] = entry
                
                # Evict if necessary
                if len(self._cache) > self.max_size:
                    # Remove least recently used
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                    self._evictions += 1
    
    async def remove(self, key: str) -> bool:
        """
        Remove entry from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if entry was removed
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all cache entries"""
        async with self._lock:
            self._cache.clear()
    
    async def cleanup_expired(self) -> int:
        """
        Remove expired entries.
        
        Returns:
            Number of entries removed
        """
        async with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]
            
            for key in expired_keys:
                del self._cache[key]
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "evictions": self._evictions,
            "total_requests": total_requests
        }


class CacheManager:
    """
    Comprehensive cache manager for all indexing operations.
    
    Manages multiple cache layers:
    - File cache: File metadata and change detection
    - Parse cache: Parsed results for unchanged files
    - Query cache: Search results with TTL
    
    TODO: Query cache (get_query_results/cache_query_results) is implemented but
    not used by core/search/engine.py (HybridSearcher). Performance analysis shows
    search targets already met without caching. Integration deferred due to cache
    invalidation complexity and user expectation of real-time results.
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        file_cache_size: int = 10000,
        parse_cache_size: int = 1000,
        query_cache_size: int = 500,
        query_cache_ttl: int = 300  # 5 minutes
    ):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory for persistent cache files
            file_cache_size: Maximum file cache entries
            parse_cache_size: Maximum parse cache entries
            query_cache_size: Maximum query cache entries
            query_cache_ttl: Query cache TTL in seconds
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".claude-indexer" / "cache"
        
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize cache layers
        self.file_cache = LRUCache[FileCacheEntry](max_size=file_cache_size)
        self.parse_cache = LRUCache[ParseResult](max_size=parse_cache_size)
        self.query_cache = LRUCache[List[SearchResult]](
            max_size=query_cache_size, 
            default_ttl=query_cache_ttl
        )
        
        # Persistent file cache
        self._file_cache_data: Dict[str, FileCacheEntry] = {}
        self._file_cache_loaded = False
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup_task()
        
        logger.info("Initialized CacheManager with all cache layers")
    
    def _start_cleanup_task(self) -> None:
        """Start background cleanup task"""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(300)  # 5 minutes
                    await self._cleanup_expired_entries()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.warning(f"Cache cleanup failed: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    async def _cleanup_expired_entries(self) -> None:
        """Clean up expired entries from all caches"""
        try:
            file_removed = await self.file_cache.cleanup_expired()
            parse_removed = await self.parse_cache.cleanup_expired()
            query_removed = await self.query_cache.cleanup_expired()
            
            total_removed = file_removed + parse_removed + query_removed
            if total_removed > 0:
                logger.debug(f"Cleaned up {total_removed} expired cache entries")
                
        except Exception as e:
            logger.warning(f"Cache cleanup failed: {e}")
    
    async def _load_file_cache(self) -> None:
        """Load persistent file cache"""
        if self._file_cache_loaded:
            return
        
        cache_file = self.cache_dir / "file_cache.json"
        
        if cache_file.exists():
            try:
                async with aiofiles.open(cache_file, 'r') as f:
                    content = await f.read()
                    raw_data = json.loads(content)
                    
                    # Convert to FileCacheEntry objects
                    for file_path, entry_dict in raw_data.items():
                        try:
                            entry = FileCacheEntry.from_dict(entry_dict)
                            self._file_cache_data[file_path] = entry
                        except Exception as e:
                            logger.warning(f"Invalid file cache entry: {e}")
                
                logger.info(f"Loaded {len(self._file_cache_data)} file cache entries")
                
            except Exception as e:
                logger.warning(f"Failed to load file cache: {e}")
        
        self._file_cache_loaded = True
    
    async def _save_file_cache(self) -> None:
        """Save persistent file cache"""
        cache_file = self.cache_dir / "file_cache.json"
        
        try:
            # Convert to serializable format
            serializable_data = {
                file_path: entry.to_dict()
                for file_path, entry in self._file_cache_data.items()
            }
            
            # Write atomically
            temp_file = cache_file.with_suffix('.tmp')
            async with aiofiles.open(temp_file, 'w') as f:
                await f.write(json.dumps(serializable_data, indent=2))
            
            temp_file.rename(cache_file)
            logger.debug(f"Saved {len(self._file_cache_data)} file cache entries")
            
        except Exception as e:
            logger.error(f"Failed to save file cache: {e}")
    
    async def get_file_cache_entry(
        self, 
        file_path: Path
    ) -> Optional[FileCacheEntry]:
        """
        Get file cache entry.
        
        Args:
            file_path: Path to file
            
        Returns:
            File cache entry or None
        """
        await self._load_file_cache()
        
        file_key = str(file_path)
        
        # Check in-memory cache first
        entry = await self.file_cache.get(file_key)
        if entry:
            return entry
        
        # Check persistent cache
        entry = self._file_cache_data.get(file_key)
        if entry:
            # Move to in-memory cache
            await self.file_cache.put(file_key, entry)
            return entry
        
        return None
    
    async def update_file_cache(
        self,
        file_path: Path,
        collection_name: str
    ) -> None:
        """
        Update file cache entry after processing.
        
        Args:
            file_path: Path to file
            collection_name: Collection name
        """
        await self._load_file_cache()
        
        try:
            # Compute file hash and metadata
            hasher = hashlib.sha256()
            stat = file_path.stat()
            
            async with aiofiles.open(file_path, 'rb') as f:
                while chunk := await f.read(8192):
                    hasher.update(chunk)
            
            file_hash = hasher.hexdigest()
            
            # Create cache entry
            entry = FileCacheEntry(
                file_path=str(file_path),
                file_hash=file_hash,
                file_size=stat.st_size,
                last_modified=stat.st_mtime
            )
            
            file_key = str(file_path)
            
            # Update both caches
            await self.file_cache.put(file_key, entry)
            self._file_cache_data[file_key] = entry
            
            # Save to disk (async)
            asyncio.create_task(self._save_file_cache())
            
        except Exception as e:
            logger.warning(f"Failed to update file cache for {file_path}: {e}")
    
    async def get_parse_result(
        self, 
        file_hash: str
    ) -> Optional[ParseResult]:
        """
        Get cached parse result.
        
        Args:
            file_hash: Hash of file content
            
        Returns:
            Cached parse result or None
        """
        return await self.parse_cache.get(file_hash)
    
    async def cache_parse_result(
        self,
        file_hash: str,
        parse_result: ParseResult
    ) -> None:
        """
        Cache parse result.
        
        Args:
            file_hash: Hash of file content
            parse_result: Parse result to cache
        """
        await self.parse_cache.put(file_hash, parse_result)
    
    async def get_query_results(
        self,
        query_key: str
    ) -> Optional[List[SearchResult]]:
        """
        Get cached query results.
        
        Args:
            query_key: Query cache key
            
        Returns:
            Cached results or None
            
        TODO: This method is unused by HybridSearcher. Performance analysis shows
        search latency targets already met without result caching.
        """
        return await self.query_cache.get(query_key)
    
    async def cache_query_results(
        self,
        query_key: str,
        results: List[SearchResult],
        ttl: Optional[int] = None
    ) -> None:
        """
        Cache query results.
        
        Args:
            query_key: Query cache key
            results: Search results to cache
            ttl: Custom TTL in seconds
            
        TODO: This method is unused by HybridSearcher. Performance analysis shows
        search latency targets already met without result caching.
        """
        await self.query_cache.put(query_key, results, ttl)
    
    def create_query_key(
        self,
        collection_name: str,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50
    ) -> str:
        """
        Create standardized query cache key.
        
        Args:
            collection_name: Collection name
            query: Search query
            filters: Search filters
            limit: Result limit
            
        Returns:
            Cache key string
        """
        key_data = {
            "collection": collection_name,
            "query": query,
            "filters": filters or {},
            "limit": limit
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    async def clear_all_caches(self) -> None:
        """Clear all cache layers"""
        await self.file_cache.clear()
        await self.parse_cache.clear()
        await self.query_cache.clear()
        
        self._file_cache_data.clear()
        
        # Remove persistent cache file
        cache_file = self.cache_dir / "file_cache.json"
        if cache_file.exists():
            cache_file.unlink()
        
        logger.info("Cleared all caches")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        return {
            "file_cache": self.file_cache.get_stats(),
            "parse_cache": self.parse_cache.get_stats(),
            "query_cache": self.query_cache.get_stats(),
            "persistent_file_entries": len(self._file_cache_data)
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Save persistent cache
        if self._file_cache_loaded:
            await self._save_file_cache()