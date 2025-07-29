"""
Indexing orchestration for claude-code-context.

This module provides the main indexing pipeline that orchestrates parsing,
embedding generation, and storage operations for efficient code indexing.

Key Components:
- HybridIndexer: Main orchestrator combining all indexing stages
- IncrementalIndexer: Smart file change detection for efficient updates
- CacheManager: Multi-level caching for performance optimization
"""

from .hybrid_indexer import HybridIndexer
from .incremental import IncrementalIndexer, FileChangeDetector
from .cache import CacheManager, FileCacheEntry, ParseCacheEntry

__all__ = [
    "HybridIndexer",
    "IncrementalIndexer", 
    "FileChangeDetector",
    "CacheManager",
    "FileCacheEntry",
    "ParseCacheEntry"
]