"""
Search package for claude-code-context.

Provides intelligent search capabilities with query optimization and result ranking.
"""

from .engine import HybridSearcher, SearchConfig, SearchMode

__all__ = [
    "HybridSearcher",
    "SearchConfig", 
    "SearchMode"
]