"""
Claude Code Context - Semantic context enrichment for Claude Code.

A sophisticated MCP (Model Context Protocol) server that enhances Claude Code with 
intelligent code search capabilities using local Stella embeddings, Qdrant vector 
storage, and Claude-powered search orchestration.
"""

__version__ = "1.0.0"
__author__ = "Claude Code Context Team"
__email__ = "noreply@anthropic.com"

# Package imports for convenient access
from core.models.entities import Entity, ASTNode, Relation
from core.models.config import ProjectConfig, QdrantConfig, StellaConfig
from core.models.storage import SearchResult, StorageResult

__all__ = [
    "Entity",
    "ASTNode", 
    "Relation",
    "ProjectConfig",
    "QdrantConfig",
    "StellaConfig",
    "SearchResult",
    "StorageResult",
    "__version__",
]