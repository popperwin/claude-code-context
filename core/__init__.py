"""
claude-code-context core package

Semantic context enrichment for Claude Code with local embeddings.
"""

__version__ = "1.0.0"
__author__ = "Claude Code Context Team"

from .models import Entity, EntityType, SourceLocation, QdrantPoint, StorageResult, ProjectConfig

__all__ = [
    "Entity",
    "EntityType", 
    "SourceLocation",
    "QdrantPoint",
    "StorageResult",
    "ProjectConfig"
]