"""
Embedding providers for claude-code-context

Local embedding generation using Stella and other models.
"""

from .base import EmbedderProtocol

__all__ = ["EmbedderProtocol"]