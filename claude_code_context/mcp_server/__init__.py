"""
MCP Server package for claude-code-context.

Provides Model Context Protocol integration with Claude Code through stdio transport,
enabling intelligent code search with Claude-powered orchestration.

Key Components:
- server.py: FastMCP-based stdio server
- orchestrator.py: Claude CLI subprocess management
- search_executor.py: Bridge to existing search infrastructure
- context_builder.py: Project context assembly
- session_manager.py: Multi-turn conversation state
- connection.py: Qdrant connection management
"""

from typing import Optional
import os

# Runtime configuration from environment
MCP_PROJECT_PATH = os.getenv("MCP_PROJECT_PATH", ".")
MCP_COLLECTION_NAME = os.getenv("MCP_COLLECTION_NAME", "auto")
MCP_MAX_CLAUDE_CALLS = int(os.getenv("MCP_MAX_CLAUDE_CALLS", "10"))
MCP_DEBUG = os.getenv("MCP_DEBUG", "false").lower() == "true"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

__all__ = [
    "MCP_PROJECT_PATH",
    "MCP_COLLECTION_NAME", 
    "MCP_MAX_CLAUDE_CALLS",
    "MCP_DEBUG",
    "QDRANT_URL",
]