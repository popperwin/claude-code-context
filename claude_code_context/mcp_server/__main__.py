"""
MCP Server entry point for claude-code-context.

Usage:
    python -m claude_code_context.mcp_server

This is the main entry point used by Claude Code when connecting to the MCP server
via stdio transport. The server will:

1. Initialize connection to Qdrant
2. Perform delta-scan to ensure index is current  
3. Start FastMCP server with stdio transport
4. Handle search requests with Claude orchestration

Environment Variables:
    MCP_PROJECT_PATH: Project root directory (default: current directory)
    MCP_COLLECTION_NAME: Qdrant collection name (default: auto-generated)
    MCP_MAX_CLAUDE_CALLS: Maximum search iterations (default: 10)
    MCP_DEBUG: Enable debug logging (default: false)
    QDRANT_URL: Qdrant server URL (default: http://localhost:6333)
"""

import asyncio
import sys
from pathlib import Path

# Ensure the project root is in Python path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from claude_code_context.mcp_server.server import main


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🔌 MCP server shutting down...")
        sys.exit(0)
    except Exception as e:
        print(f"❌ MCP server error: {e}", file=sys.stderr)
        sys.exit(1)