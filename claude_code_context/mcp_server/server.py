"""
MCP Server implementation for claude-code-context.

FastMCP-based stdio server that provides intelligent code search capabilities
through Claude orchestration and integration with existing search infrastructure.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

from claude_code_context.mcp_server.models import (
    MCPServerConfig,
    MCPServerStatus,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SearchMode,
    ServerHealthStatus,
    MCPToolDefinition,
    MCPServerInfo
)
from claude_code_context.mcp_server.connection import QdrantConnectionManager
from claude_code_context.mcp_server.search_executor import SearchExecutor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global server instance
server_instance: Optional['MCPCodeContextServer'] = None


class MCPCodeContextServer:
    """
    Claude Code Context MCP Server.
    
    Provides intelligent code search through FastMCP stdio transport with 
    Claude orchestration and integration with existing search infrastructure.
    """
    
    def __init__(self, config: Optional[MCPServerConfig] = None) -> None:
        """Initialize MCP server with configuration"""
        self.config = config or self._load_config_from_env()
        self.status = MCPServerStatus.INITIALIZING
        try:
            self.start_time = asyncio.get_event_loop().time()
        except RuntimeError:
            # No event loop running during initialization
            self.start_time = 0
        self.requests_handled = 0
        self.total_response_time = 0.0
        
        # Initialize connection manager
        self.connection_manager = QdrantConnectionManager(self.config)
        
        # Initialize search executor
        self.search_executor = SearchExecutor(self.connection_manager, self.config)
        
        # Initialize FastMCP server
        self.mcp = FastMCP("claude-code-context")
        self._register_tools()
        
        logger.info(f"🚀 MCP Server initialized for project: {self.config.project_path}")
        logger.info(f"📂 Collection: {self.config.get_collection_name()}")
    
    def _load_config_from_env(self) -> MCPServerConfig:
        """Load configuration from environment variables"""
        return MCPServerConfig(
            project_path=Path(os.getenv("MCP_PROJECT_PATH", ".")),
            collection_name=os.getenv("MCP_COLLECTION_NAME", "auto"),
            max_claude_calls=int(os.getenv("MCP_MAX_CLAUDE_CALLS", "10")),
            debug_mode=os.getenv("MCP_DEBUG", "false").lower() == "true",
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        )
    
    def _register_tools(self) -> None:
        """Register MCP tools with FastMCP"""
        
        @self.mcp.tool()
        async def search_codebase(query: str, mode: str = "auto", limit: int = 10) -> Dict[str, Any]:
            """
            Search the codebase using intelligent query analysis.
            
            Args:
                query: Natural language search query
                mode: Search mode (auto, payload, semantic, hybrid)
                limit: Maximum number of results (1-50)
                
            Returns:
                Search results with relevance scores and code snippets
            """
            try:
                # Validate and create search request
                search_request = SearchRequest(
                    request_id=f"search_{self.requests_handled + 1}",
                    query=query,
                    mode=SearchMode(mode.lower()),
                    limit=min(max(limit, 1), 50)
                )
                
                # Execute search using search executor
                response = await self.search_executor.execute_search(search_request)
                
                self.requests_handled += 1
                return response.model_dump(mode='json')
                
            except Exception as e:
                logger.error(f"Search error: {e}")
                return {
                    "success": False,
                    "error_message": str(e),
                    "results": [],
                    "total_found": 0
                }
        
        @self.mcp.tool()
        async def get_server_health() -> Dict[str, Any]:
            """
            Get server health status and performance metrics.
            
            Returns:
                Server health information including Qdrant connection status
            """
            try:
                uptime = asyncio.get_event_loop().time() - self.start_time
                avg_response_time = (
                    self.total_response_time / self.requests_handled 
                    if self.requests_handled > 0 else None
                )
                
                # Get orchestrator and search executor health
                orchestrator_health = await self._get_orchestrator_health()
                
                health = ServerHealthStatus(
                    status=self.status,
                    healthy=self.status == MCPServerStatus.READY,
                    qdrant_connected=await self._check_qdrant_connection(),
                    collection_available=await self._check_collection_available(),
                    claude_cli_available=self._check_claude_cli_available(),
                    uptime_seconds=uptime,
                    requests_handled=self.requests_handled,
                    average_response_time_ms=avg_response_time,
                    project_path=str(self.config.project_path),
                    collection_name=self.config.get_collection_name()
                )
                
                # Combine with orchestrator health
                health_dict = health.model_dump(mode='json')
                health_dict["orchestrator_health"] = orchestrator_health
                
                return health_dict
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                return {
                    "status": "error",
                    "healthy": False,
                    "error_details": {"error": str(e)}
                }
    
    
    async def _check_qdrant_connection(self) -> bool:
        """Check if Qdrant is accessible"""
        try:
            health = await self.connection_manager.health_check()
            return health.get("connected", False)
        except Exception as e:
            logger.error(f"Qdrant connection check failed: {e}")
            return False
    
    async def _check_collection_available(self) -> bool:
        """Check if collection exists and is accessible"""
        try:
            # Force fresh check by clearing cache
            self.connection_manager._last_health_check = 0.0
            health = await self.connection_manager.health_check()
            collection_exists = health.get("collection_exists")
            # Ensure we return a boolean, not None
            return bool(collection_exists) if collection_exists is not None else False
        except Exception as e:
            logger.error(f"Collection availability check failed: {e}")
            return False
    
    def _check_claude_cli_available(self) -> bool:
        """Check if Claude CLI is available"""
        try:
            import shutil
            return shutil.which("claude") is not None
        except Exception:
            return False
    
    async def _get_orchestrator_health(self) -> Dict[str, Any]:
        """Get orchestrator health information"""
        try:
            return await self.search_executor.health_check()
        except Exception as e:
            logger.error(f"Error getting orchestrator health: {e}")
            return {"error": str(e)}
    
    async def get_server_health(self) -> Dict[str, Any]:
        """
        Get server health status and performance metrics (public method for testing).
        
        Returns:
            Server health information including Qdrant connection status
        """
        try:
            uptime = asyncio.get_event_loop().time() - self.start_time
            avg_response_time = (
                self.total_response_time / self.requests_handled 
                if self.requests_handled > 0 else None
            )
            
            # Get orchestrator and search executor health
            orchestrator_health = await self._get_orchestrator_health()
            
            # Force fresh checks for more accurate health status
            self.connection_manager._last_health_check = 0.0
            
            health = ServerHealthStatus(
                status=self.status,
                healthy=self.status == MCPServerStatus.READY,
                qdrant_connected=await self._check_qdrant_connection(),
                collection_available=await self._check_collection_available(),
                claude_cli_available=self._check_claude_cli_available(),
                uptime_seconds=uptime,
                requests_handled=self.requests_handled,
                average_response_time_ms=avg_response_time,
                project_path=str(self.config.project_path),
                collection_name=self.config.get_collection_name()
            )
            
            # Combine with orchestrator health
            health_dict = health.model_dump(mode='json')
            health_dict["orchestrator_health"] = orchestrator_health
            
            return health_dict
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return {
                "status": "error",
                "healthy": False,
                "error_details": {"error": str(e)}
            }
    
    async def start(self) -> None:
        """Start the MCP server"""
        try:
            self.status = MCPServerStatus.CONNECTING
            logger.info("🔌 Starting MCP server...")
            
            # Initialize Qdrant connection
            qdrant_connected = await self.connection_manager.connect()
            if qdrant_connected:
                # Ensure collection exists
                collection_ready = await self.connection_manager.ensure_collection_exists()
                if collection_ready:
                    logger.info("✅ Qdrant and collection ready")
                else:
                    logger.warning("⚠️  Collection setup failed")
            else:
                logger.warning("⚠️  Qdrant connection not available")
            
            # Initialize search executor
            search_ready = await self.search_executor.initialize()
            if search_ready:
                logger.info("✅ Search executor ready")
            else:
                logger.warning("⚠️  Search executor using placeholder mode")
            
            if not self._check_claude_cli_available():
                logger.warning("⚠️  Claude CLI not available")
            
            self.status = MCPServerStatus.READY
            self.start_time = asyncio.get_event_loop().time()
            
            logger.info("✅ MCP server ready")
            logger.info(f"📍 Project: {self.config.project_path}")
            logger.info(f"🗄️  Collection: {self.config.get_collection_name()}")
            
            # Start FastMCP server with stdio transport
            await self.mcp.run(transport="stdio")
            
        except Exception as e:
            self.status = MCPServerStatus.ERROR
            logger.error(f"❌ Server startup failed: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the MCP server gracefully"""
        try:
            self.status = MCPServerStatus.SHUTDOWN
            logger.info("🔌 Shutting down MCP server...")
            
            # Shutdown search executor
            if self.search_executor:
                await self.search_executor.shutdown()
            
            # Disconnect from Qdrant
            if self.connection_manager:
                await self.connection_manager.disconnect()
            
            logger.info("✅ MCP server shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Server shutdown error: {e}")
            raise


async def main() -> None:
    """
    Main entry point for the MCP server.
    
    Creates and starts the MCP server with stdio transport for Claude Code integration.
    """
    global server_instance
    
    try:
        # Create server instance
        server_instance = MCPCodeContextServer()
        
        # Start server
        await server_instance.start()
        
    except KeyboardInterrupt:
        logger.info("🛑 Received shutdown signal")
        if server_instance:
            await server_instance.shutdown()
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        if server_instance:
            await server_instance.shutdown()
        raise


if __name__ == "__main__":
    # This allows the module to be run directly for testing
    asyncio.run(main())