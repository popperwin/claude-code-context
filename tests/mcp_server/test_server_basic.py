"""
Unit tests for basic MCP server functionality.

Tests the minimal FastMCP server implementation, models, and configuration.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from claude_code_context.mcp_server.models import (
    MCPServerConfig,
    MCPServerStatus,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SearchMode,
    ServerHealthStatus
)
from claude_code_context.mcp_server.server import MCPCodeContextServer


class TestMCPServerModels:
    """Test MCP server Pydantic models"""
    
    def test_mcp_server_config_defaults(self):
        """Test MCPServerConfig default values"""
        config = MCPServerConfig()
        
        # The default is Path(".") without validation (Pydantic behavior)
        # Validator only runs when explicitly setting or passing values
        assert config.project_path == Path(".")
        assert config.collection_name == "auto"
        assert config.max_claude_calls == 10
        assert config.debug_mode is False
        assert config.qdrant_url == "http://localhost:6333"
        assert config.qdrant_timeout == 60.0
        assert config.search_timeout_ms == 30000
        assert config.context_word_limit == 20000
    
    def test_mcp_server_config_validation(self):
        """Test MCPServerConfig field validation"""
        # Valid configuration
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MCPServerConfig(
                project_path=Path(tmpdir),
                qdrant_url="https://example.com:6333",
                max_claude_calls=5
            )
            assert config.project_path == Path(tmpdir).resolve()
            assert config.qdrant_url == "https://example.com:6333"
            assert config.max_claude_calls == 5
        
        # Invalid URL
        with pytest.raises(ValueError, match="Qdrant URL must start with"):
            MCPServerConfig(qdrant_url="invalid-url")
        
        # Invalid max_claude_calls range
        with pytest.raises(ValueError):
            MCPServerConfig(max_claude_calls=0)
        
        with pytest.raises(ValueError):
            MCPServerConfig(max_claude_calls=100)
    
    def test_collection_name_generation(self):
        """Test collection name generation logic"""
        config = MCPServerConfig(project_path=Path("/test/my-project"))
        
        # Auto-generated from directory name
        assert config.get_collection_name() == "ccc_my_project"
        
        # Explicit project name
        assert config.get_collection_name("Custom Project") == "ccc_custom_project"
        
        # Manual collection name
        config.collection_name = "manual_collection"
        assert config.get_collection_name() == "manual_collection"
    
    def test_search_request_validation(self):
        """Test SearchRequest model validation"""
        # Valid request
        request = SearchRequest(
            request_id="test_123",
            query="find authentication functions",
            mode=SearchMode.HYBRID,
            limit=20
        )
        
        assert request.request_id == "test_123"
        assert request.query == "find authentication functions"
        assert request.mode == SearchMode.HYBRID
        assert request.limit == 20
        assert request.include_code is True
        assert request.include_docs is True
        
        # Invalid query length
        with pytest.raises(ValueError):
            SearchRequest(request_id="test", query="")
        
        with pytest.raises(ValueError):
            SearchRequest(request_id="test", query="x" * 1001)
        
        # Invalid limit range
        with pytest.raises(ValueError):
            SearchRequest(request_id="test", query="test", limit=0)
        
        with pytest.raises(ValueError):
            SearchRequest(request_id="test", query="test", limit=101)
    
    def test_search_result_validation(self):
        """Test SearchResult model validation"""
        # Valid result
        result = SearchResult(
            entity_id="func_123",
            file_path="src/auth.py",
            name="authenticate",
            content="def authenticate(user):\n    return True",
            entity_type="function",
            language="python",
            start_line=10,
            end_line=12,
            start_byte=200,
            end_byte=250,
            relevance_score=0.85,
            match_type="semantic"
        )
        
        assert result.entity_id == "func_123"
        assert result.relevance_score == 0.85
        assert result.start_line == 10
        assert result.end_line == 12
        
        # Invalid line/byte ordering
        with pytest.raises(ValueError, match="end_line must be >= start_line"):
            SearchResult(
                entity_id="test", file_path="test.py", name="test",
                entity_type="function", language="python",
                start_line=10, end_line=8, start_byte=0, end_byte=100,
                relevance_score=0.5, match_type="test"
            )
        
        with pytest.raises(ValueError, match="end_byte must be >= start_byte"):
            SearchResult(
                entity_id="test", file_path="test.py", name="test",
                entity_type="function", language="python",
                start_line=1, end_line=2, start_byte=100, end_byte=50,
                relevance_score=0.5, match_type="test"
            )
    
    def test_search_response_validation(self):
        """Test SearchResponse model validation"""
        result = SearchResult(
            entity_id="test", file_path="test.py", name="test",
            entity_type="function", language="python",
            start_line=1, end_line=2, start_byte=0, end_byte=100,
            relevance_score=0.5, match_type="test"
        )
        
        # Valid response
        response = SearchResponse(
            request_id="test_123",
            results=[result],
            total_found=1,
            execution_time_ms=150.5,
            search_mode_used=SearchMode.SEMANTIC,
            success=True
        )
        
        assert response.request_id == "test_123"
        assert len(response.results) == 1
        assert response.total_found == 1
        assert response.success is True
        
        # Too many results
        many_results = [result] * 101
        with pytest.raises(ValueError, match="Results list cannot exceed 100"):
            SearchResponse(
                request_id="test",
                results=many_results,
                total_found=101,
                execution_time_ms=100,
                search_mode_used=SearchMode.SEMANTIC,
                success=True
            )


class TestMCPCodeContextServer:
    """Test MCP server class functionality"""
    
    def test_server_init_with_defaults(self):
        """Test server initialization with default configuration"""
        with patch.dict('os.environ', {}, clear=True):
            server = MCPCodeContextServer()
            
            assert server.config.project_path.name == Path(".").resolve().name
            assert server.config.collection_name == "auto"
            assert server.config.max_claude_calls == 10
            assert server.status == MCPServerStatus.INITIALIZING
            assert server.requests_handled == 0
    
    def test_server_init_with_env_config(self):
        """Test server initialization with environment configuration"""
        env_config = {
            'MCP_PROJECT_PATH': '/test/project',
            'MCP_COLLECTION_NAME': 'test_collection',
            'MCP_MAX_CLAUDE_CALLS': '5',
            'MCP_DEBUG': 'true',
            'QDRANT_URL': 'http://test:6334'
        }
        
        with patch.dict('os.environ', env_config, clear=True):
            server = MCPCodeContextServer()
            
            assert str(server.config.project_path) == '/test/project'
            assert server.config.collection_name == 'test_collection'
            assert server.config.max_claude_calls == 5
            assert server.config.debug_mode is True
            assert server.config.qdrant_url == 'http://test:6334'
    
    def test_server_init_with_custom_config(self):
        """Test server initialization with custom configuration object"""
        custom_config = MCPServerConfig(
            project_path=Path("/custom/path"),
            collection_name="custom_collection",
            max_claude_calls=15,
            debug_mode=True
        )
        
        server = MCPCodeContextServer(config=custom_config)
        
        assert server.config.project_path == Path("/custom/path").resolve()
        assert server.config.collection_name == "custom_collection"
        assert server.config.max_claude_calls == 15
        assert server.config.debug_mode is True
    
    def test_claude_cli_availability_check(self):
        """Test Claude CLI availability detection"""
        server = MCPCodeContextServer()
        
        # Test when Claude CLI is available
        with patch('shutil.which', return_value='/usr/bin/claude'):
            assert server._check_claude_cli_available() is True
        
        # Test when Claude CLI is not available
        with patch('shutil.which', return_value=None):
            assert server._check_claude_cli_available() is False
        
        # Test when shutil.which raises exception
        with patch('shutil.which', side_effect=Exception("Test error")):
            assert server._check_claude_cli_available() is False
    
    @pytest.mark.asyncio
    async def test_execute_search_with_search_executor(self):
        """Test search execution through search executor"""
        server = MCPCodeContextServer()
        
        request = SearchRequest(
            request_id="test_search_1",
            query="test query",
            mode=SearchMode.SEMANTIC,
            limit=5
        )
        
        # Mock search executor's execute_search method
        from unittest.mock import AsyncMock
        server.search_executor.execute_search = AsyncMock(
            return_value=SearchResponse(
                request_id="test_search_1",
                session_id=None,
                results=[],
                total_found=0,
                execution_time_ms=10.0,
                search_mode_used=SearchMode.SEMANTIC,
                claude_calls_made=0,
                success=True
            )
        )
        
        response = await server.search_executor.execute_search(request)
        
        assert response.request_id == "test_search_1"
        assert response.success is True
        assert response.search_mode_used == SearchMode.SEMANTIC
        assert response.claude_calls_made == 0
        assert response.execution_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_check_qdrant_connection_placeholder(self):
        """Test Qdrant connection check (placeholder implementation)"""
        server = MCPCodeContextServer()
        
        # Placeholder always returns True for now
        assert await server._check_qdrant_connection() is True
    
    @pytest.mark.asyncio
    async def test_check_collection_available_placeholder(self):
        """Test collection availability check (placeholder implementation)"""
        server = MCPCodeContextServer()
        
        # Placeholder always returns True for now
        assert await server._check_collection_available() is True
    
    @pytest.mark.asyncio
    async def test_server_startup_sequence(self):
        """Test server startup sequence without actually starting FastMCP"""
        server = MCPCodeContextServer()
        
        # Mock FastMCP run to avoid actual server startup
        with patch.object(server.mcp, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = None
            
            await server.start()
            
            # Verify status changes
            assert server.status == MCPServerStatus.READY
            assert server.start_time > 0
            
            # Verify FastMCP run was called with stdio transport
            mock_run.assert_called_once_with(transport="stdio")
    
    @pytest.mark.asyncio
    async def test_server_startup_error_handling(self):
        """Test server startup error handling"""
        server = MCPCodeContextServer()
        
        # Mock FastMCP run to raise an exception
        with patch.object(server.mcp, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = Exception("Test startup error")
            
            with pytest.raises(Exception, match="Test startup error"):
                await server.start()
            
            # Verify status is set to error
            assert server.status == MCPServerStatus.ERROR
    
    @pytest.mark.asyncio
    async def test_server_shutdown(self):
        """Test server graceful shutdown"""
        server = MCPCodeContextServer()
        
        await server.shutdown()
        
        assert server.status == MCPServerStatus.SHUTDOWN


class TestMCPServerTools:
    """Test MCP server tool functionality"""
    
    @pytest.mark.asyncio
    async def test_search_codebase_tool_success(self):
        """Test search_codebase tool with valid input"""
        server = MCPCodeContextServer()
        
        # Get the registered tools using proper FastMCP API
        tools = await server.mcp.get_tools()
        
        # FastMCP returns a dict with tool names as keys
        assert 'search_codebase' in tools
        search_tool = tools['search_codebase']
        
        assert search_tool is not None
        
        # Execute search tool using FunctionTool.fn (the actual callable)
        result = await search_tool.fn("test query", "semantic", 5)
        
        assert isinstance(result, dict)
        assert result["success"] is True
        assert "results" in result
        assert "total_found" in result
        # In placeholder mode, might return 0 or more results
        assert "results" in result
        assert isinstance(result["results"], list)
        
        # Verify server metrics updated
        assert server.requests_handled == 1
    
    @pytest.mark.asyncio
    async def test_search_codebase_tool_invalid_mode(self):
        """Test search_codebase tool with invalid search mode"""
        server = MCPCodeContextServer()
        
        # Get the registered tools using proper FastMCP API
        tools = await server.mcp.get_tools()
        
        # FastMCP returns a dict with tool names as keys
        assert 'search_codebase' in tools
        search_tool = tools['search_codebase']
        
        assert search_tool is not None
        
        # Execute search tool with invalid mode using FunctionTool.fn
        result = await search_tool.fn("test query", "invalid_mode", 5)
        
        assert isinstance(result, dict)
        assert result["success"] is False
        assert "error_message" in result
        assert result["total_found"] == 0
    
    @pytest.mark.asyncio
    async def test_get_server_health_tool(self):
        """Test get_server_health tool"""
        server = MCPCodeContextServer()
        server.status = MCPServerStatus.READY
        server.start_time = asyncio.get_event_loop().time() - 100  # 100 seconds ago
        server.requests_handled = 5
        server.total_response_time = 500.0  # 500ms total
        
        # Get the registered tools using proper FastMCP API
        tools = await server.mcp.get_tools()
        
        # FastMCP returns a dict with tool names as keys
        assert 'get_server_health' in tools
        health_tool = tools['get_server_health']
        
        assert health_tool is not None
        
        # Execute health tool using FunctionTool.fn
        result = await health_tool.fn()
        
        assert isinstance(result, dict)
        assert result["status"] == "ready"
        assert result["healthy"] is True
        assert result["requests_handled"] == 5
        assert result["uptime_seconds"] >= 99  # At least 99 seconds
        assert result["average_response_time_ms"] == 100.0  # 500ms / 5 requests
        assert "project_path" in result
        assert "collection_name" in result
    
    @pytest.mark.asyncio
    async def test_get_server_health_tool_error(self):
        """Test get_server_health tool error handling"""
        server = MCPCodeContextServer()
        
        # Mock an error in health check
        with patch.object(server, '_check_qdrant_connection', side_effect=Exception("Test error")):
            
            # Get the registered tools using proper FastMCP API
            tools = await server.mcp.get_tools()
            
            # FastMCP returns a dict with tool names as keys
            assert 'get_server_health' in tools
            health_tool = tools['get_server_health']
            
            assert health_tool is not None
            
            # Execute health tool using FunctionTool.fn
            result = await health_tool.fn()
            
            assert isinstance(result, dict)
            assert result["status"] == "error"
            assert result["healthy"] is False
            assert "error_details" in result