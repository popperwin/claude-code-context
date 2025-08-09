"""
Unit tests for search executor functionality.

Tests search integration, result formatting, error handling, and performance metrics.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Any, Dict, List

from claude_code_context.mcp_server.search_executor import (
    SearchExecutor, 
    SearchUnavailableError,
    SEARCH_AVAILABLE
)
from claude_code_context.mcp_server.models import (
    MCPServerConfig,
    SearchRequest,
    SearchResponse,
    SearchResult as MCPSearchResult,
    SearchMode
)
from claude_code_context.mcp_server.connection import QdrantConnectionManager


class TestSearchExecutor:
    """Test search executor functionality"""
    
    @pytest.fixture
    def mcp_config(self):
        """Create test MCP server configuration"""
        return MCPServerConfig(
            project_path="/test/project",
            collection_name="test_collection",
            qdrant_url="http://localhost:6333",
            qdrant_timeout=30.0
        )
    
    @pytest.fixture
    def connection_manager(self, mcp_config):
        """Create mock connection manager"""
        manager = Mock(spec=QdrantConnectionManager)
        manager.config = mcp_config
        manager.connect = AsyncMock(return_value=True)
        return manager
    
    @pytest.fixture
    def search_executor(self, connection_manager, mcp_config):
        """Create search executor instance"""
        return SearchExecutor(connection_manager, mcp_config)
    
    @pytest.fixture
    def sample_search_request(self):
        """Create sample search request"""
        return SearchRequest(
            request_id="test_search_123",
            query="find authentication functions",
            mode=SearchMode.HYBRID,
            limit=10,
            session_id="session_123"
        )
    
    def test_initialization(self, search_executor, connection_manager, mcp_config):
        """Test search executor initialization"""
        assert search_executor.connection_manager == connection_manager
        assert search_executor.config == mcp_config
        assert search_executor._initialized is False
        assert search_executor._search_count == 0
        assert search_executor._failed_searches == 0
    
    @pytest.mark.asyncio
    async def test_initialization_success_with_search_infrastructure(self, search_executor):
        """Test successful initialization when search infrastructure is available"""
        if not SEARCH_AVAILABLE:
            pytest.skip("Search infrastructure not available")
        
        # Mock the embedder and related components
        with patch('claude_code_context.mcp_server.search_executor.StellaEmbedder') as mock_embedder_class:
            mock_embedder = AsyncMock()
            mock_embedder.load_model = AsyncMock(return_value=True)
            mock_embedder._is_loaded = True
            mock_embedder_class.return_value = mock_embedder
            
            with patch('claude_code_context.mcp_server.search_executor.HybridQdrantClient') as mock_client_class:
                mock_client = Mock()
                mock_client.connect = AsyncMock(return_value=True)  # Add async connect method
                mock_client_class.return_value = mock_client
                
                with patch('claude_code_context.mcp_server.search_executor.HybridSearcher') as mock_searcher_class:
                    mock_searcher = Mock()
                    mock_searcher_class.return_value = mock_searcher
                    
                    result = await search_executor.initialize()
                    
                    assert result is True
                    assert search_executor._initialized is True
                    assert search_executor._embedder == mock_embedder
                    assert search_executor._qdrant_client == mock_client
                    assert search_executor._searcher == mock_searcher
    
    @pytest.mark.asyncio
    async def test_initialization_failure_no_search_infrastructure(self, search_executor):
        """Test initialization when search infrastructure is not available"""
        # This test will work regardless of SEARCH_AVAILABLE because we mock the module-level variable
        with patch('claude_code_context.mcp_server.search_executor.SEARCH_AVAILABLE', False):
            result = await search_executor.initialize()
            
            assert result is False
            assert search_executor._initialized is False
    
    @pytest.mark.asyncio 
    async def test_initialization_failure_connection_error(self, search_executor):
        """Test initialization failure due to connection error"""
        search_executor.connection_manager.connect = AsyncMock(return_value=False)
        
        if SEARCH_AVAILABLE:
            with patch('claude_code_context.mcp_server.search_executor.StellaEmbedder'):
                result = await search_executor.initialize()
                
                assert result is False
                assert search_executor._initialized is False
        else:
            # When search infrastructure not available, just test the expected behavior
            result = await search_executor.initialize()
            assert result is False
    
    @pytest.mark.asyncio
    async def test_execute_search_with_placeholder_mode(self, search_executor, sample_search_request):
        """Test search execution with placeholder mode (when search infrastructure unavailable)"""
        # Force uninitialized state
        search_executor._initialized = False
        
        with patch.object(search_executor, 'initialize', new_callable=AsyncMock) as mock_init:
            mock_init.return_value = False  # Initialization fails
            
            response = await search_executor.execute_search(sample_search_request)
            
            assert isinstance(response, SearchResponse)
            assert response.request_id == sample_search_request.request_id
            assert response.success is True  # Placeholder mode still succeeds
            assert len(response.results) > 0  # Should have placeholder results
            assert response.results[0].match_type == "placeholder"
            assert "authentication" in response.results[0].content.lower() or "function" in response.results[0].content.lower()
            assert len(response.warnings) > 0  # Should warn about placeholder mode
    
    @pytest.mark.asyncio
    async def test_execute_search_with_real_infrastructure(self, search_executor, sample_search_request):
        """Test search execution with real search infrastructure"""
        if not SEARCH_AVAILABLE:
            pytest.skip("Search infrastructure not available")
        
        # Mock successful initialization
        search_executor._initialized = True
        
        # Create mock core search results
        mock_core_result = Mock()
        mock_core_result.point = Mock()
        mock_core_result.point.payload = {
            "entity_id": "auth.py::authenticate",
            "file_path": "auth.py",
            "entity_name": "authenticate",
            "entity_type": "function",
            "language": "python",
            "content": "def authenticate(user): ...",
            "start_line": 10,
            "end_line": 15,
            "start_byte": 200, 
            "end_byte": 250
        }
        mock_core_result.score = 0.85
        mock_core_result.search_type = "hybrid"
        
        # Mock searcher
        mock_searcher = Mock()
        mock_searcher.search = AsyncMock(return_value=[mock_core_result])
        search_executor._searcher = mock_searcher
        
        response = await search_executor.execute_search(sample_search_request)
        
        assert isinstance(response, SearchResponse)
        assert response.request_id == sample_search_request.request_id
        assert response.success is True
        assert len(response.results) == 1
        assert response.results[0].entity_id == "auth.py::authenticate"
        assert response.results[0].relevance_score == 0.85
        assert response.results[0].match_type == "hybrid"
        assert response.execution_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_execute_search_error_handling(self, search_executor, sample_search_request):
        """Test search execution error handling"""
        # Force initialization to succeed but search to fail
        search_executor._initialized = True
        
        mock_searcher = Mock()
        mock_searcher.search = AsyncMock(side_effect=Exception("Search failed"))
        search_executor._searcher = mock_searcher
        
        response = await search_executor.execute_search(sample_search_request)
        
        assert isinstance(response, SearchResponse)
        assert response.request_id == sample_search_request.request_id
        assert response.success is False
        assert response.error_message == "Search failed"
        assert len(response.results) == 0
        assert response.execution_time_ms > 0
        
        # Check metrics updated
        assert search_executor._failed_searches == 1
    
    def test_translate_search_mode(self, search_executor):
        """Test search mode translation from MCP to core"""
        if not SEARCH_AVAILABLE:
            pytest.skip("Search infrastructure not available")
        
        from core.search.engine import SearchMode as CoreSearchMode
        
        # Test all mode translations
        assert search_executor._translate_search_mode(SearchMode.AUTO) == CoreSearchMode.AUTO
        assert search_executor._translate_search_mode(SearchMode.PAYLOAD) == CoreSearchMode.PAYLOAD_ONLY
        assert search_executor._translate_search_mode(SearchMode.SEMANTIC) == CoreSearchMode.SEMANTIC_ONLY
        assert search_executor._translate_search_mode(SearchMode.HYBRID) == CoreSearchMode.HYBRID
    
    @pytest.mark.asyncio
    async def test_convert_results_to_mcp(self, search_executor, sample_search_request):
        """Test conversion of core search results to MCP format"""
        if not SEARCH_AVAILABLE:
            pytest.skip("Search infrastructure not available")
        
        # Create mock core result
        mock_point = Mock()
        mock_point.payload = {
            "entity_id": "test.py::test_func",
            "file_path": "test.py",
            "entity_name": "test_func",
            "entity_type": "function",
            "language": "python",
            "content": "def test_func(): pass",
            "docstring": "Test function",
            "start_line": 5,
            "end_line": 6,
            "start_byte": 100,
            "end_byte": 120
        }
        
        mock_core_result = Mock()
        mock_core_result.point = mock_point
        mock_core_result.score = 0.92
        mock_core_result.search_type = "semantic"
        
        mcp_results = await search_executor._convert_results_to_mcp(
            [mock_core_result], sample_search_request
        )
        
        assert len(mcp_results) == 1
        mcp_result = mcp_results[0]
        
        assert isinstance(mcp_result, MCPSearchResult)
        assert mcp_result.entity_id == "test.py::test_func"
        assert mcp_result.file_path == "test.py"
        assert mcp_result.name == "test_func"
        assert mcp_result.entity_type == "function"
        assert mcp_result.language == "python"
        assert mcp_result.content == "def test_func(): pass"
        assert mcp_result.docstring == "Test function"
        assert mcp_result.start_line == 5
        assert mcp_result.end_line == 6
        assert mcp_result.start_byte == 100
        assert mcp_result.end_byte == 120
        assert mcp_result.relevance_score == 0.92
        assert mcp_result.match_type == "semantic"
    
    def test_get_match_type(self, search_executor):
        """Test match type conversion"""
        if not SEARCH_AVAILABLE:
            # Test with fallback behavior
            assert search_executor._get_match_type("unknown") == "hybrid"
        else:
            from core.search.engine import SearchMode as CoreSearchMode
            
            assert search_executor._get_match_type(CoreSearchMode.PAYLOAD_ONLY) == "exact"
            assert search_executor._get_match_type(CoreSearchMode.SEMANTIC_ONLY) == "semantic"
            assert search_executor._get_match_type(CoreSearchMode.HYBRID) == "hybrid"
            assert search_executor._get_match_type("unknown") == "hybrid"
    
    @pytest.mark.asyncio
    async def test_health_check(self, search_executor):
        """Test health check functionality"""
        # Set up some metrics
        search_executor._search_count = 5
        search_executor._failed_searches = 1
        search_executor._total_search_time = 250.0  # 250ms total
        
        health = await search_executor.health_check()
        
        assert "search_executor" in health
        assert "performance" in health
        
        se_health = health["search_executor"]
        assert "initialized" in se_health
        assert "search_infrastructure_available" in se_health
        assert "embedder_loaded" in se_health
        assert "qdrant_client_available" in se_health
        assert "searcher_available" in se_health
        
        performance = health["performance"]
        assert performance["total_searches"] == 5
        assert performance["failed_searches"] == 1
        assert performance["success_rate_percent"] == 80.0  # (5-1)/5 * 100
        assert performance["average_search_time_ms"] == 50.0  # 250/5
    
    @pytest.mark.asyncio
    async def test_shutdown(self, search_executor):
        """Test shutdown functionality"""
        # Set up components
        mock_embedder = AsyncMock()
        mock_client = AsyncMock()
        search_executor._embedder = mock_embedder
        search_executor._qdrant_client = mock_client
        search_executor._searcher = Mock()
        search_executor._initialized = True
        
        await search_executor.shutdown()
        
        # Verify cleanup
        mock_embedder.unload_model.assert_called_once()
        mock_client.disconnect.assert_called_once()
        assert search_executor._embedder is None
        assert search_executor._qdrant_client is None
        assert search_executor._searcher is None
        assert search_executor._initialized is False
    
    @pytest.mark.asyncio
    async def test_shutdown_with_errors(self, search_executor):
        """Test shutdown with component errors"""
        # Set up components that raise errors
        mock_embedder = AsyncMock()
        mock_embedder.unload_model.side_effect = Exception("Unload error")
        mock_client = AsyncMock()
        mock_client.disconnect.side_effect = Exception("Disconnect error")
        
        search_executor._embedder = mock_embedder
        search_executor._qdrant_client = mock_client
        search_executor._initialized = True
        
        # Should not raise exception despite component errors
        await search_executor.shutdown()
        
        # Should still clean up
        assert search_executor._embedder is None
        assert search_executor._qdrant_client is None
        assert search_executor._initialized is False
    
    def test_get_metrics(self, search_executor):
        """Test comprehensive metrics retrieval"""
        # Set up test metrics
        search_executor._search_count = 10
        search_executor._failed_searches = 2
        search_executor._total_search_time = 500.0
        search_executor._initialized = True
        search_executor._embedder = Mock()
        search_executor._embedder._is_loaded = True
        
        metrics = search_executor.get_metrics()
        
        assert "search_executor" in metrics
        assert "performance" in metrics
        assert "components" in metrics
        
        # Check search executor info
        se_info = metrics["search_executor"]
        assert se_info["initialized"] is True
        assert se_info["search_infrastructure_available"] == SEARCH_AVAILABLE
        
        config_info = se_info["config"]
        assert config_info["qdrant_url"] == "http://localhost:6333"
        assert config_info["collection_name"] == "test_collection"
        
        # Check performance metrics
        perf = metrics["performance"]
        assert perf["total_searches"] == 10
        assert perf["failed_searches"] == 2
        assert perf["success_rate_percent"] == 80.0
        assert perf["average_search_time_ms"] == 50.0
        assert perf["total_search_time_seconds"] == 0.5
        
        # Check component status
        components = metrics["components"]
        assert components["embedder_available"] is True
        assert components["embedder_loaded"] is True


class TestSearchExecutorPlaceholderResults:
    """Test placeholder result generation"""
    
    @pytest.fixture
    def search_executor(self):
        """Create minimal search executor for placeholder testing"""
        config = MCPServerConfig()
        connection_manager = Mock()
        return SearchExecutor(connection_manager, config)
    
    @pytest.mark.asyncio
    async def test_placeholder_function_query(self, search_executor):
        """Test placeholder results for function queries"""
        request = SearchRequest(
            request_id="test_1",
            query="find authentication function",
            mode=SearchMode.HYBRID
        )
        
        response = await search_executor._create_placeholder_response(request, time.time())
        
        assert response.success is True
        assert len(response.results) >= 1
        assert any("function" in result.content.lower() for result in response.results)
        assert any("authentication" in result.content.lower() for result in response.results)
        assert all(result.match_type == "placeholder" for result in response.results)
        assert len(response.warnings) > 0
    
    @pytest.mark.asyncio
    async def test_placeholder_class_query(self, search_executor):
        """Test placeholder results for class queries"""
        request = SearchRequest(
            request_id="test_2",
            query="find user class",
            mode=SearchMode.SEMANTIC
        )
        
        response = await search_executor._create_placeholder_response(request, time.time())
        
        assert response.success is True
        assert len(response.results) >= 1
        assert any("class" in result.content.lower() for result in response.results)
        assert any("user" in result.content.lower() for result in response.results)
        assert all(result.match_type == "placeholder" for result in response.results)
    
    @pytest.mark.asyncio
    async def test_placeholder_generic_query(self, search_executor):
        """Test placeholder results for generic queries"""
        request = SearchRequest(
            request_id="test_3",
            query="search database operations",
            mode=SearchMode.AUTO
        )
        
        response = await search_executor._create_placeholder_response(request, time.time())
        
        assert response.success is True
        assert len(response.results) >= 1
        assert all(result.match_type == "placeholder" for result in response.results)
        assert "database operations" in response.results[0].content


class TestSearchExecutorIntegration:
    """Integration tests for search executor"""
    
    @pytest.fixture
    def mcp_config(self):
        """Create test MCP server configuration"""
        return MCPServerConfig(
            project_path="/test/integration",
            collection_name="integration_test",
            qdrant_url="http://localhost:6334"  # Different port for testing
        )
    
    @pytest.mark.asyncio 
    async def test_full_search_workflow_placeholder_mode(self, mcp_config):
        """Test complete search workflow in placeholder mode"""
        connection_manager = Mock(spec=QdrantConnectionManager)
        connection_manager.config = mcp_config
        connection_manager.connect = AsyncMock(return_value=False)  # Connection fails
        
        search_executor = SearchExecutor(connection_manager, mcp_config)
        
        # Test various search requests
        test_queries = [
            "find authentication functions",
            "search for user class",
            "locate database operations",
            "get configuration settings"
        ]
        
        for query in test_queries:
            request = SearchRequest(
                request_id=f"integration_test_{hash(query)}",
                query=query,
                mode=SearchMode.HYBRID,
                limit=5
            )
            
            response = await search_executor.execute_search(request)
            
            assert response.success is True
            assert len(response.results) > 0
            assert response.execution_time_ms > 0
            assert response.search_mode_used == SearchMode.HYBRID
            assert len(response.warnings) > 0  # Should warn about placeholder mode
        
        # Check metrics
        metrics = search_executor.get_metrics()
        assert metrics["performance"]["total_searches"] == len(test_queries)
        assert metrics["performance"]["failed_searches"] == 0  # Placeholder mode doesn't fail
        assert metrics["performance"]["success_rate_percent"] == 100.0
    
    @pytest.mark.asyncio
    async def test_search_executor_resilience(self, mcp_config):
        """Test search executor resilience to various error conditions"""
        connection_manager = Mock(spec=QdrantConnectionManager)
        connection_manager.config = mcp_config
        connection_manager.connect = AsyncMock(return_value=True)
        
        search_executor = SearchExecutor(connection_manager, mcp_config)
        
        # Test with various error conditions
        error_conditions = [
            Exception("Network error"),
            TimeoutError("Search timeout"),
            ValueError("Invalid query")
        ]
        
        for error in error_conditions:
            # Mock search failure
            search_executor._initialized = True
            mock_searcher = Mock()
            mock_searcher.search = AsyncMock(side_effect=error)
            search_executor._searcher = mock_searcher
            
            request = SearchRequest(
                request_id=f"error_test_{type(error).__name__}",
                query="test query",
                mode=SearchMode.HYBRID
            )
            
            response = await search_executor.execute_search(request)
            
            # Should handle error gracefully
            assert response.success is False
            assert str(error) in response.error_message
            assert len(response.results) == 0
            assert response.execution_time_ms > 0
        
        # Check error metrics
        metrics = search_executor.get_metrics()
        assert metrics["performance"]["failed_searches"] == len(error_conditions)
        success_rate = metrics["performance"]["success_rate_percent"]
        assert success_rate < 100.0  # Should reflect the failures