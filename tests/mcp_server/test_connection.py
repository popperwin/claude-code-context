"""
Unit tests for Qdrant connection management.

Tests connection retry logic, health monitoring, collection management,
and error handling for the MCP server connection manager.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Any, Dict, List

from qdrant_client.models import VectorParams, Distance, CollectionInfo, CreateCollection
from qdrant_client.http.exceptions import ResponseHandlingException

from claude_code_context.mcp_server.connection import (
    QdrantConnectionManager, 
    ConnectionError, 
    RetryConfig
)
from claude_code_context.mcp_server.models import MCPServerConfig


class TestRetryConfig:
    """Test retry configuration"""
    
    def test_default_retry_config(self):
        """Test default retry configuration values"""
        config = RetryConfig()
        
        assert config.max_attempts == 5
        assert config.initial_delay == 1.0
        assert config.max_delay == 30.0
        assert config.backoff_factor == 2.0
        assert config.jitter is True
    
    def test_custom_retry_config(self):
        """Test custom retry configuration"""
        config = RetryConfig(
            max_attempts=3,
            initial_delay=0.5,
            max_delay=10.0,
            backoff_factor=1.5,
            jitter=False
        )
        
        assert config.max_attempts == 3
        assert config.initial_delay == 0.5
        assert config.max_delay == 10.0
        assert config.backoff_factor == 1.5
        assert config.jitter is False
    
    def test_delay_calculation(self):
        """Test exponential backoff delay calculation"""
        config = RetryConfig(
            initial_delay=1.0,
            max_delay=10.0,
            backoff_factor=2.0,
            jitter=False
        )
        
        # Test exponential growth
        assert config.get_delay(0) == 1.0
        assert config.get_delay(1) == 2.0
        assert config.get_delay(2) == 4.0
        assert config.get_delay(3) == 8.0
        
        # Test max delay capping
        assert config.get_delay(10) == 10.0
    
    def test_jitter_applied(self):
        """Test that jitter is applied when enabled"""
        config = RetryConfig(
            initial_delay=2.0,
            jitter=True
        )
        
        delays = [config.get_delay(1) for _ in range(10)]
        
        # All delays should be >= base delay
        assert all(delay >= 2.0 for delay in delays)
        # With jitter, delays should vary (very unlikely all identical)
        assert len(set(delays)) > 1


class TestQdrantConnectionManager:
    """Test Qdrant connection manager"""
    
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
        """Create connection manager instance"""
        return QdrantConnectionManager(mcp_config)
    
    def test_initialization(self, connection_manager, mcp_config):
        """Test connection manager initialization"""
        assert connection_manager.config == mcp_config
        assert connection_manager.is_connected is False
        assert connection_manager.client is None
        assert connection_manager._connection_attempts == 0
        assert connection_manager._successful_connections == 0
        assert connection_manager._failed_connections == 0
    
    @pytest.mark.asyncio
    async def test_successful_connection(self, connection_manager):
        """Test successful connection establishment"""
        mock_client = Mock()
        mock_collections = Mock()
        mock_collections.collections = [Mock(), Mock()]
        
        with patch('claude_code_context.mcp_server.connection.QdrantClient') as mock_qdrant:
            mock_qdrant.return_value = mock_client
            
            with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
                mock_to_thread.return_value = mock_collections
                
                result = await connection_manager.connect()
                
                assert result is True
                assert connection_manager.is_connected is True
                assert connection_manager._successful_connections == 1
                assert connection_manager._connection_attempts == 1
                assert connection_manager._failed_connections == 0
    
    @pytest.mark.asyncio
    async def test_connection_retry_logic(self, connection_manager):
        """Test connection retry with exponential backoff"""
        mock_client = Mock()
        
        with patch('claude_code_context.mcp_server.connection.QdrantClient') as mock_qdrant:
            mock_qdrant.return_value = mock_client
            
            with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
                # Fail first 2 attempts, succeed on 3rd
                mock_to_thread.side_effect = [
                    Exception("Connection failed"),
                    Exception("Still failing"),
                    Mock(collections=[])
                ]
                
                with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
                    result = await connection_manager.connect()
                    
                    assert result is True
                    assert connection_manager._connection_attempts == 3
                    assert connection_manager._successful_connections == 1
                    assert connection_manager._failed_connections == 2
                    
                    # Verify exponential backoff delays
                    assert mock_sleep.call_count == 2  # 2 retries
                    delays = [call[0][0] for call in mock_sleep.call_args_list]
                    assert delays[0] >= 1.0  # First retry delay
                    assert delays[1] >= 2.0  # Second retry delay (exponential)
    
    @pytest.mark.asyncio
    async def test_connection_max_retries_exceeded(self, connection_manager):
        """Test connection failure after max retries"""
        mock_client = Mock()
        
        with patch('claude_code_context.mcp_server.connection.QdrantClient') as mock_qdrant:
            mock_qdrant.return_value = mock_client
            
            with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
                mock_to_thread.side_effect = Exception("Persistent failure")
                
                with patch('asyncio.sleep', new_callable=AsyncMock):
                    with pytest.raises(ConnectionError, match="Failed to connect to Qdrant"):
                        await connection_manager.connect()
                    
                    assert connection_manager.is_connected is False
                    assert connection_manager._connection_attempts == 5  # max_attempts
                    assert connection_manager._successful_connections == 0
                    assert connection_manager._failed_connections == 5
    
    @pytest.mark.asyncio
    async def test_existing_connection_verification(self, connection_manager):
        """Test verification of existing connections"""
        mock_client = Mock()
        mock_collections = Mock(collections=[])
        
        # Set up existing connection
        connection_manager._connected = True
        connection_manager._client = mock_client
        
        with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = mock_collections
            
            result = await connection_manager.connect()
            
            assert result is True
            # Should not increment connection attempts (reused existing)
            assert connection_manager._connection_attempts == 0
    
    @pytest.mark.asyncio
    async def test_existing_connection_invalid(self, connection_manager):
        """Test handling of invalid existing connections"""
        mock_old_client = Mock()
        mock_new_client = Mock()
        mock_collections = Mock(collections=[])
        
        # Set up invalid existing connection
        connection_manager._connected = True
        connection_manager._client = mock_old_client
        
        with patch('claude_code_context.mcp_server.connection.QdrantClient') as mock_qdrant:
            mock_qdrant.return_value = mock_new_client
            
            with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
                # First call (verification) fails, second succeeds
                mock_to_thread.side_effect = [
                    Exception("Connection dead"),
                    mock_collections
                ]
                
                result = await connection_manager.connect()
                
                assert result is True
                assert connection_manager._client == mock_new_client
                assert connection_manager._connection_attempts == 1
                assert connection_manager._successful_connections == 1
    
    @pytest.mark.asyncio
    async def test_disconnect(self, connection_manager):
        """Test disconnection"""
        mock_client = Mock()
        connection_manager._connected = True
        connection_manager._client = mock_client
        
        await connection_manager.disconnect()
        
        assert connection_manager.is_connected is False
        assert connection_manager._client is None
        mock_client.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_disconnect_with_error(self, connection_manager):
        """Test disconnection with client error"""
        mock_client = Mock()
        mock_client.close.side_effect = Exception("Close error")
        connection_manager._connected = True
        connection_manager._client = mock_client
        
        # Should not raise exception
        await connection_manager.disconnect()
        
        assert connection_manager.is_connected is False
        assert connection_manager._client is None
    
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, connection_manager):
        """Test health check with healthy status"""
        mock_client = Mock()
        mock_collections = Mock()
        # Create proper mock collections with name attributes
        mock_col_1 = Mock()
        mock_col_1.name = "other_collection"
        mock_col_2 = Mock()
        mock_col_2.name = "test_collection"  # Our collection exists
        mock_col_3 = Mock()
        mock_col_3.name = "another_collection"
        
        mock_collections.collections = [mock_col_1, mock_col_2, mock_col_3]
        
        connection_manager._connected = True
        connection_manager._client = mock_client
        
        with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = mock_collections
            
            health = await connection_manager.health_check()
            
            assert health["status"] == "healthy"
            assert health["connected"] is True
            assert health["collection_exists"] is True
            assert health["total_collections"] == 3
            assert "response_time_ms" in health
            assert health["qdrant_url"] == connection_manager.config.qdrant_url
            assert health["collection_name"] == "test_collection"
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, connection_manager):
        """Test health check with unhealthy status"""
        connection_manager._connected = False
        
        with patch.object(connection_manager, 'connect', new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = False
            
            health = await connection_manager.health_check()
            
            assert health["status"] == "unhealthy"
            assert health["connected"] is False
            assert health["error"] == "Not connected"
    
    @pytest.mark.asyncio
    async def test_health_check_cached(self, connection_manager):
        """Test health check caching"""
        connection_manager._connected = True
        connection_manager._last_health_check = time.time() - 10  # 10 seconds ago
        
        health = await connection_manager.health_check()
        
        assert health["cached"] is True
        assert "response_time_ms" not in health or health["response_time_ms"] is None
    
    @pytest.mark.asyncio
    async def test_ensure_collection_exists_create_new(self, connection_manager):
        """Test creating new collection"""
        mock_client = Mock()
        mock_collections = Mock()
        mock_collections.collections = []  # No existing collections
        
        # Set up the connection manager with a mock client
        connection_manager._client = mock_client
        connection_manager._connected = True
        
        # Mock the connect method to avoid calling it
        with patch.object(connection_manager, 'connect', new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = True
            
            with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
                # First call returns empty collections, subsequent calls are for creation/indexing
                mock_to_thread.side_effect = [
                    mock_collections,  # get_collections
                    None,  # create_collection
                    None,  # create_payload_index (entity_type)
                    None,  # create_payload_index (language)
                    None,  # create_payload_index (content)
                ]
                
                result = await connection_manager.ensure_collection_exists()
                
                assert result is True
                # Verify create_collection was called with correct parameters
                assert mock_to_thread.call_count >= 2  # get_collections + create_collection + indexes
    
    @pytest.mark.asyncio
    async def test_ensure_collection_exists_already_exists(self, connection_manager):
        """Test when collection already exists"""
        mock_client = Mock()
        mock_collections = Mock()
        # Create proper mock collection with name attribute
        mock_col = Mock()
        mock_col.name = "test_collection"
        mock_collections.collections = [mock_col]
        
        # Set up the connection manager with a mock client
        connection_manager._client = mock_client
        connection_manager._connected = True
        
        # Mock the connect method to avoid calling it
        with patch.object(connection_manager, 'connect', new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = True
            
            with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
                mock_to_thread.return_value = mock_collections
                
                result = await connection_manager.ensure_collection_exists()
                
                assert result is True
                # Should only call get_collections, not create_collection
                assert mock_to_thread.call_count == 1
    
    @pytest.mark.asyncio
    async def test_ensure_collection_creation_failure(self, connection_manager):
        """Test collection creation failure"""
        mock_client = Mock()
        mock_collections = Mock()
        mock_collections.collections = []
        
        connection_manager._connected = True
        connection_manager._client = mock_client
        
        with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.side_effect = [
                mock_collections,  # get_collections succeeds
                Exception("Creation failed")  # create_collection fails
            ]
            
            result = await connection_manager.ensure_collection_exists()
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_get_client_context_manager(self, connection_manager):
        """Test client context manager"""
        mock_client = Mock()
        connection_manager._connected = True
        connection_manager._client = mock_client
        
        with patch.object(connection_manager, 'connect', new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = True
            
            async with connection_manager.get_client() as client:
                assert client == mock_client
                
        # Should track request metrics
        assert connection_manager._request_count == 1
        assert connection_manager._total_request_time > 0
    
    @pytest.mark.asyncio
    async def test_get_client_connection_failure(self, connection_manager):
        """Test client context manager with connection failure"""
        with patch.object(connection_manager, 'connect', new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = False
            
            with pytest.raises(ConnectionError, match="Unable to connect to Qdrant"):
                async with connection_manager.get_client():
                    pass
    
    @pytest.mark.asyncio
    async def test_get_client_operation_error(self, connection_manager):
        """Test client context manager with operation error"""
        mock_client = Mock()
        connection_manager._connected = True
        connection_manager._client = mock_client
        
        with patch.object(connection_manager, 'connect', new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = True
            
            with pytest.raises(Exception, match="Test error"):
                async with connection_manager.get_client():
                    raise Exception("Test error")
    
    @pytest.mark.asyncio
    async def test_get_client_qdrant_error_resets_connection(self, connection_manager):
        """Test that Qdrant-specific errors reset connection state"""
        mock_client = Mock()
        connection_manager._connected = True
        connection_manager._client = mock_client
        
        with patch.object(connection_manager, 'connect', new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = True
            
            with pytest.raises(ResponseHandlingException):
                async with connection_manager.get_client():
                    raise ResponseHandlingException("Qdrant error")
            
            # Connection should be reset
            assert connection_manager._connected is False
    
    @pytest.mark.asyncio
    async def test_collection_info(self, connection_manager):
        """Test collection info retrieval"""
        mock_client = Mock()
        mock_info = Mock()
        mock_info.name = "test_collection"
        mock_info.status = "green"
        mock_info.vectors_count = 1000
        
        connection_manager._connected = True
        connection_manager._client = mock_client
        
        with patch.object(connection_manager, 'connect', new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = True
            
            with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
                mock_to_thread.return_value = mock_info
                
                info = await connection_manager.collection_info()
                
                assert info == mock_info
    
    @pytest.mark.asyncio
    async def test_collection_info_error(self, connection_manager):
        """Test collection info retrieval with error"""
        with patch.object(connection_manager, 'get_client', new_callable=AsyncMock) as mock_get_client:
            mock_get_client.side_effect = Exception("Connection error")
            
            info = await connection_manager.collection_info()
            
            assert info is None
    
    def test_get_metrics(self, connection_manager):
        """Test metrics retrieval"""
        # Set some test metrics
        connection_manager._connection_attempts = 5
        connection_manager._successful_connections = 4
        connection_manager._failed_connections = 1
        connection_manager._request_count = 10
        connection_manager._total_request_time = 2.5
        connection_manager._connected = True
        
        metrics = connection_manager.get_metrics()
        
        assert "connection_manager" in metrics
        assert "performance" in metrics
        assert "retry_config" in metrics
        
        # Check connection manager info
        cm_info = metrics["connection_manager"]
        assert cm_info["connected"] is True
        assert cm_info["qdrant_url"] == connection_manager.config.qdrant_url
        assert cm_info["collection_name"] == "test_collection"
        
        # Check performance stats
        perf_stats = metrics["performance"]
        assert perf_stats["total_attempts"] == 5
        assert perf_stats["successful"] == 4
        assert perf_stats["failed"] == 1
        assert perf_stats["success_rate_percent"] == 80.0
        assert perf_stats["total_requests"] == 10
        assert perf_stats["avg_request_time_ms"] == 250.0  # 2.5s / 10 * 1000
        
        # Check retry config
        retry_config = metrics["retry_config"]
        assert retry_config["max_attempts"] == 5
        assert retry_config["initial_delay"] == 1.0
    
    def test_connection_stats_empty(self, connection_manager):
        """Test connection stats with no attempts"""
        stats = connection_manager._get_connection_stats()
        
        assert stats["total_attempts"] == 0
        assert stats["successful"] == 0
        assert stats["failed"] == 0
        assert stats["success_rate_percent"] == 0.0
        assert stats["total_requests"] == 0
        assert stats["avg_request_time_ms"] is None


class TestConnectionManagerIntegration:
    """Integration tests for connection manager"""
    
    @pytest.fixture
    def mcp_config(self):
        """Create test MCP server configuration"""
        return MCPServerConfig(
            project_path="/test/project",
            collection_name="integration_test_collection",
            qdrant_url="http://localhost:6334",  # Different port for testing
            qdrant_timeout=10.0
        )
    
    @pytest.mark.asyncio
    async def test_full_connection_lifecycle(self, mcp_config):
        """Test complete connection lifecycle with real workflow"""
        connection_manager = QdrantConnectionManager(mcp_config)
        
        try:
            # Initial state
            assert connection_manager.is_connected is False
            
            # Connect (this will use mocked Qdrant for CI/testing)
            with patch('claude_code_context.mcp_server.connection.QdrantClient') as mock_qdrant:
                mock_client = Mock()
                mock_qdrant.return_value = mock_client
                
                mock_collections = Mock()
                mock_collections.collections = []
                
                with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
                    mock_to_thread.return_value = mock_collections
                    
                    connected = await connection_manager.connect()
                    assert connected is True
                    assert connection_manager.is_connected is True
            
            # Health check
            with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
                mock_to_thread.return_value = mock_collections
                
                health = await connection_manager.health_check()
                assert health["status"] == "healthy"
                assert health["connected"] is True
            
            # Create collection
            connection_manager._client = mock_client
            connection_manager._connected = True
            
            with patch.object(connection_manager, 'connect', new_callable=AsyncMock) as mock_connect2:
                mock_connect2.return_value = True
                
                with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
                    mock_to_thread.side_effect = [
                        mock_collections,  # get_collections
                        None,  # create_collection
                        None, None, None  # create_payload_index calls
                    ]
                    
                    collection_created = await connection_manager.ensure_collection_exists()
                    assert collection_created is True
            
            # Get metrics
            metrics = connection_manager.get_metrics()
            assert metrics["connection_manager"]["connected"] is True
            assert metrics["performance"]["total_attempts"] >= 1
            
        finally:
            # Cleanup
            await connection_manager.disconnect()
            assert connection_manager.is_connected is False
    
    @pytest.mark.asyncio
    async def test_connection_resilience(self, mcp_config):
        """Test connection resilience under various failure conditions"""
        connection_manager = QdrantConnectionManager(mcp_config)
        
        # Test various error conditions
        error_conditions = [
            Exception("Network timeout"),
            ResponseHandlingException("Service unavailable"),
            ConnectionRefusedError("Connection refused")
        ]
        
        for error in error_conditions:
            with patch('claude_code_context.mcp_server.connection.QdrantClient') as mock_qdrant:
                mock_client = Mock()
                mock_qdrant.return_value = mock_client
                
                with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
                    mock_to_thread.side_effect = error
                    
                    with patch('asyncio.sleep', new_callable=AsyncMock):
                        # Should handle error gracefully and eventually fail with ConnectionError
                        with pytest.raises(ConnectionError):
                            await connection_manager.connect()
                        
                        assert connection_manager.is_connected is False
            
            # Reset for next test
            connection_manager._connection_attempts = 0
            connection_manager._failed_connections = 0