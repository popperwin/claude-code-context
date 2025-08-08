"""
Integration tests for enhanced CLI functionality.

Tests the enhanced ccc command-line interface with actual async operations,
including index and clean commands with QdrantConnectionManager integration.
"""

import asyncio
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from click.testing import CliRunner

from claude_code_context.cli import main, init, status, index, clean
from claude_code_context.cli import _run_indexing, _run_cleanup
from claude_code_context.mcp_server.models import MCPServerConfig
from claude_code_context.mcp_server.connection import QdrantConnectionManager


class TestEnhancedIndexCommand:
    """Test the enhanced ccc index command with actual functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.runner = CliRunner()
    
    def test_index_not_initialized(self):
        """Test index command when project not initialized"""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(index)
            
            assert result.exit_code == 0
            assert "Project not initialized" in result.output
            assert "ccc init" in result.output
    
    def test_index_config_load_error(self):
        """Test index command with invalid config"""
        with self.runner.isolated_filesystem():
            # Create invalid JSON config
            Path('.mcp.json').write_text('{"invalid": json}')
            
            result = self.runner.invoke(index)
            
            assert result.exit_code == 0
            assert "Failed to load configuration" in result.output
    
    def test_index_dry_run_mode(self):
        """Test index command with dry-run flag"""
        with self.runner.isolated_filesystem():
            # Create valid config
            mcp_config = {
                "mcpServers": {
                    "claude-code-context": {
                        "env": {
                            "MCP_PROJECT_PATH": str(Path.cwd()),
                            "MCP_COLLECTION_NAME": "ccc_test",
                            "QDRANT_URL": "http://localhost:6333"
                        }
                    }
                }
            }
            Path('.mcp.json').write_text(json.dumps(mcp_config))
            
            result = self.runner.invoke(index, ['--dry-run'])
            
            assert result.exit_code == 0
            assert "Dry run mode" in result.output
            assert "INCREMENTAL" in result.output
            assert "ccc_test" in result.output
            assert "Configuration looks good" in result.output
    
    def test_index_dry_run_full_mode(self):
        """Test index command with dry-run and full flags"""
        with self.runner.isolated_filesystem():
            # Create valid config
            mcp_config = {
                "mcpServers": {
                    "claude-code-context": {
                        "env": {
                            "MCP_PROJECT_PATH": str(Path.cwd()),
                            "MCP_COLLECTION_NAME": "ccc_test",
                            "QDRANT_URL": "http://localhost:6333"
                        }
                    }
                }
            }
            Path('.mcp.json').write_text(json.dumps(mcp_config))
            
            result = self.runner.invoke(index, ['--full', '--dry-run'])
            
            assert result.exit_code == 0
            assert "Dry run mode" in result.output
            assert "FULL" in result.output
    
    @patch('claude_code_context.cli.asyncio.run')
    def test_index_successful_execution(self, mock_asyncio_run):
        """Test successful index execution"""
        with self.runner.isolated_filesystem():
            # Create valid config
            mcp_config = {
                "mcpServers": {
                    "claude-code-context": {
                        "env": {
                            "MCP_PROJECT_PATH": str(Path.cwd()),
                            "MCP_COLLECTION_NAME": "ccc_test",
                            "QDRANT_URL": "http://localhost:6333"
                        }
                    }
                }
            }
            Path('.mcp.json').write_text(json.dumps(mcp_config))
            
            # Mock successful async execution
            mock_asyncio_run.return_value = None
            
            result = self.runner.invoke(index)
            
            assert result.exit_code == 0
            assert "Starting project indexing" in result.output
            assert "Indexing completed successfully" in result.output
            
            # Verify asyncio.run was called with correct parameters
            mock_asyncio_run.assert_called_once()
            call_args = mock_asyncio_run.call_args[0][0]
            # Should be a coroutine call to _run_indexing
            assert hasattr(call_args, '__await__')
    
    @patch('claude_code_context.cli.asyncio.run')
    def test_index_execution_failure(self, mock_asyncio_run):
        """Test index execution with failure"""
        with self.runner.isolated_filesystem():
            # Create valid config
            mcp_config = {
                "mcpServers": {
                    "claude-code-context": {
                        "env": {
                            "MCP_PROJECT_PATH": str(Path.cwd()),
                            "MCP_COLLECTION_NAME": "ccc_test",
                            "QDRANT_URL": "http://localhost:6333"
                        }
                    }
                }
            }
            Path('.mcp.json').write_text(json.dumps(mcp_config))
            
            # Mock failed async execution
            mock_asyncio_run.side_effect = Exception("Indexing failed")
            
            result = self.runner.invoke(index)
            
            assert result.exit_code == 1
            assert "Starting project indexing" in result.output
            assert "Indexing failed" in result.output


class TestEnhancedCleanCommand:
    """Test the enhanced ccc clean command with actual functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.runner = CliRunner()
    
    def test_clean_not_initialized(self):
        """Test clean command when project not initialized"""
        with self.runner.isolated_filesystem():
            # Simulate user confirming
            result = self.runner.invoke(clean, input='y\n')
            
            assert result.exit_code == 0
            assert "Project not initialized" in result.output
            assert "Nothing to clean" in result.output
    
    def test_clean_config_load_error(self):
        """Test clean command with invalid config"""
        with self.runner.isolated_filesystem():
            # Create invalid JSON config
            Path('.mcp.json').write_text('{"invalid": json}')
            
            result = self.runner.invoke(clean, input='y\n')
            
            assert result.exit_code == 0
            assert "Failed to load configuration" in result.output
    
    def test_clean_dry_run_mode(self):
        """Test clean command with dry-run flag"""
        with self.runner.isolated_filesystem():
            # Create valid config
            mcp_config = {
                "mcpServers": {
                    "claude-code-context": {
                        "env": {
                            "MCP_PROJECT_PATH": str(Path.cwd()),
                            "MCP_COLLECTION_NAME": "ccc_test",
                            "QDRANT_URL": "http://localhost:6333"
                        }
                    }
                }
            }
            Path('.mcp.json').write_text(json.dumps(mcp_config))
            
            result = self.runner.invoke(clean, ['--dry-run'])
            
            assert result.exit_code == 0
            assert "Dry run mode" in result.output
            assert "Collection to delete: ccc_test" in result.output
            assert "This would remove all indexed data" in result.output
    
    def test_clean_user_rejection(self):
        """Test clean command with user rejection"""
        with self.runner.isolated_filesystem():
            # Create valid config
            mcp_config = {
                "mcpServers": {
                    "claude-code-context": {
                        "env": {
                            "MCP_PROJECT_PATH": str(Path.cwd()),
                            "MCP_COLLECTION_NAME": "ccc_test",
                            "QDRANT_URL": "http://localhost:6333"
                        }
                    }
                }
            }
            Path('.mcp.json').write_text(json.dumps(mcp_config))
            
            # Simulate user rejecting
            result = self.runner.invoke(clean, input='n\n')
            
            assert result.exit_code == 1
            assert "Aborted" in result.output
    
    @patch('claude_code_context.cli.asyncio.run')
    def test_clean_successful_execution(self, mock_asyncio_run):
        """Test successful clean execution"""
        with self.runner.isolated_filesystem():
            # Create valid config
            mcp_config = {
                "mcpServers": {
                    "claude-code-context": {
                        "env": {
                            "MCP_PROJECT_PATH": str(Path.cwd()),
                            "MCP_COLLECTION_NAME": "ccc_test",
                            "QDRANT_URL": "http://localhost:6333"
                        }
                    }
                }
            }
            Path('.mcp.json').write_text(json.dumps(mcp_config))
            
            # Mock successful async execution
            mock_asyncio_run.return_value = None
            
            result = self.runner.invoke(clean, input='y\n')
            
            assert result.exit_code == 0
            assert "Starting project cleanup" in result.output
            assert "Cleanup completed successfully" in result.output
            assert "ccc index" in result.output  # Check for re-index tip (may have rich formatting)
            
            # Verify asyncio.run was called
            mock_asyncio_run.assert_called_once()
    
    @patch('claude_code_context.cli.asyncio.run')
    def test_clean_execution_failure(self, mock_asyncio_run):
        """Test clean execution with failure"""
        with self.runner.isolated_filesystem():
            # Create valid config
            mcp_config = {
                "mcpServers": {
                    "claude-code-context": {
                        "env": {
                            "MCP_PROJECT_PATH": str(Path.cwd()),
                            "MCP_COLLECTION_NAME": "ccc_test",
                            "QDRANT_URL": "http://localhost:6333"
                        }
                    }
                }
            }
            Path('.mcp.json').write_text(json.dumps(mcp_config))
            
            # Mock failed async execution
            mock_asyncio_run.side_effect = Exception("Cleanup failed")
            
            result = self.runner.invoke(clean, input='y\n')
            
            assert result.exit_code == 1
            assert "Starting project cleanup" in result.output
            assert "Cleanup failed" in result.output


class TestAsyncHelperFunctions:
    """Test the async helper functions _run_indexing and _run_cleanup"""
    
    @pytest.mark.asyncio
    async def test_run_indexing_success(self):
        """Test successful _run_indexing execution"""
        project_path = Path("/test/project")
        collection_name = "test_collection"
        qdrant_url = "http://localhost:6333"
        
        # Mock QdrantConnectionManager
        with patch('claude_code_context.cli.QdrantConnectionManager') as mock_connection_class:
            mock_connection_manager = AsyncMock()
            mock_connection_manager.connect.return_value = True
            mock_connection_manager.ensure_collection_exists.return_value = True
            mock_connection_manager.health_check.return_value = {
                "collection_point_count": 100
            }
            mock_connection_manager.disconnect.return_value = None
            mock_connection_class.return_value = mock_connection_manager
            
            # Should complete without exception
            await _run_indexing(project_path, collection_name, qdrant_url, False)
            
            # Verify connection manager was used correctly
            mock_connection_class.assert_called_once()
            mock_connection_manager.connect.assert_called_once()
            mock_connection_manager.ensure_collection_exists.assert_called_once()
            mock_connection_manager.health_check.assert_called_once()
            mock_connection_manager.disconnect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_indexing_connection_failure(self):
        """Test _run_indexing with connection failure"""
        project_path = Path("/test/project")
        collection_name = "test_collection"
        qdrant_url = "http://localhost:6333"
        
        # Mock connection failure
        with patch('claude_code_context.cli.QdrantConnectionManager') as mock_connection_class:
            mock_connection_manager = AsyncMock()
            mock_connection_manager.connect.return_value = False
            mock_connection_class.return_value = mock_connection_manager
            
            # Should raise exception on connection failure
            with pytest.raises(Exception, match="Failed to connect to Qdrant"):
                await _run_indexing(project_path, collection_name, qdrant_url, True)
    
    @pytest.mark.asyncio
    async def test_run_indexing_collection_failure(self):
        """Test _run_indexing with collection setup failure"""
        project_path = Path("/test/project")
        collection_name = "test_collection"
        qdrant_url = "http://localhost:6333"
        
        # Mock collection setup failure
        with patch('claude_code_context.cli.QdrantConnectionManager') as mock_connection_class:
            mock_connection_manager = AsyncMock()
            mock_connection_manager.connect.return_value = True
            mock_connection_manager.ensure_collection_exists.return_value = False
            mock_connection_class.return_value = mock_connection_manager
            
            # Should raise exception on collection failure
            with pytest.raises(Exception, match="Failed to setup collection"):
                await _run_indexing(project_path, collection_name, qdrant_url, False)
    
    @pytest.mark.asyncio
    async def test_run_cleanup_success(self):
        """Test successful _run_cleanup execution"""
        collection_name = "test_collection"
        qdrant_url = "http://localhost:6333"
        
        # Mock QdrantConnectionManager and client
        with patch('claude_code_context.cli.QdrantConnectionManager') as mock_connection_class:
            mock_connection_manager = AsyncMock()
            mock_connection_manager.connect.return_value = True
            mock_connection_manager.health_check.return_value = {
                "collection_exists": True,
                "collection_point_count": 50
            }
            mock_connection_manager.disconnect.return_value = None
            
            # Mock Qdrant client
            mock_client = Mock()
            mock_client.delete_collection.return_value = None
            mock_connection_manager._client = mock_client
            
            mock_connection_class.return_value = mock_connection_manager
            
            # Should complete without exception
            await _run_cleanup(collection_name, qdrant_url)
            
            # Verify cleanup was performed
            mock_connection_manager.connect.assert_called_once()
            mock_connection_manager.health_check.assert_called_once()
            mock_client.delete_collection.assert_called_once_with(collection_name)
            mock_connection_manager.disconnect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_cleanup_collection_not_exists(self):
        """Test _run_cleanup when collection doesn't exist"""
        collection_name = "test_collection"
        qdrant_url = "http://localhost:6333"
        
        # Mock collection not existing
        with patch('claude_code_context.cli.QdrantConnectionManager') as mock_connection_class:
            mock_connection_manager = AsyncMock()
            mock_connection_manager.connect.return_value = True
            mock_connection_manager.health_check.return_value = {
                "collection_exists": False
            }
            mock_connection_manager.disconnect.return_value = None
            mock_connection_class.return_value = mock_connection_manager
            
            # Should complete without exception (early return)
            await _run_cleanup(collection_name, qdrant_url)
            
            # Verify connection was made and disconnected
            mock_connection_manager.connect.assert_called_once()
            mock_connection_manager.health_check.assert_called_once()
            mock_connection_manager.disconnect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_cleanup_connection_failure(self):
        """Test _run_cleanup with connection failure"""
        collection_name = "test_collection"
        qdrant_url = "http://localhost:6333"
        
        # Mock connection failure
        with patch('claude_code_context.cli.QdrantConnectionManager') as mock_connection_class:
            mock_connection_manager = AsyncMock()
            mock_connection_manager.connect.return_value = False
            mock_connection_class.return_value = mock_connection_manager
            
            # Should raise exception on connection failure
            with pytest.raises(Exception, match="Failed to connect to Qdrant"):
                await _run_cleanup(collection_name, qdrant_url)
    
    @pytest.mark.asyncio
    async def test_run_cleanup_deletion_failure(self):
        """Test _run_cleanup with deletion failure"""
        collection_name = "test_collection"
        qdrant_url = "http://localhost:6333"
        
        # Mock deletion failure
        with patch('claude_code_context.cli.QdrantConnectionManager') as mock_connection_class:
            mock_connection_manager = AsyncMock()
            mock_connection_manager.connect.return_value = True
            mock_connection_manager.health_check.return_value = {
                "collection_exists": True,
                "collection_point_count": 50
            }
            
            # Mock client with deletion failure
            mock_client = Mock()
            mock_client.delete_collection.side_effect = Exception("Deletion failed")
            mock_connection_manager._client = mock_client
            
            mock_connection_class.return_value = mock_connection_manager
            
            # Should raise exception on deletion failure
            with pytest.raises(Exception, match="Deletion failed"):
                await _run_cleanup(collection_name, qdrant_url)
    
    @pytest.mark.asyncio
    async def test_run_cleanup_no_client(self):
        """Test _run_cleanup when no client is available"""
        collection_name = "test_collection"
        qdrant_url = "http://localhost:6333"
        
        # Mock no client available
        with patch('claude_code_context.cli.QdrantConnectionManager') as mock_connection_class:
            mock_connection_manager = AsyncMock()
            mock_connection_manager.connect.return_value = True
            mock_connection_manager.health_check.return_value = {
                "collection_exists": True,
                "collection_point_count": 50
            }
            mock_connection_manager._client = None  # No client
            mock_connection_class.return_value = mock_connection_manager
            
            # Should raise exception when no client available
            with pytest.raises(Exception, match="No Qdrant client available"):
                await _run_cleanup(collection_name, qdrant_url)


class TestCLIIntegrationWorkflows:
    """Test complete CLI integration workflows"""
    
    def setup_method(self):
        """Set up test environment"""
        self.runner = CliRunner()
    
    def test_complete_workflow_init_status_index_clean(self):
        """Test complete workflow: init -> status -> index (dry-run) -> clean (dry-run)"""
        with self.runner.isolated_filesystem():
            # Step 1: Initialize project
            with patch('claude_code_context.cli._check_qdrant_connection', return_value=True):
                init_result = self.runner.invoke(init)
                assert init_result.exit_code == 0
                assert Path('.mcp.json').exists()
            
            # Step 2: Check status
            with patch('claude_code_context.cli._check_qdrant_connection', return_value=True):
                with patch('claude_code_context.cli._check_claude_cli_available', return_value=True):
                    status_result = self.runner.invoke(status, ['--verbose'])
                    assert status_result.exit_code == 0
                    assert "All systems ready" in status_result.output
            
            # Step 3: Test index in dry-run mode
            index_result = self.runner.invoke(index, ['--dry-run'])
            assert index_result.exit_code == 0
            assert "Configuration looks good" in index_result.output
            
            # Step 4: Test clean in dry-run mode
            clean_result = self.runner.invoke(clean, ['--dry-run'])
            assert clean_result.exit_code == 0
            assert "This would remove all indexed data" in clean_result.output
    
    def test_error_recovery_workflow(self):
        """Test error recovery in CLI workflows"""
        with self.runner.isolated_filesystem():
            # Try to index without initialization
            index_result = self.runner.invoke(index)
            assert index_result.exit_code == 0
            assert "Project not initialized" in index_result.output
            
            # Try to clean without initialization
            clean_result = self.runner.invoke(clean, input='y\n')
            assert clean_result.exit_code == 0
            assert "Nothing to clean" in clean_result.output
            
            # Initialize project
            with patch('claude_code_context.cli._check_qdrant_connection', return_value=True):
                init_result = self.runner.invoke(init)
                assert init_result.exit_code == 0
            
            # Now operations should work (in dry-run mode)
            index_result = self.runner.invoke(index, ['--dry-run'])
            assert index_result.exit_code == 0
            assert "Configuration looks good" in index_result.output
            
            clean_result = self.runner.invoke(clean, ['--dry-run'])
            assert clean_result.exit_code == 0
            assert "This would remove all indexed data" in clean_result.output
    
    def test_configuration_consistency(self):
        """Test that configuration is consistent across all commands"""
        with self.runner.isolated_filesystem():
            custom_collection = "my_special_collection"
            custom_url = "http://my-qdrant:6333"
            
            # Initialize with custom settings
            with patch('claude_code_context.cli._check_qdrant_connection', return_value=True) as mock_check:
                init_result = self.runner.invoke(init, [
                    '--collection-name', custom_collection,
                    '--qdrant-url', custom_url
                ])
                assert init_result.exit_code == 0
                mock_check.assert_called_with(custom_url)
            
            # Verify status shows custom settings
            with patch('claude_code_context.cli._check_qdrant_connection', return_value=True):
                with patch('claude_code_context.cli._check_claude_cli_available', return_value=True):
                    status_result = self.runner.invoke(status, ['--verbose'])
                    assert status_result.exit_code == 0
                    assert custom_collection in status_result.output
            
            # Verify index uses custom settings
            index_result = self.runner.invoke(index, ['--dry-run'])
            assert index_result.exit_code == 0
            assert custom_collection in index_result.output
            assert custom_url in index_result.output
            
            # Verify clean uses custom settings
            clean_result = self.runner.invoke(clean, ['--dry-run'])
            assert clean_result.exit_code == 0
            assert custom_collection in clean_result.output
            assert custom_url in clean_result.output