"""
Unit tests for CLI functionality.

Tests the ccc command-line interface commands: init, status, index, clean.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, call, MagicMock
from click.testing import CliRunner

from claude_code_context.cli import main, init, status, index, clean
from claude_code_context.cli import _check_qdrant_connection, _check_claude_cli_available


class TestUtilityFunctions:
    """Test CLI utility functions"""
    
    def test_check_qdrant_connection_success(self):
        """Test successful Qdrant connection check"""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            result = _check_qdrant_connection("http://localhost:6333")
            
            assert result is True
            mock_get.assert_called_once_with(
                "http://localhost:6333/health", 
                timeout=2
            )
    
    def test_check_qdrant_connection_failure(self):
        """Test failed Qdrant connection check"""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_get.return_value = mock_response
            
            result = _check_qdrant_connection("http://localhost:6333")
            
            assert result is False
    
    def test_check_qdrant_connection_exception(self):
        """Test Qdrant connection check with exception"""
        with patch('requests.get', side_effect=Exception("Connection error")):
            result = _check_qdrant_connection("http://localhost:6333")
            assert result is False
    
    def test_check_claude_cli_available_found(self):
        """Test Claude CLI availability when found"""
        with patch('shutil.which', return_value='/usr/local/bin/claude'):
            result = _check_claude_cli_available()
            assert result is True
    
    def test_check_claude_cli_available_not_found(self):
        """Test Claude CLI availability when not found"""
        with patch('shutil.which', return_value=None):
            result = _check_claude_cli_available()
            assert result is False
    
    def test_check_claude_cli_available_exception(self):
        """Test Claude CLI availability check with exception"""
        with patch('shutil.which', side_effect=Exception("Test error")):
            result = _check_claude_cli_available()
            assert result is False


class TestInitCommand:
    """Test the ccc init command"""
    
    def setup_method(self):
        """Set up test environment"""
        self.runner = CliRunner()
    
    def test_init_basic_success(self):
        """Test basic init command success"""
        with self.runner.isolated_filesystem():
            # Mock Qdrant connection as available
            with patch('claude_code_context.cli._check_qdrant_connection', return_value=True):
                result = self.runner.invoke(init)
                
                assert result.exit_code == 0
                assert "Initializing claude-code-context" in result.output
                assert "Qdrant connected" in result.output
                assert "Initialization complete" in result.output
                
                # Check .mcp.json was created
                mcp_config_path = Path('.mcp.json')
                assert mcp_config_path.exists()
                
                # Validate config content
                with open(mcp_config_path) as f:
                    config = json.load(f)
                
                assert "mcpServers" in config
                assert "claude-code-context" in config["mcpServers"]
                server_config = config["mcpServers"]["claude-code-context"]
                
                assert server_config["type"] == "stdio"
                assert server_config["command"] == "python"
                assert server_config["args"] == ["-m", "claude_code_context.mcp_server"]
                
                env = server_config["env"]
                assert env["MCP_MAX_CLAUDE_CALLS"] == "10"
                assert env["MCP_DEBUG"] == "false"
                assert env["QDRANT_URL"] == "http://localhost:6333"
                assert env["MCP_COLLECTION_NAME"].startswith("ccc_")
    
    def test_init_already_initialized_no_force(self):
        """Test init when already initialized without force"""
        with self.runner.isolated_filesystem():
            # Create existing .mcp.json
            Path('.mcp.json').write_text('{"existing": "config"}')
            
            result = self.runner.invoke(init)
            
            assert result.exit_code == 0
            assert "already initialized" in result.output
            assert "Use --force to overwrite" in result.output
            
            # Config should remain unchanged
            with open('.mcp.json') as f:
                config = json.load(f)
            assert config == {"existing": "config"}
    
    def test_init_force_overwrite(self):
        """Test init with force flag overwrites existing config"""
        with self.runner.isolated_filesystem():
            # Create existing .mcp.json
            Path('.mcp.json').write_text('{"existing": "config"}')
            
            with patch('claude_code_context.cli._check_qdrant_connection', return_value=True):
                result = self.runner.invoke(init, ['--force'])
                
                assert result.exit_code == 0
                assert "Initializing claude-code-context" in result.output
                
                # Config should be overwritten
                with open('.mcp.json') as f:
                    config = json.load(f)
                assert "mcpServers" in config
                assert "existing" not in config
    
    def test_init_custom_collection_name(self):
        """Test init with custom collection name"""
        with self.runner.isolated_filesystem():
            with patch('claude_code_context.cli._check_qdrant_connection', return_value=True):
                result = self.runner.invoke(init, ['--collection-name', 'my_custom_collection'])
                
                assert result.exit_code == 0
                
                with open('.mcp.json') as f:
                    config = json.load(f)
                
                env = config["mcpServers"]["claude-code-context"]["env"]
                assert env["MCP_COLLECTION_NAME"] == "my_custom_collection"
    
    def test_init_custom_qdrant_url(self):
        """Test init with custom Qdrant URL"""
        with self.runner.isolated_filesystem():
            custom_url = "https://my-qdrant.example.com:6333"
            
            with patch('claude_code_context.cli._check_qdrant_connection', return_value=True) as mock_check:
                result = self.runner.invoke(init, ['--qdrant-url', custom_url])
                
                assert result.exit_code == 0
                mock_check.assert_called_with(custom_url)
                
                with open('.mcp.json') as f:
                    config = json.load(f)
                
                env = config["mcpServers"]["claude-code-context"]["env"]
                assert env["QDRANT_URL"] == custom_url
    
    def test_init_qdrant_unavailable(self):
        """Test init when Qdrant is not available"""
        with self.runner.isolated_filesystem():
            with patch('claude_code_context.cli._check_qdrant_connection', return_value=False):
                result = self.runner.invoke(init)
                
                assert result.exit_code == 0
                assert "Qdrant not available" in result.output
                assert "docker run" in result.output  # Installation instructions
                assert "Continuing without Qdrant connection" in result.output
                
                # Should still create config
                assert Path('.mcp.json').exists()
    
    def test_init_file_write_error(self):
        """Test init with file write error"""
        with self.runner.isolated_filesystem():
            with patch('claude_code_context.cli._check_qdrant_connection', return_value=True):
                with patch('builtins.open', side_effect=PermissionError("Access denied")):
                    result = self.runner.invoke(init)
                    
                    assert result.exit_code == 1
                    assert "Failed to create configuration" in result.output


class TestStatusCommand:
    """Test the ccc status command"""
    
    def setup_method(self):
        """Set up test environment"""
        self.runner = CliRunner()
    
    def test_status_not_initialized(self):
        """Test status when project not initialized"""
        with self.runner.isolated_filesystem():
            with patch('claude_code_context.cli._check_qdrant_connection', return_value=False):
                with patch('claude_code_context.cli._check_claude_cli_available', return_value=False):
                    result = self.runner.invoke(status)
                    
                    assert result.exit_code == 0
                    assert "Not initialized" in result.output
                    assert "Run 'ccc init' to initialize" in result.output
                    assert "Some components need attention" in result.output
    
    def test_status_fully_operational(self):
        """Test status when all components are working"""
        with self.runner.isolated_filesystem():
            # Create valid .mcp.json
            mcp_config = {
                "mcpServers": {
                    "claude-code-context": {
                        "env": {
                            "MCP_PROJECT_PATH": str(Path.cwd()),
                            "MCP_COLLECTION_NAME": "ccc_test",
                            "QDRANT_URL": "http://localhost:6333",
                            "MCP_MAX_CLAUDE_CALLS": "10"
                        }
                    }
                }
            }
            Path('.mcp.json').write_text(json.dumps(mcp_config))
            
            with patch('claude_code_context.cli._check_qdrant_connection', return_value=True):
                with patch('claude_code_context.cli._check_claude_cli_available', return_value=True):
                    result = self.runner.invoke(status)
                    
                    assert result.exit_code == 0
                    assert "✅ Initialized" in result.output
                    assert "✅ Connected" in result.output
                    assert "✅ Available" in result.output
                    assert "✅ Installed" in result.output
                    assert "All systems ready" in result.output
    
    def test_status_verbose_mode(self):
        """Test status with verbose flag"""
        with self.runner.isolated_filesystem():
            # Create valid .mcp.json with test data
            mcp_config = {
                "mcpServers": {
                    "claude-code-context": {
                        "env": {
                            "MCP_PROJECT_PATH": str(Path.cwd()),
                            "MCP_COLLECTION_NAME": "ccc_test_project",
                            "QDRANT_URL": "http://localhost:6333",
                            "MCP_MAX_CLAUDE_CALLS": "15"
                        }
                    }
                }
            }
            Path('.mcp.json').write_text(json.dumps(mcp_config))
            
            with patch('claude_code_context.cli._check_qdrant_connection', return_value=True):
                with patch('claude_code_context.cli._check_claude_cli_available', return_value=True):
                    result = self.runner.invoke(status, ['--verbose'])
                    
                    assert result.exit_code == 0
                    assert "ccc_test_project" in result.output  # Collection name
                    assert "15" in result.output  # Max Claude calls
    
    def test_status_config_parsing_error(self):
        """Test status with invalid config file"""
        with self.runner.isolated_filesystem():
            # Create invalid JSON
            Path('.mcp.json').write_text('{"invalid": json content}')
            
            with patch('claude_code_context.cli._check_qdrant_connection', return_value=False):
                with patch('claude_code_context.cli._check_claude_cli_available', return_value=False):
                    result = self.runner.invoke(status)
                    
                    assert result.exit_code == 0
                    assert "❌ Error" in result.output
                    assert "Some components need attention" in result.output
    
    def test_status_mixed_component_states(self):
        """Test status with mixed component availability"""
        with self.runner.isolated_filesystem():
            mcp_config = {
                "mcpServers": {
                    "claude-code-context": {
                        "env": {
                            "MCP_PROJECT_PATH": str(Path.cwd()),
                            "MCP_COLLECTION_NAME": "ccc_test",
                            "QDRANT_URL": "http://localhost:6333",
                            "MCP_MAX_CLAUDE_CALLS": "10"
                        }
                    }
                }
            }
            Path('.mcp.json').write_text(json.dumps(mcp_config))
            
            # Qdrant available, Claude CLI not available
            with patch('claude_code_context.cli._check_qdrant_connection', return_value=True):
                with patch('claude_code_context.cli._check_claude_cli_available', return_value=False):
                    result = self.runner.invoke(status)
                    
                    assert result.exit_code == 0
                    assert "✅ Initialized" in result.output
                    assert "✅ Connected" in result.output
                    assert "❌ Not available" in result.output
                    assert "Some components need attention" in result.output
                    assert "claude login" in result.output  # Helpful suggestion


class TestIndexCommand:
    """Test the ccc index command"""
    
    def setup_method(self):
        """Set up test environment"""
        self.runner = CliRunner()
    
    def test_index_basic(self):
        """Test basic index command without project initialization"""
        result = self.runner.invoke(index)
        
        assert result.exit_code == 0
        assert "Project not initialized" in result.output
        assert "'ccc init'" in result.output  # More flexible check for the command
    
    def test_index_full_flag(self):
        """Test index command with --full flag without project initialization"""
        result = self.runner.invoke(index, ['--full'])
        
        assert result.exit_code == 0
        assert "Project not initialized" in result.output
        assert "'ccc init'" in result.output  # More flexible check for the command
    
    def test_index_dry_run(self):
        """Test index command with --dry-run flag"""
        with self.runner.isolated_filesystem():
            # First initialize the project
            with patch('claude_code_context.cli._check_qdrant_connection', return_value=True):
                init_result = self.runner.invoke(init)
                assert init_result.exit_code == 0
            
            # Now test index dry-run
            result = self.runner.invoke(index, ['--dry-run'])
            
            assert result.exit_code == 0
            assert "Dry run mode" in result.output
            assert "Mode: INCREMENTAL" in result.output
            assert "Configuration looks good" in result.output
    
    def test_index_with_initialized_project(self):
        """Test index command with initialized project"""
        with self.runner.isolated_filesystem():
            # First initialize the project
            with patch('claude_code_context.cli._check_qdrant_connection', return_value=True):
                init_result = self.runner.invoke(init)
                assert init_result.exit_code == 0
            
            # Mock the async indexing operation
            async def mock_run_indexing(*args):
                pass
            
            # Now test actual indexing
            with patch('claude_code_context.cli.asyncio.run', side_effect=mock_run_indexing):
                result = self.runner.invoke(index)
                
                assert result.exit_code == 0
                assert "Starting project indexing" in result.output
                assert "Indexing completed successfully" in result.output


class TestCleanCommand:
    """Test the ccc clean command"""
    
    def setup_method(self):
        """Set up test environment"""
        self.runner = CliRunner()
    
    def test_clean_with_confirmation(self):
        """Test clean command without project initialization"""
        # Simulate user confirming with 'y'
        result = self.runner.invoke(clean, input='y\n')
        
        assert result.exit_code == 0
        assert "Project not initialized" in result.output
        assert "Nothing to clean" in result.output
    
    def test_clean_with_rejection(self):
        """Test clean command with user rejection"""
        with self.runner.isolated_filesystem():
            # First initialize the project
            with patch('claude_code_context.cli._check_qdrant_connection', return_value=True):
                init_result = self.runner.invoke(init)
                assert init_result.exit_code == 0
            
            # Simulate user rejecting with 'n'
            result = self.runner.invoke(clean, input='n\n')
            
            assert result.exit_code == 1  # Click exits with 1 when confirmation denied
            assert "Aborted" in result.output
    
    def test_clean_dry_run(self):
        """Test clean command with --dry-run flag"""
        with self.runner.isolated_filesystem():
            # First initialize the project
            with patch('claude_code_context.cli._check_qdrant_connection', return_value=True):
                init_result = self.runner.invoke(init)
                assert init_result.exit_code == 0
            
            # Now test clean dry-run
            result = self.runner.invoke(clean, ['--dry-run'])
            
            assert result.exit_code == 0
            assert "Dry run mode" in result.output
            assert "Collection to delete:" in result.output
            assert "would remove all indexed data" in result.output
    
    def test_clean_with_initialized_project(self):
        """Test clean command with initialized project and confirmation"""
        with self.runner.isolated_filesystem():
            # First initialize the project
            with patch('claude_code_context.cli._check_qdrant_connection', return_value=True):
                init_result = self.runner.invoke(init)
                assert init_result.exit_code == 0
            
            # Mock the async cleanup operation
            async def mock_run_cleanup(*args):
                pass
            
            # Now test actual cleanup with confirmation
            with patch('claude_code_context.cli.asyncio.run', side_effect=mock_run_cleanup):
                result = self.runner.invoke(clean, input='y\n')
                
                assert result.exit_code == 0
                assert "Starting project cleanup" in result.output
                assert "Cleanup completed successfully" in result.output


class TestMainCommand:
    """Test the main ccc command group"""
    
    def setup_method(self):
        """Set up test environment"""
        self.runner = CliRunner()
    
    def test_main_help(self):
        """Test main command help"""
        result = self.runner.invoke(main, ['--help'])
        
        assert result.exit_code == 0
        assert "Claude Code Context CLI" in result.output
        assert "Manage MCP server configuration" in result.output
        assert "init" in result.output
        assert "status" in result.output
        assert "index" in result.output
        assert "clean" in result.output
    
    def test_main_version(self):
        """Test version option"""
        result = self.runner.invoke(main, ['--version'])
        
        assert result.exit_code == 0
        assert "1.0.0" in result.output
    
    def test_subcommand_help(self):
        """Test subcommand help"""
        result = self.runner.invoke(main, ['init', '--help'])
        
        assert result.exit_code == 0
        assert "Initialize claude-code-context" in result.output
        assert "--force" in result.output
        assert "--collection-name" in result.output
        assert "--qdrant-url" in result.output


class TestCLIIntegration:
    """Integration tests for CLI functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.runner = CliRunner()
    
    def test_init_then_status_workflow(self):
        """Test complete workflow: init -> status"""
        with self.runner.isolated_filesystem():
            # Step 1: Initialize project
            with patch('claude_code_context.cli._check_qdrant_connection', return_value=True):
                init_result = self.runner.invoke(init)
                assert init_result.exit_code == 0
                assert Path('.mcp.json').exists()
            
            # Step 2: Check status
            with patch('claude_code_context.cli._check_qdrant_connection', return_value=True):
                with patch('claude_code_context.cli._check_claude_cli_available', return_value=True):
                    status_result = self.runner.invoke(status)
                    assert status_result.exit_code == 0
                    assert "All systems ready" in status_result.output
    
    def test_project_name_sanitization(self):
        """Test that project name is properly sanitized for collection name"""
        with tempfile.TemporaryDirectory(prefix="test-project with spaces") as tmpdir:
            with self.runner.isolated_filesystem(temp_dir=tmpdir):
                with patch('claude_code_context.cli._check_qdrant_connection', return_value=True):
                    result = self.runner.invoke(init)
                    
                    assert result.exit_code == 0
                    
                    with open('.mcp.json') as f:
                        config = json.load(f)
                    
                    collection_name = config["mcpServers"]["claude-code-context"]["env"]["MCP_COLLECTION_NAME"]
                    
                    # Should sanitize spaces and special characters
                    assert collection_name.startswith("ccc_")
                    assert " " not in collection_name
                    assert collection_name.replace("ccc_", "").replace("_", "").isalnum()
    
    def test_error_handling_resilience(self):
        """Test CLI resilience to various error conditions"""
        with self.runner.isolated_filesystem():
            # Test with network errors - should treat as False, not propagate exception
            with patch('claude_code_context.cli._check_qdrant_connection', return_value=False):
                result = self.runner.invoke(init)
                assert result.exit_code == 0  # Should continue despite error
                assert Path('.mcp.json').exists()
            
            # Test status with various error conditions
            with patch('claude_code_context.cli._check_qdrant_connection', return_value=False):
                with patch('claude_code_context.cli._check_claude_cli_available', return_value=False):
                    result = self.runner.invoke(status)
                    assert result.exit_code == 0  # Should handle gracefully
                    assert "Some components need attention" in result.output


class TestCLIOutput:
    """Test CLI output formatting and user experience"""
    
    def setup_method(self):
        """Set up test environment"""
        self.runner = CliRunner()
    
    def test_console_output_formatting(self):
        """Test that CLI uses rich console formatting appropriately"""
        with self.runner.isolated_filesystem():
            with patch('claude_code_context.cli._check_qdrant_connection', return_value=True):
                result = self.runner.invoke(init)
                
                # Check that rich formatting markers are present
                assert "🚀" in result.output  # Emoji usage
                assert "✅" in result.output  # Status indicators
                assert "🎉" in result.output  # Success indicators
    
    def test_helpful_error_messages(self):
        """Test that error messages provide helpful guidance"""
        with self.runner.isolated_filesystem():
            with patch('claude_code_context.cli._check_qdrant_connection', return_value=False):
                result = self.runner.invoke(init)
                
                # Should provide Docker installation command
                assert "docker run" in result.output
                assert "qdrant/qdrant" in result.output
                assert "6333:6333" in result.output
    
    def test_next_steps_guidance(self):
        """Test that successful operations provide clear next steps"""
        with self.runner.isolated_filesystem():
            with patch('claude_code_context.cli._check_qdrant_connection', return_value=True):
                result = self.runner.invoke(init)
                
                assert "Next steps:" in result.output
                assert "claude" in result.output  # Start Claude Code
                assert "mcp" in result.output   # Verify MCP connection (without slash due to rich formatting)
                assert "Find authentication functions" in result.output  # Example query