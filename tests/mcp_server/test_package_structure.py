"""
Unit tests for MCP server package structure and entry points.

Tests package imports, environment configuration, and basic module loading.
Following the same pattern as other test files in the project.
"""

import pytest
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestMCPServerPackage:
    """Test MCP server package structure and imports"""
    
    def test_package_import(self):
        """Test that claude_code_context package can be imported"""
        import claude_code_context
        
        assert hasattr(claude_code_context, '__version__')
        assert claude_code_context.__version__ == "1.0.0"
        assert hasattr(claude_code_context, '__author__')
        assert hasattr(claude_code_context, '__email__')
    
    def test_mcp_server_module_import(self):
        """Test that MCP server module can be imported"""
        from claude_code_context.mcp_server import (
            MCP_PROJECT_PATH,
            MCP_COLLECTION_NAME,
            MCP_MAX_CLAUDE_CALLS,
            MCP_DEBUG,
            QDRANT_URL
        )
        
        # Test default values (test environment uses 6334, real project uses 6333)
        assert MCP_PROJECT_PATH == "."
        assert MCP_COLLECTION_NAME == "auto"  
        assert MCP_MAX_CLAUDE_CALLS == 10
        assert MCP_DEBUG is False
        # In test environment, QDRANT_URL may be overridden to 6334
        assert QDRANT_URL in ["http://localhost:6333", "http://localhost:6334"]
    
    def test_mcp_server_environment_config(self):
        """Test environment variable configuration"""
        test_env = {
            'MCP_PROJECT_PATH': '/test/project',
            'MCP_COLLECTION_NAME': 'test_collection',
            'MCP_MAX_CLAUDE_CALLS': '5',
            'MCP_DEBUG': 'true',
            'QDRANT_URL': 'http://test:6333'
        }
        
        with patch.dict(os.environ, test_env, clear=False):
            # Reload the module to pick up new environment values
            if 'claude_code_context.mcp_server' in sys.modules:
                del sys.modules['claude_code_context.mcp_server']
            
            from claude_code_context.mcp_server import (
                MCP_PROJECT_PATH,
                MCP_COLLECTION_NAME, 
                MCP_MAX_CLAUDE_CALLS,
                MCP_DEBUG,
                QDRANT_URL
            )
            
            assert MCP_PROJECT_PATH == '/test/project'
            assert MCP_COLLECTION_NAME == 'test_collection'
            assert MCP_MAX_CLAUDE_CALLS == 5
            assert MCP_DEBUG is True
            assert QDRANT_URL == 'http://test:6333'
    
    def test_main_module_entry_point_exists(self):
        """Test that __main__.py entry point exists"""
        main_module = Path('claude_code_context/mcp_server/__main__.py')
        assert main_module.exists()
        
        # Test that the file contains expected structure
        content = main_module.read_text()
        assert 'asyncio.run(main())' in content
        assert 'from claude_code_context.mcp_server.server import main' in content
        assert '__name__ == "__main__"' in content
    
    def test_entry_point_python_path_setup(self):
        """Test that __main__.py sets up Python path correctly"""
        main_module = Path('claude_code_context/mcp_server/__main__.py')
        content = main_module.read_text()
        
        # Verify path setup logic is present
        assert 'project_root = Path(__file__).parent.parent.parent' in content
        assert 'sys.path.insert(0, str(project_root))' in content
    
    def test_package_exports(self):
        """Test that package exports expected classes"""
        import claude_code_context
        
        # Test that main model classes are available
        assert hasattr(claude_code_context, 'Entity')
        assert hasattr(claude_code_context, 'ASTNode')
        assert hasattr(claude_code_context, 'Relation')
        assert hasattr(claude_code_context, 'ProjectConfig')
        assert hasattr(claude_code_context, 'QdrantConfig')
        assert hasattr(claude_code_context, 'StellaConfig')
        assert hasattr(claude_code_context, 'SearchResult')
        assert hasattr(claude_code_context, 'StorageResult')
        
        # Test that imports work
        from claude_code_context import Entity, ProjectConfig, SearchResult
        assert Entity is not None
        assert ProjectConfig is not None
        assert SearchResult is not None


class TestMCPServerEntryPoint:
    """Test MCP server entry point functionality"""
    
    def test_main_module_can_be_imported(self):
        """Test that __main__ module can be imported without errors"""
        # Test the import structure by checking if server module exists
        try:
            from claude_code_context.mcp_server import server
            # If we get here, the main dependency imports work
            assert hasattr(server, 'main')
            assert callable(server.main)
        except ImportError as e:
            pytest.fail(f"Failed to import server module: {e}")
    
    def test_main_module_has_error_handling(self):
        """Test that __main__.py has proper error handling"""
        main_module = Path('claude_code_context/mcp_server/__main__.py')
        content = main_module.read_text()
        
        # Verify error handling is present
        assert 'try:' in content
        assert 'except KeyboardInterrupt:' in content
        assert 'except Exception as e:' in content
        assert 'sys.exit(0)' in content
        assert 'sys.exit(1)' in content
    
    @patch('claude_code_context.mcp_server.server.main')
    @patch('asyncio.run')
    def test_main_execution_path(self, mock_asyncio_run, mock_main):
        """Test main execution path without actually running the server"""
        # Mock the main function to avoid actual server startup
        mock_main.return_value = None
        
        # Test that we can import the server module and it has expected structure
        from claude_code_context.mcp_server import server
        
        # The server module should exist and be importable
        assert server is not None
        assert hasattr(server, 'main')
        assert callable(server.main)
    
    def test_environment_variables_documented(self):
        """Test that environment variables are documented in __main__.py"""
        main_module = Path('claude_code_context/mcp_server/__main__.py')
        content = main_module.read_text()
        
        # Verify documentation includes environment variables
        assert 'MCP_PROJECT_PATH' in content
        assert 'MCP_COLLECTION_NAME' in content  
        assert 'MCP_MAX_CLAUDE_CALLS' in content
        assert 'MCP_DEBUG' in content
        assert 'QDRANT_URL' in content
        
        # Verify usage examples are provided
        assert 'python -m claude_code_context.mcp_server' in content