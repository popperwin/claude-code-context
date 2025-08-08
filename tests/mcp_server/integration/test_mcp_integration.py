"""
Integration tests for MCP server with Claude orchestration.

Tests complete single-turn orchestration flow, strategy execution, 
and result synthesis in the MCP server environment.
"""

import asyncio
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from claude_code_context.mcp_server.models import (
    SearchRequest, 
    SearchResponse, 
    SearchMode,
    MCPServerConfig
)
from claude_code_context.mcp_server.search_executor import SearchExecutor
from claude_code_context.mcp_server.connection import QdrantConnectionManager
from claude_code_context.mcp_server.orchestrator import (
    ClaudeStrategyResponse,
    SearchType
)


class TestMCPOrchestrationIntegration:
    """Test complete MCP server orchestration integration"""
    
    @pytest.fixture
    def temp_project(self):
        """Create temporary project structure"""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            
            # Create project files
            (project_path / "main.py").write_text("""
def authenticate_user(username: str, password: str) -> bool:
    '''Authenticate user with username and password'''
    return validate_credentials(username, password)

def validate_credentials(username: str, password: str) -> bool:
    '''Validate user credentials against database'''
    # Implementation here
    return True

class UserManager:
    '''Manages user operations and authentication'''
    
    def login(self, credentials):
        return authenticate_user(credentials.username, credentials.password)
""")
            
            (project_path / "README.md").write_text("""
# Test Project

A Python authentication system with user management.

## Features
- User authentication
- Credential validation
- Session management
""")
            
            (project_path / "CLAUDE.md").write_text("""
# Test Project Configuration

This is a Python authentication system.

## Architecture
- authentication functions in main.py
- UserManager class for user operations
""")
            
            yield project_path
    
    @pytest.fixture
    def config(self, temp_project):
        """Create test configuration"""
        return MCPServerConfig(
            project_path=temp_project,
            collection_name="test_mcp_integration",
            qdrant_url="http://localhost:6334",
            max_claude_calls=3,
            debug_mode=False,
            context_word_limit=5000
        )
    
    @pytest.fixture
    def connection_manager(self, config):
        """Create connection manager"""
        return QdrantConnectionManager(config)
    
    @pytest.fixture
    def search_executor(self, connection_manager, config):
        """Create search executor"""
        return SearchExecutor(connection_manager, config)
    
    @pytest.mark.asyncio
    async def test_orchestration_initialization(self, search_executor):
        """Test that orchestration components initialize correctly"""
        # Mock search infrastructure to avoid Qdrant dependency
        with patch('claude_code_context.mcp_server.search_executor.SEARCH_AVAILABLE', False):
            initialized = await search_executor.initialize()
            
            # Should still initialize with placeholders
            assert initialized is False  # Search infrastructure not available
            assert search_executor._orchestration_enabled is False
    
    @pytest.mark.asyncio
    async def test_orchestration_with_mocked_claude(self, search_executor, config):
        """Test complete orchestration flow with mocked Claude CLI"""
        # Mock Claude orchestrator response
        mock_strategy = ClaudeStrategyResponse(
            search_type=SearchType.SEMANTIC,
            query="authentication login user validation functions",
            reasoning="User is looking for authentication-related functions",
            focus_areas=["auth", "security", "validation"],
            iterations_remaining=0
        )
        
        # Mock search infrastructure
        with patch('claude_code_context.mcp_server.search_executor.SEARCH_AVAILABLE', False):
            # Initialize with mocked components
            search_executor._orchestration_enabled = True
            search_executor._initialized = True
            
            # Mock orchestrator
            mock_orchestrator = AsyncMock()
            mock_orchestrator.analyze_query.return_value = mock_strategy
            search_executor._orchestrator = mock_orchestrator
            
            # Mock project context
            search_executor._project_context = "Test project with authentication"
            
            # Create search request
            request = SearchRequest(
                request_id="test_1",
                query="find authentication functions",
                mode=SearchMode.AUTO,
                limit=10
            )
            
            # Execute search
            response = await search_executor.execute_search(request)
            
            # Verify orchestration was called
            mock_orchestrator.analyze_query.assert_called_once()
            
            # Verify response structure
            assert response.success is True
            assert response.claude_calls_made == 1
            assert response.query_optimization == "authentication login user validation functions"
            assert response.search_mode_used == SearchMode.SEMANTIC
    
    @pytest.mark.asyncio
    async def test_orchestration_fallback_on_error(self, search_executor):
        """Test orchestration falls back gracefully on Claude CLI error"""
        with patch('claude_code_context.mcp_server.search_executor.SEARCH_AVAILABLE', False):
            # Initialize with orchestration enabled
            search_executor._orchestration_enabled = True
            search_executor._initialized = True
            
            # Mock orchestrator that fails
            mock_orchestrator = AsyncMock()
            mock_orchestrator.analyze_query.side_effect = Exception("Claude CLI error")
            search_executor._orchestrator = mock_orchestrator
            
            # Create search request
            request = SearchRequest(
                request_id="test_2",
                query="test query",
                mode=SearchMode.AUTO,
                limit=5
            )
            
            # Execute search - should not fail due to orchestration error
            response = await search_executor.execute_search(request)
            
            # Should succeed with fallback
            assert response.success is True
            assert response.claude_calls_made == 0  # No successful Claude calls
            assert response.search_mode_used == SearchMode.AUTO  # Original mode used
    
    @pytest.mark.asyncio
    async def test_direct_search_modes(self, search_executor):
        """Test that direct search modes bypass orchestration"""
        with patch('claude_code_context.mcp_server.search_executor.SEARCH_AVAILABLE', False):
            search_executor._orchestration_enabled = True
            search_executor._initialized = True
            
            # Mock orchestrator (should not be called)
            mock_orchestrator = AsyncMock()
            search_executor._orchestrator = mock_orchestrator
            
            # Test each direct mode
            for mode in [SearchMode.PAYLOAD, SearchMode.SEMANTIC, SearchMode.HYBRID]:
                request = SearchRequest(
                    request_id=f"test_{mode.value}",
                    query="test query",
                    mode=mode,
                    limit=5
                )
                
                response = await search_executor.execute_search(request)
                
                # Should succeed without orchestration
                assert response.success is True
                assert response.claude_calls_made == 0
                assert response.search_mode_used == mode
            
            # Orchestrator should never have been called
            mock_orchestrator.analyze_query.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_health_check_with_orchestration(self, search_executor):
        """Test health check includes orchestration status"""
        with patch('claude_code_context.mcp_server.search_executor.SEARCH_AVAILABLE', False):
            # Mock orchestration components
            search_executor._orchestration_enabled = True
            search_executor._initialized = True
            
            mock_orchestrator = AsyncMock()
            mock_orchestrator.health_check.return_value = {
                "claude_cli_available": True,
                "health_check_passed": True
            }
            search_executor._orchestrator = mock_orchestrator
            
            # Mock context builder
            search_executor._context_builder = Mock()
            search_executor._project_context = "Test context"
            
            # Get health status
            health = await search_executor.health_check()
            
            # Verify orchestration status is included
            assert "search_executor" in health
            assert health["search_executor"]["orchestration_enabled"] is True
            assert health["search_executor"]["orchestrator_available"] is True
            assert health["search_executor"]["context_builder_available"] is True
            assert health["search_executor"]["project_context_ready"] is True
            
            assert "orchestrator" in health
            assert health["orchestrator"]["claude_cli_available"] is True
    
    @pytest.mark.asyncio 
    async def test_metrics_include_orchestration(self, search_executor):
        """Test metrics include orchestration performance data"""
        with patch('claude_code_context.mcp_server.search_executor.SEARCH_AVAILABLE', False):
            # Set up orchestration state
            search_executor._orchestration_enabled = True
            search_executor._initialized = True
            search_executor._orchestration_calls = 5
            search_executor._orchestration_time = 250.0  # 250ms total
            
            # Get metrics
            metrics = search_executor.get_metrics()
            
            # Verify orchestration metrics
            assert "search_executor" in metrics
            assert metrics["search_executor"]["orchestration_enabled"] is True
            
            assert "performance" in metrics
            perf = metrics["performance"]
            assert perf["orchestration_calls"] == 5
            assert perf["average_orchestration_time_ms"] == 50.0  # 250/5
            assert perf["total_orchestration_time_seconds"] == 0.25
            
            assert "components" in metrics
            comp = metrics["components"]
            assert "orchestrator_available" in comp
            assert "context_builder_available" in comp
            assert "project_context_ready" in comp
    
    @pytest.mark.asyncio
    async def test_shutdown_cleans_up_orchestration(self, search_executor):
        """Test shutdown properly cleans up orchestration components"""
        # Set up orchestration components
        search_executor._orchestrator = Mock()
        search_executor._context_builder = Mock()
        search_executor._project_context = "test"
        search_executor._orchestration_enabled = True
        
        # Shutdown
        await search_executor.shutdown()
        
        # Verify cleanup
        assert search_executor._orchestrator is None
        assert search_executor._context_builder is None
        assert search_executor._project_context is None
        assert search_executor._orchestration_enabled is False


class TestMCPOrchestrationWithRealContext:
    """Test orchestration with real project context building"""
    
    @pytest.fixture
    def temp_project(self):
        """Create temporary project with rich content"""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            
            # Create comprehensive project structure
            (project_path / "src").mkdir()
            (project_path / "tests").mkdir()
            
            # Main application
            (project_path / "src" / "auth.py").write_text("""
import hashlib
import jwt
from typing import Optional

class AuthenticationService:
    '''Service for handling user authentication'''
    
    def __init__(self):
        self.secret_key = "test-secret"
    
    def hash_password(self, password: str) -> str:
        '''Hash password using SHA256'''
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, password: str, hash: str) -> bool:
        '''Verify password against hash'''
        return self.hash_password(password) == hash
    
    def generate_jwt_token(self, user_id: str) -> str:
        '''Generate JWT token for user'''
        payload = {"user_id": user_id}
        return jwt.encode(payload, self.secret_key, algorithm="HS256")
    
    def validate_jwt_token(self, token: str) -> Optional[dict]:
        '''Validate JWT token and return payload'''
        try:
            return jwt.decode(token, self.secret_key, algorithms=["HS256"])
        except jwt.InvalidTokenError:
            return None

def authenticate_user(username: str, password: str) -> bool:
    '''Main authentication function'''
    auth_service = AuthenticationService()
    # Implementation would check database
    return True
""")
            
            # Database models
            (project_path / "src" / "models.py").write_text("""
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class User:
    '''User model for authentication system'''
    id: str
    username: str
    email: str
    password_hash: str
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True
    
    def check_password(self, password: str) -> bool:
        '''Check if provided password matches user's password'''
        from .auth import AuthenticationService
        auth = AuthenticationService()
        return auth.verify_password(password, self.password_hash)

@dataclass
class Session:
    '''User session model'''
    id: str
    user_id: str
    token: str
    expires_at: datetime
    created_at: datetime
""")
            
            # Tests
            (project_path / "tests" / "test_auth.py").write_text("""
import pytest
from src.auth import AuthenticationService, authenticate_user

def test_hash_password():
    auth = AuthenticationService()
    hashed = auth.hash_password("test123")
    assert len(hashed) == 64  # SHA256 hex length

def test_verify_password():
    auth = AuthenticationService()
    password = "test123"
    hashed = auth.hash_password(password)
    assert auth.verify_password(password, hashed) is True
    assert auth.verify_password("wrong", hashed) is False

def test_jwt_operations():
    auth = AuthenticationService()
    token = auth.generate_jwt_token("user123")
    payload = auth.validate_jwt_token(token)
    assert payload["user_id"] == "user123"
""")
            
            # Documentation
            (project_path / "README.md").write_text("""
# Authentication System

A comprehensive Python authentication system with JWT tokens.

## Features
- Password hashing and verification
- JWT token generation and validation
- User and session models
- Comprehensive test suite

## Components
- `src/auth.py` - Core authentication service
- `src/models.py` - User and session models
- `tests/` - Test suite

## Usage
```python
from src.auth import authenticate_user
result = authenticate_user("username", "password")
```
""")
            
            (project_path / "CLAUDE.md").write_text("""
# Authentication System - Claude Configuration

This project implements a secure authentication system.

## Architecture
- **AuthenticationService**: Main authentication logic
- **User/Session models**: Data structures  
- **JWT tokens**: Stateless authentication
- **Password hashing**: SHA256 security

## Key Functions
- `authenticate_user()` - Main entry point
- `hash_password()` - Password security
- `generate_jwt_token()` - Token creation
- `validate_jwt_token()` - Token verification

## Search Tips
- Find auth functions: "authentication service methods"
- Find models: "user session dataclass models"  
- Find tests: "test authentication validation"
""")
            
            yield project_path
    
    @pytest.mark.asyncio
    async def test_real_context_building(self, temp_project):
        """Test context building with real project structure"""
        config = MCPServerConfig(
            project_path=temp_project,
            collection_name="test_context",
            qdrant_url="http://localhost:6334",
            context_word_limit=10000
        )
        
        connection_manager = QdrantConnectionManager(config)
        search_executor = SearchExecutor(connection_manager, config)
        
        # Initialize orchestration components only
        from claude_code_context.mcp_server.context_builder import ProjectContextBuilder
        from claude_code_context.mcp_server.orchestrator import ClaudeOrchestrator
        
        context_builder = ProjectContextBuilder(config)
        
        # Test context building
        assert context_builder.is_valid_project()
        context = await context_builder.build_project_context()
        
        # Verify context contains expected information
        assert temp_project.name in context
        assert "Python" in context
        assert "AuthenticationService" in context
        assert "JWT token" in context
        assert "Directory Structure" in context
        assert "CLAUDE.md" in context
        
        # Verify word limit is respected
        word_count = context_builder._count_words(context)
        assert word_count <= config.context_word_limit
        assert word_count > 100  # Should have substantial content
    
    @pytest.mark.asyncio
    async def test_orchestration_with_rich_context(self, temp_project):
        """Test orchestration leverages rich project context"""
        config = MCPServerConfig(
            project_path=temp_project,
            collection_name="test_rich_context",
            qdrant_url="http://localhost:6334",
            max_claude_calls=2,
            debug_mode=False
        )
        
        connection_manager = QdrantConnectionManager(config)
        search_executor = SearchExecutor(connection_manager, config)
        
        # Mock successful context building and orchestration
        with patch('claude_code_context.mcp_server.search_executor.SEARCH_AVAILABLE', False):
            # Set up successful orchestration
            search_executor._orchestration_enabled = True
            search_executor._initialized = True
            
            # Real context from project
            from claude_code_context.mcp_server.context_builder import ProjectContextBuilder
            context_builder = ProjectContextBuilder(config)
            search_executor._project_context = await context_builder.build_project_context()
            
            # Mock Claude response based on rich context
            mock_strategy = ClaudeStrategyResponse(
                search_type=SearchType.HYBRID,
                query="AuthenticationService JWT token validation methods",
                reasoning="Based on project context, user wants JWT-related authentication methods",
                focus_areas=["jwt", "auth", "token", "validation"],
                iterations_remaining=0
            )
            
            mock_orchestrator = AsyncMock()
            mock_orchestrator.analyze_query.return_value = mock_strategy
            search_executor._orchestrator = mock_orchestrator
            
            # Test search with authentication query
            request = SearchRequest(
                request_id="rich_context_test",
                query="find JWT token functions",
                mode=SearchMode.AUTO,
                limit=10
            )
            
            response = await search_executor.execute_search(request)
            
            # Verify orchestration received rich context
            call_args = mock_orchestrator.analyze_query.call_args[0][0]
            assert call_args.project_context is not None
            assert "AuthenticationService" in call_args.project_context
            assert "JWT" in call_args.project_context
            
            # Verify response shows orchestration worked
            assert response.success is True
            assert response.claude_calls_made == 1
            assert "AuthenticationService" in response.query_optimization
            assert "JWT" in response.query_optimization