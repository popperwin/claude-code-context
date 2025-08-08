"""
Claude CLI orchestrator with secure subprocess management.

This module provides the ClaudeOrchestrator class that manages interactions with
the Claude CLI in a secure way, including input sanitization, JSON schema validation,
and proper subprocess handling to prevent injection attacks.
"""

import asyncio
import json
import logging
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, ValidationError, Field
from .models import MCPServerConfig

logger = logging.getLogger(__name__)


class SearchType(str, Enum):
    """Valid search types returned by Claude"""
    PAYLOAD = "payload"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class ClaudeStrategyResponse(BaseModel):
    """Validated response schema from Claude CLI strategy analysis"""
    search_type: SearchType = Field(..., description="Type of search to perform")
    query: str = Field(..., min_length=1, max_length=1000, description="Optimized search query")
    reasoning: str = Field(..., min_length=1, max_length=2000, description="Why this strategy was selected")
    focus_areas: List[str] = Field(default_factory=list, max_items=10, description="Specific areas to focus on")
    iterations_remaining: int = Field(default=0, ge=0, le=10, description="How many more iterations needed")
    
    class Config:
        use_enum_values = True


@dataclass
class OrchestrationContext:
    """Context for Claude orchestration calls"""
    project_path: Path
    query: str
    iteration: int = 1
    max_iterations: int = 10
    previous_results: List[Dict[str, Any]] = field(default_factory=list)
    search_history: List[str] = field(default_factory=list)
    project_context: Optional[str] = None


class SecurityError(Exception):
    """Raised when input fails security validation"""
    pass


class ClaudeOrchestrator:
    """
    Secure orchestrator for Claude CLI interactions.
    
    Handles subprocess management, input sanitization, JSON schema validation,
    and iterative search strategy optimization.
    """
    
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.claude_cli_path = self._find_claude_cli()
        self.max_claude_calls = config.max_claude_calls
        self.debug_mode = config.debug_mode
        
        # Security constraints
        self.max_query_length = 1000
        self.max_context_length = 20000  # 20k words limit from Sprint 8
        self.timeout_seconds = 30
        
        # Input sanitization patterns
        self.dangerous_patterns = [
            r'[;&|`$]',  # Shell metacharacters
            r'\.\./',    # Path traversal
            r'<script',  # HTML injection
            r'exec\(',   # Code execution
            r'eval\(',   # Code evaluation
            r'import\s+os',  # OS imports
            r'subprocess',   # Subprocess calls
        ]
        
        logger.info(f"ClaudeOrchestrator initialized with CLI at: {self.claude_cli_path}")
    
    def _find_claude_cli(self) -> Optional[str]:
        """Find Claude CLI executable safely"""
        try:
            import shutil
            claude_path = shutil.which("claude")
            if claude_path and os.access(claude_path, os.X_OK):
                return claude_path
            else:
                logger.warning("Claude CLI not found or not executable")
                return None
        except Exception as e:
            logger.error(f"Error finding Claude CLI: {e}")
            return None
    
    def _sanitize_input(self, text: str) -> str:
        """
        Sanitize input text to prevent injection attacks.
        
        Args:
            text: Input text to sanitize
            
        Returns:
            Sanitized text
            
        Raises:
            SecurityError: If dangerous patterns are detected
        """
        if not isinstance(text, str):
            raise SecurityError("Input must be a string")
        
        # Check length limits
        if len(text) > self.max_query_length:
            raise SecurityError(f"Input too long: {len(text)} > {self.max_query_length}")
        
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                raise SecurityError(f"Dangerous pattern detected: {pattern}")
        
        # Remove control characters except whitespace
        sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Normalize whitespace
        sanitized = ' '.join(sanitized.split())
        
        return sanitized
    
    def _build_prompt(self, context: OrchestrationContext) -> str:
        """
        Build secure prompt for Claude CLI.
        
        Args:
            context: Orchestration context
            
        Returns:
            Sanitized prompt string
        """
        # Sanitize all inputs
        query = self._sanitize_input(context.query)
        
        # Build prompt components
        prompt_parts = [
            "Analyze this code search query and provide a JSON response with search strategy.",
            "IMPORTANT: Keep the optimized query simple and focused. Do not add multiple variations or expand the query unnecessarily.",
            "For payload searches, use concise keywords that match actual function/class names.",
            f"Query: {query}",
            f"Iteration: {context.iteration}/{context.max_iterations}",
        ]
        
        # Add project context if available (truncated for security)
        if context.project_context:
            context_preview = context.project_context[:500] + "..." if len(context.project_context) > 500 else context.project_context
            prompt_parts.append(f"Project context: {context_preview}")
        
        # Add search history
        if context.search_history:
            history_str = ", ".join(context.search_history[-3:])  # Last 3 queries only
            prompt_parts.append(f"Previous searches: {history_str}")
        
        # Add response format requirements
        prompt_parts.extend([
            "",
            "Respond with JSON in this exact format:",
            '{',
            '  "search_type": "payload" | "semantic" | "hybrid",',
            '  "query": "optimized search query",',
            '  "reasoning": "why this strategy was chosen",',
            '  "focus_areas": ["area1", "area2"],',
            '  "iterations_remaining": 0',
            '}',
        ])
        
        prompt = "\n".join(prompt_parts)
        
        # Final length check
        if len(prompt) > self.max_context_length:
            raise SecurityError(f"Prompt too long: {len(prompt)} > {self.max_context_length}")
        
        return prompt
    
    async def _execute_claude_cli(self, prompt: str) -> Dict[str, Any]:
        """
        Execute Claude CLI securely with proper subprocess handling.
        
        Args:
            prompt: Sanitized prompt to send to Claude
            
        Returns:
            Parsed JSON response from Claude
            
        Raises:
            subprocess.TimeoutExpired: If Claude CLI times out
            json.JSONDecodeError: If response is not valid JSON
            SecurityError: If response fails validation
        """
        if not self.claude_cli_path:
            raise RuntimeError("Claude CLI not available")
        
        # Build command args list (no shell injection possible)
        cmd_args = [
            self.claude_cli_path,
            # Note: --json not supported by all Claude CLI versions
            "--print",     # Use print mode for non-interactive output
        ]
        
        # Set secure environment
        env = os.environ.copy()
        env['CLAUDE_NO_INTERACTIVE'] = '1'  # Disable interactive prompts
        env['CLAUDE_TIMEOUT'] = str(self.timeout_seconds)
        
        try:
            logger.debug(f"Executing Claude CLI with {len(prompt)} character prompt")
            
            # Execute with timeout and security measures
            process = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=self.config.project_path,  # Safe working directory
            )
            
            # Send prompt and get response with timeout
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=prompt.encode('utf-8')),
                timeout=self.timeout_seconds
            )
            
            # Check return code
            if process.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='replace')
                logger.error(f"Claude CLI failed with code {process.returncode}: {error_msg}")
                raise RuntimeError(f"Claude CLI error: {error_msg}")
            
            # Parse response
            response_text = stdout.decode('utf-8', errors='replace')
            logger.debug(f"Claude CLI response: {len(response_text)} characters")
            
            # Extract JSON from response (Claude might include extra text)
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                raise json.JSONDecodeError("No JSON found in Claude response", response_text, 0)
            
            json_str = json_match.group(0)
            return json.loads(json_str)
            
        except asyncio.TimeoutError:
            logger.error(f"Claude CLI timed out after {self.timeout_seconds} seconds")
            raise subprocess.TimeoutExpired(cmd_args, self.timeout_seconds)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from Claude CLI: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error executing Claude CLI: {e}")
            raise
    
    def _validate_response(self, response_data: Dict[str, Any]) -> ClaudeStrategyResponse:
        """
        Validate Claude CLI response against schema.
        
        Args:
            response_data: Raw response from Claude CLI
            
        Returns:
            Validated response object
            
        Raises:
            ValidationError: If response doesn't match schema
            SecurityError: If response contains dangerous content
        """
        try:
            # Validate with Pydantic schema
            validated = ClaudeStrategyResponse(**response_data)
            
            # Additional security checks
            self._sanitize_input(validated.query)  # Will raise if dangerous
            
            # Sanitize focus areas
            for area in validated.focus_areas:
                self._sanitize_input(area)
            
            logger.debug(f"Validated Claude response: {validated.search_type} search")
            return validated
            
        except ValidationError as e:
            logger.error(f"Claude response validation failed: {e}")
            raise
        except SecurityError as e:
            logger.error(f"Claude response security check failed: {e}")
            raise
    
    async def analyze_query(self, context: OrchestrationContext) -> ClaudeStrategyResponse:
        """
        Analyze search query and get optimized strategy from Claude.
        
        Args:
            context: Orchestration context with query and history
            
        Returns:
            Validated strategy response from Claude
            
        Raises:
            SecurityError: If inputs fail security validation
            RuntimeError: If Claude CLI is not available or fails
            subprocess.TimeoutExpired: If Claude CLI times out
        """
        if not self.claude_cli_path:
            # Fallback to basic strategy without Claude
            logger.warning("Claude CLI unavailable, using fallback strategy")
            return self._fallback_strategy(context)
        
        try:
            # Build and sanitize prompt
            prompt = self._build_prompt(context)
            
            # Execute Claude CLI
            start_time = time.perf_counter()
            response_data = await self._execute_claude_cli(prompt)
            execution_time = time.perf_counter() - start_time
            
            # Validate response
            validated_response = self._validate_response(response_data)
            
            logger.info(f"Claude analysis completed in {execution_time:.2f}s: {validated_response.search_type}")
            
            return validated_response
            
        except Exception as e:
            logger.error(f"Claude analysis failed: {e}")
            if self.debug_mode:
                raise
            else:
                # Graceful fallback in production
                return self._fallback_strategy(context)
    
    def _fallback_strategy(self, context: OrchestrationContext) -> ClaudeStrategyResponse:
        """
        Provide fallback strategy when Claude CLI is unavailable.
        
        Args:
            context: Orchestration context
            
        Returns:
            Basic strategy response
        """
        query = self._sanitize_input(context.query)
        
        # Simple heuristics for search type
        if len(query.split()) <= 3 and not any(word in query.lower() for word in ['how', 'what', 'why', 'explain']):
            search_type = SearchType.PAYLOAD
        elif any(word in query.lower() for word in ['explain', 'how', 'pattern', 'implement', 'example']):
            search_type = SearchType.SEMANTIC
        else:
            search_type = SearchType.HYBRID
        
        return ClaudeStrategyResponse(
            search_type=search_type,
            query=query,
            reasoning=f"Fallback strategy: using {search_type.value} search based on query characteristics",
            focus_areas=["general"],
            iterations_remaining=0
        )
    
    async def iterative_search(self, initial_query: str, project_context: Optional[str] = None) -> List[ClaudeStrategyResponse]:
        """
        Perform iterative search with Claude orchestration.
        
        Args:
            initial_query: Initial search query
            project_context: Optional project context
            
        Returns:
            List of strategy responses from all iterations
        """
        results = []
        context = OrchestrationContext(
            project_path=self.config.project_path,
            query=initial_query,
            max_iterations=self.max_claude_calls,
            project_context=project_context
        )
        
        for iteration in range(1, self.max_claude_calls + 1):
            context.iteration = iteration
            
            try:
                strategy = await self.analyze_query(context)
                results.append(strategy)
                
                # Update context for next iteration
                context.search_history.append(strategy.query)
                
                # Check if we should continue
                if strategy.iterations_remaining == 0:
                    logger.info(f"Claude orchestration completed after {iteration} iterations")
                    break
                    
            except Exception as e:
                logger.error(f"Iteration {iteration} failed: {e}")
                break
        
        return results
    
    def is_available(self) -> bool:
        """Check if Claude CLI orchestration is available"""
        return self.claude_cli_path is not None
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Claude CLI orchestration.
        
        Returns:
            Health status information
        """
        status = {
            "claude_cli_available": self.claude_cli_path is not None,
            "claude_cli_path": self.claude_cli_path,
            "max_claude_calls": self.max_claude_calls,
            "timeout_seconds": self.timeout_seconds,
            "debug_mode": self.debug_mode,
        }
        
        if self.claude_cli_path:
            try:
                # Test basic Claude CLI execution
                test_context = OrchestrationContext(
                    project_path=self.config.project_path,
                    query="test query",
                    max_iterations=1
                )
                
                start_time = time.perf_counter()
                await self.analyze_query(test_context)
                response_time = time.perf_counter() - start_time
                
                status.update({
                    "health_check_passed": True,
                    "response_time_seconds": response_time,
                })
                
            except Exception as e:
                status.update({
                    "health_check_passed": False,
                    "error": str(e),
                })
        
        return status