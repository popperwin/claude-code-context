"""
MCP Server models for claude-code-context.

Defines Pydantic models for MCP protocol compliance, search requests/responses,
and server configuration. Follows the same patterns as core.models.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator

# Import the existing CollectionType from core system for consistency
from core.storage.schemas import CollectionType


class MCPServerStatus(Enum):
    """MCP server status states"""
    INITIALIZING = "initializing"
    CONNECTING = "connecting"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class SearchMode(Enum):
    """Search execution modes"""
    PAYLOAD = "payload"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    AUTO = "auto"


class MCPServerConfig(BaseModel):
    """MCP server configuration model"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )
    
    # Project settings
    project_path: Path = Field(default=Path("."))
    collection_name: str = Field(default="auto")
    
    # MCP settings
    max_claude_calls: int = Field(default=10, ge=1, le=50)
    debug_mode: bool = Field(default=False)
    
    # Context management (token-aware limits for Claude 4 - 200K token context)
    # Reserve ~50K tokens for Claude responses, use ~150K for input
    max_user_query_chars: int = Field(default=4000, ge=100, le=10000)      # ~1K tokens - user queries
    max_prompt_chars: int = Field(default=40000, ge=1000, le=100000)       # ~10K tokens - full prompts
    max_context_chars: int = Field(default=600000, ge=10000, le=2000000)   # ~150K tokens - conversation context
    max_results_summary_chars: int = Field(default=20000, ge=1000, le=50000) # ~5K tokens - results per iteration
    
    # Qdrant connection
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_timeout: float = Field(default=60.0, ge=1.0, le=300.0)
    
    # Performance settings
    search_timeout_ms: int = Field(default=30000, ge=1000, le=120000)
    context_word_limit: int = Field(default=20000, ge=1000, le=50000)
    
    # Result filtering settings (MCP layer, not core engine)
    # These can be overridden via environment variables:
    # MCP_MAX_RESULTS, MCP_PAYLOAD_MIN_SCORE, MCP_SEMANTIC_MIN_SCORE, MCP_HYBRID_MIN_SCORE
    max_results: int = Field(default=10, ge=1, le=100, description="Maximum results to return to user")
    payload_min_score: float = Field(default=0.15, ge=0.0, le=1.0, description="Min score for payload results")
    semantic_min_score: float = Field(default=0.4, ge=0.0, le=1.0, description="Min score for semantic results")
    hybrid_min_score: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description=(
            "Min score for hybrid results. When using RRF fusion in core, scores are scaled into ~[0,1]; "
            "0.1 is a conservative default that avoids filtering good RRF results. Set to 0.0 to disable."
        ),
    )
    
    @field_validator('qdrant_url')
    @classmethod
    def validate_qdrant_url(cls, v: str) -> str:
        """Validate Qdrant URL format"""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('Qdrant URL must start with http:// or https://')
        return v.rstrip('/')
    
    @field_validator('project_path')
    @classmethod
    def validate_project_path(cls, v: Path) -> Path:
        """Ensure project path is absolute"""
        return v.resolve()
    
    def get_collection_name(self, project_name: Optional[str] = None) -> str:
        """Generate collection name for project"""
        if self.collection_name != "auto":
            return self.collection_name
        
        if project_name:
            # Sanitize project name for collection
            safe_name = project_name.lower().replace(' ', '_').replace('-', '_')
            safe_name = ''.join(c for c in safe_name if c.isalnum() or c == '_')
            return f"ccc_{safe_name}"
        
        # Fallback to directory name
        dir_name = self.project_path.name.lower().replace(' ', '_').replace('-', '_')
        dir_name = ''.join(c for c in dir_name if c.isalnum() or c == '_')
        return f"ccc_{dir_name}"
    
    def get_typed_collection_name(self, collection_type: CollectionType = CollectionType.CODE) -> str:
        """
        Get collection name with proper core system type suffix.
        
        This ensures compatibility with the core CollectionManager which
        appends collection type suffixes like '-code', '-relations', etc.
        
        Args:
            collection_type: Type of collection from core system enum
            
        Returns:
            Collection name with type suffix for core system compatibility
        """
        base_name = self.get_collection_name()
        suffix = collection_type.value
        
        # Prevent double-suffixing if the collection name already ends with the type
        if base_name.endswith(f"-{suffix}"):
            return base_name
        
        # Use the enum's value property to get the string suffix
        return f"{base_name}-{suffix}"


class SearchRequest(BaseModel):
    """MCP search request model"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    # Request identification
    request_id: str = Field(..., min_length=1, max_length=100)
    query: str = Field(..., min_length=1, max_length=1000)
    
    # Search parameters
    mode: SearchMode = Field(default=SearchMode.AUTO)
    limit: int = Field(default=10, ge=1, le=100, description="Maximum results to return")
    min_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum relevance score")
    
    # Context and filtering
    include_code: bool = Field(default=True)
    include_docs: bool = Field(default=True)
    file_types: Optional[List[str]] = Field(default=None)
    
    # Multi-turn support
    session_id: Optional[str] = Field(default=None, max_length=100)
    previous_queries: Optional[List[str]] = Field(default=None, max_items=10)
    
    @field_validator('file_types')
    @classmethod
    def validate_file_types(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate file type filters"""
        if v is None:
            return v
        
        # Remove dots and validate extensions
        cleaned = []
        for ft in v:
            clean_ft = ft.lstrip('.').lower()
            if clean_ft and clean_ft.replace('_', '').isalnum():
                cleaned.append(clean_ft)
        
        return cleaned if cleaned else None


class SearchResult(BaseModel):
    """Individual search result item"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    # Result identification
    entity_id: str = Field(..., min_length=1)
    file_path: str = Field(..., min_length=1)
    
    # Content
    name: str = Field(..., min_length=1)
    content: str = Field(default="")
    docstring: Optional[str] = Field(default=None)
    
    # Metadata
    entity_type: str = Field(..., min_length=1)
    language: str = Field(..., min_length=1)
    
    # Location
    start_line: int = Field(..., ge=1)
    end_line: int = Field(..., ge=1)
    start_byte: int = Field(..., ge=0)
    end_byte: int = Field(..., ge=0)
    
    # Search relevance
    relevance_score: float = Field(..., ge=0.0)  # No upper limit - scores can exceed 1.0
    match_type: str = Field(..., min_length=1)  # "exact", "semantic", "hybrid"
    
    @field_validator('end_line')
    @classmethod
    def validate_line_order(cls, v: int, info) -> int:
        """Ensure end_line >= start_line"""
        if 'start_line' in info.data and v < info.data['start_line']:
            raise ValueError('end_line must be >= start_line')
        return v
    
    @field_validator('end_byte')
    @classmethod
    def validate_byte_order(cls, v: int, info) -> int:  
        """Ensure end_byte >= start_byte"""
        if 'start_byte' in info.data and v < info.data['start_byte']:
            raise ValueError('end_byte must be >= start_byte')
        return v


class SearchResponse(BaseModel):
    """MCP search response model"""
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )
    
    # Response identification
    request_id: str = Field(..., min_length=1)
    session_id: Optional[str] = Field(default=None)
    
    # Results
    results: List[SearchResult] = Field(default_factory=list)
    total_found: int = Field(..., ge=0)
    
    # Execution metadata
    execution_time_ms: float = Field(..., ge=0)
    search_mode_used: SearchMode = Field(...)
    claude_calls_made: int = Field(default=0, ge=0)
    
    # Claude orchestration
    query_optimization: Optional[str] = Field(default=None)
    
    # Status and errors
    success: bool = Field(...)
    error_message: Optional[str] = Field(default=None)
    warnings: List[str] = Field(default_factory=list)
    
    # Performance metrics
    qdrant_time_ms: Optional[float] = Field(default=None, ge=0)
    embedding_time_ms: Optional[float] = Field(default=None, ge=0)
    
    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.now)
    
    @field_validator('results')
    @classmethod
    def validate_results_limit(cls, v: List[SearchResult]) -> List[SearchResult]:
        """Ensure results don't exceed reasonable limits"""
        if len(v) > 100:
            raise ValueError('Results list cannot exceed 100 items')
        return v


class ServerHealthStatus(BaseModel):
    """Server health check response"""
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )
    
    # Overall status
    status: MCPServerStatus = Field(...)
    healthy: bool = Field(...)
    
    # Component health
    qdrant_connected: bool = Field(...)
    collection_available: bool = Field(...)
    claude_cli_available: bool = Field(...)
    
    # Performance metrics
    uptime_seconds: float = Field(..., ge=0)
    requests_handled: int = Field(default=0, ge=0)
    average_response_time_ms: Optional[float] = Field(default=None, ge=0)
    
    # Configuration
    project_path: str = Field(...)
    collection_name: str = Field(...)
    
    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Optional error details
    error_details: Optional[Dict[str, Any]] = Field(default=None)


class MCPToolDefinition(BaseModel):
    """MCP tool definition for protocol compliance"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=1, max_length=500)
    input_schema: Dict[str, Any] = Field(...)
    
    @field_validator('name')
    @classmethod
    def validate_tool_name(cls, v: str) -> str:
        """Validate tool name format"""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Tool name must be alphanumeric with underscores/hyphens')
        return v


class MCPServerInfo(BaseModel):
    """MCP server information for handshake"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    name: str = Field(default="claude-code-context")
    version: str = Field(default="1.0.0")
    protocol_version: str = Field(default="2024-11-05")
    
    # Capabilities
    tools: List[MCPToolDefinition] = Field(default_factory=list)
    supports_notifications: bool = Field(default=False)
    supports_sampling: bool = Field(default=False)