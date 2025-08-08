"""
Hook models for Claude Code integration.

Handles UserPromptSubmit hook requests and responses with <ccc> tag parsing.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator
import re


class HookType(Enum):
    """Types of Claude Code hooks"""
    USER_PROMPT_SUBMIT = "user_prompt_submit"
    USER_PROMPT_PRE_PROCESS = "user_prompt_pre_process"
    RESPONSE_POST_PROCESS = "response_post_process"


class CCCQuery(BaseModel):
    """Parsed <ccc>query</ccc> tag from user prompt"""
    model_config = ConfigDict(frozen=True)
    
    # Query content
    query: str
    
    # Location in original prompt
    start_pos: int
    end_pos: int
    
    # Parsing metadata
    tag_full_match: str  # Complete <ccc>...</ccc> match
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate query is not empty after stripping"""
        query = v.strip()
        if not query:
            raise ValueError('CCC query cannot be empty')
        return query
    
    @property
    def length(self) -> int:
        """Get query length"""
        return len(self.query)
    
    @property
    def word_count(self) -> int:
        """Get approximate word count"""
        return len(self.query.split())
    
    @classmethod
    def parse_from_prompt(cls, prompt: str) -> List['CCCQuery']:
        """Parse all <ccc>query</ccc> tags from prompt"""
        queries = []
        
        # Regular expression to match <ccc>...</ccc> tags
        pattern = r'<ccc\s*>(.*?)</ccc\s*>'
        
        for match in re.finditer(pattern, prompt, re.DOTALL | re.IGNORECASE):
            query_text = match.group(1).strip()
            if query_text:  # Only include non-empty queries
                try:
                    query = cls(
                        query=query_text,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        tag_full_match=match.group(0)
                    )
                    queries.append(query)
                except ValueError:
                    # Skip invalid queries
                    continue
        
        return queries
    
    def remove_from_prompt(self, prompt: str) -> str:
        """Remove this query tag from the prompt"""
        return prompt.replace(self.tag_full_match, '', 1)


class HookRequest(BaseModel):
    """Request data for Claude Code hooks"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )
    
    # Hook metadata
    hook_type: HookType
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Request content
    prompt: str
    
    # Context information
    working_directory: Optional[str] = None
    project_name: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Parsed CCC queries (computed)
    ccc_queries: List[CCCQuery] = Field(default_factory=list)
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def __init__(self, **data):
        super().__init__(**data)
        # Auto-parse CCC queries from prompt
        if not self.ccc_queries and self.prompt:
            self.ccc_queries = CCCQuery.parse_from_prompt(self.prompt)
    
    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        """Validate prompt is not empty"""
        if not v.strip():
            raise ValueError('Prompt cannot be empty')
        return v
    
    @property
    def has_ccc_queries(self) -> bool:
        """Check if request contains CCC queries"""
        return len(self.ccc_queries) > 0
    
    @property
    def total_query_length(self) -> int:
        """Get total length of all queries"""
        return sum(query.length for query in self.ccc_queries)
    
    def get_prompt_without_ccc_tags(self) -> str:
        """Get prompt with all CCC tags removed"""
        clean_prompt = self.prompt
        for query in self.ccc_queries:
            clean_prompt = query.remove_from_prompt(clean_prompt)
        return clean_prompt.strip()
    
    def get_all_queries_text(self) -> List[str]:
        """Get list of all query texts"""
        return [query.query for query in self.ccc_queries]


class ContextResult(BaseModel):
    """Single context result from search"""
    model_config = ConfigDict(frozen=True)
    
    # Content identification
    entity_id: str
    entity_type: str
    file_path: str
    
    # Content
    source_code: str
    signature: Optional[str] = None
    docstring: Optional[str] = None
    
    # Location
    start_line: int
    end_line: int
    
    # Relevance
    score: float = Field(ge=0.0, le=1.0)
    rank: int = Field(ge=1)
    
    # Search metadata
    matched_query: str
    search_type: str  # semantic, hybrid, keyword
    
    @field_validator('score')
    @classmethod
    def validate_score(cls, v: float) -> float:
        """Validate and round score"""
        return round(v, 4)
    
    @property
    def location_reference(self) -> str:
        """Get file:line reference"""
        return f"{self.file_path}:{self.start_line}"
    
    @property
    def is_highly_relevant(self) -> bool:
        """Check if highly relevant (>0.8 score)"""
        return self.score > 0.8
    
    def format_for_injection(self, include_metadata: bool = True) -> str:
        """Format result for context injection"""
        lines = []
        
        if include_metadata:
            lines.append(f"# {self.entity_type.title()}: {self.entity_id}")
            lines.append(f"# File: {self.location_reference}")
            lines.append(f"# Relevance: {self.score:.3f}")
            lines.append("")
        
        if self.signature:
            lines.append(f"```")
            lines.append(self.signature)
            lines.append("```")
            lines.append("")
        
        if self.docstring:
            lines.append(self.docstring)
            lines.append("")
        
        # Add source code with minimal context
        if self.source_code:
            lines.append("```python")  # TODO: Detect language
            lines.append(self.source_code)
            lines.append("```")
        
        return "\n".join(lines)


class HookResponse(BaseModel):
    """Response from Claude Code hook processing"""
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )
    
    # Response status  
    success: bool
    processing_time_ms: float
    
    # Enhanced prompt (with context injected)
    enhanced_prompt: str
    
    # Context results
    context_results: List[ContextResult] = Field(default_factory=list)
    
    # Query processing
    queries_processed: int = 0
    total_results_found: int = 0
    
    # Error handling
    error: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    
    # Metadata
    search_stats: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    
    @field_validator('enhanced_prompt')
    @classmethod
    def validate_enhanced_prompt(cls, v: str) -> str:
        """Validate enhanced prompt is not empty"""
        if not v.strip():
            raise ValueError('Enhanced prompt cannot be empty')
        return v
    
    @property
    def has_context(self) -> bool:
        """Check if response includes context"""
        return len(self.context_results) > 0
    
    @property
    def average_relevance_score(self) -> float:
        """Calculate average relevance score"""
        if not self.context_results:
            return 0.0
        return sum(r.score for r in self.context_results) / len(self.context_results)
    
    @property
    def highly_relevant_count(self) -> int:
        """Count highly relevant results"""
        return sum(1 for r in self.context_results if r.is_highly_relevant)
    
    def get_context_summary(self) -> str:
        """Get summary of context results"""
        if not self.has_context:
            return "No context found"
        
        by_type = {}
        for result in self.context_results:
            by_type[result.entity_type] = by_type.get(result.entity_type, 0) + 1
        
        type_summary = ", ".join(f"{count} {entity_type}" for entity_type, count in by_type.items())
        avg_score = self.average_relevance_score
        
        return f"Found {len(self.context_results)} results ({type_summary}) with avg relevance {avg_score:.3f}"
    
    @classmethod
    def success_response(
        cls,
        enhanced_prompt: str,
        context_results: List[ContextResult],
        processing_time_ms: float,
        queries_processed: int = 0,
        search_stats: Optional[Dict[str, Any]] = None
    ) -> 'HookResponse':
        """Create successful response"""
        return cls(
            success=True,
            processing_time_ms=processing_time_ms,
            enhanced_prompt=enhanced_prompt,
            context_results=context_results,
            queries_processed=queries_processed,
            total_results_found=len(context_results),
            search_stats=search_stats or {}
        )
    
    @classmethod
    def error_response(
        cls,
        original_prompt: str,
        error: str,
        processing_time_ms: float,
        warnings: Optional[List[str]] = None
    ) -> 'HookResponse':
        """Create error response"""
        return cls(
            success=False,
            processing_time_ms=processing_time_ms,
            enhanced_prompt=original_prompt,  # Return original on error
            error=error,
            warnings=warnings or []
        )


class HookExecutionContext(BaseModel):
    """Context for hook execution with project information"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )
    
    # Project context
    project_name: str
    project_path: str
    collection_prefix: str
    
    # Qdrant connection
    qdrant_url: str = "http://localhost:6333"
    qdrant_timeout: float = 10.0
    
    # Search settings
    max_results_per_query: int = Field(default=5, ge=1, le=20)
    min_relevance_score: float = Field(default=0.3, ge=0.0)  # No upper limit - scores can exceed 1.0
    enable_hybrid_search: bool = True
    
    # Performance settings
    max_processing_time_ms: float = Field(default=5000, ge=100, le=30000)
    
    @field_validator('project_name')
    @classmethod
    def validate_project_name(cls, v: str) -> str:
        """Validate project name"""
        if not v or not v.replace('-', '').replace('_', '').isalnum():
            raise ValueError('Project name must be alphanumeric with dashes/underscores')
        return v.lower()
    
    def get_collection_names(self) -> Dict[str, str]:
        """Get collection names for this project"""
        return {
            'code': f"{self.collection_prefix}-code",
            'relations': f"{self.collection_prefix}-relations", 
            'embeddings': f"{self.collection_prefix}-embeddings"
        }
    
    @property
    def is_qdrant_available(self) -> bool:
        """Check if Qdrant is accessible"""
        try:
            import requests
            response = requests.get(f"{self.qdrant_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False