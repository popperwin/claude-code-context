"""
Unit tests for hooks model functionality.

Tests CCC tag parsing, hook request/response processing, context results, and execution context.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch
from typing import List, Dict, Any
from pydantic import ValidationError

from core.models.hooks import (
    HookType,
    CCCQuery,
    HookRequest,
    ContextResult,
    HookResponse,
    HookExecutionContext
)


class TestHookType:
    """Test HookType enum"""
    
    def test_hook_type_values(self):
        """Test hook type enum values"""
        assert HookType.USER_PROMPT_SUBMIT.value == "user_prompt_submit"
        assert HookType.USER_PROMPT_PRE_PROCESS.value == "user_prompt_pre_process"
        assert HookType.RESPONSE_POST_PROCESS.value == "response_post_process"
    
    def test_hook_type_string_representation(self):
        """Test string representation of hook types"""
        assert str(HookType.USER_PROMPT_SUBMIT) == "HookType.USER_PROMPT_SUBMIT"


class TestCCCQuery:
    """Test CCCQuery functionality"""
    
    def test_basic_initialization(self):
        """Test basic CCC query creation"""
        query = CCCQuery(
            query="test query",
            start_pos=10,
            end_pos=25,
            tag_full_match="<ccc>test query</ccc>"
        )
        
        assert query.query == "test query"
        assert query.start_pos == 10
        assert query.end_pos == 25
        assert query.tag_full_match == "<ccc>test query</ccc>"
    
    def test_query_validation_strips_whitespace(self):
        """Test query validation strips whitespace"""
        query = CCCQuery(
            query="  test query  ",
            start_pos=0,
            end_pos=15,
            tag_full_match="<ccc>  test query  </ccc>"
        )
        
        assert query.query == "test query"
    
    def test_query_validation_empty_fails(self):
        """Test validation fails for empty query"""
        with pytest.raises(ValueError, match="CCC query cannot be empty"):
            CCCQuery(
                query="",
                start_pos=0,
                end_pos=10,
                tag_full_match="<ccc></ccc>"
            )
        
        with pytest.raises(ValueError, match="CCC query cannot be empty"):
            CCCQuery(
                query="   ",
                start_pos=0,
                end_pos=10,
                tag_full_match="<ccc>   </ccc>"
            )
    
    def test_length_property(self):
        """Test query length property"""
        query = CCCQuery(
            query="hello world",
            start_pos=0,
            end_pos=10,
            tag_full_match="<ccc>hello world</ccc>"
        )
        
        assert query.length == 11
    
    def test_word_count_property(self):
        """Test word count property"""
        query = CCCQuery(
            query="hello world test",
            start_pos=0,
            end_pos=10,
            tag_full_match="<ccc>hello world test</ccc>"
        )
        
        assert query.word_count == 3
    
    def test_word_count_single_word(self):
        """Test word count with single word"""
        query = CCCQuery(
            query="hello",
            start_pos=0,
            end_pos=10,
            tag_full_match="<ccc>hello</ccc>"
        )
        
        assert query.word_count == 1
    
    def test_parse_from_prompt_basic(self):
        """Test parsing CCC queries from prompt"""
        prompt = "Hello <ccc>search for functions</ccc> in the code"
        
        queries = CCCQuery.parse_from_prompt(prompt)
        
        assert len(queries) == 1
        assert queries[0].query == "search for functions"
        assert queries[0].start_pos == 6
        assert queries[0].end_pos == 37
        assert queries[0].tag_full_match == "<ccc>search for functions</ccc>"
    
    def test_parse_from_prompt_multiple(self):
        """Test parsing multiple CCC queries"""
        prompt = "Find <ccc>auth functions</ccc> and <ccc>validation code</ccc> examples"
        
        queries = CCCQuery.parse_from_prompt(prompt)
        
        assert len(queries) == 2
        assert queries[0].query == "auth functions"
        assert queries[1].query == "validation code"
    
    def test_parse_from_prompt_multiline(self):
        """Test parsing multiline CCC queries"""
        prompt = """Find <ccc>authentication
        validation
        error handling</ccc> examples"""
        
        queries = CCCQuery.parse_from_prompt(prompt)
        
        assert len(queries) == 1
        expected_query = "authentication\n        validation\n        error handling"
        assert queries[0].query == expected_query
    
    def test_parse_from_prompt_case_insensitive(self):
        """Test parsing is case insensitive"""
        prompt = "Find <CCC>test functions</CCC> and <Ccc>validation</ccc> code"
        
        queries = CCCQuery.parse_from_prompt(prompt)
        
        assert len(queries) == 2
        assert queries[0].query == "test functions"
        assert queries[1].query == "validation"
    
    def test_parse_from_prompt_with_spaces_in_tags(self):
        """Test parsing with spaces after ccc in tags"""
        prompt = "Find <ccc >test functions</ccc > in code"
        
        queries = CCCQuery.parse_from_prompt(prompt)
        
        assert len(queries) == 1
        assert queries[0].query == "test functions"
    
    def test_parse_from_prompt_empty_queries(self):
        """Test parsing skips empty queries"""
        prompt = "Find <ccc></ccc> and <ccc>  </ccc> and <ccc>valid query</ccc>"
        
        queries = CCCQuery.parse_from_prompt(prompt)
        
        assert len(queries) == 1
        assert queries[0].query == "valid query"
    
    def test_parse_from_prompt_no_queries(self):
        """Test parsing with no CCC queries"""
        prompt = "Regular prompt without any special tags"
        
        queries = CCCQuery.parse_from_prompt(prompt)
        
        assert len(queries) == 0
    
    def test_remove_from_prompt(self):
        """Test removing query tag from prompt"""
        prompt = "Find <ccc>auth functions</ccc> in the code"
        query = CCCQuery(
            query="auth functions",
            start_pos=5,
            end_pos=26,
            tag_full_match="<ccc>auth functions</ccc>"
        )
        
        result = query.remove_from_prompt(prompt)
        
        assert result == "Find  in the code"
    
    def test_remove_from_prompt_multiple_occurrences(self):
        """Test removing only first occurrence of tag"""
        prompt = "Find <ccc>test</ccc> and <ccc>test</ccc> again"
        query = CCCQuery(
            query="test",
            start_pos=5,
            end_pos=16,
            tag_full_match="<ccc>test</ccc>"
        )
        
        result = query.remove_from_prompt(prompt)
        
        assert result == "Find  and <ccc>test</ccc> again"
    
    def test_frozen_model(self):
        """Test that CCCQuery is frozen (immutable)"""
        query = CCCQuery(
            query="test",
            start_pos=0,
            end_pos=10,
            tag_full_match="<ccc>test</ccc>"
        )
        
        with pytest.raises(ValidationError):
            query.query = "modified"


class TestHookRequest:
    """Test HookRequest functionality"""
    
    def test_basic_initialization(self):
        """Test basic hook request creation"""
        request = HookRequest(
            hook_type=HookType.USER_PROMPT_SUBMIT,
            prompt="Test prompt"
        )
        
        assert request.hook_type == HookType.USER_PROMPT_SUBMIT
        assert request.prompt == "Test prompt"
        assert isinstance(request.timestamp, datetime)
        assert request.ccc_queries == []
    
    def test_initialization_with_ccc_queries(self):
        """Test initialization auto-parses CCC queries"""
        prompt = "Find <ccc>auth functions</ccc> in code"
        request = HookRequest(
            hook_type=HookType.USER_PROMPT_SUBMIT,
            prompt=prompt
        )
        
        assert len(request.ccc_queries) == 1
        assert request.ccc_queries[0].query == "auth functions"
    
    def test_initialization_with_explicit_ccc_queries(self):
        """Test initialization with explicit CCC queries doesn't re-parse"""
        query = CCCQuery(
            query="explicit query",
            start_pos=0,
            end_pos=10,
            tag_full_match="<ccc>explicit query</ccc>"
        )
        
        request = HookRequest(
            hook_type=HookType.USER_PROMPT_SUBMIT,
            prompt="Different prompt",
            ccc_queries=[query]
        )
        
        assert len(request.ccc_queries) == 1
        assert request.ccc_queries[0].query == "explicit query"
    
    def test_prompt_validation_strips_whitespace(self):
        """Test prompt validation strips whitespace"""
        request = HookRequest(
            hook_type=HookType.USER_PROMPT_SUBMIT,
            prompt="  test prompt  "
        )
        
        assert request.prompt == "test prompt"  # Pydantic strips whitespace
    
    def test_prompt_validation_empty_fails(self):
        """Test validation fails for empty prompt"""
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            HookRequest(
                hook_type=HookType.USER_PROMPT_SUBMIT,
                prompt=""
            )
        
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            HookRequest(
                hook_type=HookType.USER_PROMPT_SUBMIT,
                prompt="   "
            )
    
    def test_has_ccc_queries_property(self):
        """Test has_ccc_queries property"""
        # Without queries
        request = HookRequest(
            hook_type=HookType.USER_PROMPT_SUBMIT,
            prompt="Regular prompt"
        )
        assert not request.has_ccc_queries
        
        # With queries
        request_with_queries = HookRequest(
            hook_type=HookType.USER_PROMPT_SUBMIT,
            prompt="Find <ccc>functions</ccc> in code"
        )
        assert request_with_queries.has_ccc_queries
    
    def test_total_query_length_property(self):
        """Test total query length calculation"""
        request = HookRequest(
            hook_type=HookType.USER_PROMPT_SUBMIT,
            prompt="Find <ccc>auth</ccc> and <ccc>validation</ccc> code"
        )
        
        # "auth" = 4, "validation" = 10
        assert request.total_query_length == 14
    
    def test_get_prompt_without_ccc_tags(self):
        """Test getting prompt with CCC tags removed"""
        request = HookRequest(
            hook_type=HookType.USER_PROMPT_SUBMIT,
            prompt="Find <ccc>auth functions</ccc> and <ccc>validation</ccc> in code"
        )
        
        clean_prompt = request.get_prompt_without_ccc_tags()
        
        assert clean_prompt == "Find  and  in code"
    
    def test_get_prompt_without_ccc_tags_no_queries(self):
        """Test getting prompt without CCC tags when none exist"""
        request = HookRequest(
            hook_type=HookType.USER_PROMPT_SUBMIT,
            prompt="Regular prompt without special tags"
        )
        
        clean_prompt = request.get_prompt_without_ccc_tags()
        
        assert clean_prompt == "Regular prompt without special tags"
    
    def test_get_all_queries_text(self):
        """Test getting all query texts"""
        request = HookRequest(
            hook_type=HookType.USER_PROMPT_SUBMIT,
            prompt="Find <ccc>auth functions</ccc> and <ccc>validation code</ccc>"
        )
        
        queries = request.get_all_queries_text()
        
        assert queries == ["auth functions", "validation code"]
    
    def test_get_all_queries_text_empty(self):
        """Test getting query texts when no queries exist"""
        request = HookRequest(
            hook_type=HookType.USER_PROMPT_SUBMIT,
            prompt="Regular prompt"
        )
        
        queries = request.get_all_queries_text()
        
        assert queries == []
    
    def test_full_initialization(self):
        """Test full initialization with all fields"""
        metadata = {"key": "value"}
        request = HookRequest(
            hook_type=HookType.USER_PROMPT_SUBMIT,
            prompt="Test prompt",
            working_directory="/path/to/project",
            project_name="test-project",
            user_id="user123",
            session_id="session456",
            metadata=metadata
        )
        
        assert request.working_directory == "/path/to/project"
        assert request.project_name == "test-project"
        assert request.user_id == "user123"
        assert request.session_id == "session456"
        assert request.metadata == metadata


class TestContextResult:
    """Test ContextResult functionality"""
    
    def test_basic_initialization(self):
        """Test basic context result creation"""
        result = ContextResult(
            entity_id="func_123",
            entity_type="function",
            file_path="/path/to/file.py",
            source_code="def test(): pass",
            start_line=10,
            end_line=12,
            score=0.85,
            rank=1,
            matched_query="test function",
            search_type="semantic"
        )
        
        assert result.entity_id == "func_123"
        assert result.entity_type == "function"
        assert result.file_path == "/path/to/file.py"
        assert result.source_code == "def test(): pass"
        assert result.start_line == 10
        assert result.end_line == 12
        assert result.score == 0.85
        assert result.rank == 1
        assert result.matched_query == "test function"
        assert result.search_type == "semantic"
    
    def test_score_validation_and_rounding(self):
        """Test score validation and rounding"""
        result = ContextResult(
            entity_id="test",
            entity_type="function",
            file_path="/test.py",
            source_code="code",
            start_line=1,
            end_line=2,
            score=0.123456789,
            rank=1,
            matched_query="query",
            search_type="semantic"
        )
        
        assert result.score == 0.1235  # Rounded to 4 decimal places
    
    def test_score_validation_bounds(self):
        """Test score validation enforces bounds"""
        # Valid scores
        ContextResult(
            entity_id="test", entity_type="function", file_path="/test.py",
            source_code="code", start_line=1, end_line=2, score=0.0,
            rank=1, matched_query="query", search_type="semantic"
        )
        
        ContextResult(
            entity_id="test", entity_type="function", file_path="/test.py", 
            source_code="code", start_line=1, end_line=2, score=1.0,
            rank=1, matched_query="query", search_type="semantic"
        )
        
        # Invalid scores should be handled by Pydantic validation
        with pytest.raises(ValueError):
            ContextResult(
                entity_id="test", entity_type="function", file_path="/test.py",
                source_code="code", start_line=1, end_line=2, score=-0.1,
                rank=1, matched_query="query", search_type="semantic"
            )
        
        with pytest.raises(ValueError):
            ContextResult(
                entity_id="test", entity_type="function", file_path="/test.py",
                source_code="code", start_line=1, end_line=2, score=1.1,
                rank=1, matched_query="query", search_type="semantic"
            )
    
    def test_rank_validation(self):
        """Test rank validation"""
        # Valid rank
        ContextResult(
            entity_id="test", entity_type="function", file_path="/test.py",
            source_code="code", start_line=1, end_line=2, score=0.5,
            rank=1, matched_query="query", search_type="semantic"
        )
        
        # Invalid rank
        with pytest.raises(ValueError):
            ContextResult(
                entity_id="test", entity_type="function", file_path="/test.py",
                source_code="code", start_line=1, end_line=2, score=0.5,
                rank=0, matched_query="query", search_type="semantic"
            )
    
    def test_location_reference_property(self):
        """Test location reference property"""
        result = ContextResult(
            entity_id="test", entity_type="function", file_path="/path/to/file.py",
            source_code="code", start_line=42, end_line=45, score=0.5,
            rank=1, matched_query="query", search_type="semantic"
        )
        
        assert result.location_reference == "/path/to/file.py:42"
    
    def test_is_highly_relevant_property(self):
        """Test highly relevant property"""
        # Highly relevant
        high_result = ContextResult(
            entity_id="test", entity_type="function", file_path="/test.py",
            source_code="code", start_line=1, end_line=2, score=0.85,
            rank=1, matched_query="query", search_type="semantic"
        )
        assert high_result.is_highly_relevant
        
        # Not highly relevant
        low_result = ContextResult(
            entity_id="test", entity_type="function", file_path="/test.py",
            source_code="code", start_line=1, end_line=2, score=0.75,
            rank=1, matched_query="query", search_type="semantic"
        )
        assert not low_result.is_highly_relevant
    
    def test_format_for_injection_basic(self):
        """Test basic context injection formatting"""
        result = ContextResult(
            entity_id="test_function",
            entity_type="function",
            file_path="/src/utils.py",
            source_code="def test_function():\n    return True",
            start_line=10,
            end_line=12,
            score=0.85,
            rank=1,
            matched_query="test function",
            search_type="semantic"
        )
        
        formatted = result.format_for_injection()
        
        assert "# Function: test_function" in formatted
        assert "# File: /src/utils.py:10" in formatted
        assert "# Relevance: 0.850" in formatted
        assert "def test_function():" in formatted
        assert "```python" in formatted
    
    def test_format_for_injection_with_signature_and_docstring(self):
        """Test formatting with signature and docstring"""
        result = ContextResult(
            entity_id="test_func",
            entity_type="function",
            file_path="/test.py",
            source_code="def test():\n    pass",
            signature="def test() -> bool",
            docstring="Test function docstring",
            start_line=1,
            end_line=2,
            score=0.9,
            rank=1,
            matched_query="query",
            search_type="semantic"
        )
        
        formatted = result.format_for_injection()
        
        assert "def test() -> bool" in formatted
        assert "Test function docstring" in formatted
    
    def test_format_for_injection_without_metadata(self):
        """Test formatting without metadata"""
        result = ContextResult(
            entity_id="test", entity_type="function", file_path="/test.py",
            source_code="def test(): pass", start_line=1, end_line=2,
            score=0.5, rank=1, matched_query="query", search_type="semantic"
        )
        
        formatted = result.format_for_injection(include_metadata=False)
        
        assert "# Function:" not in formatted
        assert "# File:" not in formatted
        assert "# Relevance:" not in formatted
        assert "def test(): pass" in formatted
    
    def test_frozen_model(self):
        """Test that ContextResult is frozen (immutable)"""
        result = ContextResult(
            entity_id="test", entity_type="function", file_path="/test.py",
            source_code="code", start_line=1, end_line=2, score=0.5,
            rank=1, matched_query="query", search_type="semantic"
        )
        
        with pytest.raises(ValidationError):
            result.score = 0.8


class TestHookResponse:
    """Test HookResponse functionality"""
    
    def test_basic_initialization(self):
        """Test basic hook response creation"""
        response = HookResponse(
            success=True,
            processing_time_ms=150.5,
            enhanced_prompt="Enhanced prompt text"
        )
        
        assert response.success is True
        assert response.processing_time_ms == 150.5
        assert response.enhanced_prompt == "Enhanced prompt text"
        assert response.context_results == []
        assert response.queries_processed == 0
        assert response.total_results_found == 0
        assert response.error is None
        assert response.warnings == []
        assert isinstance(response.timestamp, datetime)
    
    def test_enhanced_prompt_validation_empty_fails(self):
        """Test validation fails for empty enhanced prompt"""
        with pytest.raises(ValueError, match="Enhanced prompt cannot be empty"):
            HookResponse(
                success=True,
                processing_time_ms=100.0,
                enhanced_prompt=""
            )
        
        with pytest.raises(ValueError, match="Enhanced prompt cannot be empty"):
            HookResponse(
                success=True,
                processing_time_ms=100.0,
                enhanced_prompt="   "
            )
    
    def test_has_context_property(self):
        """Test has_context property"""
        # Without context
        response = HookResponse(
            success=True,
            processing_time_ms=100.0,
            enhanced_prompt="test"
        )
        assert not response.has_context
        
        # With context
        context_result = ContextResult(
            entity_id="test", entity_type="function", file_path="/test.py",
            source_code="code", start_line=1, end_line=2, score=0.5,
            rank=1, matched_query="query", search_type="semantic"
        )
        response_with_context = HookResponse(
            success=True,
            processing_time_ms=100.0,
            enhanced_prompt="test",
            context_results=[context_result]
        )
        assert response_with_context.has_context
    
    def test_average_relevance_score_property(self):
        """Test average relevance score calculation"""
        # No results
        response = HookResponse(
            success=True,
            processing_time_ms=100.0,
            enhanced_prompt="test"
        )
        assert response.average_relevance_score == 0.0
        
        # With results
        results = [
            ContextResult(
                entity_id="test1", entity_type="function", file_path="/test.py",
                source_code="code", start_line=1, end_line=2, score=0.8,
                rank=1, matched_query="query", search_type="semantic"
            ),
            ContextResult(
                entity_id="test2", entity_type="function", file_path="/test.py",
                source_code="code", start_line=3, end_line=4, score=0.6,
                rank=2, matched_query="query", search_type="semantic"
            )
        ]
        
        response_with_results = HookResponse(
            success=True,
            processing_time_ms=100.0,
            enhanced_prompt="test",
            context_results=results
        )
        
        assert response_with_results.average_relevance_score == 0.7  # (0.8 + 0.6) / 2
    
    def test_highly_relevant_count_property(self):
        """Test highly relevant count calculation"""
        results = [
            ContextResult(
                entity_id="test1", entity_type="function", file_path="/test.py",
                source_code="code", start_line=1, end_line=2, score=0.85,  # Highly relevant
                rank=1, matched_query="query", search_type="semantic"
            ),
            ContextResult(
                entity_id="test2", entity_type="function", file_path="/test.py",
                source_code="code", start_line=3, end_line=4, score=0.75,  # Not highly relevant
                rank=2, matched_query="query", search_type="semantic"
            ),
            ContextResult(
                entity_id="test3", entity_type="function", file_path="/test.py",
                source_code="code", start_line=5, end_line=6, score=0.9,   # Highly relevant
                rank=3, matched_query="query", search_type="semantic"
            )
        ]
        
        response = HookResponse(
            success=True,
            processing_time_ms=100.0,
            enhanced_prompt="test",
            context_results=results
        )
        
        assert response.highly_relevant_count == 2
    
    def test_get_context_summary_no_context(self):
        """Test context summary with no context"""
        response = HookResponse(
            success=True,
            processing_time_ms=100.0,
            enhanced_prompt="test"
        )
        
        summary = response.get_context_summary()
        
        assert summary == "No context found"
    
    def test_get_context_summary_with_context(self):
        """Test context summary with mixed context types"""
        results = [
            ContextResult(
                entity_id="func1", entity_type="function", file_path="/test.py",
                source_code="code", start_line=1, end_line=2, score=0.8,
                rank=1, matched_query="query", search_type="semantic"
            ),
            ContextResult(
                entity_id="class1", entity_type="class", file_path="/test.py",
                source_code="code", start_line=3, end_line=4, score=0.6,
                rank=2, matched_query="query", search_type="semantic"
            ),
            ContextResult(
                entity_id="func2", entity_type="function", file_path="/test.py",
                source_code="code", start_line=5, end_line=6, score=0.7,
                rank=3, matched_query="query", search_type="semantic"
            )
        ]
        
        response = HookResponse(
            success=True,
            processing_time_ms=100.0,
            enhanced_prompt="test",
            context_results=results
        )
        
        summary = response.get_context_summary()
        
        assert "Found 3 results" in summary
        assert "2 function" in summary
        assert "1 class" in summary
        assert "avg relevance 0.700" in summary
    
    def test_success_response_factory(self):
        """Test success response factory method"""
        context_results = [
            ContextResult(
                entity_id="test", entity_type="function", file_path="/test.py",
                source_code="code", start_line=1, end_line=2, score=0.8,
                rank=1, matched_query="query", search_type="semantic"
            )
        ]
        
        search_stats = {"search_time": 50.0}
        
        response = HookResponse.success_response(
            enhanced_prompt="Enhanced prompt",
            context_results=context_results,
            processing_time_ms=120.0,
            queries_processed=2,
            search_stats=search_stats
        )
        
        assert response.success is True
        assert response.enhanced_prompt == "Enhanced prompt"
        assert response.context_results == context_results
        assert response.processing_time_ms == 120.0
        assert response.queries_processed == 2
        assert response.total_results_found == 1
        assert response.search_stats == search_stats
        assert response.error is None
    
    def test_error_response_factory(self):
        """Test error response factory method"""
        warnings = ["Warning message"]
        
        response = HookResponse.error_response(
            original_prompt="Original prompt",
            error="Something went wrong",
            processing_time_ms=50.0,
            warnings=warnings
        )
        
        assert response.success is False
        assert response.enhanced_prompt == "Original prompt"
        assert response.error == "Something went wrong"
        assert response.processing_time_ms == 50.0
        assert response.warnings == warnings
        assert response.context_results == []
    
    def test_error_response_factory_no_warnings(self):
        """Test error response factory without warnings"""
        response = HookResponse.error_response(
            original_prompt="Original prompt",
            error="Error occurred",
            processing_time_ms=30.0
        )
        
        assert response.warnings == []


class TestHookExecutionContext:
    """Test HookExecutionContext functionality"""
    
    def test_basic_initialization(self):
        """Test basic execution context creation"""
        context = HookExecutionContext(
            project_name="test-project",
            project_path="/path/to/project",
            collection_prefix="test-project"
        )
        
        assert context.project_name == "test-project"
        assert context.project_path == "/path/to/project"
        assert context.collection_prefix == "test-project"
        assert context.qdrant_url == "http://localhost:6333"
        assert context.qdrant_timeout == 10.0
        assert context.max_results_per_query == 5
        assert context.min_relevance_score == 0.3
        assert context.enable_hybrid_search is True
        assert context.max_processing_time_ms == 5000
    
    def test_project_name_validation_success(self):
        """Test project name validation with valid names"""
        valid_names = ["test", "test-project", "test_project", "test123", "project-name-123"]
        
        for name in valid_names:
            context = HookExecutionContext(
                project_name=name,
                project_path="/path",
                collection_prefix="prefix"
            )
            assert context.project_name == name.lower()
    
    def test_project_name_validation_failure(self):
        """Test project name validation with invalid names"""
        invalid_names = ["", "test@project", "test project", "test.project", "test/project"]
        
        for name in invalid_names:
            with pytest.raises(ValueError, match="Project name must be alphanumeric"):
                HookExecutionContext(
                    project_name=name,
                    project_path="/path",
                    collection_prefix="prefix"
                )
    
    def test_field_validation_constraints(self):
        """Test field validation constraints"""
        # Valid values
        context = HookExecutionContext(
            project_name="test",
            project_path="/path",
            collection_prefix="prefix",
            max_results_per_query=1,
            min_relevance_score=0.0,
            max_processing_time_ms=100
        )
        assert context.max_results_per_query == 1
        assert context.min_relevance_score == 0.0
        assert context.max_processing_time_ms == 100
        
        # Test upper bounds
        context_upper = HookExecutionContext(
            project_name="test",
            project_path="/path", 
            collection_prefix="prefix",
            max_results_per_query=20,
            min_relevance_score=1.0,
            max_processing_time_ms=30000
        )
        assert context_upper.max_results_per_query == 20
        assert context_upper.min_relevance_score == 1.0
        assert context_upper.max_processing_time_ms == 30000
        
        # Invalid values should raise validation errors
        with pytest.raises(ValueError):
            HookExecutionContext(
                project_name="test",
                project_path="/path",
                collection_prefix="prefix",
                max_results_per_query=0  # Below minimum
            )
        
        with pytest.raises(ValueError):
            HookExecutionContext(
                project_name="test",
                project_path="/path",
                collection_prefix="prefix",
                max_results_per_query=21  # Above maximum
            )
    
    def test_get_collection_names(self):
        """Test getting collection names for project"""
        context = HookExecutionContext(
            project_name="test-project",
            project_path="/path",
            collection_prefix="my-prefix"
        )
        
        collections = context.get_collection_names()
        
        expected = {
            'code': 'my-prefix-code',
            'relations': 'my-prefix-relations',
            'embeddings': 'my-prefix-embeddings'
        }
        
        assert collections == expected
    
    @patch('requests.get')
    def test_is_qdrant_available_success(self, mock_get):
        """Test Qdrant availability check success"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        context = HookExecutionContext(
            project_name="test",
            project_path="/path",
            collection_prefix="prefix",
            qdrant_url="http://localhost:6333"
        )
        
        assert context.is_qdrant_available is True
        mock_get.assert_called_once_with("http://localhost:6333/health", timeout=2)
    
    @patch('requests.get')
    def test_is_qdrant_available_failure(self, mock_get):
        """Test Qdrant availability check failure"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response
        
        context = HookExecutionContext(
            project_name="test",
            project_path="/path", 
            collection_prefix="prefix"
        )
        
        assert context.is_qdrant_available is False
    
    @patch('requests.get')
    def test_is_qdrant_available_exception(self, mock_get):
        """Test Qdrant availability check with exception"""
        mock_get.side_effect = Exception("Connection failed")
        
        context = HookExecutionContext(
            project_name="test",
            project_path="/path",
            collection_prefix="prefix"
        )
        
        assert context.is_qdrant_available is False


class TestIntegrationScenarios:
    """Integration tests for hooks models working together"""
    
    def test_complete_hook_processing_workflow(self):
        """Test complete hook processing workflow"""
        # Create hook request with CCC queries
        request = HookRequest(
            hook_type=HookType.USER_PROMPT_SUBMIT,
            prompt="Find <ccc>authentication functions</ccc> and <ccc>validation code</ccc>",
            project_name="test-project",
            working_directory="/project"
        )
        
        # Verify request parsing
        assert request.has_ccc_queries
        assert len(request.ccc_queries) == 2
        assert request.get_all_queries_text() == ["authentication functions", "validation code"]
        
        # Create mock context results
        context_results = [
            ContextResult(
                entity_id="auth_func",
                entity_type="function",
                file_path="/src/auth.py",
                source_code="def authenticate(user): pass",
                start_line=10,
                end_line=12,
                score=0.9,
                rank=1,
                matched_query="authentication functions",
                search_type="semantic"
            ),
            ContextResult(
                entity_id="validate_func", 
                entity_type="function",
                file_path="/src/validation.py",
                source_code="def validate_input(data): pass",
                start_line=5,
                end_line=7,
                score=0.85,
                rank=2,
                matched_query="validation code",
                search_type="hybrid"
            )
        ]
        
        # Create successful response
        clean_prompt = request.get_prompt_without_ccc_tags()
        enhanced_prompt = f"{clean_prompt}\n\n## Context\n{context_results[0].format_for_injection()}"
        
        response = HookResponse.success_response(
            enhanced_prompt=enhanced_prompt,
            context_results=context_results,
            processing_time_ms=250.0,
            queries_processed=2,
            search_stats={"total_search_time": 200.0}
        )
        
        # Verify response
        assert response.success
        assert response.has_context
        assert response.queries_processed == 2
        assert response.total_results_found == 2
        assert response.average_relevance_score == 0.875  # (0.9 + 0.85) / 2
        assert response.highly_relevant_count == 2
        
        # Verify context summary
        summary = response.get_context_summary()
        assert "Found 2 results" in summary
        assert "2 function" in summary
    
    def test_error_handling_workflow(self):
        """Test error handling in hook processing"""
        # Create request with invalid prompt
        try:
            request = HookRequest(
                hook_type=HookType.USER_PROMPT_SUBMIT,
                prompt=""  # Invalid empty prompt
            )
            assert False, "Should have raised validation error"
        except ValueError:
            pass
        
        # Create valid request but simulate processing error
        request = HookRequest(
            hook_type=HookType.USER_PROMPT_SUBMIT,
            prompt="Find <ccc>functions</ccc> in code"
        )
        
        # Create error response
        error_response = HookResponse.error_response(
            original_prompt=request.prompt,
            error="Qdrant connection failed",
            processing_time_ms=50.0,
            warnings=["Falling back to cache", "Search results may be stale"]
        )
        
        # Verify error response
        assert not error_response.success
        assert error_response.error == "Qdrant connection failed"
        assert error_response.enhanced_prompt == request.prompt  # Original prompt returned
        assert len(error_response.warnings) == 2
        assert not error_response.has_context
    
    def test_execution_context_integration(self):
        """Test execution context integration with other models"""
        # Create execution context
        context = HookExecutionContext(
            project_name="integration-test",
            project_path="/project/path",
            collection_prefix="integration-test"
        )
        
        # Verify collection naming
        collections = context.get_collection_names()
        assert collections["code"] == "integration-test-code"
        
        # Create request that would use this context
        request = HookRequest(
            hook_type=HookType.USER_PROMPT_SUBMIT,
            prompt="Find <ccc>test functions</ccc>",
            project_name=context.project_name,
            working_directory=context.project_path
        )
        
        # Verify project name matches
        assert request.project_name == context.project_name
        
        # Create context result that references the project
        result = ContextResult(
            entity_id="test_func",
            entity_type="function", 
            file_path=f"{context.project_path}/tests/test_utils.py",
            source_code="def test_function(): pass",
            start_line=1,
            end_line=2,
            score=0.8,
            rank=1,
            matched_query="test functions",
            search_type="semantic"
        )
        
        # Verify file path is under project path
        assert result.file_path.startswith(context.project_path)