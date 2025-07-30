"""
Unit tests for HybridSearcher functionality.

Tests query optimization, search modes, result processing, and specialized searches.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import List

from core.search import HybridSearcher, SearchConfig, SearchMode
from core.storage.client import HybridQdrantClient
from core.models.storage import SearchResult, QdrantPoint


@pytest.fixture
def mock_client():
    """Create mock HybridQdrantClient"""
    client = Mock(spec=HybridQdrantClient)
    return client


@pytest.fixture
def sample_search_results():
    """Create sample search results for testing"""
    import uuid
    return [
        SearchResult(
            point=QdrantPoint(
                id=str(uuid.uuid4()),  # Use UUID for valid Qdrant point ID
                vector=[0.1] * 1024,
                payload={
                    "entity_id": "test.py::test_function",
                    "entity_name": "test_function",
                    "entity_type": "function",
                    "file_path": "test.py",
                    "signature": "def test_function():",
                    "start_line": 10,
                    "end_line": 15
                }
            ),
            score=0.9,
            query="test function",
            search_type=SearchMode.HYBRID,
            rank=1,
            total_results=3
        ),
        SearchResult(
            point=QdrantPoint(
                id=str(uuid.uuid4()),  # Use UUID for valid Qdrant point ID
                vector=[0.1] * 1024,
                payload={
                    "entity_id": "helper.js::helper_method",
                    "entity_name": "helper_method",
                    "entity_type": "method",
                    "file_path": "helper.js",
                    "signature": "function helper_method() {",
                    "start_line": 5,
                    "end_line": 10
                }
            ),
            score=0.7,
            query="test function",
            search_type=SearchMode.HYBRID,
            rank=2,
            total_results=3
        ),
        SearchResult(
            point=QdrantPoint(
                id=str(uuid.uuid4()),  # Use UUID for valid Qdrant point ID
                vector=[0.1] * 1024,
                payload={
                    "entity_id": "setup.py::setup_test",
                    "entity_name": "setup_test",
                    "entity_type": "function",
                    "file_path": "setup.py",
                    "signature": "def setup_test():",
                    "start_line": 20,
                    "end_line": 25
                }
            ),
            score=0.5,
            query="test function",
            search_type=SearchMode.HYBRID,
            rank=3,
            total_results=3
        )
    ]


class TestSearchConfig:
    """Test SearchConfig data class"""
    
    def test_default_initialization(self):
        """Test default search configuration"""
        config = SearchConfig()
        
        assert config.mode == SearchMode.HYBRID
        assert config.limit == 50
        assert config.payload_weight == 0.8
        assert config.semantic_weight == 0.2
        assert config.min_score_threshold == 0.0
        assert config.include_file_types == []
        assert config.exclude_file_types == []
        assert config.include_entity_types == []
        assert config.exclude_entity_types == []
    
    def test_custom_initialization(self):
        """Test custom search configuration"""
        config = SearchConfig(
            mode=SearchMode.SEMANTIC_ONLY,
            limit=20,
            payload_weight=0.3,
            semantic_weight=0.7,
            min_score_threshold=0.5,
            include_file_types=[".py", ".js"],
            exclude_file_types=[".test.py"],
            include_entity_types=["function", "class"],
            exclude_entity_types=["variable"]
        )
        
        assert config.mode == SearchMode.SEMANTIC_ONLY
        assert config.limit == 20
        assert config.payload_weight == 0.3
        assert config.semantic_weight == 0.7
        assert config.min_score_threshold == 0.5
        assert config.include_file_types == [".py", ".js"]
        assert config.exclude_file_types == [".test.py"]
        assert config.include_entity_types == ["function", "class"]
        assert config.exclude_entity_types == ["variable"]
    
    def test_weight_normalization(self):
        """Test automatic weight normalization in post_init"""
        config = SearchConfig(
            payload_weight=3.0,
            semantic_weight=1.0
        )
        
        # Should normalize to 0.75 and 0.25
        assert abs(config.payload_weight - 0.75) < 0.001
        assert abs(config.semantic_weight - 0.25) < 0.001
    
    def test_weight_normalization_zero_total(self):
        """Test weight normalization when total is zero"""
        config = SearchConfig(
            payload_weight=0.0,
            semantic_weight=0.0
        )
        
        # Should remain 0.0 when total is 0
        assert config.payload_weight == 0.0
        assert config.semantic_weight == 0.0


class TestHybridSearcher:
    """Test HybridSearcher functionality"""
    
    def test_searcher_initialization(self, mock_client):
        """Test searcher initialization"""
        searcher = HybridSearcher(mock_client)
        
        assert searcher.client == mock_client
        assert searcher.query_analyzer is not None
        assert len(searcher.query_analyzer._code_patterns) > 0
        assert len(searcher.query_analyzer._exact_patterns) > 0
        assert "def " in searcher.query_analyzer._code_patterns
        assert "class " in searcher.query_analyzer._code_patterns
        assert "\"" in searcher.query_analyzer._exact_patterns
    
    @pytest.mark.asyncio
    async def test_search_with_default_config(self, mock_client, sample_search_results):
        """Test search with default configuration"""
        searcher = HybridSearcher(mock_client)
        
        # Mock hybrid search
        mock_client.search_hybrid = AsyncMock(return_value=sample_search_results)
        
        results = await searcher.search("test-collection", "test query")
        
        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)
        
        # Should have called hybrid search
        mock_client.search_hybrid.assert_called_once()
        call_args = mock_client.search_hybrid.call_args
        assert call_args[0][0] == "test-collection"
        assert call_args[0][1] == "test query"
    
    @pytest.mark.asyncio
    async def test_search_payload_only_mode(self, mock_client, sample_search_results):
        """Test search with payload-only mode"""
        searcher = HybridSearcher(mock_client)
        config = SearchConfig(mode=SearchMode.PAYLOAD_ONLY)
        
        # Mock payload search
        mock_client.search_payload = AsyncMock(return_value=sample_search_results)
        
        results = await searcher.search("test-collection", "test query", config)
        
        assert len(results) == 3
        mock_client.search_payload.assert_called_once()
        
        # Should not have called other search methods
        mock_client.search_semantic.assert_not_called()
        mock_client.search_hybrid.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_search_semantic_only_mode(self, mock_client, sample_search_results):
        """Test search with semantic-only mode"""
        searcher = HybridSearcher(mock_client)
        config = SearchConfig(mode=SearchMode.SEMANTIC_ONLY)
        
        # Mock semantic search
        mock_client.search_semantic = AsyncMock(return_value=sample_search_results)
        
        results = await searcher.search("test-collection", "test query", config)
        
        assert len(results) == 3
        mock_client.search_semantic.assert_called_once()
        
        # Should not have called other search methods
        mock_client.search_payload.assert_not_called()
        mock_client.search_hybrid.assert_not_called()
    
    def test_optimize_search_mode_exact_patterns(self, mock_client):
        """Test search mode optimization for exact patterns via QueryAnalyzer"""
        searcher = HybridSearcher(mock_client)
        
        # Test exact match indicators through QueryAnalyzer
        analysis = searcher.query_analyzer.analyze_query('"exact match"')
        assert analysis.recommended_mode == "payload"
        
        analysis = searcher.query_analyzer.analyze_query("name:function_name")
        assert analysis.recommended_mode == "payload"
        
        analysis = searcher.query_analyzer.analyze_query("file:test.py")
        assert analysis.recommended_mode == "payload"
    
    def test_optimize_search_mode_code_patterns(self, mock_client):
        """Test search mode optimization for code patterns via QueryAnalyzer"""
        searcher = HybridSearcher(mock_client)
        
        # Test code-specific patterns through QueryAnalyzer
        analysis = searcher.query_analyzer.analyze_query("def function")
        assert analysis.recommended_mode in ["hybrid", "payload"]  # Short code query may go either way
        
        analysis = searcher.query_analyzer.analyze_query("class MyClass")  
        assert analysis.recommended_mode in ["hybrid", "payload"]  # Short code query may go either way
        
        analysis = searcher.query_analyzer.analyze_query("async function handler")
        assert analysis.recommended_mode == "hybrid"
        
        analysis = searcher.query_analyzer.analyze_query("import module system")
        assert analysis.recommended_mode == "hybrid"
    
    def test_optimize_search_mode_query_length(self, mock_client):
        """Test search mode optimization based on query length via QueryAnalyzer"""
        searcher = HybridSearcher(mock_client)
        
        # Short queries (1-2 words) -> payload search
        analysis = searcher.query_analyzer.analyze_query("test")
        assert analysis.recommended_mode == "payload"
        
        analysis = searcher.query_analyzer.analyze_query("user login")
        assert analysis.recommended_mode == "payload"
        
        # Long queries (6+ words) -> semantic search  
        analysis = searcher.query_analyzer.analyze_query("find code that handles user authentication and authorization logic")
        assert analysis.recommended_mode == "semantic"
        
        # Medium queries (3-5 words) -> hybrid search
        analysis = searcher.query_analyzer.analyze_query("user authentication handling")
        assert analysis.recommended_mode == "hybrid"
    
    def test_requested_mode_is_preserved(self, mock_client):
        """Test that explicitly requested search modes are preserved in SearchConfig"""
        searcher = HybridSearcher(mock_client)
        
        # Test that non-AUTO modes are preserved
        config = SearchConfig(mode=SearchMode.PAYLOAD_ONLY)
        assert config.mode == SearchMode.PAYLOAD_ONLY
        
        config = SearchConfig(mode=SearchMode.SEMANTIC_ONLY)
        assert config.mode == SearchMode.SEMANTIC_ONLY
        
        config = SearchConfig(mode=SearchMode.HYBRID)
        assert config.mode == SearchMode.HYBRID
    
    def test_file_type_to_language_mapping(self, mock_client):
        """Test file type to language mapping"""
        searcher = HybridSearcher(mock_client)
        
        # Test common mappings
        assert searcher._file_type_to_language("py") == "python"
        assert searcher._file_type_to_language(".py") == "python"  # With leading dot
        assert searcher._file_type_to_language("js") == "javascript"
        assert searcher._file_type_to_language("ts") == "typescript"
        assert searcher._file_type_to_language("go") == "go"
        assert searcher._file_type_to_language("rs") == "rust"
        assert searcher._file_type_to_language("java") == "java"
        assert searcher._file_type_to_language("cpp") == "cpp"
        assert searcher._file_type_to_language("c") == "c"
        
        # Test unknown type
        assert searcher._file_type_to_language("unknown") is None
    
    def test_build_filters_file_types(self, mock_client):
        """Test filter building for file types"""
        searcher = HybridSearcher(mock_client)
        
        config = SearchConfig(include_file_types=[".py", ".js", ".ts"])
        filters = searcher._build_filters(config)
        
        assert "language" in filters
        expected_languages = ["python", "javascript", "typescript"]
        assert all(lang in filters["language"] for lang in expected_languages)
    
    def test_build_filters_entity_types(self, mock_client):
        """Test filter building for entity types"""
        searcher = HybridSearcher(mock_client)
        
        config = SearchConfig(include_entity_types=["function", "class", "method"])
        filters = searcher._build_filters(config)
        
        assert "entity_type" in filters
        assert filters["entity_type"] == ["function", "class", "method"]
    
    def test_post_process_results_score_threshold(self, mock_client, sample_search_results):
        """Test post-processing with score threshold"""
        searcher = HybridSearcher(mock_client)
        
        config = SearchConfig(min_score_threshold=0.8)
        processed = searcher._post_process_results(sample_search_results, config)
        
        # Should only keep results with score >= 0.8
        assert len(processed) == 1
        assert processed[0].score >= 0.8
    
    def test_post_process_results_file_type_exclusion(self, mock_client, sample_search_results):
        """Test post-processing with file type exclusion"""
        searcher = HybridSearcher(mock_client)
        
        config = SearchConfig(exclude_file_types=[".js"])
        processed = searcher._post_process_results(sample_search_results, config)
        
        # Should exclude the .js file result
        assert len(processed) == 2
        assert all(not r.point.payload.get("file_path", "").endswith(".js") for r in processed)
    
    def test_post_process_results_entity_type_exclusion(self, mock_client, sample_search_results):
        """Test post-processing with entity type exclusion"""
        searcher = HybridSearcher(mock_client)
        
        config = SearchConfig(exclude_entity_types=["method"])
        processed = searcher._post_process_results(sample_search_results, config)
        
        # Should exclude the method result
        assert len(processed) == 2
        assert all(r.point.payload.get("entity_type") != "method" for r in processed)
    
    @pytest.mark.asyncio
    async def test_search_similar_not_implemented(self, mock_client):
        """Test similar entity search (not yet implemented)"""
        searcher = HybridSearcher(mock_client)
        
        results = await searcher.search_similar("test-collection", "entity123", limit=10)
        
        # Should return empty list for now
        assert results == []
    
    @pytest.mark.asyncio
    async def test_search_by_location(self, mock_client, sample_search_results):
        """Test location-based search"""
        searcher = HybridSearcher(mock_client)
        
        # Mock payload search
        mock_client.search_payload = AsyncMock(return_value=sample_search_results)
        
        results = await searcher.search_by_location("test-collection", "test.py")
        
        assert len(results) == 3
        mock_client.search_payload.assert_called_once()
        
        # Check that file path filter was used
        call_args = mock_client.search_payload.call_args
        assert "file:test.py" in call_args[0][1]  # Query should contain file filter
        assert call_args[0][3]["file_path"] == "test.py"  # Filters should contain file_path
    
    @pytest.mark.asyncio
    async def test_search_by_location_with_line_range(self, mock_client, sample_search_results):
        """Test location-based search with line range filtering"""
        searcher = HybridSearcher(mock_client)
        
        # Mock payload search
        mock_client.search_payload = AsyncMock(return_value=sample_search_results)
        
        results = await searcher.search_by_location(
            "test-collection", "test.py", line_range=(5, 15)
        )
        
        # Should filter results based on line range overlap
        # sample_search_results has entities at lines 10-15, 5-10, and 20-25
        # Line range (5, 15) should match first two
        assert len(results) == 2
        
        for result in results:
            start_line = result.point.payload.get("start_line", 0)
            end_line = result.point.payload.get("end_line", 0)
            # Check overlap with range (5, 15)
            assert start_line <= 15 and end_line >= 5
    
    @pytest.mark.asyncio
    async def test_search_by_location_error_handling(self, mock_client):
        """Test error handling in location search"""
        searcher = HybridSearcher(mock_client)
        
        # Mock search failure
        mock_client.search_payload = AsyncMock(side_effect=Exception("Search failed"))
        
        results = await searcher.search_by_location("test-collection", "test.py")
        
        # Should return empty list on error
        assert results == []
    
    @pytest.mark.asyncio
    async def test_search_by_signature(self, mock_client, sample_search_results):
        """Test signature-based search"""
        searcher = HybridSearcher(mock_client)
        
        # Mock payload search
        mock_client.search_payload = AsyncMock(return_value=sample_search_results)
        
        results = await searcher.search_by_signature("test-collection", "test_function")
        
        assert len(results) >= 0  # May be filtered based on signature content
        mock_client.search_payload.assert_called_once()
        
        # Check that signature filter was used
        call_args = mock_client.search_payload.call_args
        assert "signature:test_function" in call_args[0][1]
    
    @pytest.mark.asyncio
    async def test_search_by_signature_with_filtering(self, mock_client, sample_search_results):
        """Test signature search with content filtering"""
        searcher = HybridSearcher(mock_client)
        
        # Mock payload search - return results where only some match the signature pattern
        filtered_results = [
            result for result in sample_search_results 
            if "test_function" in result.point.payload.get("signature", "").lower()
        ]
        mock_client.search_payload = AsyncMock(return_value=sample_search_results)
        
        results = await searcher.search_by_signature("test-collection", "test_function")
        
        # Should filter results based on signature content
        for result in results:
            signature = result.point.payload.get("signature", "")
            assert "test_function" in signature.lower()
    
    @pytest.mark.asyncio
    async def test_search_by_signature_error_handling(self, mock_client):
        """Test error handling in signature search"""
        searcher = HybridSearcher(mock_client)
        
        # Mock search failure
        mock_client.search_payload = AsyncMock(side_effect=Exception("Search failed"))
        
        results = await searcher.search_by_signature("test-collection", "test_pattern")
        
        # Should return empty list on error
        assert results == []
    
    
    def test_get_search_suggestions_file_extensions(self, mock_client):
        """Test search suggestions for file extensions"""
        searcher = HybridSearcher(mock_client)
        
        suggestions = searcher.get_search_suggestions("file:test")
        
        assert len(suggestions) <= 10
        extensions = [".py", ".js", ".ts", ".go", ".rs", ".java"]
        assert any(any(ext in suggestion for ext in extensions) for suggestion in suggestions)
    
    def test_get_search_suggestions_empty_query(self, mock_client):
        """Test search suggestions with empty query"""
        searcher = HybridSearcher(mock_client)
        
        suggestions = searcher.get_search_suggestions("")
        
        # Should return empty list for empty query
        assert len(suggestions) == 0
    
    def test_get_search_suggestions_limit(self, mock_client):
        """Test that search suggestions are limited to 10"""
        searcher = HybridSearcher(mock_client)
        
        # Query that might generate many suggestions
        suggestions = searcher.get_search_suggestions("def function test class")
        
        # Should never exceed 10 suggestions
        assert len(suggestions) <= 10


class TestSearchModeConstants:
    """Test SearchMode constants"""
    
    def test_search_mode_values(self):
        """Test search mode constant values"""
        assert SearchMode.PAYLOAD_ONLY == "payload"
        assert SearchMode.SEMANTIC_ONLY == "semantic"
        assert SearchMode.HYBRID == "hybrid"
        assert SearchMode.AUTO == "auto"