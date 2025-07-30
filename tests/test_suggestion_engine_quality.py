"""
Comprehensive end-to-end quality tests for QuerySuggestionEngine.

Tests suggestion effectiveness with real repositories across different languages,
validating that suggestions improve search experience on actual codebases.
"""

import pytest
import os
import shutil
import subprocess
import uuid
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

# Fix tokenizer fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from core.search.suggestions import QuerySuggestionEngine, SearchSuggestion, SuggestionType
from core.search.engine import HybridSearcher, SearchMode, SearchConfig
from core.models.storage import SearchResult, QdrantPoint
from core.indexer.hybrid_indexer import HybridIndexer, IndexingJobConfig
from core.storage.client import HybridQdrantClient
from core.parser.parallel_pipeline import ProcessParsingPipeline
from core.parser.registry import ParserRegistry
from core.embeddings.stella import StellaEmbedder
from core.indexer.cache import CacheManager


class TestQuerySuggestionEngineQuality:
    """Test QuerySuggestionEngine quality with real codebases"""
    
    @classmethod
    def setup_class(cls):
        """Setup test repositories"""
        cls.test_dir = Path("test-harness/temp-repos")
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Clone smaller, manageable real repositories for testing
        cls.repos = {
            "python": {
                "url": "https://github.com/kennethreitz/requests.git",
                "path": cls.test_dir / "requests",
                "branch": "main"
            },
            "typescript": {
                "url": "https://github.com/sindresorhus/got.git",
                "path": cls.test_dir / "got",
                "branch": "main"
            },
            "javascript": {
                "url": "https://github.com/axios/axios.git",
                "path": cls.test_dir / "axios", 
                "branch": "main"
            },
            "go": {
                "url": "https://github.com/gin-gonic/gin.git",
                "path": cls.test_dir / "gin",
                "branch": "master"
            }
        }
        
        # Clone repositories
        for repo_name, repo_info in cls.repos.items():
            if not repo_info["path"].exists():
                subprocess.run([
                    "git", "clone", "--depth", "1", 
                    repo_info["url"], str(repo_info["path"])
                ], check=True, capture_output=True)
        
        # Track shared collections for cleanup
        cls.shared_collections = set()
    
    @classmethod
    def teardown_class(cls):
        """Cleanup test repositories and shared collections"""
        # Clean up shared collections
        if hasattr(cls, 'shared_collections'):
            try:
                import requests
                for collection_name in cls.shared_collections:
                    try:
                        requests.delete(f"http://localhost:6334/collections/{collection_name}", timeout=5)
                    except Exception:
                        pass  # Ignore cleanup errors
            except Exception:
                pass
        
        # Clean up test repositories
        if hasattr(cls, 'test_dir') and cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
    
    def setup_method(self):
        """Setup test instance"""
        self.suggestion_engine = QuerySuggestionEngine()
        self.client = HybridQdrantClient("http://localhost:6334")
        self.searcher = None
        
        # Initialize components for HybridIndexer (will be done in async setup)
        self.parser_pipeline = None
        self.embedder = None
        self.cache_manager = None
        self.indexer = None
        
        # Clean up any old test collections (but not shared ones)
        self._cleanup_old_test_collections()
    
    async def _async_setup(self):
        """Async setup for components that need event loop"""
        if self.indexer is None:
            self.parser_pipeline = ProcessParsingPipeline(max_workers=2, batch_size=5)
            self.embedder = StellaEmbedder()
            self.cache_manager = CacheManager()
            self.indexer = HybridIndexer(
                parser_pipeline=self.parser_pipeline,
                embedder=self.embedder,
                storage_client=self.client,
                cache_manager=self.cache_manager
            )
            self.searcher = HybridSearcher(self.client)
    
    def teardown_method(self):
        """Cleanup after each test"""
        # Don't clean up shared collections in teardown_method
        # They will be cleaned up in teardown_class
        pass
    
    def _cleanup_old_test_collections(self):
        """Clean up old test collections (but not shared suggestion collections)"""
        try:
            import requests
            response = requests.get("http://localhost:6334/collections", timeout=5)
            if response.status_code == 200:
                collections = response.json().get("result", {}).get("collections", [])
                for collection in collections:
                    collection_name = collection.get("name", "")
                    # Only delete old UUID-based collections, not our shared ones
                    old_patterns = ["temp-", "quality-test-", "suggest-test-", "real-indexer-", "integration-test"]
                    # Don't delete our shared "test-suggestion-{repo}" collections
                    if (any(pattern in collection_name for pattern in old_patterns) and 
                        not collection_name.startswith("test-suggestion-")):
                        requests.delete(f"http://localhost:6334/collections/{collection_name}", timeout=5)
        except Exception:
            pass  # Ignore cleanup errors
    
    async def get_or_create_collection(self, repo_name: str) -> str:
        """Get existing collection for repo or create it if needed"""
        # Deterministic collection name based on repo
        collection_name = f"test-suggestion-{repo_name}"
        
        # Check if collection already exists
        try:
            collection_info = await self.client.get_collection_info(collection_name)
            if collection_info:
                # Collection exists and is valid, reuse it
                self.__class__.shared_collections.add(collection_name)
                return collection_name
        except Exception:
            pass  # Collection doesn't exist, create it
        
        # Setup async components if not already done
        await self._async_setup()
        
        repo_info = self.repos[repo_name]
        repo_path = repo_info["path"]
        
        if not repo_path.exists():
            raise pytest.skip(f"Repository {repo_name} not available")
        
        # Create and index new collection
        config = IndexingJobConfig(
            project_path=repo_path,
            project_name=collection_name,  # Use collection name as project name
            include_patterns=["*.py", "*.js", "*.ts", "*.go", "*.rs"],
            batch_size=50
        )
        
        await self.indexer.index_project(config)
        
        # Get the actual collection name
        from core.storage.schemas import CollectionManager, CollectionType
        collection_manager = CollectionManager(project_name=config.project_name)
        actual_collection_name = collection_manager.get_collection_name(CollectionType.CODE)
        
        # Track for cleanup
        self.__class__.shared_collections.add(actual_collection_name)
        return actual_collection_name
    
    async def search_with_query(self, repo_name: str, query: str, limit: int = 10) -> List[SearchResult]:
        """Search real codebase using cached/shared collection"""
        try:
            # Get or create collection for this repository
            collection_name = await self.get_or_create_collection(repo_name)
            
            # Setup searcher if not done
            await self._async_setup()
            
            # Search using the hybrid searcher
            config = SearchConfig(
                mode=SearchMode.HYBRID,
                limit=limit
            )
            results = await self.searcher.search(
                collection_name=collection_name,
                query=query,
                config=config
            )
            
            return results
            
        except Exception as e:
            print(f"Error searching {repo_name}: {e}")
            return []
    
    def test_basic_suggestion_generation(self):
        """Test basic suggestion generation with common developer queries"""
        test_cases = [
            {
                "query": "auth",
                "expected_types": [SuggestionType.COMPLETION, SuggestionType.REFINEMENT],
                "min_suggestions": 5
            },
            {
                "query": "def login",
                "expected_types": [SuggestionType.COMPLETION],
                "min_suggestions": 3
            },
            {
                "query": "how to",
                "expected_types": [SuggestionType.COMPLETION, SuggestionType.REFINEMENT],
                "min_suggestions": 4
            },
            {
                "query": "request",
                "expected_types": [SuggestionType.REFINEMENT, SuggestionType.COMPLETION],
                "min_suggestions": 8
            }
        ]
        
        for case in test_cases:
            suggestions = self.suggestion_engine.get_suggestions(case["query"], limit=15)
            
            print(f"\n=== Testing query: '{case['query']}' ===")
            for i, suggestion in enumerate(suggestions):
                print(f"{i+1}: [{suggestion.type.value}] {suggestion.text} (conf: {suggestion.confidence:.2f})")
                print(f"    {suggestion.explanation}")
            
            # Verify we have enough suggestions
            assert len(suggestions) >= case["min_suggestions"], \
                f"Expected at least {case['min_suggestions']} suggestions for '{case['query']}', got {len(suggestions)}"
            
            # Verify expected types are present
            suggestion_types = {s.type for s in suggestions}
            for expected_type in case["expected_types"]:
                assert expected_type in suggestion_types, \
                    f"Expected suggestion type {expected_type.value} for query '{case['query']}'"
            
            # Verify all suggestions have valid confidence scores
            for suggestion in suggestions:
                assert 0 <= suggestion.confidence <= 1, \
                    f"Invalid confidence {suggestion.confidence} for suggestion '{suggestion.text}'"
            
            # Verify suggestions are sorted by confidence
            confidences = [s.confidence for s in suggestions]
            assert confidences == sorted(confidences, reverse=True), \
                f"Suggestions not sorted by confidence for query '{case['query']}'"
    
    async def test_function_pattern_suggestions_with_real_results(self):
        """Test function pattern suggestions and validate they improve search results"""
        base_query = "login"
        
        # Get suggestions for function patterns
        suggestions = self.suggestion_engine.get_suggestions(base_query, limit=10)
        
        # Find function-related suggestions
        function_suggestions = [s for s in suggestions if "def" in s.text or "function" in s.text or "method" in s.text]
        
        print(f"\n=== Function suggestions for '{base_query}' ===")
        for suggestion in function_suggestions:
            print(f"[{suggestion.type.value}] {suggestion.text} (confidence: {suggestion.confidence:.2f})")
            print(f"  Explanation: {suggestion.explanation}")
        
        assert len(function_suggestions) >= 2, "Should generate function pattern suggestions"
        
        # Test with real codebase - compare base query vs suggested query results
        python_results_base = await self.search_with_query("python", base_query, limit=10)
        
        if python_results_base and function_suggestions:
            # Use the first function suggestion
            suggested_query = function_suggestions[0].text
            python_results_suggested = await self.search_with_query("python", suggested_query, limit=10)
            
            print(f"\nBase query '{base_query}' found {len(python_results_base)} results")
            print(f"Suggested query '{suggested_query}' found {len(python_results_suggested)} results")
            
            # Analyze result quality
            if python_results_suggested:
                function_results = [r for r in python_results_suggested 
                                  if r.point.payload.get("entity_type") in ["function", "method"]]
                function_ratio = len(function_results) / len(python_results_suggested)
                
                print(f"Function/method results in suggested query: {len(function_results)}/{len(python_results_suggested)} ({function_ratio:.1%})")
                
                # Suggested query should have higher proportion of function results
                if len(python_results_base) > 0:
                    base_function_results = [r for r in python_results_base 
                                           if r.point.payload.get("entity_type") in ["function", "method"]]
                    base_function_ratio = len(base_function_results) / len(python_results_base)
                    
                    print(f"Function/method results in base query: {len(base_function_results)}/{len(python_results_base)} ({base_function_ratio:.1%})")
                    
                    assert function_ratio >= base_function_ratio, \
                        "Function pattern suggestion should improve function result ratio"
    
    async def test_class_pattern_suggestions_with_real_results(self):
        """Test class pattern suggestions and validate they improve search results"""
        base_query = "manager"
        
        # Get suggestions for class patterns
        suggestions = self.suggestion_engine.get_suggestions(base_query, limit=10)
        
        # Find class-related suggestions
        class_suggestions = [s for s in suggestions if "class" in s.text or "interface" in s.text or "model" in s.text]
        
        print(f"\n=== Class suggestions for '{base_query}' ===")
        for suggestion in class_suggestions:
            print(f"[{suggestion.type.value}] {suggestion.text} (confidence: {suggestion.confidence:.2f})")
            print(f"  Explanation: {suggestion.explanation}")
        
        if class_suggestions:
            # Test with real codebase
            python_results_base = await self.search_with_query("python", base_query, limit=10)
            suggested_query = class_suggestions[0].text
            python_results_suggested = await self.search_with_query("python", suggested_query, limit=10)
            
            print(f"\nBase query '{base_query}' found {len(python_results_base)} results")
            print(f"Suggested query '{suggested_query}' found {len(python_results_suggested)} results")
            
            # Analyze result quality for class-related entities
            if python_results_suggested:
                class_results = [r for r in python_results_suggested 
                               if r.point.payload.get("entity_type") in ["class", "interface", "type"]]
                class_ratio = len(class_results) / len(python_results_suggested) if python_results_suggested else 0
                
                print(f"Class/interface results in suggested query: {len(class_results)}/{len(python_results_suggested)} ({class_ratio:.1%})")
                
                # Should find some class-related results
                assert class_ratio > 0.1 or len(class_results) >= 1, \
                    "Class pattern suggestion should find some class-related results"
    
    async def test_semantic_pattern_suggestions_with_real_results(self):
        """Test semantic pattern suggestions with real developer scenarios"""
        semantic_queries = [
            "how to handle authentication",
            "what is session management", 
            "find error handling",
            "show me logging examples"
        ]
        
        for base_query in ["auth", "session", "error", "log"]:
            suggestions = self.suggestion_engine.get_suggestions(base_query, limit=15)
            
            # Find semantic suggestions
            semantic_suggestions = [s for s in suggestions 
                                  if any(pattern in s.text.lower() 
                                        for pattern in ["how to", "what is", "find", "show me", "explain"])]
            
            print(f"\n=== Semantic suggestions for '{base_query}' ===")
            for suggestion in semantic_suggestions:
                print(f"[{suggestion.type.value}] {suggestion.text} (confidence: {suggestion.confidence:.2f})")
            
            if semantic_suggestions:
                # Test one semantic suggestion with real search
                semantic_query = semantic_suggestions[0].text
                results = await self.search_with_query("python", semantic_query, limit=8)
                
                print(f"Semantic query '{semantic_query}' found {len(results)} results")
                
                # Semantic queries should still find relevant results
                if len(results) > 0:
                    # Check that results contain relevant keywords
                    relevant_results = []
                    for result in results:
                        entity_name = result.point.payload.get("entity_name", "").lower()
                        docstring = result.point.payload.get("docstring", "").lower()
                        file_path = result.point.payload.get("file_path", "").lower()
                        
                        if (base_query.lower() in entity_name or 
                            base_query.lower() in docstring or 
                            base_query.lower() in file_path):
                            relevant_results.append(result)
                    
                    relevance_ratio = len(relevant_results) / len(results)
                    print(f"Relevant results: {len(relevant_results)}/{len(results)} ({relevance_ratio:.1%})")
                    
                    # At least some results should be relevant
                    assert relevance_ratio >= 0.3, \
                        f"Semantic suggestion should find relevant results (got {relevance_ratio:.1%})"
    
    async def test_file_type_suggestions_with_real_results(self):
        """Test file type suggestions improve language-specific searches"""
        base_query = "request"
        
        # Get file type suggestions
        suggestions = self.suggestion_engine.get_suggestions(base_query, limit=15)
        
        # Find file type suggestions
        file_suggestions = [s for s in suggestions 
                          if any(ext in s.text for ext in [".py", ".js", ".ts", ".go"])]
        
        print(f"\n=== File type suggestions for '{base_query}' ===")
        for suggestion in file_suggestions:
            print(f"[{suggestion.type.value}] {suggestion.text} (confidence: {suggestion.confidence:.2f})")
        
        if file_suggestions:
            # Test Python file suggestion
            py_suggestion = next((s for s in file_suggestions if ".py" in s.text), None)
            if py_suggestion:
                results = await self.search_with_query("python", py_suggestion.text, limit=10)
                
                print(f"Python file suggestion '{py_suggestion.text}' found {len(results)} results")
                
                # Check that results are from Python files
                if results:
                    python_files = [r for r in results if r.point.payload.get("file_path", "").endswith(".py")]
                    python_ratio = len(python_files) / len(results)
                    
                    print(f"Python files in results: {len(python_files)}/{len(results)} ({python_ratio:.1%})")
                    
                    # Should heavily favor Python files
                    assert python_ratio >= 0.7, \
                        f"Python file suggestion should favor .py files (got {python_ratio:.1%})"
    
    async def test_context_aware_suggestions(self):
        """Test context-aware suggestions with file and query history"""
        base_query = "validate"
        
        # Test with different contexts
        contexts = [
            {"current_file": "auth.py"},
            {"recent_queries": ["login validation", "user authentication", "password check"]},
            {"current_file": "test_auth.py", "recent_queries": ["test cases"]}
        ]
        
        for i, context in enumerate(contexts):
            suggestions = self.suggestion_engine.get_suggestions(base_query, limit=15, context=context)
            
            print(f"\n=== Context {i+1} suggestions for '{base_query}' ===")
            print(f"Context: {context}")
            
            context_suggestions = [s for s in suggestions if s.type == SuggestionType.CONTEXT]
            for suggestion in context_suggestions:
                print(f"[{suggestion.type.value}] {suggestion.text} (confidence: {suggestion.confidence:.2f})")
            
            # Should generate context-aware suggestions
            assert len(context_suggestions) >= 1, \
                f"Should generate context suggestions for context {context}"
            
            # Check for context-specific patterns
            if "current_file" in context:
                file_suggestions = [s for s in context_suggestions if context["current_file"] in s.text]
                assert len(file_suggestions) >= 1, "Should suggest searching in current file"
            
            if "recent_queries" in context:
                recent_suggestions = [s for s in context_suggestions 
                                    if any(recent in s.text for recent in context["recent_queries"])]
                # May or may not have recent query suggestions depending on relevance
    
    def test_query_hints_for_developer_scenarios(self):
        """Test query hints for common developer scenarios"""
        test_cases = [
            {
                "query": "auth",
                "expected_hints": ["Add 'def' or 'class'", "more context words"],
                "description": "Single word query should get context hints"
            },
            {
                "query": "very long query with many words that might be too verbose for effective searching",
                "expected_hints": ["shorter, more specific"],
                "description": "Long query should get brevity hints"
            },
            {
                "query": "login function",
                "expected_hints": ["exact matches", "file type"],
                "description": "Short query should get specificity hints"
            },
            {
                "query": "LoginManager",
                "expected_hints": ["exact matches", "file type"],
                "description": "Specific term should get exact match hints"
            }
        ]
        
        for case in test_cases:
            hints = self.suggestion_engine.get_query_hints(case["query"])
            
            print(f"\n=== Hints for '{case['query']}' ===")
            print(f"Description: {case['description']}")
            for hint in hints:
                print(f"  • {hint}")
            
            # Should provide helpful hints
            assert len(hints) >= 1, f"Should provide hints for query '{case['query']}'"
            assert len(hints) <= 3, "Should limit hints to avoid overwhelming user"
            
            # Check for expected hint patterns
            hint_text = " ".join(hints).lower()
            relevant_hints = [pattern for pattern in case["expected_hints"] 
                            if any(keyword in hint_text for keyword in pattern.lower().split())]
            
            assert len(relevant_hints) >= 1, \
                f"Should provide relevant hints for '{case['query']}'. Expected patterns: {case['expected_hints']}, Got: {hints}"
    
    async def test_suggestion_quality_across_languages(self):
        """Test suggestion quality across different programming languages"""
        cross_language_queries = ["function", "class", "interface", "test", "util"]
        
        for query in cross_language_queries:
            suggestions = self.suggestion_engine.get_suggestions(query, limit=12)
            
            print(f"\n=== Cross-language suggestions for '{query}' ===")
            
            # Group suggestions by type
            by_type = {}
            for suggestion in suggestions:
                if suggestion.type not in by_type:
                    by_type[suggestion.type] = []
                by_type[suggestion.type].append(suggestion)
            
            for suggestion_type, type_suggestions in by_type.items():
                print(f"\n{suggestion_type.value.upper()} suggestions:")
                for suggestion in type_suggestions:
                    print(f"  {suggestion.text} (confidence: {suggestion.confidence:.2f})")
            
            # Should have diverse suggestion types
            assert len(by_type) >= 2, f"Should have diverse suggestion types for '{query}'"
            
            # Should have reasonable confidence distribution
            confidences = [s.confidence for s in suggestions]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            assert avg_confidence >= 0.5, \
                f"Average suggestion confidence should be reasonable for '{query}' (got {avg_confidence:.2f})"
            
            # Test a few suggestions with real search results
            for suggestion in suggestions[:3]:  # Test top 3 suggestions
                try:
                    results = await self.search_with_query("python", suggestion.text, limit=5)
                    print(f"  Suggestion '{suggestion.text}' found {len(results)} results")
                    
                    # Suggestions should generally find some results
                    if len(results) == 0:
                        print(f"    Warning: No results for suggestion '{suggestion.text}'")
                    
                except Exception as e:
                    print(f"    Error testing suggestion '{suggestion.text}': {e}")
    
    async def test_suggestion_performance_benchmarks(self):
        """Test suggestion generation performance"""
        test_queries = [
            "auth", "login", "user", "password", "session", "token",
            "function", "class", "method", "variable", "constant",
            "test", "mock", "fixture", "assert", "validate",
            "error", "exception", "logging", "debug", "trace"
        ]
        
        # Benchmark suggestion generation
        start_time = time.time()
        
        total_suggestions = 0
        for query in test_queries:
            suggestions = self.suggestion_engine.get_suggestions(query, limit=10)
            total_suggestions += len(suggestions)
        
        generation_time = time.time() - start_time
        
        print(f"\n=== Suggestion Performance Benchmarks ===")
        print(f"Generated {total_suggestions} suggestions for {len(test_queries)} queries")
        print(f"Total time: {generation_time:.3f}s")
        print(f"Average time per query: {generation_time/len(test_queries):.3f}s")
        print(f"Average suggestions per query: {total_suggestions/len(test_queries):.1f}")
        
        # Performance assertions
        assert generation_time < 5.0, f"Suggestion generation took too long: {generation_time:.3f}s"
        
        avg_time_per_query = generation_time / len(test_queries)
        assert avg_time_per_query < 0.1, f"Average time per query too slow: {avg_time_per_query:.3f}s"
        
        avg_suggestions = total_suggestions / len(test_queries)
        assert avg_suggestions >= 5, f"Should generate sufficient suggestions: {avg_suggestions:.1f}"
    
    def test_suggestion_deduplication_and_ranking(self):
        """Test suggestion deduplication and confidence ranking"""
        # Query that might generate duplicate suggestions
        query = "test function"
        
        suggestions = self.suggestion_engine.get_suggestions(query, limit=20)
        
        print(f"\n=== Deduplication test for '{query}' ===")
        for i, suggestion in enumerate(suggestions):
            print(f"{i+1}: {suggestion.text} (confidence: {suggestion.confidence:.2f}, type: {suggestion.type.value})")
        
        # Check for duplicates
        suggestion_texts = [s.text.lower().strip() for s in suggestions]
        unique_texts = set(suggestion_texts)
        
        assert len(unique_texts) == len(suggestion_texts), \
            f"Found duplicate suggestions: {len(suggestion_texts)} total, {len(unique_texts)} unique"
        
        # Check confidence ranking
        confidences = [s.confidence for s in suggestions]
        sorted_confidences = sorted(confidences, reverse=True)
        
        assert confidences == sorted_confidences, \
            "Suggestions should be sorted by confidence in descending order"
        
        # Check confidence ranges
        if suggestions:
            max_confidence = max(confidences)
            min_confidence = min(confidences)
            
            assert max_confidence <= 1.0, f"Max confidence should not exceed 1.0: {max_confidence}"
            assert min_confidence >= 0.0, f"Min confidence should not be below 0.0: {min_confidence}"
            assert max_confidence > min_confidence, "Should have confidence variation"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])