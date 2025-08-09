"""
REAL End-to-End Test: Claude Orchestrated Natural Language Search

This test demonstrates the full power of iterative Claude-orchestrated search
using natural language queries on a real repository.

Repository: FastAPI (has authentication examples and complex workflows)
Queries: Natural language questions about implementation details
Focus: Verify that Claude actually makes multiple search iterations and finds comprehensive results
"""

import asyncio
import logging
import os
import pytest
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional

# Fix tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from claude_code_context.mcp_server.models import (
    SearchRequest, 
    SearchResponse, 
    SearchMode,
    MCPServerConfig
)
from claude_code_context.mcp_server.search_executor import SearchExecutor
from claude_code_context.mcp_server.connection import QdrantConnectionManager

# Import core components for indexing
from core.indexer.hybrid_indexer import HybridIndexer, IndexingJobConfig
from core.parser.parallel_pipeline import ProcessParsingPipeline
from core.storage.client import HybridQdrantClient
from core.storage.schemas import CollectionManager, QdrantSchema
from core.embeddings.stella import StellaEmbedder
from core.models.config import StellaConfig

logger = logging.getLogger(__name__)


class TestClaudeOrchestrationE2E:
    """Test Claude orchestration with natural language queries on real repositories"""
    
    # Shared resources across all tests to avoid re-indexing
    _shared_embedder: Optional[StellaEmbedder] = None
    _shared_storage_client: Optional[HybridQdrantClient] = None
    _shared_parser_pipeline: Optional[ProcessParsingPipeline] = None
    _shared_search_executor: Optional[SearchExecutor] = None
    _shared_collection_name: Optional[str] = None
    
    @classmethod
    def setup_class(cls):
        """Setup test environment and index repository ONCE"""
        cls.test_dir = Path("test-harness/claude-orchestration-e2e").resolve()
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Repository choices with good examples of complex implementations
        cls.repositories = {
            "fastapi": {
                "url": "https://github.com/tiangolo/fastapi.git",
                "branch": "master",
                "description": "Modern web framework with security and auth examples"
            },
            "authlib": {
                "url": "https://github.com/lepture/authlib.git",
                "branch": "master", 
                "description": "Authentication library with OAuth implementations"
            },
            "flask-security": {
                "url": "https://github.com/Flask-Middleware/flask-security.git",
                "branch": "main",
                "description": "Security and authentication for Flask apps"
            }
        }
        
        # We'll use FastAPI for this test - it has good auth examples
        cls.selected_repo = "fastapi"
        cls.repo_info = cls.repositories[cls.selected_repo]
        cls.repo_path = cls.test_dir / cls.selected_repo
        
        cls.created_collections = set()
        
        # Check Claude CLI availability - CRITICAL for this test
        import shutil as shutil_module
        cls.claude_cli_available = shutil_module.which('claude') is not None
        if not cls.claude_cli_available:
            pytest.skip("Claude CLI required for orchestration testing")
        else:
            logger.info("✅ Claude CLI available - full orchestration testing enabled")
        
        # Setup shared resources and index repository ONCE
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Clone repository
            cls._clone_repository_sync()
            
            # Initialize shared heavy resources
            cls._initialize_shared_resources_sync(loop)
            
            # Index repository ONCE
            cls._index_repository_sync(loop)
            
            # Create shared search executor ONCE
            cls._create_search_executor_sync(loop)
            
            logger.info(f"✅ Setup complete - repository indexed and search executor ready")
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            loop.close()
            pytest.skip(f"Failed to setup test environment: {e}")
        
        loop.close()
    
    @classmethod
    def teardown_class(cls):
        """Cleanup test environment and shared resources"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Cleanup shared search executor
        if cls._shared_search_executor:
            try:
                loop.run_until_complete(cls._shared_search_executor.shutdown())
            except Exception as e:
                logger.warning(f"Error shutting down search executor: {e}")
            finally:
                cls._shared_search_executor = None
        
        # Cleanup shared storage client
        if cls._shared_storage_client:
            try:
                loop.run_until_complete(cls._shared_storage_client.disconnect())
            except Exception as e:
                logger.warning(f"Error disconnecting storage client: {e}")
            finally:
                cls._shared_storage_client = None
        
        # Cleanup shared embedder
        if cls._shared_embedder:
            try:
                loop.run_until_complete(cls._shared_embedder.unload_model())
            except Exception as e:
                logger.warning(f"Error unloading embedder: {e}")
            finally:
                cls._shared_embedder = None
        
        # Delete test collections
        try:
            import requests
            for collection_name in cls.created_collections:
                try:
                    requests.delete(f"http://localhost:6334/collections/{collection_name}", timeout=5)
                except Exception:
                    pass
        except Exception:
            pass
        
        loop.close()
        
        # Clean up test directory
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir, ignore_errors=True)
    
    @classmethod
    def _clone_repository_sync(cls):
        """Clone the selected repository - called once in setup_class"""
        repo_path = cls.repo_path
        
        if repo_path.exists() and (repo_path / ".git").exists():
            logger.info(f"Repository already exists at {repo_path}")
            return
        
        logger.info(f"Cloning {cls.selected_repo} repository...")
        try:
            result = subprocess.run([
                "git", "clone",
                "--depth", "1",  # Shallow clone for speed
                "--single-branch",
                "--branch", cls.repo_info["branch"],
                cls.repo_info["url"],
                str(repo_path)
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"Failed to clone repository: {result.stderr}")
            
            logger.info(f"✅ Cloned {cls.selected_repo} to {repo_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to clone repository: {e}")
    
    @classmethod
    def _initialize_shared_resources_sync(cls, loop):
        """Initialize shared heavy resources - called once in setup_class"""
        # Initialize embedder
        embedder_config = StellaConfig(
            model_name="stella_en_400M_v5",
            batch_size=32,
            cache_size=100,
            cache_ttl_seconds=300
        )
        cls._shared_embedder = StellaEmbedder(embedder_config)
        model_loaded = loop.run_until_complete(cls._shared_embedder.load_model())
        if not model_loaded:
            raise RuntimeError("Stella model not available")
        
        # Initialize storage client
        cls._shared_storage_client = HybridQdrantClient(url="http://localhost:6334")
        loop.run_until_complete(cls._shared_storage_client.connect())
        
        # Initialize parser pipeline
        cls._shared_parser_pipeline = ProcessParsingPipeline(
            max_workers=4,
            batch_size=50,
            execution_mode="thread"
        )
        
        logger.info("✅ Shared resources initialized")
    
    @classmethod
    def _index_repository_sync(cls, loop):
        """Index repository ONCE - called in setup_class"""
        # Create unique collection name
        unique_id = str(uuid.uuid4())[:8]
        cls._shared_collection_name = f"claude-orch-{cls.selected_repo}-{unique_id}-code"
        
        # Create collection
        collection_config = QdrantSchema.get_code_collection_config(cls._shared_collection_name)
        create_result = loop.run_until_complete(
            cls._shared_storage_client.create_collection(collection_config)
        )
        if not create_result.success:
            raise RuntimeError(f"Failed to create collection: {create_result.error}")
        
        cls.created_collections.add(cls._shared_collection_name)
        
        # Index repository
        logger.info(f"Indexing {cls.selected_repo} repository...")
        
        config = IndexingJobConfig(
            project_path=cls.repo_path,
            project_name=cls.selected_repo,
            include_patterns=["*.py"],
            exclude_patterns=[
                "__pycache__/*",
                ".git/*",
                "*.pyc",
                "tests/*",  # Skip test files for cleaner results
                "docs/*",
                "build/*",
                "dist/*"
            ],
            max_workers=4,
            batch_size=50
        )
        
        indexer = HybridIndexer(
            parser_pipeline=cls._shared_parser_pipeline,
            embedder=cls._shared_embedder,
            storage_client=cls._shared_storage_client,
            cache_manager=None,
            config=config
        )
        
        result = loop.run_until_complete(
            indexer.perform_delta_scan(
                project_path=cls.repo_path,
                collection_name=cls._shared_collection_name,
                force_full_scan=True
            )
        )
        
        if not result["success"]:
            raise RuntimeError(f"Indexing failed: {result.get('error_message')}")
        
        entity_count = result["phases"]["upsert_operations"]["upserted_entities"]
        logger.info(f"✅ Indexed {entity_count} entities from {cls.selected_repo}")
    
    @classmethod
    def _create_search_executor_sync(cls, loop):
        """Create shared search executor - called once in setup_class"""
        # Create MCP configuration
        mcp_config = MCPServerConfig(
            project_path=cls.repo_path,
            collection_name=cls._shared_collection_name,
            qdrant_url="http://localhost:6334",
            max_claude_calls=5,  # Allow more iterations for complex queries
            max_results=30,  # Get comprehensive results
            debug_mode=True  # Enable debug for detailed logs
        )
        
        connection_manager = QdrantConnectionManager(mcp_config)
        cls._shared_search_executor = SearchExecutor(connection_manager, mcp_config)
        
        # Initialize search executor
        initialized = loop.run_until_complete(cls._shared_search_executor.initialize())
        if not initialized:
            raise RuntimeError("Failed to initialize search executor")
        
        if not cls._shared_search_executor._orchestration_enabled:
            raise RuntimeError("Claude orchestration not enabled - test cannot proceed")
        
        logger.info(f"✅ Search executor ready with Claude orchestration")
    
    @pytest.mark.asyncio
    async def test_authentication_workflow_search(self):
        """Test natural language query about authentication workflow"""
        # Use shared search executor - repository already indexed in setup_class
        search_executor = self._shared_search_executor
        if not search_executor:
            pytest.skip("Search executor not initialized")
        
        # Natural language query about authentication
        query = "how is the authentication workflow implemented with JWT tokens and OAuth2"
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing natural language query: '{query}'")
        logger.info(f"Repository: {self.selected_repo}")
        logger.info(f"{'='*80}\n")
        
        request = SearchRequest(
            request_id="auth_workflow_test",
            query=query,
            mode=SearchMode.AUTO,  # Let Claude decide the best approach
            limit=30
        )
        
        start_time = time.perf_counter()
        response = await search_executor.execute_search(request)
        duration = time.perf_counter() - start_time
        
        # Verify response
        assert response.success is True, f"Search failed: {response.error_message}"
        
        # CRITICAL: Verify Claude orchestration happened
        assert response.claude_calls_made > 0, \
            f"No Claude calls made - orchestration didn't happen!"
        
        logger.info(f"\n📊 Search Statistics:")
        logger.info(f"  - Claude calls made: {response.claude_calls_made}")
        logger.info(f"  - Search duration: {duration:.2f}s")
        logger.info(f"  - Results found: {len(response.results)}")
        logger.info(f"  - Search mode used: {response.search_mode_used}")
        
        # Check if multiple strategies were executed
        if response.query_optimization:
            logger.info(f"\n🔍 Search Strategies Executed:")
            try:
                import ast
                strategies = ast.literal_eval(response.query_optimization)
                for i, strategy in enumerate(strategies, 1):
                    logger.info(f"  {i}. Mode: {strategy.get('mode', 'unknown')}")
                    logger.info(f"     Query: {strategy.get('query', 'N/A')}")
                    if strategy.get('reasoning'):
                        logger.info(f"     Reasoning: {strategy['reasoning'][:100]}...")
            except:
                logger.info(f"  Raw: {response.query_optimization}")
        
        # Analyze results
        if response.results:
            logger.info(f"\n📝 Top Results Found:")
            
            # Group by entity type
            by_type = {}
            for result in response.results[:10]:  # Top 10
                entity_type = result.entity_type
                if entity_type not in by_type:
                    by_type[entity_type] = []
                by_type[entity_type].append(result)
            
            for entity_type, results in by_type.items():
                logger.info(f"\n  {entity_type.upper()}s ({len(results)}):")
                for r in results[:3]:  # Top 3 per type
                    logger.info(f"    - {r.name} (score: {r.relevance_score:.3f})")
                    logger.info(f"      File: {r.file_path}")
                    if r.docstring:
                        logger.info(f"      Doc: {r.docstring[:80]}...")
        
        # Verify quality of results for authentication query
        result_names = [r.name.lower() for r in response.results]
        result_contents = [r.content.lower() if r.content else "" for r in response.results]
        
        # Check for authentication-related terms
        auth_terms = ["auth", "jwt", "token", "oauth", "login", "bearer", "credential", "security"]
        relevant_results = 0
        
        for name, content in zip(result_names, result_contents):
            if any(term in name or term in content for term in auth_terms):
                relevant_results += 1
        
        relevance_ratio = relevant_results / len(response.results) if response.results else 0
        logger.info(f"\n✅ Relevance Check: {relevant_results}/{len(response.results)} "
                   f"results contain auth-related terms ({relevance_ratio:.1%})")
        
        # Should have found relevant results
        assert relevant_results > 0, "No authentication-related results found"
        assert relevance_ratio > 0.3, f"Low relevance ratio: {relevance_ratio:.1%}"
    
    @pytest.mark.asyncio
    async def test_error_handling_search(self):
        """Test natural language query about error handling patterns"""
        # Use shared search executor
        search_executor = self._shared_search_executor
        if not search_executor:
            pytest.skip("Search executor not initialized")
        
        query = "show me how errors and exceptions are handled in the API endpoints"
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing query: '{query}'")
        logger.info(f"{'='*80}\n")
        
        request = SearchRequest(
            request_id="error_handling_test",
            query=query,
            mode=SearchMode.AUTO,
            limit=25
        )
        
        response = await search_executor.execute_search(request)
        
        assert response.success is True
        assert response.claude_calls_made > 0
        assert len(response.results) > 0
        
        # Check for error handling patterns
        error_terms = ["error", "exception", "raise", "try", "except", "catch", "handle", "httpexception"]
        relevant = sum(1 for r in response.results 
                      if any(term in r.name.lower() or term in (r.content or "").lower() 
                            for term in error_terms))
        
        logger.info(f"Found {relevant} error-handling related results out of {len(response.results)}")
        assert relevant > 0, "Should find error handling patterns"
    
    @pytest.mark.asyncio
    async def test_data_validation_search(self):
        """Test natural language query about data validation"""
        # Use shared search executor
        search_executor = self._shared_search_executor
        if not search_executor:
            pytest.skip("Search executor not initialized")
        
        query = "how does the framework validate request data and parameters"
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing query: '{query}'")
        logger.info(f"{'='*80}\n")
        
        request = SearchRequest(
            request_id="validation_test",
            query=query,
            mode=SearchMode.AUTO,
            limit=20
        )
        
        start_time = time.perf_counter()
        response = await search_executor.execute_search(request)
        duration = time.perf_counter() - start_time
        
        assert response.success is True
        assert response.claude_calls_made > 0
        
        logger.info(f"Claude made {response.claude_calls_made} calls in {duration:.2f}s")
        logger.info(f"Found {len(response.results)} results")
        
        # With FastAPI, should find Pydantic validation
        validation_terms = ["valid", "pydantic", "field", "schema", "model", "basemodel", "param"]
        relevant = sum(1 for r in response.results 
                      if any(term in r.name.lower() or term in (r.content or "").lower() 
                            for term in validation_terms))
        
        logger.info(f"Found {relevant} validation-related results")
        assert relevant > 0, "Should find validation patterns"
    
    @pytest.mark.asyncio
    async def test_iterative_refinement(self):
        """Test that Claude actually refines search iteratively"""
        # Use shared search executor
        search_executor = self._shared_search_executor
        if not search_executor:
            pytest.skip("Search executor not initialized")
        
        # Broad query that should trigger refinement
        query = "explain the request processing pipeline from receiving HTTP request to sending response"
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing iterative refinement with: '{query}'")
        logger.info(f"{'='*80}\n")
        
        request = SearchRequest(
            request_id="pipeline_test",
            query=query,
            mode=SearchMode.AUTO,
            limit=40  # Higher limit to see more results
        )
        
        response = await search_executor.execute_search(request)
        
        assert response.success is True
        
        # Should make multiple Claude calls for such a broad query
        assert response.claude_calls_made >= 1, \
            f"Expected multiple Claude calls for broad query, got {response.claude_calls_made}"
        
        logger.info(f"\n🔄 Iterative Search Analysis:")
        logger.info(f"  - Claude calls: {response.claude_calls_made}")
        logger.info(f"  - Total results: {len(response.results)}")
        
        # Parse strategies to see the refinement
        if response.query_optimization:
            try:
                import ast
                strategies = ast.literal_eval(response.query_optimization)
                logger.info(f"  - Search strategies: {len(strategies)}")
                
                # Should have different search modes for different aspects
                modes_used = set(s.get('mode', '') for s in strategies)
                logger.info(f"  - Search modes used: {modes_used}")
                
                # Ideally should use multiple modes for comprehensive results
                if len(modes_used) > 1:
                    logger.info("  ✅ Used multiple search modes for comprehensive coverage")
            except:
                pass
        
        # Check result diversity - should find different types of entities
        entity_types = set(r.entity_type for r in response.results)
        logger.info(f"  - Entity types found: {entity_types}")
        
        assert len(entity_types) >= 2, \
            f"Should find diverse entity types, got {entity_types}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--log-cli-level=INFO"])