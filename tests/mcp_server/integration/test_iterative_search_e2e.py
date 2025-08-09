"""
FEAT10 End-to-End Test: Iterative Multi-Turn Search with RRF Fusion.

Tests the complete iterative search pipeline with real components:
- Real repository cloning (validators for consistency with other tests)
- Real indexing with HybridIndexer
- Real Qdrant vector database operations
- Real Claude orchestration (if available)
- Multi-list RRF fusion algorithm
- Stop conditions and session continuity

NO MOCKS - Full end-to-end integration testing.
"""

import asyncio
import json
import logging
import os
import pytest
import shutil
import subprocess
import tempfile
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
from claude_code_context.mcp_server.orchestrator import ClaudeOrchestrator

# Import core components for indexing
from core.indexer.hybrid_indexer import HybridIndexer, IndexingJobConfig
from core.parser.parallel_pipeline import ProcessParsingPipeline
from core.storage.client import HybridQdrantClient
from core.storage.schemas import CollectionManager, QdrantSchema
from core.embeddings.stella import StellaEmbedder
from core.models.config import StellaConfig
from core.models.storage import SearchResult as CoreSearchResult

logger = logging.getLogger(__name__)


class TestIterativeSearchE2E:
    """Comprehensive E2E tests for iterative multi-turn search"""

    # Shared resources across all tests to avoid repeated heavy initialization
    _shared_embedder: Optional[StellaEmbedder] = None
    _shared_storage_client: Optional[HybridQdrantClient] = None
    _shared_parser_pipeline: Optional[ProcessParsingPipeline] = None

    @classmethod
    def setup_class(cls):
        """Setup test environment with real repository"""
        cls.test_dir = Path("test-harness/iterative-search-e2e").resolve()
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Use validators repo for consistency with other tests
        cls.repo_info = {
            "url": "https://github.com/python-validators/validators.git",
            "path": cls.test_dir / "validators",
            "branch": "master"
        }
        
        cls.created_collections = set()
        cls.search_executors = []  # Track for cleanup
        
        # Clean up any stray test collections from previous runs
        cls._cleanup_old_test_collections()
        
        # Check Claude CLI availability
        import shutil as shutil_module
        cls.claude_cli_available = shutil_module.which('claude') is not None
        if cls.claude_cli_available:
            logger.info("✅ Claude CLI available - full iterative testing enabled")
        else:
            logger.warning("⚠️  Claude CLI not available - testing fusion without orchestration")
        
        # Initialize shared heavy resources once
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        if cls._shared_embedder is None:
            embedder_config = StellaConfig(
                model_name="stella_en_400M_v5",
                batch_size=32,
                cache_size=100,
                cache_ttl_seconds=300
            )
            cls._shared_embedder = StellaEmbedder(embedder_config)
            model_loaded = loop.run_until_complete(cls._shared_embedder.load_model())
            if not model_loaded:
                pytest.skip("Stella model not available for testing")
        
        if cls._shared_storage_client is None:
            cls._shared_storage_client = HybridQdrantClient(url="http://localhost:6334")
            loop.run_until_complete(cls._shared_storage_client.connect())
        
        if cls._shared_parser_pipeline is None:
            cls._shared_parser_pipeline = ProcessParsingPipeline(
                max_workers=2,
                batch_size=20,
                execution_mode="thread"
            )
        
        loop.close()
    
    @classmethod
    def teardown_class(cls):
        """Cleanup test environment and shared resources"""
        # Shutdown search executors
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        for executor in cls.search_executors:
            try:
                loop.run_until_complete(executor.shutdown())
            except Exception as e:
                logger.warning(f"Error shutting down search executor: {e}")
        
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
        
        # Disconnect shared storage client
        if cls._shared_storage_client is not None:
            loop.run_until_complete(cls._shared_storage_client.disconnect())
            cls._shared_storage_client = None
        
        # Unload shared embedder
        if cls._shared_embedder is not None:
            loop.run_until_complete(cls._shared_embedder.unload_model())
            cls._shared_embedder = None
        
        loop.close()
        
        # Clean up test directory
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir, ignore_errors=True)
    
    @classmethod
    def _cleanup_old_test_collections(cls):
        """Remove leftover collections from previous test runs"""
        try:
            import requests
            response = requests.get("http://localhost:6334/collections", timeout=5)
            if response.status_code == 200:
                collections = response.json().get("result", {}).get("collections", [])
                for collection in collections:
                    name = collection.get("name", "")
                    if "iterative" in name and "-code" in name:
                        try:
                            requests.delete(f"http://localhost:6334/collections/{name}", timeout=5)
                        except Exception:
                            pass
        except Exception:
            pass
    
    def get_or_clone_repository(self) -> Path:
        """Get repository, cloning if needed, with reuse across tests"""
        repo_path = self.repo_info["path"]
        
        if repo_path.exists() and (repo_path / ".git").exists():
            # Repository exists, use as-is
            logger.info(f"Using existing repository at {repo_path}")
        else:
            # Clone repository
            try:
                result = subprocess.run([
                    "git", "clone",
                    "--depth", "1",  # Shallow clone for speed
                    self.repo_info["url"],
                    str(repo_path)
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    error_msg = f"Git clone failed: {result.stderr}"
                    logger.error(error_msg)
                    pytest.skip(f"Failed to clone repository: {error_msg}")
                
                logger.info(f"Cloned repository to {repo_path}")
            except FileNotFoundError:
                pytest.skip("Git command not found - skipping integration test")
            except Exception as e:
                pytest.skip(f"Failed to clone repository: {e}")
        
        return repo_path
    
    def add_test_files(self, repo_path: Path) -> List[Path]:
        """Add test files to repository for search testing"""
        added_files = []
        
        # Add authentication-related file
        auth_file = repo_path / "test_auth.py"
        auth_file.write_text('''
"""
Authentication module for testing iterative search.
"""
import hashlib
from typing import Optional

def authenticate_user(username: str, password: str) -> bool:
    """Authenticate user with credentials."""
    password_hash = hash_password(password)
    return verify_credentials(username, password_hash)

def hash_password(password: str) -> str:
    """Hash password using SHA256."""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_credentials(username: str, password_hash: str) -> bool:
    """Verify user credentials against database."""
    # Mock verification
    return True

def generate_auth_token(user_id: str) -> str:
    """Generate authentication token for user."""
    return f"token_{user_id}_{hash_password(user_id)[:8]}"
''')
        added_files.append(auth_file)
        
        # Add session management file
        session_file = repo_path / "test_session.py"
        session_file.write_text('''
"""
Session management for testing iterative search.
"""
from datetime import datetime, timedelta
from typing import Dict, Optional

class SessionManager:
    """Manages user sessions."""
    
    def __init__(self):
        self.sessions: Dict[str, dict] = {}
    
    def create_session(self, user_id: str, token: str) -> str:
        """Create new user session."""
        session_id = f"session_{user_id}_{datetime.now().timestamp()}"
        self.sessions[session_id] = {
            "user_id": user_id,
            "token": token,
            "created_at": datetime.now()
        }
        return session_id
    
    def validate_session(self, session_id: str) -> bool:
        """Validate if session is active."""
        return session_id in self.sessions
    
    def invalidate_session(self, session_id: str) -> bool:
        """Invalidate user session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

def validate_user_session(session_id: str) -> bool:
    """Main function to validate user session."""
    manager = SessionManager()
    return manager.validate_session(session_id)
''')
        added_files.append(session_file)
        
        # Add utility file with various patterns
        util_file = repo_path / "test_utils.py"
        util_file.write_text('''
"""
Utility functions for testing.
"""

def encrypt_password(password: str) -> str:
    """Encrypt password for storage."""
    return ''.join(chr(ord(c) ^ 42) for c in password)

def validate_email(email: str) -> bool:
    """Validate email format."""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def check_password_strength(password: str) -> bool:
    """Check if password meets requirements."""
    return len(password) >= 8 and any(c.isupper() for c in password)

def generate_session_key(user_id: str, timestamp: float) -> str:
    """Generate unique session key."""
    return f"key_{user_id}_{int(timestamp)}"
''')
        added_files.append(util_file)
        
        return added_files
    
    async def create_collection_for_test(self, storage_client: HybridQdrantClient, test_name: str) -> str:
        """Create test collection with unique name"""
        unique_id = str(uuid.uuid4())[:8]
        collection_name = f"iterative-{test_name}-{unique_id}-code"
        
        # Create collection with proper schema
        collection_config = QdrantSchema.get_code_collection_config(collection_name)
        create_result = await storage_client.create_collection(collection_config)
        
        if not create_result.success:
            raise RuntimeError(f"Failed to create collection: {create_result.error}")
        
        # Track for cleanup
        self.__class__.created_collections.add(collection_name)
        
        return collection_name
    
    async def index_repository(self, repo_path: Path, collection_name: str) -> bool:
        """Index repository using HybridIndexer with shared resources"""
        cls = self.__class__
        
        # Create indexer config
        config = IndexingJobConfig(
            project_path=repo_path,
            project_name=collection_name.split('-')[0],
            include_patterns=["*.py"],
            exclude_patterns=[
                "__pycache__/*",
                ".git/*",
                "*.pyc",
                ".tox/*",
                "build/*",
                "dist/*"
            ],
            max_workers=2,
            batch_size=20
        )
        
        # Create indexer with shared components
        indexer = HybridIndexer(
            parser_pipeline=cls._shared_parser_pipeline,
            embedder=cls._shared_embedder,
            storage_client=cls._shared_storage_client,
            cache_manager=None,
            config=config
        )
        
        try:
            # Perform full scan
            result = await indexer.perform_delta_scan(
                project_path=repo_path,
                collection_name=collection_name,
                force_full_scan=True
            )
            
            if result["success"]:
                entity_count = result["phases"]["upsert_operations"]["upserted_entities"]
                logger.info(f"Indexed {entity_count} entities in {collection_name}")
                return True
            else:
                logger.error(f"Indexing failed: {result.get('error_message')}")
                return False
                
        except Exception as e:
            logger.error(f"Indexing error: {e}")
            return False
    
    @pytest.mark.asyncio
    async def test_iterative_search_with_real_repository(self):
        """Test iterative search with real repository and RRF fusion"""
        # Setup repository
        repo_path = self.get_or_clone_repository()
        
        # Add test files for better search testing
        test_files = self.add_test_files(repo_path)
        
        # Create collection
        collection_name = await self.create_collection_for_test(
            self._shared_storage_client, "full"
        )
        
        # Index repository
        indexed = await self.index_repository(repo_path, collection_name)
        assert indexed, "Failed to index repository"
        
        # Create MCP configuration
        config = MCPServerConfig(
            project_path=repo_path,
            collection_name=collection_name,
            qdrant_url="http://localhost:6334",
            max_claude_calls=3,  # Limit for testing
            max_results=20,
            debug_mode=False
        )
        
        # Initialize search components
        connection_manager = QdrantConnectionManager(config)
        search_executor = SearchExecutor(connection_manager, config)
        self.__class__.search_executors.append(search_executor)
        
        # Initialize search infrastructure
        initialized = await search_executor.initialize()
        assert initialized or not search_executor._orchestration_enabled
        
        # Test 1: Multi-word query that benefits from iterative search
        logger.info("Test 1: Multi-word query 'authenticate_user validate_session'")
        
        request = SearchRequest(
            request_id="test_iterative_multiword",
            query="authenticate_user validate_session",  # Multi-word query
            mode=SearchMode.AUTO,  # Triggers iterative search
            limit=20
        )
        
        start_time = time.perf_counter()
        response = await search_executor.execute_search(request)
        search_duration = time.perf_counter() - start_time
        
        # Verify response structure
        assert response.success is True
        assert response.request_id == "test_iterative_multiword"
        assert response.search_mode_used == SearchMode.AUTO
        assert response.execution_time_ms > 0
        
        # CRITICAL: Verify iterative search actually happened
        if search_executor._orchestration_enabled:
            # Check Claude was called (at least once for sufficiency check)
            assert response.claude_calls_made > 0, \
                f"Should have made Claude calls, but made {response.claude_calls_made}"
            logger.info(f"Claude calls made: {response.claude_calls_made}")
            
            # Check if query optimization happened (indicates multiple strategies)
            if response.query_optimization:
                logger.info(f"Query optimization/strategies: {response.query_optimization}")
                # Parse strategies if it's a string representation
                if isinstance(response.query_optimization, str) and "mode" in response.query_optimization:
                    import ast
                    try:
                        strategies = ast.literal_eval(response.query_optimization)
                        logger.info(f"Executed {len(strategies)} search strategies")
                        assert len(strategies) >= 1, "Should have at least baseline strategy"
                    except:
                        pass
        else:
            logger.warning("Claude orchestration disabled - testing direct search only")
        
        # Verify results were found
        assert len(response.results) > 0, "Should find results with iterative search"
        logger.info(f"Found {len(response.results)} results in {search_duration:.2f}s")
        
        # Check that both authenticate_user and validate_session are found
        result_names = [r.name for r in response.results]
        result_contents = [r.content for r in response.results]
        
        found_authenticate = any("authenticate_user" in name or "authenticate_user" in content 
                                for name, content in zip(result_names, result_contents))
        found_validate = any("validate" in name or "validate_session" in content 
                            for name, content in zip(result_names, result_contents))
        
        logger.info(f"Found authenticate_user: {found_authenticate}")
        logger.info(f"Found validate_session: {found_validate}")
        
        # At least one should be found
        assert found_authenticate or found_validate, \
            "Should find at least one of the search terms"
        
        # Test 2: Single specific function search
        logger.info("Test 2: Specific function search 'hash_password'")
        
        specific_request = SearchRequest(
            request_id="test_specific",
            query="hash_password",
            mode=SearchMode.AUTO,
            limit=10
        )
        
        specific_response = await search_executor.execute_search(specific_request)
        
        assert specific_response.success is True
        assert len(specific_response.results) > 0
        
        # Should find hash_password function
        found_hash = any("hash_password" in r.name or "hash_password" in r.content 
                        for r in specific_response.results)
        assert found_hash, "Should find hash_password function"
        
        # Verify deduplication
        entity_ids = [r.entity_id for r in specific_response.results]
        assert len(entity_ids) == len(set(entity_ids)), "Results should be deduplicated"
        
        # Verify scores are in valid range
        for result in specific_response.results:
            assert 0.0 <= result.relevance_score <= 1.0, \
                f"Score {result.relevance_score} out of range"
        
        # Performance check
        assert search_duration < 60, f"Search took too long: {search_duration:.2f}s"
        
        logger.info(f"✅ Iterative search test completed: {len(response.results)} results")
        
        # Cleanup
        await search_executor.shutdown()
        await connection_manager.disconnect()
        
        # Clean up test files
        for test_file in test_files:
            if test_file.exists():
                test_file.unlink()
    
    @pytest.mark.asyncio
    async def test_rrf_fusion_with_multiple_strategies(self):
        """Test RRF fusion combining results from different search strategies"""
        # Setup repository
        repo_path = self.get_or_clone_repository()
        test_files = self.add_test_files(repo_path)
        
        # Create collection
        collection_name = await self.create_collection_for_test(
            self._shared_storage_client, "fusion"
        )
        
        # Index repository
        indexed = await self.index_repository(repo_path, collection_name)
        assert indexed, "Failed to index repository"
        
        # Create configuration
        config = MCPServerConfig(
            project_path=repo_path,
            collection_name=collection_name,
            qdrant_url="http://localhost:6334",
            max_claude_calls=2,
            max_results=30
        )
        
        connection_manager = QdrantConnectionManager(config)
        search_executor = SearchExecutor(connection_manager, config)
        self.__class__.search_executors.append(search_executor)
        
        await search_executor.initialize()
        
        # Test complex query that should trigger multiple strategies
        logger.info("Testing RRF fusion with complex query")
        
        request = SearchRequest(
            request_id="test_fusion",
            query="authentication session validation password",
            mode=SearchMode.AUTO,
            limit=30
        )
        
        response = await search_executor.execute_search(request)
        
        assert response.success is True
        assert len(response.results) > 0
        
        # Analyze result distribution
        entity_types = {}
        for result in response.results:
            entity_type = result.entity_type
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        logger.info(f"Entity type distribution: {entity_types}")
        
        # Should have diverse entity types if fusion worked
        assert len(entity_types) >= 2, "Should find multiple entity types"
        
        # Check score distribution
        scores = [r.relevance_score for r in response.results]
        assert all(0.0 <= s <= 1.0 for s in scores), "All scores should be in [0,1]"
        
        # Scores should be descending (properly ranked)
        for i in range(1, len(scores)):
            assert scores[i-1] >= scores[i], "Scores should be in descending order"
        
        logger.info(f"✅ RRF fusion test: {len(response.results)} fused results")
        
        # Cleanup
        await search_executor.shutdown()
        await connection_manager.disconnect()
        
        for test_file in test_files:
            if test_file.exists():
                test_file.unlink()
    
    @pytest.mark.asyncio
    async def test_stop_conditions_and_limits(self):
        """Test iterative search stop conditions and Claude call limits"""
        # Setup repository
        repo_path = self.get_or_clone_repository()
        test_files = self.add_test_files(repo_path)
        
        # Create collection
        collection_name = await self.create_collection_for_test(
            self._shared_storage_client, "limits"
        )
        
        # Index repository
        indexed = await self.index_repository(repo_path, collection_name)
        assert indexed, "Failed to index repository"
        
        # Create configuration with strict limits
        config = MCPServerConfig(
            project_path=repo_path,
            collection_name=collection_name,
            qdrant_url="http://localhost:6334",
            max_claude_calls=1,  # Very limited
            max_results=5  # Small result set
        )
        
        connection_manager = QdrantConnectionManager(config)
        search_executor = SearchExecutor(connection_manager, config)
        self.__class__.search_executors.append(search_executor)
        
        await search_executor.initialize()
        
        # Test with broad query that would benefit from more iterations
        logger.info("Testing stop conditions with limited Claude calls")
        
        request = SearchRequest(
            request_id="test_limits",
            query="user authentication session management validation",
            mode=SearchMode.AUTO,
            limit=10
        )
        
        start_time = time.perf_counter()
        response = await search_executor.execute_search(request)
        duration = time.perf_counter() - start_time
        
        # Should complete successfully despite limits
        assert response.success is True
        
        # CRITICAL: Verify stop conditions are working
        if search_executor._orchestration_enabled:
            # 1. Check max_claude_calls limit is respected
            assert response.claude_calls_made <= config.max_claude_calls, \
                f"Claude calls {response.claude_calls_made} exceeded limit {config.max_claude_calls}"
            
            # With max_claude_calls=1, should make exactly 1 call (for sufficiency check)
            assert response.claude_calls_made == 1, \
                f"Expected exactly 1 Claude call with limit=1, got {response.claude_calls_made}"
            
            logger.info(f"✅ Stop condition 1: max_claude_calls limit respected ({response.claude_calls_made}/{config.max_claude_calls})")
            
            # 2. Check that results are limited
            assert len(response.results) <= config.max_results, \
                f"Results {len(response.results)} exceeded limit {config.max_results}"
            
            logger.info(f"✅ Stop condition 2: max_results limit respected ({len(response.results)}/{config.max_results})")
            
            # 3. Parse query_optimization to see if followups were suggested but not executed
            if response.query_optimization:
                try:
                    import ast
                    strategies = ast.literal_eval(response.query_optimization)
                    # With max_claude_calls=1, should have at most 1-2 strategies (baseline + maybe 1 followup)
                    assert len(strategies) <= 2, \
                        f"Too many strategies executed with limit=1: {len(strategies)}"
                    logger.info(f"✅ Strategies executed: {len(strategies)} (baseline + limited followups)")
                except:
                    pass
        else:
            logger.warning("Claude orchestration disabled - cannot test stop conditions fully")
        
        # Should still return some results
        assert len(response.results) > 0, "Should have at least some results"
        
        # Should complete reasonably quickly with limits
        assert duration < 30, f"Limited search took too long: {duration:.2f}s"
        
        logger.info(f"✅ Stop conditions test complete: {response.claude_calls_made} Claude calls, "
                   f"{len(response.results)} results in {duration:.2f}s")
        
        # Cleanup
        await search_executor.shutdown()
        await connection_manager.disconnect()
        
        for test_file in test_files:
            if test_file.exists():
                test_file.unlink()
    
    @pytest.mark.asyncio
    async def test_direct_rrf_fusion_algorithm(self):
        """Test RRF fusion algorithm directly with synthetic data"""
        config = MCPServerConfig(
            project_path=Path("."),
            qdrant_url="http://localhost:6334"
        )
        
        connection_manager = QdrantConnectionManager(config)
        search_executor = SearchExecutor(connection_manager, config)
        
        # Import QdrantPoint for proper object creation
        from core.models.storage import QdrantPoint
        
        # List 1: Payload search results
        list1 = [
            CoreSearchResult(
                point=QdrantPoint(
                    id=1,
                    vector=[0.1] * 1024,  # Dummy 1024-dim vector for Stella
                    payload={
                        "entity_id": "func_auth_1",
                        "entity_name": "authenticate_user",
                        "entity_type": "function",
                        "file_path": "auth.py"
                    }
                ),
                score=0.95,
                query="authenticate",
                search_type="payload",
                rank=1,
                total_results=4
            ),
            CoreSearchResult(
                point=QdrantPoint(
                    id=2,
                    vector=[0.2] * 1024,  # Dummy 1024-dim vector
                    payload={
                        "entity_id": "func_session_1",
                        "entity_name": "validate_session",
                        "entity_type": "function",
                        "file_path": "session.py"
                    }
                ),
                score=0.75,
                query="authenticate",
                search_type="payload",
                rank=2,
                total_results=4
            ),
            CoreSearchResult(
                point=QdrantPoint(
                    id=3,
                    vector=[0.3] * 1024,  # Dummy 1024-dim vector
                    payload={
                        "entity_id": "func_hash_1",
                        "entity_name": "hash_password",
                        "entity_type": "function",
                        "file_path": "utils.py"
                    }
                ),
                score=0.60,
                query="authenticate",
                search_type="payload",
                rank=3,
                total_results=4
            ),
            CoreSearchResult(
                point=QdrantPoint(
                    id=4,
                    vector=[0.4] * 1024,  # Dummy 1024-dim vector
                    payload={
                        "entity_id": "class_auth_1",
                        "entity_name": "AuthManager",
                        "entity_type": "class",
                        "file_path": "auth.py"
                    }
                ),
                score=0.40,
                query="authenticate",
                search_type="payload",
                rank=4,
                total_results=4
            )
        ]
        
        # List 2: Semantic search with overlap
        list2 = [
            CoreSearchResult(
                point=QdrantPoint(
                    id=5,
                    vector=[0.5] * 1024,  # Dummy 1024-dim vector
                    payload={
                        "entity_id": "func_session_1",  # Same as list1[1]
                        "entity_name": "validate_session",
                        "entity_type": "function",
                        "file_path": "session.py"
                    }
                ),
                score=0.88,
                query="session",
                search_type="semantic",
                rank=1,
                total_results=3
            ),
            CoreSearchResult(
                point=QdrantPoint(
                    id=6,
                    vector=[0.6] * 1024,  # Dummy 1024-dim vector
                    payload={
                        "entity_id": "func_session_2",
                        "entity_name": "create_session",
                        "entity_type": "function",
                        "file_path": "session.py"
                    }
                ),
                score=0.65,
                query="session",
                search_type="semantic",
                rank=2,
                total_results=3
            ),
            CoreSearchResult(
                point=QdrantPoint(
                    id=7,
                    vector=[0.7] * 1024,  # Dummy 1024-dim vector
                    payload={
                        "entity_id": "func_auth_1",  # Same as list1[0]
                        "entity_name": "authenticate_user",
                        "entity_type": "function",
                        "file_path": "auth.py"
                    }
                ),
                score=0.45,
                query="session",
                search_type="semantic",
                rank=3,
                total_results=3
            )
        ]
        
        # List 3: Another search with partial overlap
        list3 = [
            CoreSearchResult(
                point=QdrantPoint(
                    id=8,
                    vector=[0.8] * 1024,  # Dummy 1024-dim vector
                    payload={
                        "entity_id": "func_validate_1",
                        "entity_name": "validate_token",
                        "entity_type": "function",
                        "file_path": "auth.py"
                    }
                ),
                score=0.70,
                query="validate",
                search_type="hybrid",
                rank=1,
                total_results=2
            ),
            CoreSearchResult(
                point=QdrantPoint(
                    id=9,
                    vector=[0.9] * 1024,  # Dummy 1024-dim vector
                    payload={
                        "entity_id": "func_session_1",  # Same as list1[1] and list2[0]
                        "entity_name": "validate_session",
                        "entity_type": "function",
                        "file_path": "session.py"
                    }
                ),
                score=0.55,
                query="validate",
                search_type="hybrid",
                rank=2,
                total_results=2
            )
        ]
        
        # Test multi-list RRF fusion
        fused_results = search_executor._fuse_multi_lists_rrf([list1, list2, list3], k=60)
        
        # Verify fusion results
        assert len(fused_results) == 6, "Should have 6 unique entities after fusion"
        
        # Get entity IDs in fused order
        fused_ids = [r.point.payload["entity_id"] for r in fused_results]
        
        # func_session_1 appears in all 3 lists, should rank highest
        assert fused_ids[0] == "func_session_1", \
            "Entity appearing in all lists should rank first"
        
        # func_auth_1 appears in 2 lists (ranks 1 and 3), should rank high
        assert "func_auth_1" in fused_ids[:3], \
            "Entity appearing in multiple lists should rank high"
        
        # Verify RRF scores
        for i, result in enumerate(fused_results):
            # Scores should be in [0, 1]
            assert 0.0 <= result.score <= 1.0, \
                f"Fused score {result.score} out of range"
            
            # Scores should be descending
            if i > 0:
                assert result.score <= fused_results[i-1].score, \
                    "Scores should be in descending order"
            
            # Verify rank is correct
            assert result.rank == i + 1, f"Rank should be {i+1}, got {result.rank}"
        
        # Verify top result has high score (appears in all lists)
        assert fused_results[0].score > 0.7, \
            "Top result (in all lists) should have high fused score"
        
        logger.info(f"✅ Direct RRF fusion test: {len(fused_results)} unique entities")
        logger.info(f"Top 3 fused entities: {fused_ids[:3]}")
        
        # Cleanup
        await connection_manager.disconnect()
    
    @pytest.mark.asyncio
    async def test_context_snippet_generation(self):
        """Test search context snippet generation for Claude"""
        config = MCPServerConfig(
            project_path=Path("."),
            qdrant_url="http://localhost:6334"
        )
        
        connection_manager = QdrantConnectionManager(config)
        search_executor = SearchExecutor(connection_manager, config)
        
        # Import QdrantPoint for proper object creation
        from core.models.storage import QdrantPoint
        
        results = [
            CoreSearchResult(
                point=QdrantPoint(
                    id=1,
                    vector=[0.1] * 1024,  # Dummy 1024-dim vector
                    payload={
                        "entity_id": "func1",
                        "entity_name": "authenticate_user",
                        "entity_type": "function",
                        "file_path": "auth.py"
                    }
                ),
                score=0.92,
                query="test",
                search_type="hybrid",
                rank=1,
                total_results=5
            ),
            CoreSearchResult(
                point=QdrantPoint(
                    id=2,
                    vector=[0.2] * 1024,  # Dummy 1024-dim vector
                    payload={
                        "entity_id": "class1",
                        "entity_name": "SessionManager",
                        "entity_type": "class",
                        "file_path": "session.py"
                    }
                ),
                score=0.78,
                query="test",
                search_type="hybrid",
                rank=2,
                total_results=5
            ),
            CoreSearchResult(
                point=QdrantPoint(
                    id=3,
                    vector=[0.3] * 1024,  # Dummy 1024-dim vector
                    payload={
                        "entity_id": "func2",
                        "entity_name": "validate_session",
                        "entity_type": "function",
                        "file_path": "session.py"
                    }
                ),
                score=0.65,
                query="test",
                search_type="hybrid",
                rank=3,
                total_results=5
            ),
            CoreSearchResult(
                point=QdrantPoint(
                    id=4,
                    vector=[0.4] * 1024,  # Dummy 1024-dim vector
                    payload={
                        "entity_id": "func3",
                        "entity_name": "hash_password",
                        "entity_type": "function",
                        "file_path": "utils.py"
                    }
                ),
                score=0.52,
                query="test",
                search_type="hybrid",
                rank=4,
                total_results=5
            ),
            CoreSearchResult(
                point=QdrantPoint(
                    id=5,
                    vector=[0.5] * 1024,  # Dummy 1024-dim vector
                    payload={
                        "entity_id": "const1",
                        "entity_name": "MAX_LOGIN_ATTEMPTS",
                        "entity_type": "constant",
                        "file_path": "config.py"
                    }
                ),
                score=0.38,
                query="test",
                search_type="hybrid",
                rank=5,
                total_results=5
            )
        ]
        
        # Test snippet generation with limit
        snippet = search_executor._build_search_context_snippet(
            "authentication query", results, max_results=3
        )
        
        # Verify snippet content
        assert "Found 5 results" in snippet
        assert "authentication query" in snippet
        assert "authenticate_user" in snippet
        assert "SessionManager" in snippet
        assert "validate_session" in snippet
        assert "... and 2 more results" in snippet
        assert "Entity types found:" in snippet
        assert "function: 3" in snippet
        assert "class: 1" in snippet
        assert "constant: 1" in snippet
        
        # Test with empty results
        empty_snippet = search_executor._build_search_context_snippet("test", [])
        assert empty_snippet == "No results found."
        
        # Test with all results shown
        full_snippet = search_executor._build_search_context_snippet(
            "test", results[:2], max_results=10
        )
        assert "... and" not in full_snippet  # No truncation message
        
        logger.info("✅ Context snippet generation test passed")
        
        # Cleanup
        await connection_manager.disconnect()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--log-cli-level=INFO"])