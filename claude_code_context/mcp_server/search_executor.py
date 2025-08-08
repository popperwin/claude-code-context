"""
Search executor for MCP server integration.

Bridges MCP search requests with the existing search infrastructure,
providing intelligent search execution with Claude orchestration and result formatting.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from claude_code_context.mcp_server.models import (
    SearchRequest, 
    SearchResponse, 
    SearchResult as MCPSearchResult,
    SearchMode,
    MCPServerConfig
)
from claude_code_context.mcp_server.connection import QdrantConnectionManager
from claude_code_context.mcp_server.orchestrator import (
    ClaudeOrchestrator, 
    OrchestrationContext,
    SearchType,
    SecurityError
)
from claude_code_context.mcp_server.context_builder import ProjectContextBuilder

# Import existing search infrastructure
try:
    from core.search.engine import HybridSearcher, SearchConfig, SearchMode as CoreSearchMode
    from core.storage.client import HybridQdrantClient
    from core.models.storage import SearchResult as CoreSearchResult
    from core.embeddings.stella import StellaEmbedder
    from core.models.config import StellaConfig
    SEARCH_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Search infrastructure not available: {e}")
    SEARCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class SearchUnavailableError(Exception):
    """Raised when search infrastructure is not available"""
    pass


class SearchExecutor:
    """
    Executes search requests using the existing search infrastructure with Claude orchestration.
    
    Features:
    - Integration with HybridSearcher and search engine
    - Claude orchestration for intelligent query analysis
    - Mode translation between MCP and core search modes
    - Result formatting and pagination
    - Error handling and fallback responses
    - Performance metrics tracking
    """
    
    def __init__(self, connection_manager: QdrantConnectionManager, config: MCPServerConfig):
        """Initialize search executor with connection manager and orchestration"""
        self.connection_manager = connection_manager
        self.config = config
        
        # Search infrastructure
        self._embedder: Optional[StellaEmbedder] = None
        self._qdrant_client: Optional[HybridQdrantClient] = None
        self._searcher: Optional[HybridSearcher] = None
        self._initialized = False
        
        # Claude orchestration
        self._orchestrator: Optional[ClaudeOrchestrator] = None
        self._context_builder: Optional[ProjectContextBuilder] = None
        self._project_context: Optional[str] = None
        self._orchestration_enabled = True
        
        # Performance metrics
        self._search_count = 0
        self._total_search_time = 0.0
        self._failed_searches = 0
        self._orchestration_calls = 0
        self._orchestration_time = 0.0
        
        logger.info("Initialized SearchExecutor with Claude orchestration")
    
    async def initialize(self) -> bool:
        """
        Initialize search infrastructure and orchestration components.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if not SEARCH_AVAILABLE:
            logger.warning("Search infrastructure not available - using placeholder mode")
            self._orchestration_enabled = False
            return False
        
        if self._initialized:
            return True
        
        try:
            # Initialize embedder for semantic search with proper config
            embedder_config = StellaConfig(
                model_name="stella_en_400M_v5",
                batch_size=32,
                cache_size=100,
                cache_ttl_seconds=300
            )
            self._embedder = StellaEmbedder(embedder_config)
            model_loaded = await self._embedder.load_model()
            if not model_loaded:
                logger.warning("Failed to load Stella model - semantic search disabled")
                self._embedder = None
            
            # Create Qdrant client using connection manager's client
            if not await self.connection_manager.connect():
                raise SearchUnavailableError("Unable to connect to Qdrant")
            
            # Create hybrid client wrapper
            self._qdrant_client = HybridQdrantClient(
                url=self.config.qdrant_url,
                timeout=self.config.qdrant_timeout
            )
            
            # Connect the client
            await self._qdrant_client.connect()
            
            # Initialize searcher with embedder if available
            if self._embedder:
                # Set embedder in client for semantic search
                self._qdrant_client.embedder = self._embedder
            
            self._searcher = HybridSearcher(self._qdrant_client)
            
            # Initialize Claude orchestration components
            try:
                self._orchestrator = ClaudeOrchestrator(self.config)
                self._context_builder = ProjectContextBuilder(self.config)
                
                # Build project context for Claude
                if self._context_builder.is_valid_project():
                    self._project_context = await self._context_builder.build_project_context()
                    logger.info("✅ Project context built for Claude orchestration")
                else:
                    logger.warning("⚠️  Invalid project path for context building")
                    self._orchestration_enabled = False
                    
            except Exception as e:
                logger.warning(f"⚠️  Claude orchestration initialization failed: {e}")
                self._orchestration_enabled = False
            
            self._initialized = True
            logger.info("✅ Search infrastructure initialized successfully")
            if self._orchestration_enabled:
                logger.info("✅ Claude orchestration enabled")
            else:
                logger.info("⚠️  Claude orchestration disabled - using direct search")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize search infrastructure: {e}")
            self._initialized = False
            return False
    
    async def execute_search(self, request: SearchRequest) -> SearchResponse:
        """
        Execute search request with Claude orchestration and return formatted response.
        
        Args:
            request: MCP search request
            
        Returns:
            MCP search response with results and Claude orchestration info
        """
        start_time = time.time()
        orchestration_start = time.time()
        self._search_count += 1
        claude_calls_made = 0
        
        try:
            # Initialize if needed
            if not self._initialized:
                if not await self.initialize():
                    return await self._create_placeholder_response(request, start_time)
            
            # Determine search strategy using Claude orchestration (if enabled)
            search_mode = request.mode
            optimized_query = request.query
            
            if self._orchestration_enabled and self._orchestrator and request.mode == SearchMode.AUTO:
                try:
                    # Create orchestration context
                    context = OrchestrationContext(
                        project_path=self.config.project_path,
                        query=request.query,
                        iteration=1,
                        max_iterations=1,  # Single-turn for now
                        project_context=self._project_context
                    )
                    
                    # Get strategy from Claude
                    strategy = await self._orchestrator.analyze_query(context)
                    claude_calls_made = 1
                    self._orchestration_calls += 1
                    
                    # Translate strategy to search mode
                    search_mode = self._translate_strategy_to_mode(strategy.search_type)
                    optimized_query = strategy.query
                    
                    logger.info(f"🧠 Claude strategy: {strategy.search_type} -> '{optimized_query}'")
                    logger.debug(f"Claude reasoning: {strategy.reasoning}")
                    
                except Exception as e:
                    logger.warning(f"⚠️  Claude orchestration failed: {e} - using direct search")
                    # Continue with original request parameters
                    pass
            
            orchestration_time = (time.time() - orchestration_start) * 1000
            self._orchestration_time += orchestration_time
            
            # Execute search using optimized parameters
            modified_request = SearchRequest(
                request_id=request.request_id,
                session_id=request.session_id,
                query=optimized_query,
                mode=search_mode,
                limit=request.limit,
                min_score=request.min_score,
                file_types=request.file_types
            )
            
            # Check if we should use placeholder response
            if not SEARCH_AVAILABLE or not self._searcher:
                placeholder_response = await self._create_placeholder_response(modified_request, start_time, claude_calls_made)
                # Add query optimization info if available
                if optimized_query != request.query:
                    placeholder_response.query_optimization = optimized_query
                return placeholder_response
            
            results = await self._execute_core_search(modified_request)
            
            # Convert to MCP format
            mcp_results = await self._convert_results_to_mcp(results, modified_request)
            
            execution_time = (time.time() - start_time) * 1000
            self._total_search_time += execution_time
            
            return SearchResponse(
                request_id=request.request_id,
                session_id=request.session_id,
                results=mcp_results,
                total_found=len(mcp_results),
                execution_time_ms=execution_time,
                search_mode_used=search_mode,
                claude_calls_made=claude_calls_made,
                success=True,
                query_optimization=optimized_query if optimized_query != request.query else None
            )
            
        except Exception as e:
            self._failed_searches += 1
            execution_time = (time.time() - start_time) * 1000
            
            logger.error(f"Search execution failed: {e}")
            
            return SearchResponse(
                request_id=request.request_id,
                session_id=request.session_id,
                results=[],
                total_found=0,
                execution_time_ms=execution_time,
                search_mode_used=request.mode,
                claude_calls_made=claude_calls_made,
                success=False,
                error_message=str(e)
            )
    
    async def _execute_core_search(self, request: SearchRequest) -> List[CoreSearchResult]:
        """Execute search using core search infrastructure"""
        if not self._searcher:
            raise SearchUnavailableError("Search infrastructure not initialized")
        
        # Translate MCP search mode to core search mode
        core_mode = self._translate_search_mode(request.mode)
        
        # Create search configuration
        search_config = SearchConfig(
            mode=core_mode,
            limit=min(request.limit, 100),  # Cap at 100 results
            min_score_threshold=request.min_score,
            include_file_types=request.file_types or []
        )
        
        # Get collection name with proper type suffix for core system
        # The core system stores code entities in collections with "-code" suffix
        # Import CollectionType from core system for robust type handling
        from core.storage.schemas import CollectionType
        collection_name = self.config.get_typed_collection_name(CollectionType.CODE)
        
        # Execute search
        results = await self._searcher.search(
            collection_name=collection_name,
            query=request.query,
            config=search_config
        )
        
        return results
    
    def _translate_strategy_to_mode(self, search_type: SearchType) -> SearchMode:
        """Translate Claude strategy search type to MCP search mode"""
        type_map = {
            SearchType.PAYLOAD: SearchMode.PAYLOAD,
            SearchType.SEMANTIC: SearchMode.SEMANTIC,
            SearchType.HYBRID: SearchMode.HYBRID
        }
        return type_map.get(search_type, SearchMode.HYBRID)
    
    def _translate_search_mode(self, mcp_mode: SearchMode) -> str:
        """Translate MCP search mode to core search mode"""
        if SEARCH_AVAILABLE:
            mode_map = {
                SearchMode.AUTO: CoreSearchMode.AUTO,
                SearchMode.PAYLOAD: CoreSearchMode.PAYLOAD_ONLY,
                SearchMode.SEMANTIC: CoreSearchMode.SEMANTIC_ONLY,
                SearchMode.HYBRID: CoreSearchMode.HYBRID
            }
            return mode_map.get(mcp_mode, CoreSearchMode.HYBRID)
        else:
            # Fallback mode names when search infrastructure unavailable
            mode_map = {
                SearchMode.AUTO: "auto",
                SearchMode.PAYLOAD: "payload",
                SearchMode.SEMANTIC: "semantic",
                SearchMode.HYBRID: "hybrid"
            }
            return mode_map.get(mcp_mode, "hybrid")
    
    async def _convert_results_to_mcp(
        self, 
        core_results: List[CoreSearchResult], 
        request: SearchRequest
    ) -> List[MCPSearchResult]:
        """Convert core search results to MCP format"""
        mcp_results = []
        
        for core_result in core_results:
            # Extract entity information from core result
            point = core_result.point
            payload = point.payload
            
            # Create MCP search result
            mcp_result = MCPSearchResult(
                entity_id=payload.get("entity_id", "unknown"),
                file_path=payload.get("file_path", "unknown"),
                name=payload.get("entity_name", payload.get("name", "unknown")),
                content=payload.get("content", ""),
                docstring=payload.get("docstring"),
                entity_type=payload.get("entity_type", "unknown"),
                language=payload.get("language", "unknown"),
                start_line=payload.get("start_line", 1),
                end_line=payload.get("end_line", 1),
                start_byte=payload.get("start_byte", 0),
                end_byte=payload.get("end_byte", 0),
                relevance_score=core_result.score,
                match_type=self._get_match_type(core_result.search_type)
            )
            
            mcp_results.append(mcp_result)
        
        return mcp_results
    
    def _get_match_type(self, search_type: str) -> str:
        """Get match type string from search type"""
        if SEARCH_AVAILABLE:
            type_map = {
                CoreSearchMode.PAYLOAD_ONLY: "exact",
                CoreSearchMode.SEMANTIC_ONLY: "semantic",
                CoreSearchMode.HYBRID: "hybrid"
            }
            return type_map.get(search_type, "hybrid")
        else:
            # Direct string matching when search infrastructure unavailable
            type_map = {
                "payload": "exact",
                "semantic": "semantic",
                "hybrid": "hybrid"
            }
            return type_map.get(search_type, "hybrid")
    
    async def _create_placeholder_response(
        self, 
        request: SearchRequest, 
        start_time: float,
        claude_calls_made: int = 0
    ) -> SearchResponse:
        """Create placeholder response when search infrastructure unavailable"""
        execution_time = (time.time() - start_time) * 1000
        
        # Create placeholder results that demonstrate the expected format
        placeholder_results = []
        
        if "function" in request.query.lower():
            placeholder_results.append(
                MCPSearchResult(
                    entity_id="placeholder_function_1",
                    file_path="example/code.py",
                    name="example_function",
                    content=f"def example_function():\n    \"\"\"Placeholder function matching '{request.query}'\"\"\"\n    pass",
                    entity_type="function",
                    language="python",
                    start_line=10,
                    end_line=13,
                    start_byte=200,
                    end_byte=280,
                    relevance_score=0.85,
                    match_type="placeholder"
                )
            )
        
        if "class" in request.query.lower():
            placeholder_results.append(
                MCPSearchResult(
                    entity_id="placeholder_class_1",
                    file_path="example/models.py",
                    name="ExampleClass",
                    content=f"class ExampleClass:\n    \"\"\"Placeholder class matching '{request.query}'\"\"\"\n    pass",
                    entity_type="class",
                    language="python",
                    start_line=5,
                    end_line=8,
                    start_byte=100,
                    end_byte=180,
                    relevance_score=0.75,
                    match_type="placeholder"
                )
            )
        
        # If no specific matches, create a generic result
        if not placeholder_results:
            placeholder_results.append(
                MCPSearchResult(
                    entity_id="placeholder_generic_1",
                    file_path="example/search_result.py",
                    name="search_result",
                    content=f"# Search result for: {request.query}\n# This is a placeholder result",
                    entity_type="comment",
                    language="python",
                    start_line=1,
                    end_line=2,
                    start_byte=0,
                    end_byte=50,
                    relevance_score=0.50,
                    match_type="placeholder"
                )
            )
        
        return SearchResponse(
            request_id=request.request_id,
            session_id=request.session_id,
            results=placeholder_results,
            total_found=len(placeholder_results),
            execution_time_ms=execution_time,
            search_mode_used=request.mode,
            claude_calls_made=claude_calls_made,
            success=True,
            warnings=["Search infrastructure not fully initialized - showing placeholder results"]
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Get search executor health status"""
        # Get orchestrator health if available
        orchestrator_health = {}
        if self._orchestrator:
            try:
                orchestrator_health = await self._orchestrator.health_check()
            except Exception as e:
                orchestrator_health = {"error": str(e)}
        
        health = {
            "search_executor": {
                "initialized": self._initialized,
                "search_infrastructure_available": SEARCH_AVAILABLE,
                "embedder_loaded": (
                    self._embedder is not None and 
                    hasattr(self._embedder, '_is_loaded') and 
                    self._embedder._is_loaded
                ) if self._embedder else False,
                "qdrant_client_available": self._qdrant_client is not None,
                "searcher_available": self._searcher is not None,
                "orchestration_enabled": self._orchestration_enabled,
                "orchestrator_available": self._orchestrator is not None,
                "context_builder_available": self._context_builder is not None,
                "project_context_ready": self._project_context is not None
            },
            "performance": {
                "total_searches": self._search_count,
                "failed_searches": self._failed_searches,
                "success_rate_percent": (
                    ((self._search_count - self._failed_searches) / max(self._search_count, 1)) * 100
                ),
                "average_search_time_ms": (
                    self._total_search_time / max(self._search_count, 1)
                    if self._search_count > 0 else None
                ),
                "orchestration_calls": self._orchestration_calls,
                "average_orchestration_time_ms": (
                    self._orchestration_time / max(self._orchestration_calls, 1)
                    if self._orchestration_calls > 0 else None
                )
            },
            "orchestrator": orchestrator_health
        }
        
        return health
    
    async def shutdown(self) -> None:
        """Shutdown search executor and cleanup resources"""
        # Handle each component independently to ensure cleanup even if some fail
        if self._embedder:
            try:
                await self._embedder.unload_model()
            except Exception as e:
                logger.error(f"Error unloading embedder: {e}")
            finally:
                self._embedder = None
        
        if self._qdrant_client:
            try:
                await self._qdrant_client.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting Qdrant client: {e}")
            finally:
                self._qdrant_client = None
        
        # Clean up orchestration components
        if self._orchestrator:
            try:
                # No async cleanup needed for orchestrator currently
                pass
            except Exception as e:
                logger.error(f"Error cleaning up orchestrator: {e}")
            finally:
                self._orchestrator = None
        
        # Always clean up remaining state
        self._searcher = None
        self._context_builder = None
        self._project_context = None
        self._initialized = False
        self._orchestration_enabled = False
        
        logger.info("Search executor shutdown complete")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive search executor metrics"""
        return {
            "search_executor": {
                "initialized": self._initialized,
                "search_infrastructure_available": SEARCH_AVAILABLE,
                "orchestration_enabled": self._orchestration_enabled,
                "config": {
                    "qdrant_url": self.config.qdrant_url,
                    "collection_name": self.config.get_collection_name(),
                    "timeout": self.config.qdrant_timeout,
                    "max_claude_calls": self.config.max_claude_calls,
                    "context_word_limit": self.config.context_word_limit
                }
            },
            "performance": {
                "total_searches": self._search_count,
                "failed_searches": self._failed_searches,
                "success_rate_percent": round(
                    ((self._search_count - self._failed_searches) / max(self._search_count, 1)) * 100, 2
                ),
                "average_search_time_ms": round(
                    self._total_search_time / max(self._search_count, 1), 2
                ) if self._search_count > 0 else None,
                "total_search_time_seconds": round(self._total_search_time / 1000, 2),
                "orchestration_calls": self._orchestration_calls,
                "average_orchestration_time_ms": round(
                    self._orchestration_time / max(self._orchestration_calls, 1), 2
                ) if self._orchestration_calls > 0 else None,
                "total_orchestration_time_seconds": round(self._orchestration_time / 1000, 2)
            },
            "components": {
                "embedder_available": self._embedder is not None,
                "embedder_loaded": (
                    self._embedder._is_loaded 
                    if self._embedder and hasattr(self._embedder, '_is_loaded') 
                    else False
                ),
                "qdrant_client_available": self._qdrant_client is not None,
                "searcher_available": self._searcher is not None,
                "orchestrator_available": self._orchestrator is not None,
                "context_builder_available": self._context_builder is not None,
                "project_context_ready": self._project_context is not None
            }
        }