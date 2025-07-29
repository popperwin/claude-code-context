"""
Hybrid Qdrant client for claude-code-context.

Provides combined payload and vector search with configurable weights,
collection management, and performance optimization.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, Filter, FieldCondition, 
    MatchValue, MatchText, SearchParams, WithPayloadInterface, ScoredPoint,
    CreateCollection, CollectionInfo, CountRequest, SearchRequest,
    TextIndexParams, TokenizerType, PayloadSchemaType
)
from qdrant_client.http.exceptions import ResponseHandlingException

from .schemas import (
    CollectionConfig, CollectionType, CollectionManager, 
    QdrantSchema, DistanceMetric
)
from ..models.storage import (
    QdrantPoint, SearchResult, StorageResult, OperationStatus
)
from ..embeddings.base import BaseEmbedder

logger = logging.getLogger(__name__)


class SearchMode:
    """Search mode constants"""
    PAYLOAD_ONLY = "payload"
    SEMANTIC_ONLY = "semantic" 
    HYBRID = "hybrid"


class HybridQdrantClient:
    """
    Hybrid Qdrant client supporting payload, semantic, and combined search.
    
    Features:
    - Automatic collection creation with optimized schemas
    - Hybrid search with configurable payload/semantic weights
    - Batch operations with performance tracking
    - Connection management with retries
    - Comprehensive error handling
    """
    
    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        embedder: Optional[BaseEmbedder] = None,
        default_payload_weight: float = 0.8,
        default_semantic_weight: float = 0.2
    ):
        """
        Initialize hybrid Qdrant client.
        
        Args:
            url: Qdrant server URL
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            embedder: Embedder for semantic search
            default_payload_weight: Default weight for payload search results
            default_semantic_weight: Default weight for semantic search results
        """
        self.url = url
        self.api_key = api_key
        self.timeout = timeout
        self.embedder = embedder
        
        # Search configuration
        self.default_payload_weight = default_payload_weight
        self.default_semantic_weight = default_semantic_weight
        
        # Normalize weights
        total_weight = self.default_payload_weight + self.default_semantic_weight
        if total_weight > 0:
            self.default_payload_weight /= total_weight
            self.default_semantic_weight /= total_weight
        
        # Initialize client
        self._client: Optional[QdrantClient] = None
        self._connection_lock = asyncio.Lock()
        self._connected = False
        
        # Performance tracking
        self._total_requests = 0
        self._total_request_time = 0.0
        self._failed_requests = 0
        
        logger.info(f"Initialized HybridQdrantClient: {url}")
    
    @property
    def client(self) -> QdrantClient:
        """Get Qdrant client instance"""
        if self._client is None:
            self._client = QdrantClient(
                url=self.url,
                api_key=self.api_key,
                timeout=self.timeout
            )
        return self._client
    
    async def connect(self) -> bool:
        """
        Establish connection to Qdrant server.
        
        Returns:
            True if connection successful, False otherwise
        """
        async with self._connection_lock:
            if self._connected:
                return True
            
            try:
                # Test connection
                start_time = time.time()
                collections = await asyncio.to_thread(self.client.get_collections)
                elapsed = time.time() - start_time
                
                self._connected = True
                logger.info(f"Connected to Qdrant in {elapsed:.3f}s")
                return True
                
            except Exception as e:
                logger.error(f"Failed to connect to Qdrant: {e}")
                self._connected = False
                return False
    
    async def disconnect(self) -> None:
        """Disconnect from Qdrant server"""
        async with self._connection_lock:
            if self._client:
                try:
                    self._client.close()
                except:
                    pass
                self._client = None
            
            self._connected = False
            logger.info("Disconnected from Qdrant")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check Qdrant server health.
        
        Returns:
            Health status information
        """
        try:
            start_time = time.time()
            
            # Get collections to test basic functionality
            collections = await asyncio.to_thread(self.client.get_collections)
            
            elapsed = time.time() - start_time
            
            return {
                "status": "healthy",
                "response_time_ms": elapsed * 1000,
                "collections_count": len(collections.collections),
                "connected": self._connected,
                "url": self.url
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "connected": False,
                "url": self.url
            }
    
    async def create_collection(
        self,
        config: CollectionConfig,
        recreate: bool = False
    ) -> StorageResult:
        """
        Create collection with schema.
        
        Args:
            config: Collection configuration
            recreate: Whether to recreate if collection exists
            
        Returns:
            Storage operation result
        """
        start_time = time.time()
        
        try:
            # Check if collection exists
            collections = await asyncio.to_thread(self.client.get_collections)
            collection_names = [c.name for c in collections.collections]
            
            if config.name in collection_names:
                if not recreate:
                    processing_time = (time.time() - start_time) * 1000
                    return StorageResult.successful_insert(
                        config.name, 0, processing_time
                    )
                else:
                    # Delete existing collection
                    await asyncio.to_thread(
                        self.client.delete_collection, config.name
                    )
                    logger.info(f"Deleted existing collection: {config.name}")
            
            # Create vectors configuration
            vectors_config = VectorParams(
                size=config.vector_size,
                distance=self._map_distance_metric(config.distance_metric),
                on_disk=config.on_disk_payload
            )
            
            # Create collection
            await asyncio.to_thread(
                self.client.create_collection,
                collection_name=config.name,
                vectors_config=vectors_config,
                replication_factor=config.replication_factor,
                write_consistency_factor=config.write_consistency_factor,
                optimizers_config=QdrantSchema.get_qdrant_config_params(config)["optimizers_config"],
                hnsw_config=QdrantSchema.get_qdrant_config_params(config)["hnsw_config"]
            )
            
            # Create payload indexes
            await self._create_payload_indexes(config)
            
            processing_time = (time.time() - start_time) * 1000
            
            logger.info(
                f"Created collection '{config.name}' "
                f"({config.collection_type.value}) in {processing_time:.2f}ms"
            )
            
            return StorageResult.successful_insert(
                config.name, 1, processing_time
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            error_msg = f"Failed to create collection {config.name}: {e}"
            logger.error(error_msg)
            
            return StorageResult.failed_operation(
                "create_collection", config.name, error_msg, processing_time
            )
    
    async def _create_payload_indexes(self, config: CollectionConfig) -> None:
        """Create payload indexes for collection with full-text search support"""
        for index_config in config.payload_indexes:
            try:
                if index_config.field_type == "text":
                    # Create full-text index for text fields to support tiered matching
                    # PREFIX tokenizer creates prefixes for each word for better matching
                    text_index_params = TextIndexParams(
                        type="text",  # Required field for Qdrant
                        tokenizer=TokenizerType.PREFIX,  # Enables prefix matching
                        min_token_len=2,
                        max_token_len=20,
                        lowercase=True
                    )
                    
                    await asyncio.to_thread(
                        self.client.create_payload_index,
                        collection_name=config.name,
                        field_name=index_config.field_name,
                        field_schema=PayloadSchemaType.TEXT,
                        field_index_params=text_index_params
                    )
                else:
                    # Create regular index for non-text fields
                    schema_type_map = {
                        'keyword': PayloadSchemaType.KEYWORD,
                        'integer': PayloadSchemaType.INTEGER,
                        'float': PayloadSchemaType.FLOAT,
                        'bool': PayloadSchemaType.BOOL,
                        'geo': PayloadSchemaType.GEO
                    }
                    
                    schema_type = schema_type_map.get(index_config.field_type, PayloadSchemaType.KEYWORD)
                    
                    await asyncio.to_thread(
                        self.client.create_payload_index,
                        collection_name=config.name,
                        field_name=index_config.field_name,
                        field_schema=schema_type
                    )
                
                logger.debug(
                    f"Created {index_config.field_type} index on {config.name}.{index_config.field_name}"
                )
                
            except Exception as e:
                logger.warning(
                    f"Failed to create index on {config.name}.{index_config.field_name}: {e}"
                )
    
    def _map_distance_metric(self, metric: DistanceMetric) -> Distance:
        """Map distance metric enum to Qdrant Distance"""
        mapping = {
            DistanceMetric.COSINE: Distance.COSINE,
            DistanceMetric.EUCLIDEAN: Distance.EUCLID,
            DistanceMetric.DOT_PRODUCT: Distance.DOT
        }
        return mapping[metric]
    
    async def upsert_points(
        self,
        collection_name: str,
        points: List[QdrantPoint],
        batch_size: int = 100
    ) -> StorageResult:
        """
        Upsert points to collection with batching.
        
        Args:
            collection_name: Target collection name
            points: Points to upsert
            batch_size: Batch size for upsert operations
            
        Returns:
            Storage operation result
        """
        start_time = time.time()
        total_points = len(points)
        
        if not points:
            return StorageResult.successful_insert(collection_name, 0, 0)
        
        try:
            successful_count = 0
            
            # Process in batches
            for i in range(0, total_points, batch_size):
                batch = points[i:i + batch_size]
                
                # Convert to Qdrant format
                qdrant_points = []
                for point in batch:
                    # Convert string ID to integer for Qdrant
                    qdrant_id = int(point.id) if point.id.lstrip('-').isdigit() else point.id
                    qdrant_points.append(PointStruct(
                        id=qdrant_id,
                        vector=point.vector,
                        payload=point.payload
                    ))
                
                # Upsert batch
                await asyncio.to_thread(
                    self.client.upsert,
                    collection_name=collection_name,
                    points=qdrant_points
                )
                
                successful_count += len(batch)
                
                logger.debug(
                    f"Upserted batch {i//batch_size + 1}: "
                    f"{len(batch)} points to {collection_name}"
                )
            
            processing_time = (time.time() - start_time) * 1000
            
            logger.info(
                f"Upserted {successful_count}/{total_points} points "
                f"to {collection_name} in {processing_time:.2f}ms"
            )
            
            return StorageResult.successful_insert(
                collection_name, successful_count, processing_time
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            error_msg = f"Failed to upsert points to {collection_name}: {e}"
            logger.error(error_msg)
            
            return StorageResult.failed_operation(
                "upsert", collection_name, error_msg, processing_time,
                error_details={"total_points": total_points}
            )
    
    async def search_payload(
        self,
        collection_name: str,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search collection using payload fields.
        
        Args:
            collection_name: Collection to search
            query: Search query
            limit: Maximum results to return
            filters: Additional filters
            
        Returns:
            List of search results
        """
        start_time = time.time()
        
        try:
            # Build search filters
            search_filter = self._build_payload_filter(query, filters)
            
            # Use scroll for payload-only search
            scroll_result = await asyncio.to_thread(
                self.client.scroll,
                collection_name=collection_name,
                scroll_filter=search_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            # Convert results
            results = []
            for i, point in enumerate(scroll_result[0]):  # scroll returns (points, next_page_offset)
                qdrant_point = QdrantPoint(
                    id=str(point.id),
                    vector=[0.0] * 1024,  # Dummy vector for payload search (validation requirement)
                    payload=point.payload or {}
                )
                
                # Calculate relevance score based on text matching
                score = self._calculate_payload_score(query, point.payload)
                
                result = SearchResult(
                    point=qdrant_point,
                    score=score,
                    query=query,
                    search_type=SearchMode.PAYLOAD_ONLY,
                    rank=i + 1,
                    total_results=len(scroll_result[0]),
                    keyword_score=score
                )
                results.append(result)
            
            # Sort by score descending
            results.sort(key=lambda r: r.score, reverse=True)
            
            processing_time = (time.time() - start_time) * 1000
            logger.debug(
                f"Payload search in {collection_name}: "
                f"{len(results)} results in {processing_time:.2f}ms"
            )
            
            return results[:limit]
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Payload search failed: {e}")
            return []
    
    async def search_semantic(
        self,
        collection_name: str,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search collection using semantic similarity.
        
        Args:
            collection_name: Collection to search
            query: Search query
            limit: Maximum results to return
            filters: Additional filters
            
        Returns:
            List of search results
        """
        if not self.embedder:
            logger.warning("No embedder configured for semantic search")
            return []
        
        start_time = time.time()
        
        try:
            # Generate query embedding
            query_embedding = await self.embedder.embed_single(query)
            
            # Build additional filters
            search_filter = None
            if filters:
                search_filter = self._build_additional_filters(filters)
            
            # Perform vector search
            search_results = await asyncio.to_thread(
                self.client.search,
                collection_name=collection_name,
                query_vector=query_embedding,
                query_filter=search_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            # Convert results
            results = []
            for i, scored_point in enumerate(search_results):
                qdrant_point = QdrantPoint(
                    id=str(scored_point.id),
                    vector=[0.0] * 1024,  # Dummy vector for search results (validation requirement)
                    payload=scored_point.payload or {}
                )
                
                result = SearchResult(
                    point=qdrant_point,
                    score=scored_point.score,
                    query=query,
                    search_type=SearchMode.SEMANTIC_ONLY,
                    rank=i + 1,
                    total_results=len(search_results),
                    semantic_score=scored_point.score
                )
                results.append(result)
            
            processing_time = (time.time() - start_time) * 1000
            logger.debug(
                f"Semantic search in {collection_name}: "
                f"{len(results)} results in {processing_time:.2f}ms"
            )
            
            return results
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Semantic search failed: {e}")
            return []
    
    async def search_hybrid(
        self,
        collection_name: str,
        query: str,
        limit: int = 50,
        payload_weight: Optional[float] = None,
        semantic_weight: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search collection using hybrid payload + semantic approach.
        
        Args:
            collection_name: Collection to search
            query: Search query
            limit: Maximum results to return
            payload_weight: Weight for payload search results
            semantic_weight: Weight for semantic search results
            filters: Additional filters
            
        Returns:
            List of search results ranked by combined score
        """
        # Use default weights if not specified
        if payload_weight is None:
            payload_weight = self.default_payload_weight
        if semantic_weight is None:
            semantic_weight = self.default_semantic_weight
        
        # Normalize weights
        total_weight = payload_weight + semantic_weight
        if total_weight > 0:
            payload_weight /= total_weight
            semantic_weight /= total_weight
        
        start_time = time.time()
        
        try:
            # Perform both searches concurrently
            payload_task = self.search_payload(collection_name, query, limit, filters)
            semantic_task = self.search_semantic(collection_name, query, limit, filters)
            
            payload_results, semantic_results = await asyncio.gather(
                payload_task, semantic_task, return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(payload_results, Exception):
                logger.warning(f"Payload search failed: {payload_results}")
                payload_results = []
            
            if isinstance(semantic_results, Exception):
                logger.warning(f"Semantic search failed: {semantic_results}")
                semantic_results = []
            
            # Combine and rank results
            combined_results = self._combine_search_results(
                query, payload_results, semantic_results,
                payload_weight, semantic_weight
            )
            
            processing_time = (time.time() - start_time) * 1000
            logger.debug(
                f"Hybrid search in {collection_name}: "
                f"{len(combined_results)} results in {processing_time:.2f}ms "
                f"(payload: {len(payload_results)}, semantic: {len(semantic_results)})"
            )
            
            return combined_results[:limit]
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Hybrid search failed: {e}")
            return []
    
    def _build_payload_filter(
        self,
        query: str,
        additional_filters: Optional[Dict[str, Any]] = None
    ) -> Optional[Filter]:
        """Build intelligent tiered filter for payload search following IDE patterns"""
        conditions = []
        query_length = len(query.strip())
        
        # Tier 1: Exact Match (always active, highest priority)
        # Direct hash-based lookup for perfect matches
        exact_fields = ["entity_name", "qualified_name"]
        for field in exact_fields:
            conditions.append(
                FieldCondition(
                    key=field,
                    match=MatchValue(value=query)
                )
            )
        
        # Tier 2: Prefix Match (3+ chars) - leverages PREFIX tokenizer for indexed search
        # PREFIX tokenizer creates indexed prefixes for efficient word-start matching
        if query_length >= 3:
            prefix_fields = ["entity_name", "qualified_name"] 
            for field in prefix_fields:
                conditions.append(
                    FieldCondition(
                        key=field,
                        match=MatchText(text=query)
                    )
                )
        
        # Tier 3: Unindexed Content Search (5+ chars) - for discovery in content fields
        # Uses unindexed MatchText for true substring matching in larger text fields
        # Performance trade-off: slower but finds relevant content not in names
        if query_length >= 5:
            content_fields = ["signature", "docstring"]
            for field in content_fields:
                conditions.append(
                    FieldCondition(
                        key=field,
                        match=MatchText(text=query)
                    )
                )
        
        # Short queries (1-2 chars): exact match only to prevent noise explosion
        
        # Add additional filters with AND logic
        if additional_filters:
            for field, value in additional_filters.items():
                conditions.append(
                    FieldCondition(
                        key=field,
                        match=MatchValue(value=value)
                    )
                )
        
        if conditions:
            return Filter(should=conditions)  # OR logic for text search tiers
        
        return None
    
    def _build_additional_filters(
        self,
        filters: Dict[str, Any]
    ) -> Optional[Filter]:
        """Build filter from additional filter parameters"""
        conditions = []
        
        for field, value in filters.items():
            conditions.append(
                FieldCondition(
                    key=field,
                    match=MatchValue(value=value)
                )
            )
        
        if conditions:
            return Filter(must=conditions)  # AND logic for additional filters
        
        return None
    
    def _calculate_payload_score(
        self,
        query: str,
        payload: Dict[str, Any]
    ) -> float:
        """Calculate intelligent relevance score for simplified tiered payload search"""
        if not payload:
            return 0.0
        
        query_lower = query.lower().strip()
        score = 0.0
        
        # Field importance weights for three-tier approach
        field_weights = {
            "entity_name": 1.0,         # Primary identifier (Tier 1 & 2)
            "qualified_name": 0.9,      # Full qualified name (Tier 1 & 2)  
            "signature": 0.4,           # Function signature (Tier 3)
            "docstring": 0.2            # Documentation (Tier 3)
        }
        
        # Match type scoring reflecting true IDE search hierarchy
        match_type_scores = {
            "exact": 1.0,       # Tier 1: Perfect match (highest priority)
            "prefix": 0.8,      # Tier 2: Starts with query (high priority)
            "contains": 0.5     # Tier 3: Contains query (lower priority)
        }
        
        # Entity type bonuses (more specific entities ranked higher)
        entity_type_bonus = {
            "class": 0.2,       # Classes are highly specific
            "function": 0.15,   # Functions are specific  
            "method": 0.15,     # Methods are specific
            "variable": 0.05,   # Variables are common
            "constant": 0.1,    # Constants are moderately specific
            "import": 0.02      # Imports are utility
        }
        
        # Calculate base score from field matches (all three tiers)
        for field, field_weight in field_weights.items():
            field_value = payload.get(field, "")
            if not isinstance(field_value, str):
                continue
                
            field_value_lower = field_value.lower()
            
            # Determine match type and calculate score
            match_score = 0.0
            
            if field_value_lower == query_lower:
                # Tier 1: Exact match - highest priority  
                match_score = field_weight * match_type_scores["exact"]
            elif field_value_lower.startswith(query_lower):
                # Tier 2: Prefix match - high priority
                match_score = field_weight * match_type_scores["prefix"]
            elif query_lower in field_value_lower:
                # Tier 3: Contains match - lower priority (for content fields)
                match_score = field_weight * match_type_scores["contains"]
            
            score += match_score
        
        # Add entity type bonus for final ranking
        entity_type = payload.get("entity_type", "").lower()
        if entity_type in entity_type_bonus:
            score += entity_type_bonus[entity_type]
        
        # Normalize to 0-1 range as required by SearchResult validation
        # Updated max for three-tier scoring (max 1.0 + 0.9 + 0.4 + 0.2 + 0.2 = 2.7)
        max_theoretical_score = 2.7
        return min(score / max_theoretical_score, 1.0)
    
    def _combine_search_results(
        self,
        query: str,
        payload_results: List[SearchResult],
        semantic_results: List[SearchResult],
        payload_weight: float,
        semantic_weight: float
    ) -> List[SearchResult]:
        """Combine and rank payload and semantic search results"""
        # Create lookup for semantic scores
        semantic_scores = {result.point.id: result.score for result in semantic_results}
        payload_scores = {result.point.id: result.score for result in payload_results}
        
        # Collect all unique results
        all_ids = set()
        all_ids.update(semantic_scores.keys())
        all_ids.update(payload_scores.keys())
        
        # Create combined results
        combined_results = []
        result_map = {}
        
        # Build result map from both sources
        for result in payload_results + semantic_results:
            if result.point.id not in result_map:
                result_map[result.point.id] = result
        
        # Calculate combined scores
        for point_id in all_ids:
            if point_id not in result_map:
                continue
            
            base_result = result_map[point_id]
            
            # Get individual scores
            payload_score = payload_scores.get(point_id, 0.0)
            semantic_score = semantic_scores.get(point_id, 0.0)
            
            # Calculate weighted combined score
            combined_score = (
                payload_score * payload_weight +
                semantic_score * semantic_weight
            )
            
            # Create temporary result data for sorting (not final SearchResult yet)
            combined_data = {
                'point': base_result.point,
                'score': combined_score,
                'query': query,
                'search_type': SearchMode.HYBRID,
                'relevance_score': combined_score,
                'semantic_score': semantic_score,
                'keyword_score': payload_score
            }
            
            combined_results.append(combined_data)
        
        # Sort by combined score descending
        combined_results.sort(key=lambda r: r['score'], reverse=True)
        
        # Create final immutable SearchResult objects with correct ranks
        final_results = []
        total_count = len(combined_results)
        
        for i, result_data in enumerate(combined_results):
            final_result = SearchResult(
                point=result_data['point'],
                score=result_data['score'],
                query=result_data['query'],
                search_type=result_data['search_type'],
                rank=i + 1,  # Correct rank after sorting
                total_results=total_count,  # Correct total count
                relevance_score=result_data['relevance_score'],
                semantic_score=result_data['semantic_score'],
                keyword_score=result_data['keyword_score']
            )
            final_results.append(final_result)
        
        return final_results
    
    async def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a collection"""
        try:
            info = await asyncio.to_thread(
                self.client.get_collection, collection_name
            )
            
            # Handle OptimizersStatusOneOf enum properly
            optimizer_status = None
            if info.optimizer_status:
                # OptimizersStatusOneOf is an enum-like object, access its value directly
                if hasattr(info.optimizer_status, 'value'):
                    optimizer_status = info.optimizer_status.value
                elif hasattr(info.optimizer_status, 'name'):
                    optimizer_status = info.optimizer_status.name.lower()
                else:
                    # Fallback: convert to string and extract status
                    optimizer_status = str(info.optimizer_status).split('.')[-1].lower()
            
            return {
                "name": collection_name,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "status": info.status,
                "optimizer_status": optimizer_status,
                "config": {
                    "vector_size": info.config.params.vectors.size,
                    "distance": info.config.params.vectors.distance.name
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection info for {collection_name}: {e}")
            return None
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get client performance metrics"""
        return {
            "total_requests": self._total_requests,
            "total_request_time_s": self._total_request_time,
            "failed_requests": self._failed_requests,
            "average_request_time_ms": (
                self._total_request_time / max(1, self._total_requests) * 1000
            ),
            "success_rate": (
                (self._total_requests - self._failed_requests) / max(1, self._total_requests)
            ),
            "connected": self._connected,
            "url": self.url
        }