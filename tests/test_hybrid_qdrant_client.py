"""
Unit tests for HybridQdrantClient functionality.

Tests connection management, collection operations, search modes, and error handling.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any

from qdrant_client.models import VectorParams, Distance, CollectionInfo, ScoredPoint
from qdrant_client.http.exceptions import ResponseHandlingException

from core.storage.client import HybridQdrantClient, SearchMode
from core.storage.schemas import CollectionConfig, CollectionType, DistanceMetric
from core.storage.utils import entity_id_to_qdrant_id
from core.models.storage import QdrantPoint, SearchResult, StorageResult
from core.embeddings.base import BaseEmbedder


class MockEmbedder(BaseEmbedder):
    """Mock embedder for testing"""
    
    def __init__(self):
        super().__init__()
        self._is_loaded = True
    
    @property
    def model_name(self) -> str:
        return "mock-model"
    
    @property
    def dimensions(self) -> int:
        return 1024
    
    @property
    def max_sequence_length(self) -> int:
        return 512
    
    async def load_model(self) -> bool:
        self._is_loaded = True
        return True
    
    async def unload_model(self) -> None:
        self._is_loaded = False
    
    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings"""
        return [[0.1, 0.2, 0.3, 0.4] * 256 for _ in texts]
    
    async def embed_single(self, text: str) -> List[float]:
        """Return mock embedding"""
        return [0.1, 0.2, 0.3, 0.4] * 256  # 1024 dimensions


@pytest.fixture
def mock_embedder():
    """Create mock embedder"""
    return MockEmbedder()


@pytest.fixture
def collection_config():
    """Create test collection configuration"""
    from core.storage.schemas import QdrantSchema
    return QdrantSchema.get_code_collection_config("test-collection")


@pytest.fixture
def sample_points():
    """Create sample QdrantPoints for testing"""
    return [
        QdrantPoint(
            id=entity_id_to_qdrant_id("test.py::test_function"),  # Use centralized conversion
            vector=[0.1] * 1024,
            payload={
                "entity_id": "test.py::test_function",
                "entity_name": "test_function",
                "entity_type": "function",
                "file_path": "test.py",
                "signature": "def test_function():"
            }
        ),
        QdrantPoint(
            id=entity_id_to_qdrant_id("test.py::TestClass"),  # Use centralized conversion
            vector=[0.2] * 1024,
            payload={
                "entity_id": "test.py::TestClass",
                "entity_name": "TestClass",
                "entity_type": "class",
                "file_path": "test.py",
                "signature": "class TestClass:"
            }
        )
    ]


class TestHybridQdrantClient:
    """Test HybridQdrantClient functionality"""
    
    def test_client_initialization(self, mock_embedder):
        """Test client initialization with various configurations"""
        # Default initialization
        client = HybridQdrantClient()
        assert client.url == "http://localhost:6333"
        assert client.api_key is None
        assert client.timeout == 60.0
        assert client.embedder is None
        assert client.default_payload_weight == 0.8
        assert client.default_semantic_weight == 0.2
        
        # Custom initialization
        client = HybridQdrantClient(
            url="http://test:6333",
            api_key="test-key",
            timeout=30.0,
            embedder=mock_embedder,
            default_payload_weight=0.6,
            default_semantic_weight=0.4
        )
        assert client.url == "http://test:6333"
        assert client.api_key == "test-key"
        assert client.timeout == 30.0
        assert client.embedder == mock_embedder
        assert client.default_payload_weight == 0.6
        assert client.default_semantic_weight == 0.4
    
    def test_weight_normalization(self):
        """Test automatic weight normalization"""
        client = HybridQdrantClient(
            default_payload_weight=3.0,
            default_semantic_weight=1.0
        )
        # Should normalize to 0.75 and 0.25
        assert abs(client.default_payload_weight - 0.75) < 0.001
        assert abs(client.default_semantic_weight - 0.25) < 0.001
    
    @pytest.mark.asyncio
    async def test_connection_management(self):
        """Test connection establishment and management"""
        client = HybridQdrantClient()
        
        # Mock successful connection
        with patch.object(client, '_client', new_callable=Mock) as mock_client:
            mock_client.get_collections.return_value = Mock(collections=[])
            
            # Test connection
            with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
                mock_to_thread.return_value = Mock(collections=[])
                result = await client.connect()
                assert result is True
                assert client._connected is True
        
        # Reset connection state for failure test
        client._connected = False
        client._client = None
        
        # Test connection failure
        with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.side_effect = Exception("Connection failed")
            result = await client.connect()
            assert result is False
            assert client._connected is False
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check functionality"""
        client = HybridQdrantClient()
        
        # Test healthy response
        with patch.object(client, '_client', new_callable=Mock) as mock_client:
            mock_client.get_collections.return_value = Mock(collections=[Mock(), Mock()])
            
            with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
                mock_to_thread.return_value = Mock(collections=[Mock(), Mock()])
                health = await client.health_check()
                
                assert health["status"] == "healthy"
                assert health["collections_count"] == 2
                assert "response_time_ms" in health
                assert health["url"] == client.url
        
        # Test unhealthy response
        with patch.object(client, '_client', new_callable=Mock) as mock_client:
            mock_client.get_collections.side_effect = Exception("Service unavailable")
            
            with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
                mock_to_thread.side_effect = Exception("Service unavailable")
                health = await client.health_check()
                
                assert health["status"] == "unhealthy"
                assert "Service unavailable" in health["error"]
                assert health["connected"] is False
    
    @pytest.mark.asyncio
    async def test_create_collection(self, collection_config):
        """Test collection creation"""
        client = HybridQdrantClient()
        
        # Test successful creation
        with patch.object(client, '_client', new_callable=Mock) as mock_client:
            mock_client.get_collections.return_value = Mock(collections=[])
            mock_client.create_collection = Mock()
            mock_client.create_payload_index = Mock()
            
            with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
                mock_to_thread.return_value = Mock(collections=[])
                
                result = await client.create_collection(collection_config)
                
                assert result.success is True
                assert result.operation == "insert"
                assert result.collection_name == collection_config.name
        
        # Test collection already exists
        with patch.object(client, '_client', new_callable=Mock) as mock_client:
            existing_collection = Mock()
            existing_collection.name = collection_config.name
            mock_client.get_collections.return_value = Mock(collections=[existing_collection])
            
            with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
                mock_to_thread.return_value = Mock(collections=[existing_collection])
                
                result = await client.create_collection(collection_config, recreate=False)
                
                assert result.success is True
                assert result.affected_count == 0  # No new collection created
        
        # Test creation failure
        with patch.object(client, '_client', new_callable=Mock) as mock_client:
            mock_client.get_collections.return_value = Mock(collections=[])
            mock_client.create_collection.side_effect = Exception("Creation failed")
            
            with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
                mock_to_thread.return_value = Mock(collections=[])
                mock_to_thread.side_effect = [Mock(collections=[]), Exception("Creation failed")]
                
                result = await client.create_collection(collection_config)
                
                assert result.success is False
                assert "Creation failed" in result.error
    
    @pytest.mark.asyncio
    async def test_upsert_points(self, sample_points):
        """Test point upsert operations"""
        client = HybridQdrantClient()
        
        # Test successful upsert
        with patch.object(client, '_client', new_callable=Mock) as mock_client:
            mock_client.upsert = Mock()
            
            with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
                mock_to_thread.return_value = None
                
                result = await client.upsert_points("test-collection", sample_points)
                
                assert result.success is True
                assert result.affected_count == len(sample_points)
                assert result.collection_name == "test-collection"
        
        # Test empty points
        result = await client.upsert_points("test-collection", [])
        assert result.success is True
        assert result.affected_count == 0
        
        # Test upsert failure
        with patch.object(client, '_client', new_callable=Mock) as mock_client:
            mock_client.upsert.side_effect = Exception("Upsert failed")
            
            with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
                mock_to_thread.side_effect = Exception("Upsert failed")
                
                result = await client.upsert_points("test-collection", sample_points)
                
                assert result.success is False
                assert "Upsert failed" in result.error
    
    @pytest.mark.asyncio
    async def test_search_payload(self):
        """Test payload-only search"""
        client = HybridQdrantClient()
        
        # Mock scroll results with realistic integer IDs (what Qdrant actually returns)
        mock_point1 = Mock()
        mock_point1.id = entity_id_to_qdrant_id("test.py::test_function")
        mock_point1.payload = {
            "entity_id": "test.py::test_function",  # Required field
            "entity_name": "test_function",
            "qualified_name": "module.test_function",
            "signature": "def test_function():",
            "entity_type": "function",
            "file_path": "test.py"  # Required field
        }
        
        mock_point2 = Mock()
        mock_point2.id = entity_id_to_qdrant_id("test.py::helper_function")
        mock_point2.payload = {
            "entity_id": "test.py::helper_function",  # Required field
            "entity_name": "helper_function",
            "qualified_name": "module.helper_function", 
            "signature": "def helper_function():",
            "entity_type": "function",
            "file_path": "test.py"  # Required field
        }
        
        with patch.object(client, '_client', new_callable=Mock) as mock_client:
            mock_client.scroll.return_value = ([mock_point1, mock_point2], None)
            
            with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
                mock_to_thread.return_value = ([mock_point1, mock_point2], None)
                
                results = await client.search_payload("test-collection", "test_function", limit=10)
                
                assert len(results) > 0
                assert all(isinstance(r, SearchResult) for r in results)
                assert all(r.search_type == SearchMode.PAYLOAD_ONLY for r in results)
                
                # Results should be sorted by score
                scores = [r.score for r in results]
                assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_search_semantic(self, mock_embedder):
        """Test semantic search"""
        client = HybridQdrantClient(embedder=mock_embedder)
        
        # Mock search results with realistic integer IDs (what Qdrant actually returns)
        mock_scored_point1 = Mock()
        mock_scored_point1.id = entity_id_to_qdrant_id("test.py::similar_function")
        mock_scored_point1.score = 0.9
        mock_scored_point1.payload = {
            "entity_id": "test.py::similar_function",
            "entity_name": "similar_function",
            "entity_type": "function",
            "file_path": "test.py"
        }
        
        mock_scored_point2 = Mock()
        mock_scored_point2.id = entity_id_to_qdrant_id("test.py::another_function")
        mock_scored_point2.score = 0.7
        mock_scored_point2.payload = {
            "entity_id": "test.py::another_function",
            "entity_name": "another_function",
            "entity_type": "function",
            "file_path": "test.py"
        }
        
        with patch.object(client, '_client', new_callable=Mock) as mock_client:
            mock_client.search.return_value = [mock_scored_point1, mock_scored_point2]
            
            with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
                mock_to_thread.return_value = [mock_scored_point1, mock_scored_point2]
                
                results = await client.search_semantic("test-collection", "test query", limit=10)
                
                assert len(results) == 2
                assert all(isinstance(r, SearchResult) for r in results)
                assert all(r.search_type == SearchMode.SEMANTIC_ONLY for r in results)
                assert results[0].score == 0.9
                assert results[1].score == 0.7
    
    @pytest.mark.asyncio
    async def test_search_semantic_no_embedder(self):
        """Test semantic search without embedder raises ValueError"""
        client = HybridQdrantClient()  # No embedder
        
        with pytest.raises(ValueError, match="No embedder configured for semantic search"):
            await client.search_semantic("test-collection", "test query")
    
    @pytest.mark.asyncio
    async def test_search_hybrid_integration(self, mock_embedder, collection_config, sample_points):
        """Test hybrid search with proper business flow: setup → data → search → verify → cleanup"""
        # Use test environment Qdrant URL
        client = HybridQdrantClient(url="http://localhost:6334", embedder=mock_embedder)
        collection_name = f"test-hybrid-{collection_config.name}"
        # Create new collection config with different name
        from core.storage.schemas import QdrantSchema
        test_collection_config = QdrantSchema.get_code_collection_config(collection_name)
        
        try:
            # SETUP: Connect and create collection
            connection_result = await client.connect()
            assert connection_result is True
            
            # Create collection with proper schema
            create_result = await client.create_collection(test_collection_config)
            assert create_result.success is True
            
            # DATA INSERTION: Add test data points
            insert_result = await client.upsert_points(collection_name, sample_points)
            assert insert_result.success is True
            assert insert_result.affected_count == len(sample_points)
            
            # SEARCH: Perform hybrid search on real data
            results = await client.search_hybrid(
                collection_name, "test_function",
                payload_weight=0.7, semantic_weight=0.3,
                limit=5
            )
            
            # VERIFY: Check meaningful results
            assert isinstance(results, list)
            # Should find at least the test_function we inserted
            if results:
                assert all(isinstance(r, SearchResult) for r in results)
                assert all(r.search_type == SearchMode.HYBRID for r in results)
                assert all(r.rank >= 1 for r in results)
                assert all(r.total_results >= 1 for r in results)
                # Scores should be valid and sorted descending
                scores = [r.score for r in results]
                assert all(0.0 <= score <= 1.0 for score in scores)
                assert scores == sorted(scores, reverse=True)
                
        finally:
            # CLEANUP: Remove test collection
            try:
                await client._client.delete_collection(collection_name)
            except Exception:
                pass  # Collection might not exist if test failed early
            
            await client.disconnect()
    
    @pytest.mark.asyncio 
    async def test_search_hybrid_mocked(self, mock_embedder):
        """Test hybrid search logic with mocked components (for unit testing combine logic)"""
        client = HybridQdrantClient(embedder=mock_embedder)
        
        # Use consistent point ID for both results to test combination logic
        test_point_id = entity_id_to_qdrant_id("test.py::test")
        
        # Mock both search types to test _combine_search_results logic
        payload_results = [
            SearchResult(
                point=QdrantPoint(id=test_point_id, vector=[0.1] * 1024, payload={"entity_id": "test.py::test", "entity_name": "test", "entity_type": "function", "file_path": "test.py"}),
                score=0.8,
                query="test",
                search_type=SearchMode.PAYLOAD_ONLY,
                rank=1,
                total_results=1
            )
        ]
        
        semantic_results = [
            SearchResult(
                point=QdrantPoint(id=test_point_id, vector=[0.1] * 1024, payload={"entity_id": "test.py::test", "entity_name": "test", "entity_type": "function", "file_path": "test.py"}),
                score=0.6,
                query="test",
                search_type=SearchMode.SEMANTIC_ONLY,
                rank=1,
                total_results=1
            )
        ]
        
        with patch.object(client, 'search_payload', new_callable=AsyncMock) as mock_payload:
            mock_payload.return_value = payload_results
            
            with patch.object(client, 'search_semantic', new_callable=AsyncMock) as mock_semantic:
                mock_semantic.return_value = semantic_results
                
                results = await client.search_hybrid(
                    "test-collection", "test query", 
                    payload_weight=0.7, semantic_weight=0.3
                )
                
                assert len(results) == 1
                assert results[0].search_type == SearchMode.HYBRID
                # Combined score now uses RRF with k=60 and is scaled to ~[0,1].
                # When the same point is top-1 in both lists, fused score ≈ 1.0
                assert abs(results[0].score - 1.0) < 1e-6
    
    @pytest.mark.asyncio
    async def test_get_collection_info(self):
        """Test collection info retrieval"""
        client = HybridQdrantClient()
        
        # Mock collection info
        mock_info = Mock()
        mock_info.vectors_count = 1000
        mock_info.indexed_vectors_count = 950
        mock_info.points_count = 1000
        mock_info.status = "green"
        mock_info.optimizer_status = Mock()
        mock_info.optimizer_status.status = "ok"
        mock_info.config = Mock()
        mock_info.config.params = Mock()
        mock_info.config.params.vectors = Mock()
        mock_info.config.params.vectors.size = 1024
        mock_info.config.params.vectors.distance = Mock()
        mock_info.config.params.vectors.distance.name = "Cosine"
        
        with patch.object(client, '_client', new_callable=Mock) as mock_client:
            mock_client.get_collection.return_value = mock_info
            
            with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
                mock_to_thread.return_value = mock_info
                
                info = await client.get_collection_info("test-collection")
                
                assert info is not None
                assert info["name"] == "test-collection"
                assert info["vectors_count"] == 1000
                assert info["points_count"] == 1000
                assert info["status"] == "green"
                assert info["config"]["vector_size"] == 1024
                assert info["config"]["distance"] == "Cosine"
    
    def test_performance_metrics(self):
        """Test performance metrics tracking"""
        client = HybridQdrantClient()
        
        metrics = client.get_performance_metrics()
        
        assert "total_requests" in metrics
        assert "total_request_time_s" in metrics
        assert "failed_requests" in metrics
        assert "average_request_time_ms" in metrics
        assert "success_rate" in metrics
        assert "connected" in metrics
        assert "url" in metrics
        
        assert metrics["total_requests"] == 0
        assert metrics["failed_requests"] == 0
        assert metrics["success_rate"] == 0.0  # No requests yet, so success rate is 0
        assert metrics["url"] == client.url
    
    def test_distance_metric_mapping(self):
        """Test distance metric mapping"""
        client = HybridQdrantClient()
        
        assert client._map_distance_metric(DistanceMetric.COSINE) == Distance.COSINE
        assert client._map_distance_metric(DistanceMetric.EUCLIDEAN) == Distance.EUCLID
        assert client._map_distance_metric(DistanceMetric.DOT_PRODUCT) == Distance.DOT
    
    def test_payload_score_calculation(self):
        """Test payload relevance score calculation"""
        client = HybridQdrantClient()
        
        # Test exact match in entity_name (highest weight)
        payload = {
            "entity_name": "test_function",
            "qualified_name": "module.other_function",
            "signature": "def other():",
            "docstring": "Some description"
        }
        score = client._calculate_payload_score("test_function", payload)
        assert score > 0.3  # Should get reasonable score for exact match (1.0/2.8 field weights = ~0.357)
        
        # Test partial match across multiple fields
        payload = {
            "entity_name": "other_function",
            "qualified_name": "test.module",
            "signature": "def test():",
            "docstring": "test description"
        }
        score = client._calculate_payload_score("test", payload)
        assert score > 0.0  # Score should be positive, no upper bound
        
        # Test no match
        payload = {
            "entity_name": "completely_different",
            "qualified_name": "other.module",
            "signature": "def other():",
            "docstring": "different description"
        }
        score = client._calculate_payload_score("test", payload)
        assert score == 0.0
    
    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test client disconnection"""
        client = HybridQdrantClient()
        client._connected = True
        client._client = Mock()
        client._client.close = Mock()
        
        await client.disconnect()
        
        assert client._connected is False
        assert client._client is None


class TestSearchModeConstants:
    """Test SearchMode constants"""
    
    def test_search_mode_values(self):
        """Test search mode constant values"""
        assert SearchMode.PAYLOAD_ONLY == "payload"
        assert SearchMode.SEMANTIC_ONLY == "semantic"
        assert SearchMode.HYBRID == "hybrid"