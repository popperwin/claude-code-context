"""
Tests for CollectionStateAnalyzer entity-aware state detection.

Validates entity count analysis, collection health scoring, state detection,
and entity metadata extraction for scan mode selection in HybridIndexer workflow.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import List, Dict, Any

from core.indexer.state_analyzer import (
    CollectionStateAnalyzer, CollectionState, EntityMetadata, CollectionStateInfo
)
from core.storage.client import HybridQdrantClient
from core.models.entities import Entity, EntityType, SourceLocation


class TestCollectionStateAnalyzer:
    """Test suite for CollectionStateAnalyzer entity-aware functionality."""
    
    @pytest.fixture
    def temp_project_dir(self):
        """Create temporary project directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def mock_qdrant_client(self):
        """Create mock Qdrant client."""
        client = Mock(spec=HybridQdrantClient)
        client.get_collection_info = AsyncMock(return_value={"points_count": 0})
        client.search_payload = AsyncMock(return_value=[])
        return client
    
    @pytest.fixture
    def state_analyzer(self, mock_qdrant_client):
        """Create CollectionStateAnalyzer instance."""
        return CollectionStateAnalyzer(
            storage_client=mock_qdrant_client,
            staleness_threshold_hours=24,
            min_health_score=0.7,
            max_analysis_time_ms=5000.0
        )
    
    def create_test_entity(
        self, 
        name: str = "test_function",
        file_path: Path = None,
        entity_type: EntityType = EntityType.FUNCTION,
        start_line: int = 10
    ) -> Entity:
        """Create a test entity."""
        if file_path is None:
            file_path = Path("/test/file.py")
        
        location = SourceLocation(
            file_path=file_path,
            start_line=start_line,
            start_column=4,
            end_line=start_line + 5,
            end_column=14,
            start_byte=start_line * 80 + 4,
            end_byte=(start_line + 5) * 80 + 14
        )
        
        return Entity(
            id=f"file::{file_path}::{entity_type.value}::{name}::{start_line}",
            name=name,
            qualified_name=name,
            entity_type=entity_type,
            location=location,
            source_code=f"def {name}():\\n    pass",
            docstring=f"Test {name} function",
            signature=f"{name}()" if entity_type == EntityType.FUNCTION else None
        )

    @pytest.mark.asyncio
    async def test_analyze_collection_state_empty_collection(
        self, state_analyzer, mock_qdrant_client, temp_project_dir
    ):
        """Test analysis of empty collection."""
        # Mock empty collection
        mock_qdrant_client.get_collection_info.return_value = None
        
        # Analyze collection state
        result = await state_analyzer.analyze_collection_state("empty-collection")
        
        # Should detect empty state
        assert result.state == CollectionState.EMPTY
        assert result.collection_exists is False
        assert result.entity_metadata.total_entities == 0
        assert result.health_score == 0.0
        assert "full scan" in str(result.recommendations)
        assert result.analysis_duration_ms > 0
    
    @pytest.mark.asyncio
    async def test_analyze_collection_state_healthy_collection(
        self, state_analyzer, mock_qdrant_client, temp_project_dir
    ):
        """Test analysis of healthy collection with diverse entities."""
        # Mock collection with entities
        mock_qdrant_client.get_collection_info.return_value = {
            "points_count": 150,
            "optimizer_status": "ready"
        }
        
        # Mock search results with diverse entities
        current_time = datetime.now()
        mock_entities = [
            {"point": {"payload": {
                "entity_type": "function",
                "file_path": str(temp_project_dir / "file1.py"),
                "indexed_at": current_time.isoformat()
            }}},
            {"point": {"payload": {
                "entity_type": "class", 
                "file_path": str(temp_project_dir / "file2.py"),
                "indexed_at": (current_time - timedelta(hours=1)).isoformat()
            }}},
            {"point": {"payload": {
                "entity_type": "variable",
                "file_path": str(temp_project_dir / "file3.py"),
                "indexed_at": (current_time - timedelta(hours=2)).isoformat()
            }}}
        ]
        
        mock_qdrant_client.search_payload.return_value = mock_entities
        
        # Analyze collection state
        result = await state_analyzer.analyze_collection_state("healthy-collection")
        
        # Should detect healthy state
        assert result.state == CollectionState.HEALTHY
        assert result.collection_exists is True
        assert result.entity_metadata.total_entities == 150
        assert result.entity_metadata.file_coverage == 3
        assert result.entity_metadata.last_update_time is not None
        assert len(result.entity_metadata.entity_types) == 3
        assert result.health_score > 0.7
        assert "Collection is healthy" in str(result.recommendations)
    
    @pytest.mark.asyncio
    async def test_analyze_collection_state_stale_collection(
        self, state_analyzer, mock_qdrant_client, temp_project_dir
    ):
        """Test analysis of stale collection."""
        # Mock collection with stale entities
        mock_qdrant_client.get_collection_info.return_value = {
            "points_count": 50,
            "optimizer_status": "ready"
        }
        
        # Mock search results with old timestamps
        old_time = datetime.now() - timedelta(hours=48)  # 48 hours old
        mock_entities = [
            {"point": {"payload": {
                "entity_type": "function",
                "file_path": str(temp_project_dir / "old_file.py"),
                "indexed_at": old_time.isoformat()
            }}}
        ]
        
        mock_qdrant_client.search_payload.return_value = mock_entities
        
        # Analyze collection state
        result = await state_analyzer.analyze_collection_state("stale-collection")
        
        # Should detect stale state
        assert result.state == CollectionState.STALE
        assert result.entity_metadata.total_entities == 50
        assert "48.0 hours old" in str(result.issues_found)
        assert "incremental scan" in str(result.recommendations)
        assert result.health_score < 0.7
    
    @pytest.mark.asyncio
    async def test_analyze_collection_state_inconsistent_collection(
        self, state_analyzer, mock_qdrant_client, temp_project_dir
    ):
        """Test analysis of collection with consistency issues."""
        # Mock collection with inconsistent entities
        mock_qdrant_client.get_collection_info.return_value = {
            "points_count": 100,
            "optimizer_status": "ready"
        }
        
        # Mock search results with high unknown entity ratio
        current_time = datetime.now()
        mock_entities = [
            {"point": {"payload": {
                "entity_type": "unknown",
                "file_path": str(temp_project_dir / "bad_file1.py"),
                "indexed_at": current_time.isoformat()
            }}},
            {"point": {"payload": {
                "entity_type": "unknown",
                "file_path": str(temp_project_dir / "bad_file2.py"), 
                "indexed_at": current_time.isoformat()
            }}},
            {"point": {"payload": {
                "entity_type": "function",
                "file_path": str(temp_project_dir / "good_file.py"),
                "indexed_at": current_time.isoformat()
            }}}
        ]
        
        mock_qdrant_client.search_payload.return_value = mock_entities
        
        # Analyze collection state
        result = await state_analyzer.analyze_collection_state("inconsistent-collection")
        
        # Should detect inconsistent state
        assert result.state == CollectionState.INCONSISTENT
        assert "High ratio of unknown entity types" in str(result.issues_found)
        assert "consistency validation" in str(result.recommendations)
        assert result.health_score < 0.7
    
    @pytest.mark.asyncio
    async def test_analyze_collection_state_inaccessible_collection(
        self, state_analyzer, mock_qdrant_client, temp_project_dir
    ):
        """Test analysis when collection is inaccessible."""
        # Mock connection error
        mock_qdrant_client.get_collection_info.side_effect = Exception("Connection failed")
        
        # Analyze collection state
        result = await state_analyzer.analyze_collection_state("inaccessible-collection")
        
        # Should detect inaccessible state
        assert result.state == CollectionState.INACCESSIBLE
        assert result.health_score == 0.0
        assert "Collection is inaccessible" in str(result.recommendations)
        assert "Connection failed" in str(result.issues_found)
    
    @pytest.mark.asyncio
    async def test_get_entity_count_success(self, state_analyzer, mock_qdrant_client):
        """Test successful entity count retrieval."""
        mock_qdrant_client.get_collection_info.return_value = {"points_count": 250}
        
        count = await state_analyzer.get_entity_count("test-collection")
        
        assert count == 250
        mock_qdrant_client.get_collection_info.assert_called_once_with("test-collection")
    
    @pytest.mark.asyncio
    async def test_get_entity_count_no_collection(self, state_analyzer, mock_qdrant_client):
        """Test entity count when collection doesn't exist."""
        mock_qdrant_client.get_collection_info.return_value = None
        
        count = await state_analyzer.get_entity_count("nonexistent-collection")
        
        assert count == 0
    
    @pytest.mark.asyncio
    async def test_get_entity_count_error(self, state_analyzer, mock_qdrant_client):
        """Test entity count with connection error."""
        mock_qdrant_client.get_collection_info.side_effect = Exception("Connection error")
        
        count = await state_analyzer.get_entity_count("error-collection")
        
        assert count == 0
    
    @pytest.mark.asyncio
    async def test_get_last_entity_update_success(self, state_analyzer, mock_qdrant_client):
        """Test successful last entity update retrieval."""
        test_time = datetime.now()
        mock_entities = [
            {"point": {"payload": {"indexed_at": test_time.isoformat()}}},
            {"point": {"payload": {"indexed_at": (test_time - timedelta(hours=1)).isoformat()}}}
        ]
        
        mock_qdrant_client.search_payload.return_value = mock_entities
        
        last_update = await state_analyzer.get_last_entity_update("test-collection")
        
        assert last_update is not None
        assert abs((last_update - test_time).total_seconds()) < 1.0
    
    @pytest.mark.asyncio
    async def test_get_last_entity_update_no_entities(self, state_analyzer, mock_qdrant_client):
        """Test last entity update when no entities exist."""
        mock_qdrant_client.search_payload.return_value = []
        
        last_update = await state_analyzer.get_last_entity_update("empty-collection")
        
        assert last_update is None
    
    @pytest.mark.asyncio
    async def test_get_last_entity_update_invalid_timestamps(
        self, state_analyzer, mock_qdrant_client
    ):
        """Test last entity update with invalid timestamp formats."""
        mock_entities = [
            {"point": {"payload": {"indexed_at": "invalid-timestamp"}}},
            {"point": {"payload": {}}}  # Missing timestamp
        ]
        
        mock_qdrant_client.search_payload.return_value = mock_entities
        
        last_update = await state_analyzer.get_last_entity_update("test-collection")
        
        assert last_update is None
    
    @pytest.mark.asyncio
    async def test_is_collection_healthy_healthy_collection(
        self, state_analyzer, mock_qdrant_client
    ):
        """Test health check for healthy collection."""
        # Mock healthy collection
        mock_qdrant_client.get_collection_info.return_value = {"points_count": 100}
        current_time = datetime.now()
        mock_entities = [
            {"point": {"payload": {
                "entity_type": "function",
                "file_path": "/test/file.py",
                "indexed_at": current_time.isoformat()
            }}}
        ]
        mock_qdrant_client.search_payload.return_value = mock_entities
        
        is_healthy = await state_analyzer.is_collection_healthy("test-collection")
        
        assert is_healthy is True
    
    @pytest.mark.asyncio
    async def test_is_collection_healthy_insufficient_entities(
        self, state_analyzer, mock_qdrant_client
    ):
        """Test health check with insufficient entities."""
        mock_qdrant_client.get_collection_info.return_value = {"points_count": 0}
        
        is_healthy = await state_analyzer.is_collection_healthy(
            "test-collection", min_entities=10
        )
        
        assert is_healthy is False
    
    @pytest.mark.asyncio
    async def test_is_collection_healthy_stale_collection(
        self, state_analyzer, mock_qdrant_client
    ):
        """Test health check for stale collection."""
        # Mock stale collection
        mock_qdrant_client.get_collection_info.return_value = {"points_count": 50}
        old_time = datetime.now() - timedelta(hours=48)
        mock_entities = [
            {"point": {"payload": {
                "entity_type": "function",
                "file_path": "/test/file.py",
                "indexed_at": old_time.isoformat()
            }}}
        ]
        mock_qdrant_client.search_payload.return_value = mock_entities
        
        is_healthy = await state_analyzer.is_collection_healthy(
            "test-collection", max_staleness_hours=24
        )
        
        assert is_healthy is False
    
    @pytest.mark.asyncio
    async def test_cache_functionality(self, state_analyzer, mock_qdrant_client):
        """Test analysis result caching."""
        mock_qdrant_client.get_collection_info.return_value = {"points_count": 10}
        mock_qdrant_client.search_payload.return_value = []
        
        # First analysis
        result1 = await state_analyzer.analyze_collection_state("test-collection")
        
        # Second analysis should use cache
        result2 = await state_analyzer.analyze_collection_state("test-collection")
        
        # Should be the same object (cached)
        assert result1 is result2
        assert mock_qdrant_client.get_collection_info.call_count == 1
        
        # Force refresh should bypass cache
        result3 = await state_analyzer.analyze_collection_state(
            "test-collection", force_refresh=True
        )
        
        assert result3 is not result1
        assert mock_qdrant_client.get_collection_info.call_count == 2
    
    def test_clear_analysis_cache(self, state_analyzer):
        """Test cache clearing functionality."""
        # Populate cache
        state_analyzer._analysis_cache["test1"] = Mock()
        state_analyzer._analysis_cache["test2"] = Mock()
        
        # Clear specific collection
        state_analyzer.clear_analysis_cache("test1")
        assert "test1" not in state_analyzer._analysis_cache
        assert "test2" in state_analyzer._analysis_cache
        
        # Clear all cache
        state_analyzer.clear_analysis_cache()
        assert len(state_analyzer._analysis_cache) == 0
    
    def test_get_cache_stats(self, state_analyzer):
        """Test cache statistics retrieval."""
        # Empty cache
        stats = state_analyzer.get_cache_stats()
        assert stats["cached_collections"] == 0
        assert stats["cache_ttl_seconds"] == 300
        
        # Populate cache with mock data
        now = datetime.now()
        mock_state_info = Mock()
        mock_state_info.last_analyzed = now - timedelta(seconds=100)
        state_analyzer._analysis_cache["test"] = mock_state_info
        
        stats = state_analyzer.get_cache_stats()
        assert stats["cached_collections"] == 1
        assert 90 < stats["average_age_seconds"] < 110
        assert stats["cache_hit_eligible"] == 1
    
    @pytest.mark.asyncio
    async def test_entity_metadata_extraction(self, state_analyzer, mock_qdrant_client):
        """Test extraction of entity metadata from search results."""
        # Mock collection with diverse entities
        mock_qdrant_client.get_collection_info.return_value = {"points_count": 100}
        
        current_time = datetime.now()
        mock_entities = [
            {"point": {"payload": {
                "entity_type": "function",
                "file_path": "/project/file1.py",
                "indexed_at": current_time.isoformat()
            }}},
            {"point": {"payload": {
                "entity_type": "function", 
                "file_path": "/project/file1.py",
                "indexed_at": (current_time - timedelta(minutes=30)).isoformat()
            }}},
            {"point": {"payload": {
                "entity_type": "class",
                "file_path": "/project/file2.py",
                "indexed_at": (current_time - timedelta(hours=2)).isoformat()
            }}},
            {"point": {"payload": {
                "entity_type": "variable",
                "file_path": "/project/file3.py",
                "created_at": (current_time - timedelta(hours=5)).isoformat()
            }}}
        ]
        
        mock_qdrant_client.search_payload.return_value = mock_entities
        
        # Analyze collection state
        result = await state_analyzer.analyze_collection_state("metadata-test")
        
        # Verify metadata extraction
        metadata = result.entity_metadata
        assert metadata.total_entities == 100
        assert metadata.file_coverage == 3
        assert metadata.average_entities_per_file > 1.0
        assert len(metadata.entity_types) == 3
        assert metadata.entity_types["function"] == 2
        assert metadata.entity_types["class"] == 1
        assert metadata.entity_types["variable"] == 1
        assert metadata.last_update_time is not None
        assert metadata.oldest_entity_time is not None
        assert metadata.newest_entity_time is not None
    
    @pytest.mark.asyncio 
    async def test_health_score_calculation(self, state_analyzer, mock_qdrant_client):
        """Test health score calculation algorithm."""
        # Test perfect health scenario
        mock_qdrant_client.get_collection_info.return_value = {"points_count": 100}
        
        current_time = datetime.now()
        mock_entities = [
            {"point": {"payload": {
                "entity_type": "function",
                "file_path": "/project/file1.py",
                "indexed_at": current_time.isoformat()
            }}},
            {"point": {"payload": {
                "entity_type": "class",
                "file_path": "/project/file2.py",
                "indexed_at": current_time.isoformat()
            }}},
            {"point": {"payload": {
                "entity_type": "variable",
                "file_path": "/project/file3.py",
                "indexed_at": current_time.isoformat()
            }}},
            {"point": {"payload": {
                "entity_type": "module",
                "file_path": "/project/file4.py",
                "indexed_at": current_time.isoformat()
            }}},
            {"point": {"payload": {
                "entity_type": "interface",
                "file_path": "/project/file5.py",
                "indexed_at": current_time.isoformat()
            }}}
        ]
        
        mock_qdrant_client.search_payload.return_value = mock_entities
        
        result = await state_analyzer.analyze_collection_state("perfect-health")
        
        # Should have high health score (entity count + recency + diversity + coverage)
        assert result.health_score > 0.8
        assert result.state == CollectionState.HEALTHY


class TestCollectionStateAnalyzerEdgeCases:
    """Test edge cases and error conditions for CollectionStateAnalyzer."""
    
    @pytest.fixture
    def state_analyzer(self):
        """Create minimal CollectionStateAnalyzer for edge case testing."""
        mock_client = Mock(spec=HybridQdrantClient)
        mock_client.get_collection_info = AsyncMock(return_value=None)
        mock_client.search_payload = AsyncMock(return_value=[])
        
        return CollectionStateAnalyzer(
            storage_client=mock_client,
            staleness_threshold_hours=24,
            min_health_score=0.7
        )
    
    @pytest.mark.asyncio
    async def test_analyze_with_malformed_entity_data(self, state_analyzer):
        """Test analysis with malformed entity data."""
        # Mock collection with invalid entity data
        state_analyzer.storage_client.get_collection_info.return_value = {"points_count": 3}
        
        malformed_entities = [
            {"point": {"payload": {
                "entity_type": "function",
                "file_path": "/valid/file.py",
                "indexed_at": datetime.now().isoformat()
            }}},
            {"point": {"payload": None}},  # Null payload
            {"point": {}},  # Missing payload
            {},  # Empty entity
            {"point": {"payload": {
                # Missing file_path
                "entity_type": "function",
                "indexed_at": "invalid-timestamp"
            }}}
        ]
        
        state_analyzer.storage_client.search_payload.return_value = malformed_entities
        
        # Should handle malformed data gracefully
        result = await state_analyzer.analyze_collection_state("malformed-collection")
        
        assert result.state in [CollectionState.HEALTHY, CollectionState.INCONSISTENT]
        assert result.entity_metadata.total_entities == 3
        # Should extract what it can from valid entities
        assert result.entity_metadata.file_coverage >= 0
    
    @pytest.mark.asyncio
    async def test_analyze_with_mixed_timestamp_formats(self, state_analyzer):
        """Test analysis with various timestamp formats."""
        state_analyzer.storage_client.get_collection_info.return_value = {"points_count": 4}
        
        current_time = datetime.now()
        mixed_entities = [
            {"point": {"payload": {
                "entity_type": "function",
                "file_path": "/test/file1.py",
                "indexed_at": current_time.isoformat()  # ISO format
            }}},
            {"point": {"payload": {
                "entity_type": "function",
                "file_path": "/test/file2.py",
                "created_at": current_time.isoformat() + "Z"  # ISO with Z
            }}},
            {"point": {"payload": {
                "entity_type": "function",
                "file_path": "/test/file3.py",
                "indexed_at": "not-a-timestamp"  # Invalid format
            }}},
            {"point": {"payload": {
                "entity_type": "function",
                "file_path": "/test/file4.py"
                # No timestamp
            }}}
        ]
        
        state_analyzer.storage_client.search_payload.return_value = mixed_entities
        
        result = await state_analyzer.analyze_collection_state("mixed-timestamps")
        
        # Should handle mixed formats and extract valid timestamps
        assert result.entity_metadata.last_update_time is not None
        assert result.entity_metadata.file_coverage == 4
    
    @pytest.mark.asyncio
    async def test_analyze_with_large_collection(self, state_analyzer):
        """Test analysis with large collection (sampling behavior)."""
        # Mock very large collection
        state_analyzer.storage_client.get_collection_info.return_value = {"points_count": 50000}
        
        # Mock should sample only 1000 entities for analysis
        sample_entities = [
            {"point": {"payload": {
                "entity_type": "function",
                "file_path": f"/test/file{i}.py",
                "indexed_at": datetime.now().isoformat()
            }}}
            for i in range(100)  # Sample subset
        ]
        
        state_analyzer.storage_client.search_payload.return_value = sample_entities
        
        result = await state_analyzer.analyze_collection_state("large-collection")
        
        # Should handle large collections by sampling
        assert result.entity_metadata.total_entities == 50000
        assert result.entity_metadata.file_coverage == 100  # From sample
        # Should still have called search with limit
        state_analyzer.storage_client.search_payload.assert_called_once()
        call_args = state_analyzer.storage_client.search_payload.call_args
        assert call_args[1]["limit"] == 1000  # Should limit sample size
    
    @pytest.mark.asyncio
    async def test_analyze_with_timeout_considerations(self, state_analyzer):
        """Test analysis respects maximum analysis time."""
        # Set very short timeout
        state_analyzer.max_analysis_time_ms = 10.0
        
        state_analyzer.storage_client.get_collection_info.return_value = {"points_count": 100}
        state_analyzer.storage_client.search_payload.return_value = []
        
        result = await state_analyzer.analyze_collection_state("timeout-test")
        
        # Should complete quickly due to simple mock data
        assert result.analysis_duration_ms < 1000  # Should be fast with mocks
        assert result.state == CollectionState.EMPTY  # No entities in search results
    
    @pytest.mark.asyncio
    async def test_recommendations_generation_logic(self, state_analyzer):
        """Test recommendation generation for different scenarios."""
        # Test empty collection recommendations
        state_analyzer.storage_client.get_collection_info.return_value = None
        
        result = await state_analyzer.analyze_collection_state("empty-test")
        assert result.state == CollectionState.EMPTY
        assert any("full scan" in rec.lower() for rec in result.recommendations)
        
        # Test stale collection recommendations
        state_analyzer.storage_client.get_collection_info.return_value = {"points_count": 50}
        old_time = datetime.now() - timedelta(hours=48)
        stale_entities = [
            {"point": {"payload": {
                "entity_type": "function",
                "file_path": "/test/file.py",
                "indexed_at": old_time.isoformat()
            }}}
        ]
        state_analyzer.storage_client.search_payload.return_value = stale_entities
        
        result = await state_analyzer.analyze_collection_state("stale-test")
        assert result.state == CollectionState.STALE
        assert any("incremental scan" in rec.lower() for rec in result.recommendations)
        
        # Test very stale collection (should recommend full rescan)
        very_old_time = datetime.now() - timedelta(hours=168)  # 7 days
        very_stale_entities = [
            {"point": {"payload": {
                "entity_type": "function",
                "file_path": "/test/file.py",
                "indexed_at": very_old_time.isoformat()
            }}}
        ]
        state_analyzer.storage_client.search_payload.return_value = very_stale_entities
        
        result = await state_analyzer.analyze_collection_state("very-stale-test")
        assert result.state == CollectionState.STALE
        assert any("full rescan" in rec.lower() for rec in result.recommendations)
    
    @pytest.mark.asyncio
    async def test_concurrent_analysis_requests(self, state_analyzer):
        """Test handling of concurrent analysis requests."""
        state_analyzer.storage_client.get_collection_info.return_value = {"points_count": 10}
        state_analyzer.storage_client.search_payload.return_value = []
        
        # Launch multiple concurrent analyses
        tasks = [
            state_analyzer.analyze_collection_state("concurrent-test")
            for _ in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should complete successfully
        assert len(results) == 5
        for result in results:
            assert result.state == CollectionState.EMPTY
            assert result.analysis_duration_ms > 0
        
        # Should use cache for subsequent requests
        assert len(state_analyzer._analysis_cache) == 1
    
    def test_state_info_serialization(self, state_analyzer):
        """Test CollectionStateInfo can be properly serialized."""
        # Create a state info with various data types
        metadata = EntityMetadata(
            total_entities=100,
            last_update_time=datetime.now(),
            oldest_entity_time=datetime.now() - timedelta(hours=24),
            newest_entity_time=datetime.now(),
            file_coverage=25,
            entity_types={"function": 50, "class": 30, "variable": 20},
            average_entities_per_file=4.0
        )
        
        state_info = CollectionStateInfo(
            collection_name="test-collection",
            state=CollectionState.HEALTHY,
            entity_metadata=metadata,
            health_score=0.85,
            last_analyzed=datetime.now(),
            analysis_duration_ms=250.5,
            recommendations=["Enable real-time sync"],
            issues_found=[],
            collection_exists=True,
            collection_info={"points_count": 100},
            is_indexing=False
        )
        
        # Should be able to convert to dict (for JSON serialization)
        state_dict = state_info.__dict__
        assert "state" in state_dict
        assert "entity_metadata" in state_dict
        assert "health_score" in state_dict
        assert state_dict["health_score"] == 0.85
        assert len(state_dict["recommendations"]) == 1