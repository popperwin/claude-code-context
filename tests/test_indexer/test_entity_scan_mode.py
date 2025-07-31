"""
Tests for EntityScanModeSelector and entity-level scan configuration.

Validates scan mode selection logic, configuration options, and integration
with CollectionStateAnalyzer for optimal indexing strategy selection.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import List, Dict, Any

from core.indexer.scan_mode import (
    EntityScanModeSelector, EntityScanMode, ScanModeDecision
)
from core.indexer.state_analyzer import (
    CollectionStateAnalyzer, CollectionState, EntityMetadata, CollectionStateInfo
)
from core.indexer.hybrid_indexer import IndexingJobConfig
from core.storage.client import HybridQdrantClient
from core.storage.schemas import CollectionType


class TestEntityScanMode:
    """Test suite for EntityScanMode enum and related types."""
    
    def test_entity_scan_mode_values(self):
        """Test EntityScanMode enum values."""
        assert EntityScanMode.FULL_RESCAN.value == "full_rescan"
        assert EntityScanMode.ENTITY_SYNC.value == "entity_sync"
        assert EntityScanMode.SYNC_ONLY.value == "sync_only"
        assert EntityScanMode.AUTO.value == "auto"
    
    def test_scan_mode_decision_creation(self):
        """Test ScanModeDecision dataclass creation."""
        decision = ScanModeDecision(
            selected_mode=EntityScanMode.ENTITY_SYNC,
            confidence=0.8,
            reasoning=["Collection is healthy", "Entity sync recommended"],
            entity_count_estimate=150,
            expected_duration_minutes=2.5,
            performance_tier="fast"
        )
        
        assert decision.selected_mode == EntityScanMode.ENTITY_SYNC
        assert decision.confidence == 0.8
        assert len(decision.reasoning) == 2
        assert decision.entity_count_estimate == 150
        assert decision.expected_duration_minutes == 2.5
        assert decision.performance_tier == "fast"
        assert decision.detected_issues == []  # Default empty list
    
    def test_scan_mode_decision_with_issues(self):
        """Test ScanModeDecision with detected issues."""
        issues = ["Stale entities detected", "Low health score"]
        decision = ScanModeDecision(
            selected_mode=EntityScanMode.FULL_RESCAN,
            confidence=0.9,
            reasoning=["Issues require full rescan"],
            entity_count_estimate=500,
            expected_duration_minutes=10.0,
            performance_tier="slow",
            detected_issues=issues
        )
        
        assert decision.detected_issues == issues
        assert len(decision.detected_issues) == 2


class TestEntityScanModeSelector:
    """Test suite for EntityScanModeSelector functionality."""
    
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
    def mock_state_analyzer(self, mock_qdrant_client):
        """Create mock CollectionStateAnalyzer."""
        analyzer = Mock(spec=CollectionStateAnalyzer)
        analyzer.storage_client = mock_qdrant_client
        analyzer.staleness_threshold_hours = 24
        analyzer.min_health_score = 0.7
        return analyzer
    
    @pytest.fixture
    def scan_mode_selector(self, mock_qdrant_client, mock_state_analyzer):
        """Create EntityScanModeSelector instance."""
        return EntityScanModeSelector(
            storage_client=mock_qdrant_client,
            state_analyzer=mock_state_analyzer,
            performance_threshold_entities=10000,
            staleness_threshold_hours=24,
            health_threshold=0.7
        )
    
    def create_mock_state_info(
        self,
        collection_name: str,
        state: CollectionState,
        entity_count: int = 100,
        health_score: float = 0.8,
        last_update: datetime = None,
        issues: List[str] = None
    ) -> CollectionStateInfo:
        """Create mock CollectionStateInfo for testing."""
        if last_update is None:
            last_update = datetime.now()
        
        metadata = EntityMetadata(
            total_entities=entity_count,
            last_update_time=last_update,
            oldest_entity_time=last_update - timedelta(hours=1),
            newest_entity_time=last_update,
            file_coverage=max(1, entity_count // 10),
            entity_types={"function": entity_count // 2, "class": entity_count // 2},
            average_entities_per_file=2.0
        )
        
        return CollectionStateInfo(
            collection_name=collection_name,
            state=state,
            entity_metadata=metadata,
            health_score=health_score,
            last_analyzed=datetime.now(),
            analysis_duration_ms=100.0,
            recommendations=[],
            issues_found=issues or [],
            collection_exists=(state != CollectionState.EMPTY),
            collection_info={"points_count": entity_count} if entity_count > 0 else None,
            is_indexing=False
        )
    
    @pytest.mark.asyncio
    async def test_selector_initialization(self, mock_qdrant_client):
        """Test EntityScanModeSelector initialization."""
        selector = EntityScanModeSelector(
            storage_client=mock_qdrant_client,
            performance_threshold_entities=5000,
            staleness_threshold_hours=12,
            health_threshold=0.8
        )
        
        assert selector.storage_client == mock_qdrant_client
        assert selector.performance_threshold_entities == 5000
        assert selector.staleness_threshold_hours == 12
        assert selector.health_threshold == 0.8
        assert selector.state_analyzer is not None  # Should create one
    
    @pytest.mark.asyncio
    async def test_select_scan_mode_empty_collection(
        self, scan_mode_selector, mock_state_analyzer, temp_project_dir
    ):
        """Test scan mode selection for empty collection."""
        # Mock empty collection state
        empty_state = self.create_mock_state_info(
            collection_name="empty-collection",
            state=CollectionState.EMPTY,
            entity_count=0,
            health_score=0.0
        )
        mock_state_analyzer.analyze_collection_state = AsyncMock(return_value=empty_state)
        
        # Select scan mode
        decision = await scan_mode_selector.select_scan_mode(
            collection_name="empty-collection",
            project_path=temp_project_dir,
            requested_mode="auto"
        )
        
        # Should recommend full rescan for empty collection
        assert decision.selected_mode == EntityScanMode.FULL_RESCAN
        assert decision.confidence == 1.0
        assert "Collection is empty" in str(decision.reasoning)
        assert decision.entity_count_estimate == 0
    
    @pytest.mark.asyncio
    async def test_select_scan_mode_healthy_collection(
        self, scan_mode_selector, mock_state_analyzer, temp_project_dir
    ):
        """Test scan mode selection for healthy collection."""
        # Mock healthy collection state
        healthy_state = self.create_mock_state_info(
            collection_name="healthy-collection",
            state=CollectionState.HEALTHY,
            entity_count=150,
            health_score=0.9,
            last_update=datetime.now() - timedelta(hours=2)
        )
        mock_state_analyzer.analyze_collection_state = AsyncMock(return_value=healthy_state)
        
        # Select scan mode
        decision = await scan_mode_selector.select_scan_mode(
            collection_name="healthy-collection",
            project_path=temp_project_dir,
            requested_mode="auto"
        )
        
        # Should recommend entity sync for healthy collection
        assert decision.selected_mode == EntityScanMode.ENTITY_SYNC
        assert decision.confidence == 0.8
        assert "Collection is healthy" in str(decision.reasoning)
        assert decision.entity_count_estimate == 150
        assert decision.performance_tier in ["fast", "medium"]
    
    @pytest.mark.asyncio
    async def test_select_scan_mode_very_healthy_recent_collection(
        self, scan_mode_selector, mock_state_analyzer, temp_project_dir
    ):
        """Test scan mode selection for very healthy and recent collection."""
        # Mock very healthy and very recent collection
        very_healthy_state = self.create_mock_state_info(
            collection_name="recent-collection",
            state=CollectionState.HEALTHY,
            entity_count=200,
            health_score=0.95,
            last_update=datetime.now() - timedelta(minutes=30)  # Very recent
        )
        mock_state_analyzer.analyze_collection_state = AsyncMock(return_value=very_healthy_state)
        
        # Select scan mode
        decision = await scan_mode_selector.select_scan_mode(
            collection_name="recent-collection",
            project_path=temp_project_dir,
            requested_mode="auto"
        )
        
        # Should recommend sync only for very healthy recent collection
        assert decision.selected_mode == EntityScanMode.SYNC_ONLY
        assert decision.confidence == 0.9
        assert "very healthy and recent" in str(decision.reasoning)
        assert decision.performance_tier == "fast"
    
    @pytest.mark.asyncio
    async def test_select_scan_mode_stale_collection(
        self, scan_mode_selector, mock_state_analyzer, temp_project_dir
    ):
        """Test scan mode selection for stale collection."""
        # Mock stale collection state
        stale_state = self.create_mock_state_info(
            collection_name="stale-collection",
            state=CollectionState.STALE,
            entity_count=100,
            health_score=0.6,
            last_update=datetime.now() - timedelta(hours=48)
        )
        mock_state_analyzer.analyze_collection_state = AsyncMock(return_value=stale_state)
        
        # Select scan mode
        decision = await scan_mode_selector.select_scan_mode(
            collection_name="stale-collection",
            project_path=temp_project_dir,
            requested_mode="auto"
        )
        
        # Should recommend entity sync for stale collection
        assert decision.selected_mode == EntityScanMode.ENTITY_SYNC
        assert decision.confidence == 0.8
        assert "Collection is stale" in str(decision.reasoning)
        assert decision.entity_count_estimate == 100
    
    @pytest.mark.asyncio
    async def test_select_scan_mode_very_stale_collection(
        self, scan_mode_selector, mock_state_analyzer, temp_project_dir
    ):
        """Test scan mode selection for very stale collection."""
        # Mock very stale collection state (3x staleness threshold)
        very_stale_state = self.create_mock_state_info(
            collection_name="very-stale-collection",
            state=CollectionState.STALE,
            entity_count=200,
            health_score=0.4,
            last_update=datetime.now() - timedelta(hours=72)  # 3x threshold
        )
        mock_state_analyzer.analyze_collection_state = AsyncMock(return_value=very_stale_state)
        
        # Select scan mode
        decision = await scan_mode_selector.select_scan_mode(
            collection_name="very-stale-collection",
            project_path=temp_project_dir,
            requested_mode="auto"
        )
        
        # Should recommend full rescan for very stale collection
        assert decision.selected_mode == EntityScanMode.FULL_RESCAN
        assert decision.confidence == 0.9
        assert "very stale" in str(decision.reasoning)
        assert decision.entity_count_estimate == 200
    
    @pytest.mark.asyncio
    async def test_select_scan_mode_inconsistent_collection(
        self, scan_mode_selector, mock_state_analyzer, temp_project_dir
    ):
        """Test scan mode selection for inconsistent collection."""
        # Mock inconsistent collection state
        inconsistent_state = self.create_mock_state_info(
            collection_name="inconsistent-collection",
            state=CollectionState.INCONSISTENT,
            entity_count=150,
            health_score=0.5,
            issues=["High ratio of unknown entity types", "Parse errors detected"]
        )
        mock_state_analyzer.analyze_collection_state = AsyncMock(return_value=inconsistent_state)
        
        # Select scan mode
        decision = await scan_mode_selector.select_scan_mode(
            collection_name="inconsistent-collection",
            project_path=temp_project_dir,
            requested_mode="auto"
        )
        
        # Should recommend full rescan for inconsistent collection
        assert decision.selected_mode == EntityScanMode.FULL_RESCAN
        assert decision.confidence == 0.85
        assert "consistency issues" in str(decision.reasoning)
        assert decision.entity_count_estimate == 150
        assert len(decision.detected_issues) == 2
    
    @pytest.mark.asyncio
    async def test_select_scan_mode_inaccessible_collection(
        self, scan_mode_selector, mock_state_analyzer, temp_project_dir
    ):
        """Test scan mode selection for inaccessible collection."""
        # Mock inaccessible collection state
        inaccessible_state = self.create_mock_state_info(
            collection_name="inaccessible-collection",
            state=CollectionState.INACCESSIBLE,
            entity_count=0,
            health_score=0.0
        )
        mock_state_analyzer.analyze_collection_state = AsyncMock(return_value=inaccessible_state)
        
        # Select scan mode
        decision = await scan_mode_selector.select_scan_mode(
            collection_name="inaccessible-collection",
            project_path=temp_project_dir,
            requested_mode="auto"
        )
        
        # Should recommend full rescan for inaccessible collection
        assert decision.selected_mode == EntityScanMode.FULL_RESCAN
        assert decision.confidence == 0.7
        assert "inaccessible" in str(decision.reasoning)
        assert len(decision.detected_issues) > 0
    
    @pytest.mark.asyncio
    async def test_select_scan_mode_forced_mode(
        self, scan_mode_selector, temp_project_dir
    ):
        """Test forced scan mode selection."""
        # Select forced mode (should skip analysis)
        decision = await scan_mode_selector.select_scan_mode(
            collection_name="test-collection",
            project_path=temp_project_dir,
            requested_mode="full_rescan",
            force_mode=True
        )
        
        # Should return forced mode without analysis
        assert decision.selected_mode == EntityScanMode.FULL_RESCAN
        assert decision.confidence == 0.5  # Lower confidence for forced modes
        assert "forced by user" in str(decision.reasoning)
        assert "Skipping collection analysis" in str(decision.reasoning)
    
    @pytest.mark.asyncio
    async def test_validate_requested_mode_sync_only_empty_collection(
        self, scan_mode_selector, mock_state_analyzer, temp_project_dir
    ):
        """Test validation of sync-only mode on empty collection."""
        # Mock empty collection
        empty_state = self.create_mock_state_info(
            collection_name="empty-collection",
            state=CollectionState.EMPTY,
            entity_count=0,
            health_score=0.0
        )
        mock_state_analyzer.analyze_collection_state = AsyncMock(return_value=empty_state)
        
        # Request sync-only mode on empty collection
        decision = await scan_mode_selector.select_scan_mode(
            collection_name="empty-collection",
            project_path=temp_project_dir,
            requested_mode="sync_only"
        )
        
        # Should override to full rescan
        assert decision.selected_mode == EntityScanMode.FULL_RESCAN
        assert decision.confidence == 0.9
        assert "Cannot use sync-only mode on empty collection" in str(decision.reasoning)
        assert "Overriding to full rescan" in str(decision.reasoning)
    
    @pytest.mark.asyncio
    async def test_validate_requested_mode_sync_only_unhealthy_collection(
        self, scan_mode_selector, mock_state_analyzer, temp_project_dir
    ):
        """Test validation of sync-only mode on unhealthy collection."""
        # Mock unhealthy collection
        unhealthy_state = self.create_mock_state_info(
            collection_name="unhealthy-collection",
            state=CollectionState.HEALTHY,
            entity_count=100,
            health_score=0.5  # Below threshold
        )
        mock_state_analyzer.analyze_collection_state = AsyncMock(return_value=unhealthy_state)
        
        # Request sync-only mode on unhealthy collection
        decision = await scan_mode_selector.select_scan_mode(
            collection_name="unhealthy-collection",
            project_path=temp_project_dir,
            requested_mode="sync_only"
        )
        
        # Should override to entity sync
        assert decision.selected_mode == EntityScanMode.ENTITY_SYNC
        assert decision.confidence == 0.8
        assert "Collection health too low for sync-only mode" in str(decision.reasoning)
        assert "Recommending entity sync instead" in str(decision.reasoning)
    
    @pytest.mark.asyncio
    async def test_validate_requested_mode_entity_sync_empty_collection(
        self, scan_mode_selector, mock_state_analyzer, temp_project_dir
    ):
        """Test validation of entity sync mode on empty collection."""
        # Mock empty collection
        empty_state = self.create_mock_state_info(
            collection_name="empty-collection",
            state=CollectionState.EMPTY,
            entity_count=0,
            health_score=0.0
        )
        mock_state_analyzer.analyze_collection_state = AsyncMock(return_value=empty_state)
        
        # Request entity sync mode on empty collection
        decision = await scan_mode_selector.select_scan_mode(
            collection_name="empty-collection",
            project_path=temp_project_dir,
            requested_mode="entity_sync"
        )
        
        # Should override to full rescan
        assert decision.selected_mode == EntityScanMode.FULL_RESCAN
        assert decision.confidence == 0.9
        assert "Cannot use entity sync on empty collection" in str(decision.reasoning)
        assert "Overriding to full rescan" in str(decision.reasoning)
    
    @pytest.mark.asyncio
    async def test_validate_requested_mode_valid_request(
        self, scan_mode_selector, mock_state_analyzer, temp_project_dir
    ):
        """Test validation of valid requested mode."""
        # Mock healthy collection
        healthy_state = self.create_mock_state_info(
            collection_name="healthy-collection",
            state=CollectionState.HEALTHY,
            entity_count=150,
            health_score=0.8
        )
        mock_state_analyzer.analyze_collection_state = AsyncMock(return_value=healthy_state)
        
        # Request valid entity sync mode
        decision = await scan_mode_selector.select_scan_mode(
            collection_name="healthy-collection",
            project_path=temp_project_dir,
            requested_mode="entity_sync"
        )
        
        # Should accept the requested mode
        assert decision.selected_mode == EntityScanMode.ENTITY_SYNC
        assert decision.confidence == 0.8
        assert "User-requested mode 'entity_sync' is appropriate" in str(decision.reasoning)
    
    @pytest.mark.asyncio
    async def test_analysis_failure_fallback(
        self, scan_mode_selector, mock_state_analyzer, temp_project_dir
    ):
        """Test fallback behavior when collection analysis fails."""
        # Mock analysis failure
        mock_state_analyzer.analyze_collection_state = AsyncMock(
            side_effect=Exception("Connection failed")
        )
        
        # Select scan mode with analysis failure
        decision = await scan_mode_selector.select_scan_mode(
            collection_name="error-collection",
            project_path=temp_project_dir,
            requested_mode="auto"
        )
        
        # Should fallback to full rescan for safety
        assert decision.selected_mode == EntityScanMode.FULL_RESCAN
        assert decision.confidence == 0.5
        assert "Collection analysis failed" in str(decision.reasoning)
        assert "Defaulting to full rescan for safety" in str(decision.reasoning)
    
    def test_get_mode_recommendations_speed_priority(self, scan_mode_selector):
        """Test mode recommendations with speed priority."""
        recommendations = scan_mode_selector.get_mode_recommendations(
            collection_name="test-collection",
            performance_priority="speed"
        )
        
        assert recommendations["performance_priority"] == "speed"
        assert recommendations["recommended_order"] == ["sync_only", "entity_sync", "full_rescan"]
        assert "speed over completeness" in recommendations["note"]
        assert "sync_only" in recommendations["mode_descriptions"]
        assert "entity_sync" in recommendations["mode_descriptions"]
        assert "full_rescan" in recommendations["mode_descriptions"]
    
    def test_get_mode_recommendations_accuracy_priority(self, scan_mode_selector):
        """Test mode recommendations with accuracy priority."""
        recommendations = scan_mode_selector.get_mode_recommendations(
            collection_name="test-collection",
            performance_priority="accuracy"
        )
        
        assert recommendations["performance_priority"] == "accuracy"
        assert recommendations["recommended_order"] == ["full_rescan", "entity_sync", "sync_only"]
        assert "accuracy over speed" in recommendations["note"]
    
    def test_get_mode_recommendations_balanced_priority(self, scan_mode_selector):
        """Test mode recommendations with balanced priority."""
        recommendations = scan_mode_selector.get_mode_recommendations(
            collection_name="test-collection",
            performance_priority="balanced"
        )
        
        assert recommendations["performance_priority"] == "balanced"
        assert recommendations["recommended_order"] == ["entity_sync", "sync_only", "full_rescan"]
        assert "Balanced approach" in recommendations["note"]
    
    def test_get_status(self, scan_mode_selector):
        """Test status information retrieval."""
        status = scan_mode_selector.get_status()
        
        assert "performance_threshold_entities" in status
        assert "staleness_threshold_hours" in status
        assert "health_threshold" in status
        assert "state_analyzer_configured" in status
        assert "supported_modes" in status
        
        assert status["performance_threshold_entities"] == 10000
        assert status["staleness_threshold_hours"] == 24
        assert status["health_threshold"] == 0.7
        assert status["state_analyzer_configured"] is True
        assert len(status["supported_modes"]) == 4


class TestIndexingJobConfigEntityScanOptions:
    """Test entity-level scan configuration options in IndexingJobConfig."""
    
    def test_default_entity_scan_config(self):
        """Test default entity scan configuration values."""
        config = IndexingJobConfig(
            project_path=Path("/tmp/test"),
            project_name="test-project"
        )
        
        # Check entity-level configuration defaults
        assert config.entity_scan_mode == "auto"
        assert config.enable_entity_monitoring is True
        assert config.entity_batch_size == 50
        assert config.entity_change_detection is True
        assert config.entity_content_hashing is True
    
    def test_custom_entity_scan_config(self):
        """Test custom entity scan configuration."""
        config = IndexingJobConfig(
            project_path=Path("/tmp/test"),
            project_name="test-project",
            entity_scan_mode="full_rescan",
            enable_entity_monitoring=False,
            entity_batch_size=100,
            entity_change_detection=False,
            entity_content_hashing=False
        )
        
        assert config.entity_scan_mode == "full_rescan"
        assert config.enable_entity_monitoring is False
        assert config.entity_batch_size == 100
        assert config.entity_change_detection is False
        assert config.entity_content_hashing is False
    
    def test_entity_scan_mode_values(self):
        """Test valid entity scan mode values."""
        valid_modes = ["auto", "full_rescan", "entity_sync", "sync_only"]
        
        for mode in valid_modes:
            config = IndexingJobConfig(
                project_path=Path("/tmp/test"),
                project_name="test-project",
                entity_scan_mode=mode
            )
            assert config.entity_scan_mode == mode
    
    def test_entity_batch_size_validation(self):
        """Test entity batch size values."""
        # Test various batch sizes
        for batch_size in [1, 25, 50, 100, 500]:
            config = IndexingJobConfig(
                project_path=Path("/tmp/test"),
                project_name="test-project",
                entity_batch_size=batch_size
            )
            assert config.entity_batch_size == batch_size


class TestEntityScanModeIntegration:
    """Test integration between scan mode selection and HybridIndexer."""
    
    @pytest.fixture
    def mock_hybrid_indexer_components(self):
        """Create mock components for HybridIndexer."""
        mock_parser = Mock()
        mock_embedder = Mock()
        mock_storage = Mock(spec=HybridQdrantClient)
        
        return mock_parser, mock_embedder, mock_storage
    
    def test_hybrid_indexer_has_scan_mode_selector(self, mock_hybrid_indexer_components):
        """Test that HybridIndexer includes scan mode selector."""
        from core.indexer.hybrid_indexer import HybridIndexer
        
        parser, embedder, storage = mock_hybrid_indexer_components
        
        # Create HybridIndexer (will initialize scan mode selector)
        indexer = HybridIndexer(
            parser_pipeline=parser,
            embedder=embedder,
            storage_client=storage
        )
        
        # Should have scan mode selector and state analyzer
        assert hasattr(indexer, 'scan_mode_selector')
        assert hasattr(indexer, 'state_analyzer')
        assert indexer.scan_mode_selector is not None
        assert indexer.state_analyzer is not None
    
    @pytest.mark.asyncio
    async def test_hybrid_indexer_select_entity_scan_mode(self, mock_hybrid_indexer_components):
        """Test HybridIndexer select_entity_scan_mode method."""
        from core.indexer.hybrid_indexer import HybridIndexer
        
        parser, embedder, storage = mock_hybrid_indexer_components
        
        # Create HybridIndexer
        indexer = HybridIndexer(
            parser_pipeline=parser,
            embedder=embedder,
            storage_client=storage
        )
        
        # Mock the scan mode selector
        mock_decision = ScanModeDecision(
            selected_mode=EntityScanMode.ENTITY_SYNC,
            confidence=0.8,
            reasoning=["Test decision"],
            entity_count_estimate=100,
            expected_duration_minutes=2.0,
            performance_tier="fast"
        )
        
        indexer.scan_mode_selector.select_scan_mode = AsyncMock(return_value=mock_decision)
        
        # Test method
        project_path = Path("/tmp/test-project")
        collection_name = "test-collection"
        
        decision = await indexer.select_entity_scan_mode(
            collection_name=collection_name,
            project_path=project_path
        )
        
        # Should return the mocked decision
        assert decision == mock_decision
        assert decision.selected_mode == EntityScanMode.ENTITY_SYNC
        
        # Should have called scan mode selector with correct parameters
        indexer.scan_mode_selector.select_scan_mode.assert_called_once_with(
            collection_name=collection_name,
            project_path=project_path,
            requested_mode="auto",  # Default from config
            force_mode=False
        )
    
    def test_hybrid_indexer_performance_metrics_include_entity_components(
        self, mock_hybrid_indexer_components
    ):
        """Test that performance metrics include entity-level components."""
        from core.indexer.hybrid_indexer import HybridIndexer
        
        parser, embedder, storage = mock_hybrid_indexer_components
        
        # Mock the required methods for performance metrics
        parser.max_workers = 4
        parser.batch_size = 100
        embedder.config = Mock()
        embedder.config.model_name = "test-model"
        embedder.config.device = "cpu" 
        embedder.config.batch_size = 32
        
        # Create HybridIndexer
        indexer = HybridIndexer(
            parser_pipeline=parser,
            embedder=embedder,
            storage_client=storage
        )
        
        # Mock methods needed for metrics
        indexer.batch_indexer = Mock()
        indexer.batch_indexer.get_performance_metrics = Mock(return_value={})
        
        # Get performance metrics
        metrics = indexer.get_performance_metrics()
        
        # Should include entity-level components
        assert "state_analyzer" in metrics
        assert "scan_mode_selector" in metrics
        
        # Check state analyzer metrics
        state_metrics = metrics["state_analyzer"]
        assert "staleness_threshold_hours" in state_metrics
        assert "min_health_score" in state_metrics
        assert "cache_stats" in state_metrics
        
        # Check scan mode selector metrics
        selector_metrics = metrics["scan_mode_selector"]
        assert "performance_threshold_entities" in selector_metrics
        assert "staleness_threshold_hours" in selector_metrics
        assert "health_threshold" in selector_metrics
        assert "supported_modes" in selector_metrics