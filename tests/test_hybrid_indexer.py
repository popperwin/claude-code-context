"""
Comprehensive unit tests for hybrid_indexer.py

Tests all components: IndexingJobConfig, IndexingJobMetrics, HybridIndexer
"""

import pytest
import asyncio
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import List
import logging

from core.indexer.hybrid_indexer import (
    IndexingJobConfig, IndexingJobMetrics, HybridIndexer
)

from core.indexer.cache import CacheManager
from core.indexer.state_analyzer import CollectionStateAnalyzer, CollectionState, CollectionStateInfo
from core.indexer.scan_mode import EntityScanModeSelector, EntityScanMode, ScanModeDecision
from core.parser.parallel_pipeline import ProcessParsingPipeline, PipelineStats
from core.parser.base import ParseResult
from core.embeddings.stella import StellaEmbedder
from core.storage.client import HybridQdrantClient
from core.storage.indexing import BatchIndexer, IndexingResult
from core.models.entities import Entity, EntityType, SourceLocation
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse


@pytest.fixture
async def cleanup_test_collections():
    """Fixture to clean up test collections before and after tests"""
    test_collections = [
        "test-collection", "custom-collection", "test", "empty-test", 
        "test-new-files", "error-test", "test-incremental"
    ]
    
    # Setup: Clean before tests
    client = QdrantClient(url="http://localhost:6334")
    for collection_name in test_collections:
        try:
            await asyncio.to_thread(client.delete_collection, collection_name)
        except (UnexpectedResponse, Exception):
            pass  # Collection doesn't exist or other error
    
    yield
    
    # Teardown: Clean after tests  
    for collection_name in test_collections:
        try:
            await asyncio.to_thread(client.delete_collection, collection_name)
        except (UnexpectedResponse, Exception):
            pass  # Collection doesn't exist or other error


class TestIndexingJobConfig:
    """Test IndexingJobConfig dataclass"""
    
    @pytest.mark.usefixtures("cleanup_test_collections")
    def test_config_creation_with_defaults(self, tmp_path):
        """Test config creation with default values"""
        config = IndexingJobConfig(
            project_path=tmp_path,
            project_name="test-collection"
        )
        
        assert config.project_path == tmp_path
        assert config.project_name == "test-collection"

        assert config.max_workers == 4
        assert config.batch_size == 100
        assert config.enable_caching is True
        assert config.cache_size_mb == 512
        assert config.progress_callback_interval == 1.0
        
        # Check default patterns
        assert "*.py" in config.include_patterns
        assert "*.js" in config.include_patterns
        assert "node_modules/*" in config.exclude_patterns
        assert ".git/*" in config.exclude_patterns
    
    @pytest.mark.usefixtures("cleanup_test_collections")
    def test_config_custom_values(self, tmp_path):
        """Test config with custom values"""
        custom_include = ["*.py", "*.ts"]
        custom_exclude = ["test/*"]
        
        config = IndexingJobConfig(
            project_path=tmp_path,
            project_name="custom-collection",

            max_workers=8,
            batch_size=50,
            include_patterns=custom_include,
            exclude_patterns=custom_exclude,
            enable_caching=False,
            cache_size_mb=1024
        )
        

        assert config.max_workers == 8
        assert config.batch_size == 50
        assert config.include_patterns == custom_include
        assert config.exclude_patterns == custom_exclude
        assert config.enable_caching is False
        assert config.cache_size_mb == 1024


class TestIndexingJobMetrics:
    """Test IndexingJobMetrics dataclass and computed properties"""
    
    @pytest.mark.usefixtures("cleanup_test_collections")
    def test_metrics_initialization(self):
        """Test metrics initialization with defaults"""
        metrics = IndexingJobMetrics()
        
        assert isinstance(metrics.start_time, datetime)
        assert metrics.end_time is None
        assert metrics.total_duration_seconds == 0.0
        assert metrics.files_discovered == 0
        assert metrics.files_processed == 0
        assert metrics.entities_extracted == 0
        assert metrics.cache_hits == 0
        assert metrics.errors == []
    
    @pytest.mark.usefixtures("cleanup_test_collections")
    def test_files_per_second_calculation(self):
        """Test files per second calculation"""
        metrics = IndexingJobMetrics()
        metrics.files_processed = 100
        metrics.total_duration_seconds = 50.0
        
        assert metrics.files_per_second == 2.0
        
        # Test zero duration
        metrics.total_duration_seconds = 0.0
        assert metrics.files_per_second == 0.0
    
    @pytest.mark.usefixtures("cleanup_test_collections")
    def test_entities_per_second_calculation(self):
        """Test entities per second calculation"""
        metrics = IndexingJobMetrics()
        metrics.entities_indexed = 500
        metrics.total_duration_seconds = 25.0
        
        assert metrics.entities_per_second == 20.0
        
        # Test zero duration
        metrics.total_duration_seconds = 0.0
        assert metrics.entities_per_second == 0.0
    
    @pytest.mark.usefixtures("cleanup_test_collections")
    def test_success_rate_calculation(self):
        """Test success rate calculation"""
        metrics = IndexingJobMetrics()
        
        # Test with no files processed
        assert metrics.success_rate == 1.0
        
        # Test with successful processing
        metrics.files_processed = 80
        metrics.files_failed = 20
        assert metrics.success_rate == 0.8
        
        # Test with all failures
        metrics.files_processed = 0
        metrics.files_failed = 100
        assert metrics.success_rate == 0.0
    
    @pytest.mark.usefixtures("cleanup_test_collections")
    def test_cache_hit_rate_calculation(self):
        """Test cache hit rate calculation"""
        metrics = IndexingJobMetrics()
        
        # Test with no cache requests
        assert metrics.cache_hit_rate == 0.0
        
        # Test with cache hits and misses
        metrics.cache_hits = 75
        metrics.cache_misses = 25
        assert metrics.cache_hit_rate == 0.75
        
        # Test with all hits
        metrics.cache_hits = 100
        metrics.cache_misses = 0
        assert metrics.cache_hit_rate == 1.0
    
    @pytest.mark.usefixtures("cleanup_test_collections")
    def test_to_dict_conversion(self):
        """Test metrics to dictionary conversion"""
        metrics = IndexingJobMetrics()
        metrics.files_processed = 50
        metrics.entities_indexed = 200
        metrics.total_duration_seconds = 10.0
        metrics.cache_hits = 30
        metrics.cache_misses = 20
        metrics.errors = ["error1", "error2"]
        
        result_dict = metrics.to_dict()
        
        assert result_dict["files_processed"] == 50
        assert result_dict["entities_indexed"] == 200
        assert result_dict["duration_seconds"] == 10.0
        assert result_dict["files_per_second"] == 5.0
        assert result_dict["entities_per_second"] == 20.0
        assert result_dict["success_rate"] == 1.0
        assert result_dict["cache_hit_rate"] == 0.6
        assert result_dict["error_count"] == 2


class TestHybridIndexer:
    """Test HybridIndexer class"""
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components for testing"""
        mock_parser = Mock(spec=ProcessParsingPipeline)
        mock_parser.registry = Mock()
        mock_parser.max_workers = 4
        mock_parser.batch_size = 10
        
        mock_embedder = Mock(spec=StellaEmbedder)
        mock_embedder.config = Mock()
        mock_embedder.config.model_name = "test-model"
        mock_embedder.config.device = "cpu"
        mock_embedder.config.batch_size = 32
        
        mock_storage = Mock(spec=HybridQdrantClient)
        mock_cache = Mock(spec=CacheManager)
        
        return {
            "parser": mock_parser,
            "embedder": mock_embedder, 
            "storage": mock_storage,
            "cache": mock_cache
        }
    
    @pytest.fixture
    def indexer(self, mock_components):
        """Create HybridIndexer with mock components"""
        return HybridIndexer(
            parser_pipeline=mock_components["parser"],
            embedder=mock_components["embedder"],
            storage_client=mock_components["storage"],
            cache_manager=mock_components["cache"]
        )
    
    @pytest.mark.usefixtures("cleanup_test_collections")
    def test_indexer_initialization(self, indexer, mock_components):
        """Test indexer initialization"""
        assert indexer.parser_pipeline == mock_components["parser"]
        assert indexer.embedder == mock_components["embedder"]
        assert indexer.storage_client == mock_components["storage"]
        assert indexer.cache_manager == mock_components["cache"]
        assert isinstance(indexer.batch_indexer, BatchIndexer)

        assert indexer._progress_callbacks == []
    
    @pytest.mark.usefixtures("cleanup_test_collections")
    def test_progress_callback_management(self, indexer):
        """Test adding and removing progress callbacks"""
        callback1 = Mock()
        callback2 = Mock()
        
        # Add callbacks
        indexer.add_progress_callback(callback1)
        indexer.add_progress_callback(callback2)
        
        assert len(indexer._progress_callbacks) == 2
        assert callback1 in indexer._progress_callbacks
        assert callback2 in indexer._progress_callbacks
        
        # Remove callback
        indexer.remove_progress_callback(callback1)
        assert len(indexer._progress_callbacks) == 1
        assert callback1 not in indexer._progress_callbacks
        assert callback2 in indexer._progress_callbacks
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_discover_files(self, indexer, mock_components, tmp_path):
        """Test file discovery with filtering"""
        # Create test files
        (tmp_path / "test.py").write_text("# python file")
        (tmp_path / "test.js").write_text("// javascript file")
        (tmp_path / "test.txt").write_text("text file")
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "lib.js").write_text("// library")
        
        # Mock parser registry
        all_files = [
            tmp_path / "test.py",
            tmp_path / "test.js", 
            tmp_path / "test.txt",
            tmp_path / "node_modules" / "lib.js"
        ]
        mock_components["parser"].registry.discover_files.return_value = all_files
        
        # Create config
        config = IndexingJobConfig(
            project_path=tmp_path,
            project_name="test",
            include_patterns=["*.py", "*.js"],
            exclude_patterns=["node_modules/*"]
        )
        
        metrics = IndexingJobMetrics()
        
        # Test file discovery
        files = await indexer._discover_files(config, metrics)
        
        # Should include .py and .js files but exclude node_modules
        expected_files = {tmp_path / "test.py", tmp_path / "test.js"}
        assert set(files) == expected_files
        assert metrics.files_discovered == 2
    

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_parse_files(self, indexer, mock_components, tmp_path):
        """Test file parsing with progress tracking"""
        files = [tmp_path / "test1.py", tmp_path / "test2.py"]
        
        # Create mock parse results
        mock_result1 = Mock(spec=ParseResult)
        mock_result1.success = True
        mock_result1.entities = [Mock(), Mock()]
        mock_result1.relations = [Mock()]
        
        mock_result2 = Mock(spec=ParseResult)
        mock_result2.success = True
        mock_result2.entities = [Mock()]
        mock_result2.relations = []
        
        parse_results = [mock_result1, mock_result2]
        
        # Mock pipeline stats
        mock_stats = Mock(spec=PipelineStats)
        mock_stats.successful_files = 2
        mock_stats.failed_files = 0
        mock_stats.total_entities = 3
        mock_stats.total_relations = 1
        
        # Mock parser pipeline
        mock_components["parser"].parse_files.return_value = (parse_results, mock_stats)
        
        config = IndexingJobConfig(project_path=tmp_path, project_name="test")
        metrics = IndexingJobMetrics()
        
        # Test parsing
        result = await indexer._parse_files(files, config, metrics, show_progress=True)
        
        assert result == parse_results
        assert metrics.files_processed == 2
        assert metrics.files_failed == 0
        assert metrics.entities_extracted == 3
        assert metrics.relations_extracted == 1
        assert metrics.parse_time_seconds > 0
        
        # Verify parser was called correctly
        mock_components["parser"].parse_files.assert_called_once()
        call_args = mock_components["parser"].parse_files.call_args
        assert call_args[0][0] == files  # First positional arg should be files
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_extract_entities_relations(self, indexer):
        """Test entity and relation extraction from parse results"""
        # Create mock entities and relations
        entity1 = Mock(spec=Entity)
        entity2 = Mock(spec=Entity)
        relation1 = Mock()
        
        # Create mock parse results
        result1 = Mock(spec=ParseResult)
        result1.success = True
        result1.entities = [entity1]
        result1.relations = [relation1]
        
        result2 = Mock(spec=ParseResult)
        result2.success = True
        result2.entities = [entity2]
        result2.relations = []
        
        # Failed result should be ignored
        result3 = Mock(spec=ParseResult)
        result3.success = False
        result3.entities = [Mock()]
        result3.relations = [Mock()]
        
        parse_results = [result1, result2, result3]
        metrics = IndexingJobMetrics()
        
        # Test extraction
        entities, relations = await indexer._extract_entities_relations(
            parse_results, metrics
        )
        
        assert entities == [entity1, entity2]
        assert relations == [relation1]
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_index_entities(self, indexer, mock_components):
        """Test entity indexing with progress tracking"""
        entities = [Mock(spec=Entity), Mock(spec=Entity)]
        project_name = "test-collection"
        metrics = IndexingJobMetrics()
        
        # Mock indexing result
        mock_result = Mock(spec=IndexingResult)
        mock_result.successful_entities = 2
        mock_result.failed_entities = 0
        mock_result.errors = []
        
        # Mock batch indexer
        indexer.batch_indexer = Mock()
        indexer.batch_indexer.index_entities = AsyncMock(return_value=mock_result)
        indexer.batch_indexer.add_progress_callback = Mock()
        indexer.batch_indexer.remove_progress_callback = Mock()
        
        # Test indexing
        # Get actual collection name from project name using CollectionManager
        from core.storage.schemas import CollectionManager, CollectionType
        collection_manager = CollectionManager(project_name=project_name)
        collection_name = collection_manager.get_collection_name(CollectionType.CODE)
        
        await indexer._index_entities(entities, collection_name, metrics, show_progress=True)
        
        assert metrics.entities_indexed == 2
        assert metrics.entities_failed == 0
        assert metrics.index_time_seconds > 0
        
        # Verify batch indexer was called correctly
        indexer.batch_indexer.index_entities.assert_called_once_with(
            entities=entities,
            collection_name=collection_name,
            show_progress=True,
            description="Indexing entities"
        )
        
        # Verify progress callback management
        indexer.batch_indexer.add_progress_callback.assert_called_once()
        indexer.batch_indexer.remove_progress_callback.assert_called_once()
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_update_cache_state(self, indexer, mock_components, tmp_path):
        """Test cache state updates"""
        files = [tmp_path / "file1.py", tmp_path / "file2.py"]
        config = IndexingJobConfig(project_path=tmp_path, project_name="test")
        metrics = IndexingJobMetrics()
        
        # Mock cache manager
        mock_components["cache"].update_file_cache = AsyncMock()
        
        # Test cache update - need to derive collection name from project name
        from core.storage.schemas import CollectionManager, CollectionType
        collection_manager = CollectionManager(project_name=config.project_name)
        collection_name = collection_manager.get_collection_name(config.collection_type)
        
        await indexer._update_cache_state(files, collection_name, metrics)
        
        # Verify cache updates
        assert mock_components["cache"].update_file_cache.call_count == 2
        for file_path in files:
            mock_components["cache"].update_file_cache.assert_any_call(
                file_path, collection_name
            )
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_index_single_file(self, indexer, tmp_path):
        """Test single file indexing"""
        test_file = tmp_path / "test.py"
        test_file.write_text("# test file")
        
        # Mock the main index_project method
        indexer.index_project = AsyncMock()
        expected_metrics = IndexingJobMetrics()
        indexer.index_project.return_value = expected_metrics
        
        # Test single file indexing
        result = await indexer.index_single_file(
            test_file, "test-collection", force_reindex=True
        )
        
        assert result == expected_metrics
        
        # Verify index_project was called with correct config
        indexer.index_project.assert_called_once()
        call_args = indexer.index_project.call_args[0][0]  # First arg is config
        
        assert call_args.project_path == test_file.parent
        assert call_args.project_name == "test-collection"

        assert call_args.max_workers == 1
        assert call_args.batch_size == 10
        assert "test.py" in call_args.include_patterns
    
    @pytest.mark.usefixtures("cleanup_test_collections")
    def test_get_performance_metrics(self, indexer, mock_components):
        """Test performance metrics collection"""
        # Mock batch indexer metrics
        batch_metrics = {"total_indexed": 100, "average_speed": 10.5}
        indexer.batch_indexer = Mock()
        indexer.batch_indexer.get_performance_metrics.return_value = batch_metrics
        
        # Mock cache manager stats
        cache_stats = {"hit_rate": 0.8, "size": 500}
        mock_components["cache"].get_stats.return_value = cache_stats
        
        # Test metrics collection
        metrics = indexer.get_performance_metrics()
        
        assert "parser_pipeline" in metrics
        assert metrics["parser_pipeline"]["max_workers"] == 4
        assert metrics["parser_pipeline"]["batch_size"] == 10
        
        assert "batch_indexer" in metrics
        assert metrics["batch_indexer"] == batch_metrics
        
        assert "embedder" in metrics
        assert metrics["embedder"]["model_name"] == "test-model"
        assert metrics["embedder"]["device"] == "cpu"
        assert metrics["embedder"]["batch_size"] == 32
        
        assert "cache_manager" in metrics
        assert metrics["cache_manager"] == cache_stats


class TestHybridIndexerIntegration:
    """Integration tests for HybridIndexer with real components"""
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_empty_project_indexing(self, tmp_path):
        """Test indexing an empty project"""
        # Create real components with minimal setup
        mock_parser = Mock()
        mock_parser.registry = Mock()
        mock_parser.registry.discover_files.return_value = []
        
        mock_embedder = Mock()
        mock_storage = Mock()
        
        indexer = HybridIndexer(
            parser_pipeline=mock_parser,
            embedder=mock_embedder,
            storage_client=mock_storage
        )
        
        config = IndexingJobConfig(
            project_path=tmp_path,
            project_name="empty-test",

        )
        
        # Test indexing
        metrics = await indexer.index_project(config, show_progress=False)
        
        assert metrics.files_discovered == 0
        assert metrics.files_processed == 0
        assert metrics.entities_extracted == 0
        assert metrics.success_rate == 1.0
        assert isinstance(metrics.total_duration_seconds, float)
        assert metrics.total_duration_seconds >= 0
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_error_handling_in_indexing(self, tmp_path):
        """Test error handling during indexing"""
        # Create mock that raises exception
        mock_parser = Mock()
        mock_parser.registry = Mock()
        mock_parser.registry.discover_files.side_effect = Exception("Discovery failed")
        
        mock_embedder = Mock()
        mock_embedder.dimensions = 1024
        
        # Mock storage client properly for async operations
        mock_storage = Mock()
        mock_storage.get_collection_info = AsyncMock(return_value=None)  # Collection doesn't exist
        mock_storage.create_collection = AsyncMock(return_value=Mock(success=True))
        
        indexer = HybridIndexer(
            parser_pipeline=mock_parser,
            embedder=mock_embedder,
            storage_client=mock_storage
        )
        
        config = IndexingJobConfig(
            project_path=tmp_path,
            project_name="error-test"
        )
        
        # Test error handling
        metrics = await indexer.index_project(config, show_progress=False)
        
        assert len(metrics.errors) > 0
        assert "Discovery failed" in metrics.errors[0]
        assert metrics.end_time is not None
        assert metrics.total_duration_seconds > 0


class TestHybridIndexerEntityLevelOperations:
    """Test HybridIndexer entity-level operations and sync integration."""
    
    @pytest.fixture
    def mock_components_with_entity_support(self):
        """Create mock components with entity-level operation support."""
        mock_parser = Mock(spec=ProcessParsingPipeline)
        mock_parser.registry = Mock()
        mock_parser.max_workers = 4
        mock_parser.batch_size = 10
        
        mock_embedder = Mock(spec=StellaEmbedder)
        mock_embedder.config = Mock()
        mock_embedder.config.model_name = "test-model"
        mock_embedder.config.device = "cpu"
        mock_embedder.config.batch_size = 32
        mock_embedder.dimensions = 1024
        
        mock_storage = Mock(spec=HybridQdrantClient)
        mock_storage.get_collection_info = AsyncMock(return_value=None)
        mock_storage.create_collection = AsyncMock(return_value=Mock(success=True))
        mock_storage.search_payload = AsyncMock(return_value=[])
        
        mock_cache = Mock(spec=CacheManager)
        
        return {
            "parser": mock_parser,
            "embedder": mock_embedder,
            "storage": mock_storage,
            "cache": mock_cache
        }
    
    @pytest.fixture
    def entity_indexer(self, mock_components_with_entity_support):
        """Create HybridIndexer with entity operation support."""
        return HybridIndexer(
            parser_pipeline=mock_components_with_entity_support["parser"],
            embedder=mock_components_with_entity_support["embedder"],
            storage_client=mock_components_with_entity_support["storage"],
            cache_manager=mock_components_with_entity_support["cache"]
        )
    
    def create_mock_state_info(
        self,
        collection_name: str,
        state: CollectionState,
        entity_count: int = 100
    ) -> CollectionStateInfo:
        """Create mock CollectionStateInfo for testing."""
        from core.indexer.state_analyzer import EntityMetadata
        from datetime import datetime, timedelta
        
        metadata = EntityMetadata(
            total_entities=entity_count,
            last_update_time=datetime.now(),
            oldest_entity_time=datetime.now() - timedelta(hours=1),
            newest_entity_time=datetime.now(),
            file_coverage=max(1, entity_count // 10),
            entity_types={"function": entity_count // 2, "class": entity_count // 2},
            average_entities_per_file=2.0
        )
        
        return CollectionStateInfo(
            collection_name=collection_name,
            state=state,
            entity_metadata=metadata,
            health_score=0.8,
            last_analyzed=datetime.now(),
            analysis_duration_ms=100.0,
            recommendations=[],
            issues_found=[],
            collection_exists=(state != CollectionState.EMPTY),
            collection_info={"points_count": entity_count} if entity_count > 0 else None,
            is_indexing=False
        )
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_entity_level_components_initialization(self, entity_indexer):
        """Test that HybridIndexer initializes entity-level components."""
        # Check entity-level components are present
        assert hasattr(entity_indexer, 'state_analyzer')
        assert hasattr(entity_indexer, 'scan_mode_selector')
        assert isinstance(entity_indexer.state_analyzer, CollectionStateAnalyzer)
        assert isinstance(entity_indexer.scan_mode_selector, EntityScanModeSelector)
        
        # Check sync engine is not initialized by default
        assert entity_indexer.sync_engine is None
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_select_entity_scan_mode(
        self, entity_indexer, mock_components_with_entity_support, tmp_path
    ):
        """Test entity scan mode selection."""
        collection_name = "test-entity-scan"
        
        # Mock state analyzer to return healthy collection
        healthy_state = self.create_mock_state_info(
            collection_name, CollectionState.HEALTHY, 150
        )
        entity_indexer.state_analyzer.analyze_collection_state = AsyncMock(
            return_value=healthy_state
        )
        
        # Test scan mode selection
        decision = await entity_indexer.select_entity_scan_mode(
            collection_name=collection_name,
            project_path=tmp_path
        )
        
        assert isinstance(decision, ScanModeDecision)
        assert decision.selected_mode in [EntityScanMode.ENTITY_SYNC, EntityScanMode.SYNC_ONLY]
        assert decision.confidence > 0.0
        assert len(decision.reasoning) > 0
        assert decision.entity_count_estimate == 150
        
        # Verify analyzer was called
        entity_indexer.state_analyzer.analyze_collection_state.assert_called_once()
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_index_project_full_entity_scan(
        self, entity_indexer, mock_components_with_entity_support, tmp_path
    ):
        """Test full entity scan operation."""
        # Create test files
        (tmp_path / "test.py").write_text("def hello(): pass")
        (tmp_path / "test2.py").write_text("class Test: pass")
        
        # Mock components for full entity scan
        mock_components_with_entity_support["parser"].registry.discover_files.return_value = [
            tmp_path / "test.py", tmp_path / "test2.py"
        ]
        
        # Mock parse results
        mock_result1 = Mock(spec=ParseResult)
        mock_result1.success = True
        mock_result1.entities = [Mock(spec=Entity), Mock(spec=Entity)]
        mock_result1.relations = []
        
        mock_stats = Mock(spec=PipelineStats)
        mock_stats.successful_files = 2
        mock_stats.failed_files = 0
        mock_stats.total_entities = 2
        mock_stats.total_relations = 0
        
        mock_components_with_entity_support["parser"].parse_files.return_value = (
            [mock_result1], mock_stats
        )
        
        # Mock batch indexer
        entity_indexer.batch_indexer = Mock()
        entity_indexer.batch_indexer.index_entities = AsyncMock(
            return_value=Mock(successful_entities=2, failed_entities=0, errors=[])
        )
        entity_indexer.batch_indexer.add_progress_callback = Mock()
        entity_indexer.batch_indexer.remove_progress_callback = Mock()
        
        # Mock state analyzer to return empty collection (forces full rescan)
        empty_state = self.create_mock_state_info(
            "test-entity-full-scan-code", CollectionState.EMPTY, 0
        )
        entity_indexer.state_analyzer.analyze_collection_state = AsyncMock(
            return_value=empty_state
        )
        
        # Create config for full entity scan
        config = IndexingJobConfig(
            project_path=tmp_path,
            project_name="test-entity-full-scan",
            entity_scan_mode="full_rescan",
            enable_entity_monitoring=False,

        )
        
        # Test full entity scan
        metrics = await entity_indexer.index_project(config, show_progress=False)
        
        # Verify results
        assert metrics.files_discovered == 2
        assert metrics.files_processed == 2
        assert metrics.entities_extracted == 2
        assert metrics.entities_indexed == 2
        assert metrics.total_duration_seconds > 0
        assert len(metrics.errors) == 0
        
        # Verify entity indexing was called
        entity_indexer.batch_indexer.index_entities.assert_called_once()
        call_args = entity_indexer.batch_indexer.index_entities.call_args
        assert len(call_args.kwargs["entities"]) == 2
        assert call_args.kwargs["collection_name"] == "test-entity-full-scan-code"
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_index_project_entity_sync_mode(
        self, entity_indexer, mock_components_with_entity_support, tmp_path
    ):
        """Test entity sync mode operation."""
        # Mock state analyzer to return healthy collection
        healthy_state = self.create_mock_state_info(
            "test-entity-sync-code", CollectionState.HEALTHY, 100
        )
        entity_indexer.state_analyzer.analyze_collection_state = AsyncMock(
            return_value=healthy_state
        )
        
        # Mock scan mode selector to return entity sync
        entity_indexer.scan_mode_selector.select_scan_mode = AsyncMock(
            return_value=ScanModeDecision(
                selected_mode=EntityScanMode.ENTITY_SYNC,
                confidence=0.8,
                reasoning=["Collection is healthy, entity sync appropriate"],
                entity_count_estimate=100,
                expected_duration_minutes=2.0,
                performance_tier="fast"
            )
        )
        
        # Mock entity sync implementation (delegates to full scan for now)
        entity_indexer._perform_entity_sync = AsyncMock()
        
        # Create config for entity sync
        config = IndexingJobConfig(
            project_path=tmp_path,
            project_name="test-entity-sync",
            entity_scan_mode="auto",  # Will be determined by analyzer
            enable_entity_monitoring=True
        )
        
        # Test entity sync mode
        metrics = await entity_indexer.index_project(config, show_progress=False)
        
        # Verify entity sync was called
        entity_indexer._perform_entity_sync.assert_called_once()
        
        # Verify metrics
        assert metrics.total_duration_seconds > 0
        assert len(metrics.errors) == 0
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_index_project_sync_only_mode(
        self, entity_indexer, mock_components_with_entity_support, tmp_path
    ):
        """Test sync-only mode operation."""
        # Mock state analyzer to return very healthy recent collection
        from datetime import datetime, timedelta
        very_healthy_state = self.create_mock_state_info(
            "test-sync-only-code", CollectionState.HEALTHY, 200
        )
        very_healthy_state.health_score = 0.95
        very_healthy_state.entity_metadata.last_update_time = datetime.now() - timedelta(minutes=30)
        
        entity_indexer.state_analyzer.analyze_collection_state = AsyncMock(
            return_value=very_healthy_state
        )
        
        # Mock scan mode selector to return sync only
        entity_indexer.scan_mode_selector.select_scan_mode = AsyncMock(
            return_value=ScanModeDecision(
                selected_mode=EntityScanMode.SYNC_ONLY,
                confidence=0.9,
                reasoning=["Collection is very healthy and recent, sync-only sufficient"],
                entity_count_estimate=200,
                expected_duration_minutes=0.5,
                performance_tier="fast"
            )
        )
        
        # Mock sync-only implementation
        entity_indexer._enable_entity_sync_only = AsyncMock()
        entity_indexer._enable_entity_sync = AsyncMock(return_value=True)
        
        # Create config for sync-only mode
        config = IndexingJobConfig(
            project_path=tmp_path,
            project_name="test-sync-only",
            entity_scan_mode="auto",
            enable_entity_monitoring=True
        )
        
        # Test sync-only mode
        metrics = await entity_indexer.index_project(config, show_progress=False)
        
        # Verify sync-only was called
        entity_indexer._enable_entity_sync_only.assert_called_once()
        
        # Verify entity monitoring was enabled
        entity_indexer._enable_entity_sync.assert_called_once()
        
        # Verify metrics (minimal processing)
        assert metrics.total_duration_seconds > 0
        assert len(metrics.errors) == 0
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_enable_entity_sync(
        self, entity_indexer, mock_components_with_entity_support, tmp_path
    ):
        """Test entity sync engine integration."""
        # Mock sync engine
        mock_sync_engine = Mock()
        mock_sync_engine.is_running = False
        mock_sync_engine.start_monitoring = AsyncMock(return_value=True)
        mock_sync_engine.add_project = AsyncMock(return_value=True)
        
        entity_indexer.sync_engine = mock_sync_engine
        
        # Create config with sync settings
        config = IndexingJobConfig(
            project_path=tmp_path,
            project_name="test-sync",
            enable_realtime_sync=True,
            sync_debounce_ms=250,
            sync_batch_size=5,
            sync_worker_count=1
        )
        
        # Test entity sync enablement
        result = await entity_indexer._enable_entity_sync(
            project_path=tmp_path,
            collection_name="test-sync-code",
            config=config
        )
        
        assert result is True
        
        # Verify sync engine was started and project added
        mock_sync_engine.start_monitoring.assert_called_once()
        mock_sync_engine.add_project.assert_called_once_with(
            project_path=tmp_path,
            collection_name="test-sync-code",
            debounce_ms=250,
            start_monitoring=True
        )
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_entity_cache_state_update(
        self, entity_indexer, mock_components_with_entity_support
    ):
        """Test entity-level cache state updates."""
        # Mock cache manager
        mock_cache = mock_components_with_entity_support["cache"]
        mock_cache.update_collection_cache = AsyncMock()
        
        # Create metrics with entity data
        metrics = IndexingJobMetrics()
        metrics.entities_indexed = 150
        metrics.relations_extracted = 25
        
        # Test cache state update
        await entity_indexer._update_entity_cache_state(
            collection_name="test-cache-collection",
            metrics=metrics
        )
        
        # Verify cache update was called with entity data
        mock_cache.update_collection_cache.assert_called_once()
        call_args = mock_cache.update_collection_cache.call_args
        
        assert call_args.kwargs["collection_name"] == "test-cache-collection"
        assert call_args.kwargs["entity_count"] == 150
        assert call_args.kwargs["relation_count"] == 25
        assert "last_update" in call_args.kwargs
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_sync_status_reporting(
        self, entity_indexer, mock_components_with_entity_support
    ):
        """Test sync engine status reporting."""
        # Test without sync engine
        status = entity_indexer.get_sync_status()
        assert status["sync_enabled"] is False
        assert status["sync_engine_initialized"] is False
        
        # Mock sync engine with status
        mock_sync_engine = Mock()
        mock_sync_engine.get_status.return_value = {
            "is_running": True,
            "projects_count": 2,
            "events_processed": 100
        }
        entity_indexer.sync_engine = mock_sync_engine
        
        # Test with sync engine
        status = entity_indexer.get_sync_status()
        assert status["sync_enabled"] is True
        assert status["sync_engine_initialized"] is True
        assert status["is_running"] is True
        assert status["projects_count"] == 2
        assert status["events_processed"] == 100
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_entity_performance_metrics(
        self, entity_indexer, mock_components_with_entity_support
    ):
        """Test that performance metrics include entity-level components."""
        # Mock batch indexer metrics
        batch_metrics = {"total_indexed": 100, "average_speed": 10.5}
        entity_indexer.batch_indexer = Mock()
        entity_indexer.batch_indexer.get_performance_metrics.return_value = batch_metrics
        
        # Mock cache manager stats
        cache_stats = {"hit_rate": 0.8, "size": 500}
        mock_components_with_entity_support["cache"].get_stats.return_value = cache_stats
        
        # Test metrics collection
        metrics = entity_indexer.get_performance_metrics()
        
        # Check entity-level component metrics are included
        assert "state_analyzer" in metrics
        assert "scan_mode_selector" in metrics
        
        state_analyzer_metrics = metrics["state_analyzer"]
        assert "staleness_threshold_hours" in state_analyzer_metrics
        assert "min_health_score" in state_analyzer_metrics
        assert "cache_stats" in state_analyzer_metrics
        
        scan_selector_metrics = metrics["scan_mode_selector"]
        assert "performance_threshold_entities" in scan_selector_metrics
        assert "staleness_threshold_hours" in scan_selector_metrics
        assert "health_threshold" in scan_selector_metrics
        assert "supported_modes" in scan_selector_metrics
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_entity_config_defaults(self, tmp_path):
        """Test that IndexingJobConfig includes entity-level defaults."""
        config = IndexingJobConfig(
            project_path=tmp_path,
            project_name="test-entity-config"
        )
        
        # Check entity-level configuration defaults
        assert config.entity_scan_mode == "auto"
        assert config.enable_entity_monitoring is True
        assert config.entity_batch_size == 50
        assert config.entity_change_detection is True
        assert config.entity_content_hashing is True
    
    @pytest.mark.asyncio
    @pytest.mark.usefixtures("cleanup_test_collections")
    async def test_entity_config_customization(self, tmp_path):
        """Test entity-level configuration customization."""
        config = IndexingJobConfig(
            project_path=tmp_path,
            project_name="test-custom-entity",
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