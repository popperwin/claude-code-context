"""
Hybrid indexer orchestrating parsing, embedding, and storage operations.

This module provides the main HybridIndexer class that combines all indexing stages
into a unified, high-performance pipeline with progress tracking and error handling.
"""

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field

from ..parser.parallel_pipeline import ProcessParsingPipeline, PipelineStats
from ..parser.base import ParseResult
from ..embeddings.stella import StellaEmbedder
from ..storage.client import HybridQdrantClient
from ..storage.indexing import BatchIndexer, IndexingResult
from ..models.entities import Entity, Relation
from ..storage.schemas import CollectionManager, CollectionType
from .cache import CacheManager
from .state_analyzer import CollectionStateAnalyzer
from .scan_mode import EntityScanModeSelector, EntityScanMode

logger = logging.getLogger(__name__)


@dataclass
class IndexingJobConfig:
    """Configuration for indexing jobs"""
    project_path: Path
    project_name: str
    collection_type: CollectionType = CollectionType.CODE
    incremental: bool = True
    max_workers: int = 4
    batch_size: int = 100
    
    # File filtering
    include_patterns: List[str] = field(default_factory=lambda: [
        "*.py", "*.js", "*.ts", "*.jsx", "*.tsx", "*.go", "*.rs", 
        "*.java", "*.c", "*.cpp", "*.h", "*.hpp", "*.cs", "*.rb"
    ])
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "node_modules/*", ".git/*", "__pycache__/*", "*.pyc",
        ".venv/*", "venv/*", "build/*", "dist/*", ".cache/*"
    ])
    
    # Performance tuning
    enable_caching: bool = True
    cache_size_mb: int = 512
    progress_callback_interval: float = 1.0  # seconds
    
    # Real-time synchronization options
    enable_realtime_sync: bool = False
    sync_debounce_ms: int = 500
    sync_batch_size: int = 10
    sync_worker_count: int = 2
    sync_auto_repair: bool = True
    sync_validation_interval_minutes: int = 5
    
    # Entity-level scan configuration
    entity_scan_mode: str = "auto"  # "full_rescan", "entity_sync", "sync_only", "auto"
    enable_entity_monitoring: bool = True
    entity_batch_size: int = 50
    entity_change_detection: bool = True
    entity_content_hashing: bool = True


@dataclass 
class IndexingJobMetrics:
    """Comprehensive metrics for indexing jobs"""
    # Timing metrics
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_duration_seconds: float = 0.0
    
    # File processing metrics
    files_discovered: int = 0
    files_processed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    
    # Entity metrics
    entities_extracted: int = 0
    entities_indexed: int = 0
    entities_failed: int = 0
    relations_extracted: int = 0
    
    # Performance metrics
    parse_time_seconds: float = 0.0
    embed_time_seconds: float = 0.0
    index_time_seconds: float = 0.0
    
    # Cache metrics
    cache_hits: int = 0
    cache_misses: int = 0
    
    # Synchronization metrics
    sync_time_seconds: float = 0.0
    orphaned_entities_cleaned: int = 0
    stale_entities_detected: int = 0
    files_requiring_reindex: int = 0
    
    # Error tracking
    errors: List[str] = field(default_factory=list)
    
    @property
    def files_per_second(self) -> float:
        """Calculate files processing rate"""
        if self.total_duration_seconds <= 0:
            return 0.0
        return self.files_processed / self.total_duration_seconds
    
    @property
    def entities_per_second(self) -> float:
        """Calculate entity processing rate"""
        if self.total_duration_seconds <= 0:
            return 0.0
        return self.entities_indexed / self.total_duration_seconds
    
    @property
    def success_rate(self) -> float:
        """Calculate overall success rate"""
        total_files = self.files_processed + self.files_failed
        if total_files == 0:
            return 1.0
        return self.files_processed / total_files
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total_requests = self.cache_hits + self.cache_misses
        if total_requests == 0:
            return 0.0
        return self.cache_hits / total_requests
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging/reporting"""
        return {
            "duration_seconds": self.total_duration_seconds,
            "files_discovered": self.files_discovered,
            "files_processed": self.files_processed,
            "files_skipped": self.files_skipped,
            "files_failed": self.files_failed,
            "entities_extracted": self.entities_extracted,
            "entities_indexed": self.entities_indexed,
            "relations_extracted": self.relations_extracted,
            "files_per_second": self.files_per_second,
            "entities_per_second": self.entities_per_second,
            "success_rate": self.success_rate,
            "cache_hit_rate": self.cache_hit_rate,
            "parse_time_seconds": self.parse_time_seconds,
            "embed_time_seconds": self.embed_time_seconds,
            "index_time_seconds": self.index_time_seconds,
            "sync_time_seconds": self.sync_time_seconds,
            "orphaned_entities_cleaned": self.orphaned_entities_cleaned,
            "stale_entities_detected": self.stale_entities_detected,
            "files_requiring_reindex": self.files_requiring_reindex,
            "error_count": len(self.errors)
        }


class HybridIndexer:
    """
    Main indexing orchestrator combining parsing, embedding, and storage.
    
    Features:
    - Orchestrates ProcessParsingPipeline → StellaEmbedder → BatchIndexer
    - Incremental indexing with file change detection  
    - Multi-level caching for performance optimization
    - Comprehensive progress tracking and metrics
    - Error recovery and retry logic
    - Configurable parallelism and batch processing
    """
    
    def __init__(
        self,
        parser_pipeline: ProcessParsingPipeline,
        embedder: StellaEmbedder,
        storage_client: HybridQdrantClient,
        cache_manager: Optional[CacheManager] = None,
        config: Optional[IndexingJobConfig] = None
    ):
        """
        Initialize hybrid indexer with required components.
        
        Args:
            parser_pipeline: Tree-sitter parsing pipeline
            embedder: Stella embedding generator  
            storage_client: Qdrant storage client
            cache_manager: Optional cache manager for performance
            config: Optional default indexing configuration
        """
        self.parser_pipeline = parser_pipeline
        self.embedder = embedder
        self.storage_client = storage_client
        self.cache_manager = cache_manager
        self.default_config = config
        
        # Create batch indexer for storage operations
        self.batch_indexer = BatchIndexer(
            client=storage_client,
            embedder=embedder,
            batch_size=100,
            max_retries=3
        )
        
        
        # Initialize entity-level components
        self.state_analyzer = CollectionStateAnalyzer(
            storage_client=storage_client,
            staleness_threshold_hours=24,
            min_health_score=0.7
        )
        self.scan_mode_selector = EntityScanModeSelector(
            storage_client=storage_client,
            state_analyzer=self.state_analyzer
        )
        
        # Progress tracking
        self._progress_callbacks: List[Callable[[IndexingJobMetrics], None]] = []
        
        # Real-time synchronization engine (if enabled)
        from ..sync.engine import ProjectCollectionSyncEngine
        self.sync_engine: Optional[ProjectCollectionSyncEngine] = None
        
        # Initialize synchronization if enabled in default config
        if self.default_config and self.default_config.enable_realtime_sync:
            self._initialize_sync_engine()
        
        logger.info("Initialized HybridIndexer with all components")
    
    def _initialize_sync_engine(self) -> None:
        """Initialize the real-time synchronization engine with default config."""
        if not self.default_config:
            return
        
        try:
            from ..sync.engine import ProjectCollectionSyncEngine
            
            self.sync_engine = ProjectCollectionSyncEngine(
                storage_client=self.storage_client,
                max_queue_size=1000,
                max_batch_size=self.default_config.sync_batch_size,
                worker_count=self.default_config.sync_worker_count
            )
            
            logger.info(f"Initialized synchronization engine with {self.default_config.sync_worker_count} workers")
            
        except Exception as e:
            logger.error(f"Failed to initialize synchronization engine: {e}")
            self.sync_engine = None
    
    async def enable_realtime_sync(
        self,
        project_path: Path,
        collection_name: str,
        config: Optional[IndexingJobConfig] = None
    ) -> bool:
        """
        Enable real-time synchronization for a project (delegates to entity-level sync).
        
        Args:
            project_path: Root path of the project
            collection_name: Name of the collection to synchronize
            config: Optional configuration for sync settings
            
        Returns:
            True if synchronization was enabled successfully
        """
        if not config:
            config = self.default_config
            
        if not config or not config.enable_realtime_sync:
            logger.warning("Real-time synchronization is not enabled in configuration")
            return False
        
        # Delegate to entity-level sync method
        return await self._enable_entity_sync(
            project_path=project_path,
            collection_name=collection_name,
            config=config
        )
    
    async def disable_realtime_sync(self, project_path: Path) -> bool:
        """
        Disable real-time synchronization for a project.
        
        Args:
            project_path: Root path of the project
            
        Returns:
            True if synchronization was disabled successfully
        """
        if not self.sync_engine:
            return True  # Already disabled
        
        try:
            success = await self.sync_engine.remove_project(project_path)
            if success:
                logger.info(f"Disabled real-time synchronization for {project_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error disabling real-time synchronization: {e}")
            return False
    
    async def _ensure_collection_exists(
        self,
        project_name: str,
        collection_type: CollectionType
    ) -> str:
        """
        Ensure collection exists using CollectionManager abstraction.
        
        Args:
            project_name: Project name for collection
            collection_type: Type of collection to create
            
        Returns:
            The actual collection name created/verified
        """
        # Create CollectionManager for this project and delegate to it
        collection_manager = CollectionManager(project_name=project_name)
        return await collection_manager.ensure_collection_exists(
            collection_type=collection_type,
            storage_client=self.storage_client,
            vector_size=self.embedder.dimensions
        )
    
    def add_progress_callback(
        self, 
        callback: Callable[[IndexingJobMetrics], None]
    ) -> None:
        """Add progress callback for real-time updates"""
        self._progress_callbacks.append(callback)
    
    def remove_progress_callback(
        self, 
        callback: Callable[[IndexingJobMetrics], None]
    ) -> None:
        """Remove progress callback"""
        if callback in self._progress_callbacks:
            self._progress_callbacks.remove(callback)
    
    async def select_entity_scan_mode(
        self,
        collection_name: str,
        project_path: Path,
        config: Optional[IndexingJobConfig] = None
    ):
        """
        Select optimal entity scan mode for indexing operation.
        
        Args:
            collection_name: Name of collection to analyze
            project_path: Project root path
            config: Optional configuration with scan mode preferences
            
        Returns:
            ScanModeDecision with selected mode and reasoning
        """
        if not config:
            config = self.default_config
        
        requested_mode = config.entity_scan_mode if config else "auto"
        
        return await self.scan_mode_selector.select_scan_mode(
            collection_name=collection_name,
            project_path=project_path,
            requested_mode=requested_mode,
            force_mode=False
        )
    
    async def index_project(
        self,
        config: IndexingJobConfig,
        show_progress: bool = True
    ) -> IndexingJobMetrics:
        """
        Index an entire project using pure entity-level operations.
        
        Args:
            config: Indexing job configuration
            show_progress: Whether to show progress output
            
        Returns:
            Comprehensive indexing metrics
        """
        metrics = IndexingJobMetrics()
        
        try:
            logger.info(f"Starting entity-level project indexing: {config.project_path}")
            
            # Phase 0: Ensure Collection Exists
            collection_name = await self._ensure_collection_exists(
                config.project_name, config.collection_type
            )
            
            # Phase 1: Entity Scan Mode Selection
            scan_decision = await self.select_entity_scan_mode(
                collection_name=collection_name,
                project_path=config.project_path,
                config=config
            )
            
            logger.info(f"Selected entity scan mode: {scan_decision.selected_mode.value} "
                       f"(confidence: {scan_decision.confidence:.2f})")
            for reason in scan_decision.reasoning:
                logger.debug(f"Scan mode reasoning: {reason}")
            
            # Phase 2: Execute Entity-Level Operations Based on Scan Mode
            if scan_decision.selected_mode == EntityScanMode.FULL_RESCAN:
                await self._perform_full_entity_scan(
                    config, collection_name, metrics, show_progress
                )
            elif scan_decision.selected_mode == EntityScanMode.ENTITY_SYNC:
                await self._perform_entity_sync(
                    config, collection_name, metrics, show_progress
                )
            elif scan_decision.selected_mode == EntityScanMode.SYNC_ONLY:
                await self._enable_entity_sync_only(
                    config, collection_name, metrics
                )
            
            # Phase 3: Enable Entity Monitoring (if configured)
            if config.enable_entity_monitoring:
                sync_success = await self._enable_entity_sync(
                    project_path=config.project_path,
                    collection_name=collection_name,
                    config=config
                )
                if sync_success:
                    logger.info(f"Entity monitoring enabled for {config.project_path}")
                else:
                    logger.warning(f"Failed to enable entity monitoring for {config.project_path}")
            
            # Phase 4: Update Entity Cache State
            if config.enable_caching and self.cache_manager:
                await self._update_entity_cache_state(collection_name, metrics)
            
        except Exception as e:
            metrics.errors.append(f"Indexing failed: {str(e)}")
            logger.error(f"Project indexing failed: {e}", exc_info=True)
        
        finally:
            metrics.end_time = datetime.now()
            metrics.total_duration_seconds = (
                (metrics.end_time - metrics.start_time).total_seconds()
            )
            
            # Final progress callback
            for callback in self._progress_callbacks:
                try:
                    callback(metrics)
                except Exception as e:
                    logger.warning(f"Progress callback failed: {e}")
        
        logger.info(
            f"Project indexing completed: {metrics.entities_indexed} entities "
            f"in {metrics.total_duration_seconds:.2f}s "
            f"({metrics.entities_per_second:.1f} entities/sec)"
        )
        
        return metrics
    
    async def _discover_files(
        self,
        config: IndexingJobConfig,
        metrics: IndexingJobMetrics
    ) -> List[Path]:
        """Discover files matching the configured patterns"""
        start_time = time.perf_counter()
        
        # Use parser registry to discover files
        files = self.parser_pipeline.registry.discover_files(
            config.project_path,
            recursive=True
        )
        
        # Apply additional filtering if needed
        filtered_files = []
        for file_path in files:
            # Check include patterns
            if config.include_patterns:
                if not any(file_path.match(pattern) for pattern in config.include_patterns):
                    continue
            
            # Check exclude patterns
            if config.exclude_patterns:
                if any(file_path.match(pattern) for pattern in config.exclude_patterns):
                    continue
            
            filtered_files.append(file_path)
        
        metrics.files_discovered = len(filtered_files)
        
        discovery_time = time.perf_counter() - start_time
        logger.info(
            f"Discovered {len(filtered_files)} files in {discovery_time:.2f}s"
        )
        
        return filtered_files
    
    
    async def _parse_files(
        self,
        files: List[Path],
        config: IndexingJobConfig,
        metrics: IndexingJobMetrics,
        show_progress: bool
    ) -> List[ParseResult]:
        """Parse files using the parallel pipeline"""
        start_time = time.perf_counter()
        
        # Setup progress callback for parsing
        def parse_progress_callback(current: int, total: int, stats: PipelineStats):
            metrics.files_processed = current
            metrics.entities_extracted = stats.total_entities
            metrics.relations_extracted = stats.total_relations
            
            # Notify progress callbacks
            for callback in self._progress_callbacks:
                try:
                    callback(metrics)
                except Exception as e:
                    logger.warning(f"Progress callback failed: {e}")
        
        # Parse files
        parse_results, parse_stats = self.parser_pipeline.parse_files(
            files,
            progress_callback=parse_progress_callback if show_progress else None
        )
        
        # Update metrics
        metrics.parse_time_seconds = time.perf_counter() - start_time
        metrics.files_processed = parse_stats.successful_files
        metrics.files_failed = parse_stats.failed_files
        metrics.entities_extracted = parse_stats.total_entities
        metrics.relations_extracted = parse_stats.total_relations
        
        logger.info(
            f"Parsing completed: {parse_stats.successful_files}/{len(files)} files, "
            f"{parse_stats.total_entities} entities in {metrics.parse_time_seconds:.2f}s"
        )
        
        return parse_results
    
    async def _extract_entities_relations(
        self,
        parse_results: List[ParseResult],
        metrics: IndexingJobMetrics
    ) -> tuple[List[Entity], List[Relation]]:
        """Extract entities and relations from parse results"""
        entities = []
        relations = []
        
        for result in parse_results:
            if result.success:
                entities.extend(result.entities)
                relations.extend(result.relations)
        
        logger.info(
            f"Extracted {len(entities)} entities and {len(relations)} relations"
        )
        
        return entities, relations
    
    async def _index_entities(
        self,
        entities: List[Entity],
        collection_name: str,
        metrics: IndexingJobMetrics,
        show_progress: bool
    ) -> None:
        """Index entities with embeddings"""
        start_time = time.perf_counter()
        
        # Setup progress callback for indexing
        def indexing_progress_callback(progress):
            metrics.entities_indexed = progress.successful_entities
            metrics.entities_failed = progress.failed_entities
            
            # Notify progress callbacks
            for callback in self._progress_callbacks:
                try:
                    callback(metrics)
                except Exception as e:
                    logger.warning(f"Progress callback failed: {e}")
        
        # Add progress callback to batch indexer
        self.batch_indexer.add_progress_callback(indexing_progress_callback)
        
        try:
            # Index entities
            indexing_result = await self.batch_indexer.index_entities(
                entities=entities,
                collection_name=collection_name,
                show_progress=show_progress,
                description="Indexing entities"
            )
            
            # Update metrics
            metrics.index_time_seconds = time.perf_counter() - start_time
            metrics.entities_indexed = indexing_result.successful_entities
            metrics.entities_failed = indexing_result.failed_entities
            
            # Add any errors
            if indexing_result.errors:
                metrics.errors.extend(indexing_result.errors)
            
        finally:
            # Remove progress callback
            self.batch_indexer.remove_progress_callback(indexing_progress_callback)
    
    async def _index_relations(
        self,
        relations: List[Relation],
        collection_name: str,
        metrics: IndexingJobMetrics
    ) -> None:
        """Index relations (simplified for now)"""
        # For now, just log relations - full relation indexing can be added later
        logger.info(f"Found {len(relations)} relations (relation indexing TBD)")
    
    
    async def _perform_full_entity_scan(
        self,
        config: IndexingJobConfig,
        collection_name: str,
        metrics: IndexingJobMetrics,
        show_progress: bool
    ) -> None:
        """
        Perform complete entity-level scan of the project.
        
        Args:
            config: Indexing job configuration
            collection_name: Collection name for storage
            metrics: Metrics to update
            show_progress: Whether to show progress
        """
        logger.info(f"Starting full entity scan for {config.project_path}")
        
        # Phase 1: File Discovery (entity-aware)
        files = await self._discover_files(config, metrics)
        if not files:
            logger.warning("No files found for entity scan")
            return
        
        # Phase 2: Parallel Parsing with Entity Focus
        parse_results = await self._parse_files(
            files, config, metrics, show_progress
        )
        if not parse_results:
            logger.warning("No files parsed successfully during entity scan")
            return
        
        # Phase 3: Entity Extraction and Processing
        entities, relations = await self._extract_entities_relations(
            parse_results, metrics
        )
        
        # Phase 4: Entity Storage with Batching
        if entities:
            await self._index_entities(
                entities, collection_name, metrics, show_progress
            )
        
        # Phase 5: Relation Storage (if applicable)
        if relations:
            await self._index_relations(
                relations, collection_name, metrics
            )
        
        logger.info(f"Full entity scan completed: {metrics.entities_indexed} entities indexed")
    
    async def _perform_entity_sync(
        self,
        config: IndexingJobConfig,
        collection_name: str,
        metrics: IndexingJobMetrics,
        show_progress: bool
    ) -> None:
        """
        Perform entity-level synchronization (changed entities only).
        
        Args:
            config: Indexing job configuration  
            collection_name: Collection name for storage
            metrics: Metrics to update
            show_progress: Whether to show progress
        """
        logger.info(f"Starting entity sync for {config.project_path}")
        
        # For now, use similar logic to full scan but with entity change detection
        # This will be enhanced when EntityChangeDetector is implemented
        await self._perform_full_entity_scan(config, collection_name, metrics, show_progress)
        
        # Update sync metrics
        metrics.sync_time_seconds = metrics.total_duration_seconds
        
        logger.info(f"Entity sync completed: {metrics.entities_indexed} entities processed")
    
    async def _enable_entity_sync_only(
        self,
        config: IndexingJobConfig,
        collection_name: str,
        metrics: IndexingJobMetrics
    ) -> None:
        """
        Enable sync-only mode (no immediate indexing).
        
        Args:
            config: Indexing job configuration
            collection_name: Collection name for monitoring
            metrics: Metrics to update
        """
        logger.info(f"Enabling sync-only mode for {config.project_path}")
        
        # Just enable monitoring without immediate indexing
        sync_success = await self._enable_entity_sync(
            project_path=config.project_path,
            collection_name=collection_name,
            config=config
        )
        
        if sync_success:
            logger.info("Sync-only mode enabled successfully")
            metrics.sync_time_seconds = 0.1  # Minimal time for setup
        else:
            logger.warning("Failed to enable sync-only mode")
            metrics.errors.append("Failed to enable sync-only mode")
    
    async def _enable_entity_sync(
        self,
        project_path: Path,
        collection_name: str,
        config: IndexingJobConfig
    ) -> bool:
        """
        Enable entity-level real-time synchronization.
        
        Args:
            project_path: Root path of the project
            collection_name: Name of the collection to synchronize
            config: Configuration for sync settings
            
        Returns:
            True if synchronization was enabled successfully
        """
        try:
            # Initialize sync engine if not already done
            if not self.sync_engine:
                self._initialize_sync_engine()
            
            if not self.sync_engine:
                raise Exception("Failed to initialize synchronization engine")
            
            # Start the sync engine if not running
            if not self.sync_engine.is_running:
                success = await self.sync_engine.start_monitoring()
                if not success:
                    raise Exception("Failed to start synchronization engine")
            
            # Add this project to entity-level synchronization
            success = await self.sync_engine.add_project(
                project_path=project_path,
                collection_name=collection_name,
                debounce_ms=config.sync_debounce_ms,
                start_monitoring=True
            )
            
            if success:
                logger.info(f"Enabled entity synchronization for {project_path}")
            else:
                logger.error(f"Failed to add project {project_path} to entity synchronization")
            
            return success
            
        except Exception as e:
            logger.error(f"Error enabling entity synchronization: {e}")
            return False
    
    async def _update_entity_cache_state(
        self,
        collection_name: str,
        metrics: IndexingJobMetrics
    ) -> None:
        """
        Update entity-level cache state after successful indexing.
        
        Args:
            collection_name: Collection name
            metrics: Indexing metrics
        """
        if not self.cache_manager:
            return
        
        try:
            # TODO: Implement collection-level cache metadata updates
            # For now, just log that we completed the entity operations
            logger.debug(f"Entity cache operations completed for {collection_name}: "
                        f"{metrics.entities_indexed} entities, {metrics.relations_extracted} relations")
            
        except Exception as e:
            logger.warning(f"Failed to update entity cache state: {e}")
    
    async def index_single_file(
        self,
        file_path: Path,
        project_name: str,
        collection_type: CollectionType = CollectionType.CODE,
        force_reindex: bool = False
    ) -> IndexingJobMetrics:
        """
        Index a single file (useful for file watchers).
        
        Args:
            file_path: Path to file to index
            project_name: Project name for collection
            collection_type: Type of collection to use
            force_reindex: Skip incremental checks
            
        Returns:
            Indexing metrics for the operation
        """
        # Create minimal config for single file
        config = IndexingJobConfig(
            project_path=file_path.parent,
            project_name=project_name,
            collection_type=collection_type,
            incremental=not force_reindex,
            max_workers=1,
            batch_size=10
        )
        
        # Filter to just this file
        config.include_patterns = [file_path.name]
        
        return await self.index_project(config, show_progress=False)
    
    def get_sync_status(self) -> Dict[str, Any]:
        """
        Get comprehensive synchronization status.
        
        Returns:
            Dictionary with synchronization status information
        """
        if not self.sync_engine:
            return {
                "sync_enabled": False,
                "sync_engine_initialized": False
            }
        
        status = self.sync_engine.get_status()
        status["sync_enabled"] = True
        status["sync_engine_initialized"] = True
        
        return status
    
    async def shutdown(self) -> None:
        """
        Gracefully shutdown the indexer and all associated components.
        """
        logger.info("Shutting down HybridIndexer")
        
        try:
            # Stop synchronization engine if running
            if self.sync_engine:
                await self.sync_engine.stop_monitoring()
                logger.info("Stopped synchronization engine")
        
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
        logger.info("HybridIndexer shutdown complete")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get overall performance metrics from all components"""
        metrics = {
            "parser_pipeline": {
                "max_workers": self.parser_pipeline.max_workers,
                "batch_size": self.parser_pipeline.batch_size
            },
            "batch_indexer": self.batch_indexer.get_performance_metrics(),
            "embedder": {
                "model_name": self.embedder.config.model_name,
                "device": self.embedder.config.device,
                "batch_size": self.embedder.config.batch_size
            }
        }
        
        if self.cache_manager:
            metrics["cache_manager"] = self.cache_manager.get_stats()
        
        # Add entity-level components metrics
        metrics["state_analyzer"] = {
            "staleness_threshold_hours": self.state_analyzer.staleness_threshold_hours,
            "min_health_score": self.state_analyzer.min_health_score,
            "cache_stats": self.state_analyzer.get_cache_stats()
        }
        
        metrics["scan_mode_selector"] = self.scan_mode_selector.get_status()
        
        # Add synchronization metrics if available
        if self.sync_engine:
            metrics["sync_engine"] = self.sync_engine.get_status()
        
        return metrics