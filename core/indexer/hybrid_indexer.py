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
from .incremental import IncrementalIndexer, FileChangeDetector
from .cache import CacheManager

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
        
        # Initialize incremental indexer
        self.incremental_indexer = IncrementalIndexer()
        
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
        Enable real-time synchronization for a project.
        
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
            
            # Add this project to synchronization
            success = await self.sync_engine.add_project(
                project_path=project_path,
                collection_name=collection_name,
                debounce_ms=config.sync_debounce_ms,
                start_monitoring=True
            )
            
            if success:
                logger.info(f"Enabled real-time synchronization for {project_path}")
            else:
                logger.error(f"Failed to add project {project_path} to synchronization")
            
            return success
            
        except Exception as e:
            logger.error(f"Error enabling real-time synchronization: {e}")
            return False
    
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
        # Create CollectionManager for this project
        collection_manager = CollectionManager(project_name=project_name)
        collection_name = collection_manager.get_collection_name(collection_type)
        
        # Check if collection already exists
        existing_info = await self.storage_client.get_collection_info(collection_name)
        
        if existing_info:
            logger.debug(f"Collection '{collection_name}' already exists")
            return collection_name
        
        # Create collection with proper schema
        config = collection_manager.create_collection_config(
            collection_type=collection_type,
            vector_size=self.embedder.dimensions
        )
        
        result = await self.storage_client.create_collection(config, recreate=False)
        
        if result.success:
            logger.info(f"Created collection '{collection_name}' for project '{project_name}'")
        else:
            error_msg = f"Failed to create collection '{collection_name}': {result.error}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        return collection_name
    
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
    
    async def index_project(
        self,
        config: IndexingJobConfig,
        show_progress: bool = True
    ) -> IndexingJobMetrics:
        """
        Index an entire project with comprehensive tracking.
        
        Args:
            config: Indexing job configuration
            show_progress: Whether to show progress output
            
        Returns:
            Comprehensive indexing metrics
        """
        metrics = IndexingJobMetrics()
        
        try:
            logger.info(f"Starting project indexing: {config.project_path}")
            
            # Phase 0: Ensure Collection Exists
            collection_name = await self._ensure_collection_exists(
                config.project_name, config.collection_type
            )
            
            # Phase 1: File Discovery
            files = await self._discover_files(config, metrics)
            if not files:
                logger.warning("No files found to index")
                return metrics
            
            # Phase 2: Incremental Filtering (if enabled)
            if config.incremental:
                files = await self._filter_incremental_files(
                    files, collection_name, metrics
                )
                if not files:
                    logger.info("All files are up to date")
                    return metrics
            
            # Phase 3: Parallel Parsing
            parse_results = await self._parse_files(
                files, config, metrics, show_progress
            )
            if not parse_results:
                logger.warning("No files parsed successfully")
                return metrics
            
            # Phase 4: Entity and Relation Extraction
            entities, relations = await self._extract_entities_relations(
                parse_results, metrics
            )
            
            # Phase 5: Embedding and Storage
            if entities:
                await self._index_entities(
                    entities, collection_name, metrics, show_progress
                )
            
            # Phase 6: Relation Storage (if applicable)
            if relations:
                await self._index_relations(
                    relations, collection_name, metrics
                )
            
            # Phase 7: Enable Real-time Synchronization (if configured)
            if config.enable_realtime_sync:
                sync_success = await self.enable_realtime_sync(
                    project_path=config.project_path,
                    collection_name=collection_name,
                    config=config
                )
                if sync_success:
                    logger.info(f"Real-time synchronization enabled for {config.project_path}")
                else:
                    logger.warning(f"Failed to enable real-time synchronization for {config.project_path}")
            
            # Update cache state
            if config.enable_caching and self.cache_manager:
                await self._update_cache_state(files, collection_name, metrics)
            
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
    
    async def _filter_incremental_files(
        self,
        files: List[Path],
        collection_name: str,
        metrics: IndexingJobMetrics
    ) -> List[Path]:
        """Filter files for incremental indexing"""
        if not self.incremental_indexer:
            return files
        
        start_time = time.perf_counter()
        
        # Get changed files
        changed_files = await self.incremental_indexer.get_changed_files(
            files, collection_name
        )
        
        metrics.files_skipped = len(files) - len(changed_files)
        
        filter_time = time.perf_counter() - start_time
        logger.info(
            f"Incremental filtering: {len(changed_files)}/{len(files)} files "
            f"need processing ({filter_time:.2f}s)"
        )
        
        return changed_files
    
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
    
    async def _update_cache_state(
        self,
        files: List[Path],
        collection_name: str,
        metrics: IndexingJobMetrics
    ) -> None:
        """Update cache and incremental state after successful indexing"""
        # Update file cache entries
        if self.cache_manager:
            for file_path in files:
                try:
                    await self.cache_manager.update_file_cache(
                        file_path, collection_name
                    )
                except Exception as e:
                    logger.warning(f"Failed to update cache for {file_path}: {e}")
        
        # Update incremental indexer state
        if self.incremental_indexer:
            # Calculate entities per file (rough estimate)
            entities_per_file = metrics.entities_indexed // max(metrics.files_processed, 1)
            relations_per_file = metrics.relations_extracted // max(metrics.files_processed, 1)
            
            for file_path in files:
                try:
                    await self.incremental_indexer.update_file_state(
                        file_path=file_path,
                        collection_name=collection_name,
                        entity_count=entities_per_file,
                        relation_count=relations_per_file,
                        success=True
                    )
                except Exception as e:
                    logger.warning(f"Failed to update incremental state for {file_path}: {e}")
    
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
        
        # Add synchronization metrics if available
        if self.sync_engine:
            metrics["sync_engine"] = self.sync_engine.get_status()
        
        return metrics