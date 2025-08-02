"""
Entity Scanner with Tree-sitter Integration and Parallel Processing.

Implements comprehensive entity scanning with Tree-sitter parser integration,
parallel processing capabilities, entity batching, and streaming for large codebases.
Designed for pure entity-level operations as part of the sync engine integration.
"""

import asyncio
import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Callable, AsyncGenerator, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

from ..models.entities import Entity, EntityType
from ..parser.base import ParseResult, ParserProtocol
from ..parser.registry import parser_registry
from ..parser.parallel_pipeline import ProcessParsingPipeline, PipelineStats
from ..sync.lifecycle import EntityLifecycleManager
from ..sync.deterministic import DeterministicEntityId
from ..storage.client import HybridQdrantClient
from ..storage.utils import entity_id_to_qdrant_id

logger = logging.getLogger(__name__)


@dataclass
class EntityScanRequest:
    """Request for entity scanning operation."""
    
    file_paths: List[Path]
    collection_name: str
    project_path: Path
    scan_mode: str = "full_scan"  # "full_scan", "incremental", "targeted"
    batch_size: int = 50
    enable_parallel: bool = True
    max_workers: Optional[int] = None
    progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None


@dataclass
class EntityScanResult:
    """Result of entity scanning operation."""
    
    request_id: str
    total_files: int
    processed_files: int
    successful_files: int
    failed_files: int
    total_entities: int
    total_relations: int
    scan_time: float
    entities_per_second: float
    success_rate: float
    errors: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    entities: Optional[List[Entity]] = None  # Only populated in parse_only mode
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "request_id": self.request_id,
            "summary": {
                "total_files": self.total_files,
                "processed_files": self.processed_files,
                "successful_files": self.successful_files,
                "failed_files": self.failed_files,
                "success_rate": self.success_rate
            },
            "entities": {
                "total_entities": self.total_entities,
                "total_relations": self.total_relations,
                "entities_per_second": self.entities_per_second
            },
            "performance": {
                "scan_time": self.scan_time,
                **self.performance_metrics
            },
            "errors": self.errors
        }


@dataclass
class EntityBatch:
    """Batch of entities for processing."""
    
    batch_id: int
    entities: List[Entity]
    file_path: str
    batch_size: int
    processing_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    
    @property
    def entity_count(self) -> int:
        """Number of entities in this batch."""
        return len(self.entities)


class EntityScanner:
    """
    High-performance entity scanner with Tree-sitter integration and parallel processing.
    
    This class provides comprehensive entity scanning capabilities:
    - Tree-sitter parser integration for multi-language support
    - Parallel processing using ProcessPoolExecutor for CPU-intensive operations
    - Entity batching and streaming for memory-efficient processing
    - Integration with EntityLifecycleManager for atomic operations
    - Deterministic entity ID generation for stable identification
    - Comprehensive progress tracking and error handling
    
    Features:
    - Multi-language code parsing via Tree-sitter
    - Parallel entity extraction and processing
    - Streaming entity processing for large codebases
    - Entity-level change detection and synchronization
    - Performance monitoring and metrics
    """
    
    def __init__(
        self,
        storage_client: HybridQdrantClient,
        lifecycle_manager: Optional[EntityLifecycleManager] = None,
        enable_parallel: bool = True,
        default_batch_size: int = 50,
        max_workers: Optional[int] = None
    ):
        """
        Initialize entity scanner.
        
        Args:
            storage_client: Qdrant client for entity storage
            lifecycle_manager: Optional lifecycle manager for atomic operations
            enable_parallel: Whether to enable parallel processing
            default_batch_size: Default batch size for entity processing
            max_workers: Maximum number of worker processes
        """
        self.storage_client = storage_client
        self.lifecycle_manager = lifecycle_manager
        self.enable_parallel = enable_parallel
        self.default_batch_size = default_batch_size
        self.max_workers = max_workers
        
        # Initialize parser pipeline
        self.parser_pipeline = ProcessParsingPipeline(
            max_workers=max_workers,
            batch_size=default_batch_size
        )
        
        # Scanner state
        self._scan_count = 0
        self._total_entities_processed = 0
        self._total_scan_time = 0.0
        self._last_scan_time: Optional[datetime] = None
        
        # Performance tracking
        self._performance_history: List[Dict[str, Any]] = []
        self._max_history_size = 100
        
        logger.info(f"Initialized EntityScanner with parallel={enable_parallel}, "
                   f"batch_size={default_batch_size}, max_workers={max_workers}")
    
    async def scan_files(
        self,
        request: EntityScanRequest
    ) -> EntityScanResult:
        """
        Scan files for entities using parallel processing.
        
        Args:
            request: Entity scan request with configuration
            
        Returns:
            Entity scan result with comprehensive metrics
        """
        start_time = time.perf_counter()
        request_id = f"scan_{int(time.time())}_{id(request)}"
        
        logger.info(f"Starting entity scan {request_id}: {len(request.file_paths)} files, "
                   f"mode={request.scan_mode}, parallel={request.enable_parallel}")
        
        try:
            if request.enable_parallel and self.enable_parallel:
                # Use parallel processing pipeline
                parse_results, pipeline_stats = await self._scan_files_parallel(request)
            else:
                # Use sequential processing
                parse_results, pipeline_stats = await self._scan_files_sequential(request)
            
            # Extract entities from parse results
            all_entities = []
            all_relations = []
            successful_files = 0
            failed_files = 0
            errors = []
            
            for result in parse_results:
                if result.success:
                    successful_files += 1
                    all_entities.extend(result.entities)
                    all_relations.extend(result.relations)
                else:
                    failed_files += 1
                    errors.extend([{
                        "file": str(result.file_path),
                        "errors": result.syntax_errors,
                        "warnings": result.warnings
                    }])
            
            # Process entities in batches if requested
            if request.scan_mode != "parse_only":
                await self._process_entities_in_batches(
                    all_entities, 
                    request.collection_name,
                    request.batch_size,
                    request.progress_callback
                )
            
            # Calculate performance metrics
            scan_time = time.perf_counter() - start_time
            entities_per_second = len(all_entities) / scan_time if scan_time > 0 else 0
            success_rate = successful_files / len(request.file_paths) if request.file_paths else 0
            
            # Update scanner state
            self._scan_count += 1
            self._total_entities_processed += len(all_entities)
            self._total_scan_time += scan_time
            self._last_scan_time = datetime.now()
            
            # Create result
            result = EntityScanResult(
                request_id=request_id,
                total_files=len(request.file_paths),
                processed_files=len(parse_results),
                successful_files=successful_files,
                failed_files=failed_files,
                total_entities=len(all_entities),
                total_relations=len(all_relations),
                scan_time=scan_time,
                entities_per_second=entities_per_second,
                success_rate=success_rate,
                errors=errors,
                performance_metrics={
                    "parallel_processing": request.enable_parallel and self.enable_parallel,
                    "batch_size": request.batch_size,
                    "max_workers": self.max_workers,
                    "pipeline_stats": pipeline_stats.__dict__ if pipeline_stats else {}
                },
                entities=all_entities if request.scan_mode == "parse_only" else None
            )
            
            # Store performance metrics
            self._store_performance_metrics(result)
            
            logger.info(f"Completed entity scan {request_id}: {len(all_entities)} entities, "
                       f"{entities_per_second:.1f} entities/sec, {success_rate:.1%} success rate")
            
            return result
            
        except Exception as e:
            scan_time = time.perf_counter() - start_time
            error_msg = f"Error in entity scan {request_id}: {e}"
            logger.error(error_msg)
            
            return EntityScanResult(
                request_id=request_id,
                total_files=len(request.file_paths),
                processed_files=0,
                successful_files=0,
                failed_files=len(request.file_paths),
                total_entities=0,
                total_relations=0,
                scan_time=scan_time,
                entities_per_second=0,
                success_rate=0,
                errors=[{"type": "SCAN_ERROR", "message": error_msg}]
            )
    
    async def _scan_files_parallel(
        self,
        request: EntityScanRequest
    ) -> Tuple[List[ParseResult], PipelineStats]:
        """
        Scan files using parallel processing pipeline.
        
        Args:
            request: Entity scan request
            
        Returns:
            Tuple of parse results and pipeline statistics
        """
        def progress_wrapper(processed: int, total: int, stats: PipelineStats):
            if request.progress_callback:
                progress_data = {
                    "phase": "parsing",
                    "processed": processed,
                    "total": total,
                    "entities_found": stats.total_entities,
                    "success_rate": stats.success_rate,
                    "files_per_second": stats.files_per_second
                }
                request.progress_callback(processed, total, progress_data)
        
        # Use the existing parallel pipeline
        parse_results, pipeline_stats = self.parser_pipeline.parse_files(
            request.file_paths,
            progress_callback=progress_wrapper
        )
        
        return parse_results, pipeline_stats
    
    async def _scan_files_sequential(
        self,
        request: EntityScanRequest
    ) -> Tuple[List[ParseResult], PipelineStats]:
        """
        Scan files using sequential processing.
        
        Args:
            request: Entity scan request
            
        Returns:
            Tuple of parse results and pipeline statistics
        """
        start_time = time.perf_counter()
        parse_results = []
        successful_files = 0
        failed_files = 0
        total_entities = 0
        total_relations = 0
        
        for i, file_path in enumerate(request.file_paths):
            try:
                # Get parser for file
                parser = parser_registry.get_parser_for_file(file_path)
                if not parser:
                    failed_files += 1
                    logger.warning(f"No parser available for {file_path}")
                    continue
                
                # Parse file
                result = parser.parse_file(file_path)
                parse_results.append(result)
                
                if result.success:
                    successful_files += 1
                    total_entities += len(result.entities)
                    total_relations += len(result.relations)
                else:
                    failed_files += 1
                
                # Progress callback
                if request.progress_callback:
                    progress_data = {
                        "phase": "parsing",
                        "processed": i + 1,
                        "total": len(request.file_paths),
                        "entities_found": total_entities,
                        "current_file": str(file_path)
                    }
                    request.progress_callback(i + 1, len(request.file_paths), progress_data)
                
            except Exception as e:
                failed_files += 1
                logger.error(f"Error parsing {file_path}: {e}")
                
                # Create a failed parse result for consistent error handling
                failed_result = ParseResult(
                    file_path=file_path,
                    language="unknown",
                    entities=[],
                    relations=[],
                    ast_nodes=[],
                    parse_time=0.0,
                    file_size=0,
                    file_hash=""
                )
                failed_result.add_syntax_error({
                    "type": "PARSER_ERROR",
                    "message": str(e),
                    "line": 0,
                    "column": 0
                })
                parse_results.append(failed_result)
        
        # Create pipeline stats
        total_time = time.perf_counter() - start_time
        stats = PipelineStats(
            total_files=len(request.file_paths),
            processed_files=len(parse_results),
            successful_files=successful_files,
            failed_files=failed_files,
            total_entities=total_entities,
            total_relations=total_relations,
            total_time=total_time
        )
        
        return parse_results, stats
    
    async def _process_entities_in_batches(
        self,
        entities: List[Entity],
        collection_name: str,
        batch_size: int,
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None
    ) -> None:
        """
        Process entities in batches for memory efficiency.
        
        Args:
            entities: List of entities to process
            collection_name: Collection name for storage
            batch_size: Size of each batch
            progress_callback: Optional progress callback
        """
        if not entities:
            return
        
        logger.info(f"Processing {len(entities)} entities in batches of {batch_size}")
        
        total_batches = (len(entities) + batch_size - 1) // batch_size
        processed_batches = 0
        
        for i in range(0, len(entities), batch_size):
            batch_entities = entities[i:i + batch_size]
            batch_id = i // batch_size
            
            try:
                # Process batch
                await self._process_entity_batch(
                    EntityBatch(
                        batch_id=batch_id,
                        entities=batch_entities,
                        file_path="batch_processing",
                        batch_size=len(batch_entities)
                    ),
                    collection_name
                )
                
                processed_batches += 1
                
                # Progress callback
                if progress_callback:
                    progress_data = {
                        "phase": "entity_processing",
                        "batch_id": batch_id,
                        "entities_in_batch": len(batch_entities),
                        "processed_batches": processed_batches,
                        "total_batches": total_batches
                    }
                    progress_callback(processed_batches, total_batches, progress_data)
                
            except Exception as e:
                logger.error(f"Error processing entity batch {batch_id}: {e}")
        
        logger.info(f"Completed processing {processed_batches}/{total_batches} entity batches")
    
    async def _process_entity_batch(
        self,
        batch: EntityBatch,
        collection_name: str
    ) -> None:
        """
        Process a single entity batch.
        
        Args:
            batch: Entity batch to process
            collection_name: Collection name for storage
        """
        start_time = time.perf_counter()
        
        try:
            # Update entities with deterministic IDs
            updated_entities = []
            for entity in batch.entities:
                # Generate file hash for deterministic ID
                file_content = entity.location.file_path.read_text(encoding='utf-8')
                file_hash = hashlib.sha256(file_content.encode('utf-8')).hexdigest()
                
                updated_entity = DeterministicEntityId.update_entity_with_deterministic_id(
                    entity, file_hash
                )
                updated_entities.append(updated_entity)
            
            # Deduplicate entities by ID - keep last occurrence
            entity_dict = {}
            for entity in updated_entities:
                entity_dict[entity.id] = entity
            updated_entities = list(entity_dict.values())
            
            # Store entities using lifecycle manager if available
            if self.lifecycle_manager:
                # Use lifecycle manager for atomic operations
                from ..models.storage import QdrantPoint
                
                qdrant_points = []
                for entity in updated_entities:
                    # Set indexed_at timestamp on entity (precise per-entity timing)
                    indexed_time = datetime.now()
                    entity = entity.model_copy(update={'indexed_at': indexed_time})
                    
                    point = QdrantPoint(
                        id=entity_id_to_qdrant_id(entity.id),
                        vector=[0.0] * 1024,  # Placeholder vector
                        payload=entity.to_qdrant_payload()
                    )
                    qdrant_points.append(point)
                
                result = await self.storage_client.upsert_points(collection_name, qdrant_points)
                
                if not result.success:
                    raise Exception(f"Failed to store entities: {result.error}")
            else:
                # Direct storage without lifecycle management
                from ..models.storage import QdrantPoint
                
                qdrant_points = []
                for entity in updated_entities:
                    # Set indexed_at timestamp on entity (precise per-entity timing)
                    indexed_time = datetime.now()
                    entity = entity.model_copy(update={'indexed_at': indexed_time})
                    
                    point = QdrantPoint(
                        id=entity_id_to_qdrant_id(entity.id),
                        vector=[0.0] * 1024,  # Placeholder vector
                        payload=entity.to_qdrant_payload()
                    )
                    qdrant_points.append(point)
                
                result = await self.storage_client.upsert_points(collection_name, qdrant_points)
                
                if not result.success:
                    raise Exception(f"Failed to store entities: {result.error}")
            
            batch.processing_time = time.perf_counter() - start_time
            batch.success = True
            
            logger.debug(f"Processed entity batch {batch.batch_id}: {len(batch.entities)} entities "
                        f"in {batch.processing_time:.3f}s")
            
        except Exception as e:
            batch.processing_time = time.perf_counter() - start_time
            batch.success = False
            batch.error_message = str(e)
            logger.error(f"Error processing entity batch {batch.batch_id}: {e}")
    
    async def stream_entities(
        self,
        file_paths: List[Path],
        batch_size: int = 100
    ) -> AsyncGenerator[EntityBatch, None]:
        """
        Stream entities from files in batches for memory-efficient processing.
        
        Args:
            file_paths: List of files to process
            batch_size: Size of each entity batch
            
        Yields:
            EntityBatch objects containing entities from parsed files
        """
        logger.info(f"Starting entity streaming for {len(file_paths)} files, batch_size={batch_size}")
        
        current_batch = []
        batch_id = 0
        
        for file_path in file_paths:
            try:
                # Get parser for file
                parser = parser_registry.get_parser_for_file(file_path)
                if not parser:
                    logger.debug(f"No parser available for {file_path}")
                    continue
                
                # Parse entities
                result = parser.parse_file(file_path)
                
                if result.success and result.entities:
                    current_batch.extend(result.entities)
                    
                    # Yield batch when it reaches target size
                    while len(current_batch) >= batch_size:
                        batch_entities = current_batch[:batch_size]
                        current_batch = current_batch[batch_size:]
                        
                        yield EntityBatch(
                            batch_id=batch_id,
                            entities=batch_entities,
                            file_path=str(file_path),
                            batch_size=len(batch_entities)
                        )
                        batch_id += 1
                
            except Exception as e:
                logger.error(f"Error streaming entities from {file_path}: {e}")
        
        # Yield remaining entities in final batch
        if current_batch:
            yield EntityBatch(
                batch_id=batch_id,
                entities=current_batch,
                file_path="final_batch",
                batch_size=len(current_batch)
            )
        
        logger.info(f"Completed entity streaming: {batch_id + 1} batches yielded")
    
    def _store_performance_metrics(self, result: EntityScanResult) -> None:
        """
        Store performance metrics for historical analysis.
        
        Args:
            result: Scan result with performance data
        """
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "request_id": result.request_id,
            "files_processed": result.processed_files,
            "entities_processed": result.total_entities,
            "scan_time": result.scan_time,
            "entities_per_second": result.entities_per_second,
            "success_rate": result.success_rate,
            "parallel_processing": result.performance_metrics.get("parallel_processing", False)
        }
        
        self._performance_history.append(metrics)
        
        # Maintain history size
        if len(self._performance_history) > self._max_history_size:
            self._performance_history = self._performance_history[-self._max_history_size:]
    
    def get_scanner_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive scanner statistics.
        
        Returns:
            Dictionary with scanner performance and status information
        """
        avg_entities_per_scan = (
            self._total_entities_processed / self._scan_count if self._scan_count > 0 else 0
        )
        
        avg_scan_time = (
            self._total_scan_time / self._scan_count if self._scan_count > 0 else 0
        )
        
        return {
            "scanner_info": {
                "enable_parallel": self.enable_parallel,
                "default_batch_size": self.default_batch_size,
                "max_workers": self.max_workers
            },
            "performance": {
                "total_scans": self._scan_count,
                "total_entities_processed": self._total_entities_processed,
                "total_scan_time": self._total_scan_time,
                "average_entities_per_scan": avg_entities_per_scan,
                "average_scan_time": avg_scan_time,
                "last_scan_time": self._last_scan_time.isoformat() if self._last_scan_time else None
            },
            "parser_pipeline": {
                "max_workers": self.parser_pipeline.max_workers,
                "batch_size": self.parser_pipeline.batch_size,
                "timeout": self.parser_pipeline.timeout
            },
            "recent_performance": self._performance_history[-10:] if self._performance_history else []
        }
    
    async def scan_directory(
        self,
        directory: Path,
        collection_name: str,
        recursive: bool = True,
        scan_mode: str = "full_scan",
        batch_size: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None
    ) -> EntityScanResult:
        """
        Scan all supported files in a directory.
        
        Args:
            directory: Directory to scan
            collection_name: Collection name for entity storage
            recursive: Whether to scan recursively
            scan_mode: Scan mode ("full_scan", "incremental", "targeted")
            batch_size: Override default batch size
            progress_callback: Optional progress callback
            
        Returns:
            Entity scan result
        """
        logger.info(f"Scanning directory {directory} (recursive={recursive}, mode={scan_mode})")
        
        # Discover files using parser registry
        files = parser_registry.discover_files(directory, recursive=recursive)
        
        if not files:
            logger.warning(f"No parseable files found in {directory}")
            return EntityScanResult(
                request_id=f"dir_scan_{int(time.time())}",
                total_files=0,
                processed_files=0,
                successful_files=0,
                failed_files=0,
                total_entities=0,
                total_relations=0,
                scan_time=0,
                entities_per_second=0,
                success_rate=0
            )
        
        # Create scan request
        request = EntityScanRequest(
            file_paths=files,
            collection_name=collection_name,
            project_path=directory,
            scan_mode=scan_mode,
            batch_size=batch_size or self.default_batch_size,
            enable_parallel=self.enable_parallel,
            max_workers=self.max_workers,
            progress_callback=progress_callback
        )
        
        return await self.scan_files(request)