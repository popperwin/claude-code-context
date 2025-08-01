"""
Entity Lifecycle Integration with EntityScanner and Sync Engine.

Provides comprehensive integration between EntityScanner, EntityLifecycleManager,
and ProjectCollectionSyncEngine for atomic entity operations, entity-to-collection
mapping, and cascade operations in the pure entity-level synchronization system.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

from ..models.entities import Entity, EntityType
from ..storage.client import HybridQdrantClient
from ..storage.schemas import CollectionManager, CollectionType
from ..sync.lifecycle import EntityLifecycleManager
from ..sync.engine import ProjectCollectionSyncEngine, ProjectSyncState
from ..sync.events import FileSystemEvent, EventType
from .entity_scanner import EntityScanner, EntityScanRequest, EntityScanResult, EntityBatch
from .entity_detector import EntityChangeDetector, EntityChangeInfo

logger = logging.getLogger(__name__)


@dataclass
class EntityOperationResult:
    """Result of an entity lifecycle operation."""
    
    operation_type: str  # "bulk_create", "bulk_update", "bulk_delete", "atomic_replace"
    success: bool
    entities_affected: int = 0
    entities_created: int = 0
    entities_updated: int = 0
    entities_deleted: int = 0
    operation_time_ms: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EntityMappingState:
    """State information for entity-to-collection mapping."""
    
    total_entities: int = 0
    mapped_entities: int = 0
    orphaned_entities: int = 0
    inconsistent_mappings: int = 0
    last_sync_time: Optional[datetime] = None
    health_score: float = 1.0  # 0.0 to 1.0


class EntityLifecycleIntegrator:
    """
    High-level integrator for entity lifecycle operations with sync engine.
    
    This class provides comprehensive integration between:
    - EntityScanner for entity discovery and processing
    - EntityLifecycleManager for atomic entity operations
    - ProjectCollectionSyncEngine for real-time monitoring
    - EntityChangeDetector for intelligent change detection
    
    Features:
    - Atomic entity operations with rollback support
    - Entity-to-collection mapping maintenance
    - Cascade operations for entity relationships
    - Real-time sync engine integration
    - Comprehensive error handling and recovery
    """
    
    def __init__(
        self,
        storage_client: HybridQdrantClient,
        project_path: Path,
        collection_name: str,
        enable_real_time_sync: bool = True,
        batch_size: int = 50,
        collection_manager: Optional[CollectionManager] = None
    ):
        """
        Initialize entity lifecycle integrator.
        
        Args:
            storage_client: Qdrant client for entity storage
            project_path: Root path of the project
            collection_name: Name of the collection to manage
            enable_real_time_sync: Whether to enable real-time synchronization
            batch_size: Default batch size for entity operations
            collection_manager: Optional CollectionManager for collection lifecycle operations
        """
        self.storage_client = storage_client
        self.project_path = Path(project_path).resolve()
        self.collection_name = collection_name
        self.enable_real_time_sync = enable_real_time_sync
        self.batch_size = batch_size
        
        # Initialize collection manager for lifecycle operations
        if collection_manager is None:
            # Extract project name from collection name (assumes format: "project-name-code")
            project_name = collection_name.rsplit('-', 1)[0] if '-' in collection_name else collection_name
            collection_manager = CollectionManager(project_name=project_name)
        self.collection_manager = collection_manager
        
        # Initialize core components
        self.lifecycle_manager = EntityLifecycleManager(
            storage_client=storage_client,
            collection_name=collection_name,
            project_path=project_path
        )
        
        self.entity_scanner = EntityScanner(
            storage_client=storage_client,
            lifecycle_manager=self.lifecycle_manager,
            enable_parallel=True,
            default_batch_size=batch_size
        )
        
        self.change_detector = EntityChangeDetector(
            storage_client=storage_client
        )
        
        # Sync engine (optional for real-time monitoring)
        self.sync_engine: Optional[ProjectCollectionSyncEngine] = None
        if enable_real_time_sync:
            self.sync_engine = ProjectCollectionSyncEngine(storage_client)
        
        # State tracking
        self._operation_count = 0
        self._total_operations_time = 0.0
        self._last_operation_time: Optional[datetime] = None
        self._mapping_state = EntityMappingState()
        
        logger.info(f"Initialized EntityLifecycleIntegrator for {project_path} -> {collection_name}")
    
    async def _ensure_collection_exists(self) -> str:
        """
        Ensure the target collection exists before performing operations.
        
        This is the critical fix for the "0 results" bug - EntityLifecycleIntegrator
        must create collections before trying to store entities in them.
        
        Returns:
            The collection name that was ensured to exist
        """
        try:
            collection_name = await self.collection_manager.ensure_collection_exists(
                collection_type=CollectionType.CODE,
                storage_client=self.storage_client
            )
            
            # Update our collection name in case it was normalized
            if collection_name != self.collection_name:
                logger.info(f"Collection name updated from '{self.collection_name}' to '{collection_name}'")
                self.collection_name = collection_name
                
                # Update lifecycle manager with new collection name
                self.lifecycle_manager.collection_name = collection_name
            
            return collection_name
            
        except Exception as e:
            error_msg = f"Failed to ensure collection exists: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    async def bulk_entity_create(
        self,
        file_paths: List[Path],
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None
    ) -> EntityOperationResult:
        """
        Create entities from multiple files with atomic operations.
        
        Args:
            file_paths: List of files to process for entity creation
            progress_callback: Optional progress callback
            
        Returns:
            Entity operation result with comprehensive metrics
        """
        start_time = time.perf_counter()
        operation_id = f"bulk_create_{int(time.time())}"
        
        logger.info(f"Starting bulk entity creation {operation_id}: {len(file_paths)} files")
        
        try:
            # CRITICAL FIX: Ensure collection exists before creating entities
            await self._ensure_collection_exists()
            # Create scan request
            scan_request = EntityScanRequest(
                file_paths=file_paths,
                collection_name=self.collection_name,
                project_path=self.project_path,
                scan_mode="full_scan",
                batch_size=self.batch_size,
                progress_callback=progress_callback
            )
            
            # Scan files for entities
            scan_result = await self.entity_scanner.scan_files(scan_request)
            
            if not scan_result.success_rate > 0:
                return EntityOperationResult(
                    operation_type="bulk_create",
                    success=False,
                    error_message=f"Entity scanning failed: {len(scan_result.errors)} errors",
                    operation_time_ms=(time.perf_counter() - start_time) * 1000
                )
            
            # Update tracking
            self._operation_count += 1
            operation_time_ms = (time.perf_counter() - start_time) * 1000
            self._total_operations_time += operation_time_ms
            self._last_operation_time = datetime.now()
            
            
            # Update mapping state
            self._mapping_state.total_entities += scan_result.total_entities
            self._mapping_state.mapped_entities += scan_result.total_entities
            self._mapping_state.last_sync_time = datetime.now()
            
            logger.info(f"Completed bulk entity creation {operation_id}: "
                       f"{scan_result.total_entities} entities in {operation_time_ms:.1f}ms")
            
            return EntityOperationResult(
                operation_type="bulk_create",
                success=True,
                entities_affected=scan_result.total_entities,
                entities_created=scan_result.total_entities,
                operation_time_ms=operation_time_ms,
                metadata={
                    "files_processed": scan_result.processed_files,
                    "success_rate": scan_result.success_rate,
                    "entities_per_second": scan_result.entities_per_second
                }
            )
            
        except Exception as e:
            operation_time_ms = (time.perf_counter() - start_time) * 1000
            error_msg = f"Error in bulk entity creation {operation_id}: {e}"
            logger.error(error_msg)
            
            return EntityOperationResult(
                operation_type="bulk_create",
                success=False,
                error_message=error_msg,
                operation_time_ms=operation_time_ms
            )
    
    async def bulk_entity_update(
        self,
        file_paths: List[Path],
        detect_changes: bool = True,
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None
    ) -> EntityOperationResult:
        """
        Update entities from multiple files with change detection and atomic operations.
        
        Args:
            file_paths: List of files to process for entity updates
            detect_changes: Whether to use change detection to optimize updates
            progress_callback: Optional progress callback
            
        Returns:
            Entity operation result with comprehensive metrics
        """
        start_time = time.perf_counter()
        operation_id = f"bulk_update_{int(time.time())}"
        
        logger.info(f"Starting bulk entity update {operation_id}: {len(file_paths)} files, "
                   f"change_detection={detect_changes}")
        
        try:
            # CRITICAL FIX: Ensure collection exists before updating entities
            await self._ensure_collection_exists()
            entities_updated = 0
            entities_created = 0
            entities_deleted = 0
            
            # Process files individually for atomic replacement
            # Pure entity-level approach - let EntityLifecycleManager handle all optimization
            for i, file_path in enumerate(file_paths):
                try:
                    # CRITICAL FIX: Use absolute paths consistently (entities are stored with absolute paths)
                    absolute_path = file_path.resolve() if not file_path.is_absolute() else file_path
                    
                    event = FileSystemEvent.create_file_modified(absolute_path)
                    result = await self.lifecycle_manager.handle_file_modification(event)
                    
                    if result["success"]:
                        entities_created += result.get("entities_added", 0)
                        entities_deleted += result.get("entities_removed", 0)
                        entities_updated += result.get("entities_unchanged", 0)
                    
                    # Progress callback
                    if progress_callback:
                        progress_data = {
                            "phase": "entity_update",
                            "current_file": str(file_path),
                            "entities_updated": entities_updated,
                            "entities_created": entities_created,
                            "entities_deleted": entities_deleted
                        }
                        progress_callback(i + 1, len(file_paths), progress_data)
                
                except Exception as e:
                    logger.error(f"Error updating entities for {file_path}: {e}")
                    continue
            
            # Update tracking
            self._operation_count += 1
            operation_time_ms = (time.perf_counter() - start_time) * 1000
            self._total_operations_time += operation_time_ms
            self._last_operation_time = datetime.now()
            
            # Update mapping state
            total_affected = entities_updated + entities_created + entities_deleted
            self._mapping_state.total_entities += entities_created - entities_deleted
            self._mapping_state.last_sync_time = datetime.now()
            
            logger.info(f"Completed bulk entity update {operation_id}: "
                       f"{total_affected} entities affected in {operation_time_ms:.1f}ms")
            
            return EntityOperationResult(
                operation_type="bulk_update",
                success=True,
                entities_affected=total_affected,
                entities_created=entities_created,
                entities_updated=entities_updated,
                entities_deleted=entities_deleted,
                operation_time_ms=operation_time_ms,
                metadata={
                    "files_processed": len(file_paths),
                    "change_detection_used": detect_changes
                }
            )
            
        except Exception as e:
            operation_time_ms = (time.perf_counter() - start_time) * 1000
            error_msg = f"Error in bulk entity update {operation_id}: {e}"
            logger.error(error_msg)
            
            return EntityOperationResult(
                operation_type="bulk_update",
                success=False,
                error_message=error_msg,
                operation_time_ms=operation_time_ms
            )
    
    async def bulk_entity_delete(
        self,
        file_paths: List[Path],
        cascade_relationships: bool = True
    ) -> EntityOperationResult:
        """
        Delete entities from multiple files with cascade operations.
        
        Args:
            file_paths: List of files to process for entity deletion
            cascade_relationships: Whether to cascade delete related entities
            
        Returns:
            Entity operation result with comprehensive metrics
        """
        start_time = time.perf_counter()
        operation_id = f"bulk_delete_{int(time.time())}"
        
        logger.info(f"Starting bulk entity delete {operation_id}: {len(file_paths)} files, "
                   f"cascade={cascade_relationships}")
        
        try:
            # CRITICAL FIX: Ensure collection exists before deleting entities
            await self._ensure_collection_exists()
            entities_deleted = 0
            
            # Process file deletions
            for file_path in file_paths:
                try:
                    # CRITICAL FIX: Use absolute paths consistently (entities are stored with absolute paths)
                    absolute_path = file_path.resolve() if not file_path.is_absolute() else file_path
                    
                    event = FileSystemEvent.create_file_deleted(absolute_path)
                    result = await self.lifecycle_manager.handle_file_deletion(event)
                    
                    if result["success"]:
                        entities_deleted += result["entities_deleted"]
                
                except Exception as e:
                    logger.error(f"Error deleting entities for {file_path}: {e}")
                    continue
            
            # Update tracking
            self._operation_count += 1
            operation_time_ms = (time.perf_counter() - start_time) * 1000
            self._total_operations_time += operation_time_ms
            self._last_operation_time = datetime.now()
            
            # Update mapping state
            self._mapping_state.total_entities -= entities_deleted
            self._mapping_state.last_sync_time = datetime.now()
            
            logger.info(f"Completed bulk entity delete {operation_id}: "
                       f"{entities_deleted} entities deleted in {operation_time_ms:.1f}ms")
            
            return EntityOperationResult(
                operation_type="bulk_delete",
                success=True,
                entities_affected=entities_deleted,
                entities_deleted=entities_deleted,
                operation_time_ms=operation_time_ms,
                metadata={
                    "files_processed": len(file_paths),
                    "cascade_used": cascade_relationships
                }
            )
            
        except Exception as e:
            operation_time_ms = (time.perf_counter() - start_time) * 1000
            error_msg = f"Error in bulk entity delete {operation_id}: {e}"
            logger.error(error_msg)
            
            return EntityOperationResult(
                operation_type="bulk_delete",
                success=False,
                error_message=error_msg,
                operation_time_ms=operation_time_ms
            )
    
    async def atomic_entity_replacement(
        self,
        file_path: Path,
        new_entities: Optional[List[Entity]] = None
    ) -> EntityOperationResult:
        """
        Perform atomic entity replacement for a single file.
        
        Args:
            file_path: Path to the file to replace entities for
            new_entities: Optional list of new entities (if None, will scan file)
            
        Returns:
            Entity operation result with comprehensive metrics
        """
        start_time = time.perf_counter()
        operation_id = f"atomic_replace_{int(time.time())}"
        
        logger.info(f"Starting atomic entity replacement {operation_id}: {file_path}")
        
        try:
            # CRITICAL FIX: Ensure collection exists before replacing entities
            await self._ensure_collection_exists()
            
            # CRITICAL FIX: Use absolute paths consistently (entities are stored with absolute paths)
            absolute_path = file_path.resolve() if not file_path.is_absolute() else file_path
            normalized_path = str(absolute_path)
            
            # Get existing entities for the file
            existing_entity_ids = await self.lifecycle_manager._get_entities_for_file(normalized_path)
            
            # Scan for new entities if not provided
            if new_entities is None:
                scan_request = EntityScanRequest(
                    file_paths=[file_path],
                    collection_name=self.collection_name,
                    project_path=self.project_path,
                    scan_mode="parse_only",  # Use parse_only to get entities without storing them
                    batch_size=self.batch_size
                )
                
                scan_result = await self.entity_scanner.scan_files(scan_request)
                new_entities = scan_result.entities or []
                
            # Perform atomic replacement
            replacement_result = await self.lifecycle_manager.atomic_entity_replacement(
                normalized_path,
                existing_entity_ids,
                new_entities or []
            )
            
            # Update tracking
            self._operation_count += 1
            operation_time_ms = (time.perf_counter() - start_time) * 1000
            self._total_operations_time += operation_time_ms
            self._last_operation_time = datetime.now()
            
            # Update mapping state
            entities_added = replacement_result.get("entities_added", 0)
            entities_removed = replacement_result.get("entities_removed", 0)
            self._mapping_state.total_entities += entities_added - entities_removed
            self._mapping_state.last_sync_time = datetime.now()
            
            logger.info(f"Completed atomic entity replacement {operation_id}: "
                       f"+{entities_added}/-{entities_removed} entities in {operation_time_ms:.1f}ms")
            
            return EntityOperationResult(
                operation_type="atomic_replace",
                success=replacement_result["success"],
                entities_affected=entities_added + entities_removed,
                entities_created=entities_added,
                entities_deleted=entities_removed,
                operation_time_ms=operation_time_ms,
                error_message=replacement_result.get("error"),
                metadata=replacement_result
            )
            
        except Exception as e:
            operation_time_ms = (time.perf_counter() - start_time) * 1000
            error_msg = f"Error in atomic entity replacement {operation_id}: {e}"
            logger.error(error_msg)
            
            return EntityOperationResult(
                operation_type="atomic_replace",
                success=False,
                error_message=error_msg,
                operation_time_ms=operation_time_ms
            )
    
    async def enable_sync(self) -> bool:
        """
        Enable real-time synchronization for the project.
        
        Returns:
            True if real-time sync was successfully enabled
        """
        if not self.sync_engine:
            # Initialize sync engine dynamically when enabling sync
            self.sync_engine = ProjectCollectionSyncEngine(self.storage_client)
            logger.info("Initialized sync engine for real-time synchronization")
        
        try:
            # Start the sync engine if not already running
            if not self.sync_engine.is_running:
                engine_started = await self.sync_engine.start_monitoring()
                if not engine_started:
                    logger.error("Failed to start sync engine")
                    return False
                logger.info("Started sync engine")
            
            # Add project to sync engine
            project_added = await self.sync_engine.add_project(
                self.project_path,
                self.collection_name,
                start_monitoring=True
            )
            
            if not project_added:
                logger.error("Failed to add project to sync engine")
                return False
            
            logger.info(f"Enabled real-time sync for {self.project_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to enable real-time sync: {e}")
            return False
    
    async def disable_real_time_sync(self) -> bool:
        """
        Disable real-time synchronization for the project.
        
        Returns:
            True if real-time sync was successfully disabled
        """
        if not self.sync_engine:
            return True  # Already disabled
        
        try:
            # Remove project from sync engine
            await self.sync_engine.remove_project(self.project_path)
            
            logger.info(f"Disabled real-time sync for {self.project_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to disable real-time sync: {e}")
            return False
    
    async def rebuild_entity_mappings(self) -> EntityOperationResult:
        """
        Rebuild entity-to-collection mappings from scratch.
        
        Returns:
            Entity operation result with mapping rebuild statistics
        """
        start_time = time.perf_counter()
        operation_id = f"rebuild_mappings_{int(time.time())}"
        
        logger.info(f"Starting entity mapping rebuild {operation_id}")
        
        try:
            # Use lifecycle manager to rebuild mappings
            rebuild_result = await self.lifecycle_manager.rebuild_entity_mappings()
            
            # Update tracking
            self._operation_count += 1
            operation_time_ms = (time.perf_counter() - start_time) * 1000
            self._total_operations_time += operation_time_ms
            self._last_operation_time = datetime.now()
            
            # Update mapping state
            if rebuild_result["success"]:
                self._mapping_state.total_entities = rebuild_result.get("entities_mapped", 0)
                self._mapping_state.mapped_entities = rebuild_result.get("entities_mapped", 0)
                self._mapping_state.last_sync_time = datetime.now()
                self._mapping_state.health_score = 1.0
            
            logger.info(f"Completed entity mapping rebuild {operation_id}: "
                       f"{rebuild_result.get('entities_mapped', 0)} entities in {operation_time_ms:.1f}ms")
            
            return EntityOperationResult(
                operation_type="rebuild_mappings",
                success=rebuild_result["success"],
                entities_affected=rebuild_result.get("entities_mapped", 0),
                operation_time_ms=operation_time_ms,
                error_message=rebuild_result.get("error"),
                metadata=rebuild_result
            )
            
        except Exception as e:
            operation_time_ms = (time.perf_counter() - start_time) * 1000
            error_msg = f"Error in entity mapping rebuild {operation_id}: {e}"
            logger.error(error_msg)
            
            return EntityOperationResult(
                operation_type="rebuild_mappings",
                success=False,
                error_message=error_msg,
                operation_time_ms=operation_time_ms
            )
    
    def get_integration_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of the entity lifecycle integration.
        
        Returns:
            Dictionary with integration status and metrics
        """
        avg_operation_time = (
            self._total_operations_time / self._operation_count if self._operation_count > 0 else 0
        )
        
        return {
            "integrator_info": {
                "project_path": str(self.project_path),
                "collection_name": self.collection_name,
                "real_time_sync_enabled": self.sync_engine is not None,
                "batch_size": self.batch_size
            },
            "performance": {
                "total_operations": self._operation_count,
                "total_operations_time_ms": self._total_operations_time,
                "average_operation_time_ms": avg_operation_time,
                "last_operation_time": self._last_operation_time.isoformat() if self._last_operation_time else None
            },
            "mapping_state": {
                "total_entities": self._mapping_state.total_entities,
                "mapped_entities": self._mapping_state.mapped_entities,
                "orphaned_entities": self._mapping_state.orphaned_entities,
                "inconsistent_mappings": self._mapping_state.inconsistent_mappings,
                "health_score": self._mapping_state.health_score,
                "last_sync_time": self._mapping_state.last_sync_time.isoformat() if self._mapping_state.last_sync_time else None
            },
            "component_status": {
                "lifecycle_manager": self.lifecycle_manager.get_status(),
                "entity_scanner": self.entity_scanner.get_scanner_stats(),
                "change_detector": self.change_detector.get_detector_stats() if hasattr(self.change_detector, 'get_detector_stats') else {},
                "sync_engine": self.sync_engine.get_status() if self.sync_engine else None,
                "collection_manager": self.collection_manager.get_status()
            }
        }
    
    async def close(self) -> None:
        """
        Clean up resources and close the integrator.
        """
        logger.info("Closing EntityLifecycleIntegrator")
        
        # Disable real-time sync
        if self.sync_engine:
            await self.disable_real_time_sync()
        
        # Clear any caches
        if hasattr(self.change_detector, 'clear_cache'):
            self.change_detector.clear_cache()
        
        logger.info("EntityLifecycleIntegrator closed successfully")