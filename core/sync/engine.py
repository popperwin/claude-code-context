"""
Project Collection Synchronization Engine.

Central coordinator for real-time synchronization between project filesystems
and Qdrant collections with background processing and multi-project support.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict

from ..storage.client import HybridQdrantClient
from .watcher import ProjectFileSystemWatcher
from .lifecycle import EntityLifecycleManager
from .queue import PriorityEventQueue
from .events import FileSystemEvent, EventType, EventPriority

logger = logging.getLogger(__name__)


@dataclass
class SyncEngineMetrics:
    """Comprehensive metrics for the synchronization engine."""
    
    # Event processing
    events_processed: int = 0
    events_failed: int = 0  
    events_per_second: float = 0.0
    
    # Processing times
    avg_processing_time_ms: float = 0.0
    max_processing_time_ms: float = 0.0
    
    # Queue metrics
    queue_size: int = 0
    queue_max_size_reached: int = 0
    
    # File operations
    files_created: int = 0
    files_modified: int = 0
    files_deleted: int = 0
    files_moved: int = 0
    
    # Entity operations
    entities_created: int = 0
    entities_updated: int = 0
    entities_removed: int = 0
    
    # System metrics
    uptime_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    # Error tracking
    consecutive_errors: int = 0
    last_error_time: Optional[datetime] = None
    last_error_message: Optional[str] = None


@dataclass  
class ProjectSyncState:
    """State information for a synchronized project."""
    
    project_path: Path
    collection_name: str
    is_monitoring: bool = False
    
    # Components
    watcher: Optional[ProjectFileSystemWatcher] = None
    lifecycle_manager: Optional[EntityLifecycleManager] = None
    
    # Metrics
    start_time: Optional[datetime] = None
    events_processed: int = 0
    last_activity: Optional[datetime] = None
    
    # Configuration
    debounce_ms: int = 500
    custom_extensions: Optional[Set[str]] = None


class ProjectCollectionSyncEngine:
    """
    Central coordinator for real-time project-collection synchronization.
    
    This engine manages multiple synchronized projects, coordinates file system
    monitoring with entity lifecycle management, and provides comprehensive
    monitoring and error handling.
    
    Features:
    - Multi-project synchronization support
    - Background event processing with priority queues
    - Comprehensive performance monitoring
    - Automatic error recovery and circuit breaker patterns
    - Graceful shutdown and resource management
    """
    
    def __init__(
        self,
        storage_client: HybridQdrantClient,
        max_queue_size: int = 1000,
        max_batch_size: int = 10,
        worker_count: int = 2,
        metrics_callback: Optional[Callable[[SyncEngineMetrics], None]] = None
    ):
        """
        Initialize the synchronization engine.
        
        Args:
            storage_client: Qdrant client for entity storage
            max_queue_size: Maximum events in queue
            max_batch_size: Maximum events per batch
            worker_count: Number of background workers
            metrics_callback: Optional callback for metrics updates
        """
        self.storage_client = storage_client
        self.max_queue_size = max_queue_size
        self.max_batch_size = max_batch_size
        self.worker_count = worker_count
        self.metrics_callback = metrics_callback
        
        # Central event queue for all projects
        self.event_queue = PriorityEventQueue(
            max_queue_size=max_queue_size,
            max_batch_size=max_batch_size
        )
        
        # Project management
        self.projects: Dict[str, ProjectSyncState] = {}  # project_path -> state
        self.projects_lock = asyncio.Lock()
        
        # Background processing
        self.workers: List[asyncio.Task] = []
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        # Metrics and monitoring
        self.metrics = SyncEngineMetrics()
        self.start_time: Optional[datetime] = None
        self.metrics_update_interval = 30.0  # seconds
        self.metrics_task: Optional[asyncio.Task] = None
        
        logger.info(f"Initialized ProjectCollectionSyncEngine with {worker_count} workers")
    
    async def start_monitoring(self) -> bool:
        """
        Start the synchronization engine and background processing.
        
        Returns:
            True if startup was successful, False otherwise
        """
        if self.is_running:
            logger.warning("Synchronization engine is already running")
            return True
        
        try:
            logger.info("Starting ProjectCollectionSyncEngine")
            
            # Start the central event queue
            await self.event_queue.start()
            
            # Start background workers
            self.workers = []
            for i in range(self.worker_count):
                worker = asyncio.create_task(self._event_processing_worker(f"worker-{i}"))
                self.workers.append(worker)
            
            # Start metrics collection
            self.metrics_task = asyncio.create_task(self._metrics_collection_worker())
            
            self.is_running = True
            self.start_time = datetime.now()
            self.shutdown_event.clear()
            
            logger.info(f"Started synchronization engine with {len(self.workers)} workers")
            return True
            
        except Exception as e:
            error_msg = f"Failed to start synchronization engine: {e}"
            logger.error(error_msg)
            self.metrics.last_error_message = error_msg
            self.metrics.last_error_time = datetime.now()
            return False
    
    async def stop_monitoring(self) -> None:
        """Stop the synchronization engine and cleanup resources."""
        if not self.is_running:
            return
        
        logger.info("Stopping ProjectCollectionSyncEngine")
        self.is_running = False
        self.shutdown_event.set()
        
        # Stop all project monitoring
        async with self.projects_lock:
            for project_state in self.projects.values():
                if project_state.watcher and project_state.is_monitoring:
                    await project_state.watcher.stop_monitoring()
                project_state.is_monitoring = False
        
        # Stop metrics collection
        if self.metrics_task:
            self.metrics_task.cancel()
            try:
                await self.metrics_task
            except asyncio.CancelledError:
                pass
        
        # Stop workers
        for worker in self.workers:
            worker.cancel()
        
        if self.workers:
            try:
                # Add timeout protection to prevent hanging
                await asyncio.wait_for(
                    asyncio.gather(*self.workers, return_exceptions=True),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for workers to stop - forcing shutdown")
            except Exception as e:
                logger.warning(f"Error stopping workers: {e}")
        
        # Stop event queue
        await self.event_queue.stop()
        
        # Calculate final metrics
        if self.start_time:
            self.metrics.uptime_seconds = (datetime.now() - self.start_time).total_seconds()
        
        logger.info("Stopped ProjectCollectionSyncEngine")
    
    async def add_project(
        self,
        project_path: Path,
        collection_name: str,
        debounce_ms: int = 500,
        custom_extensions: Optional[Set[str]] = None,
        start_monitoring: bool = True
    ) -> bool:
        """
        Add a project for synchronization monitoring.
        
        Args:
            project_path: Root path of the project
            collection_name: Qdrant collection name
            debounce_ms: Debounce interval for file events
            custom_extensions: Additional file extensions to monitor
            start_monitoring: Whether to start monitoring immediately
            
        Returns:
            True if project was added successfully
        """
        project_path = Path(project_path).resolve()
        project_key = str(project_path)
        
        try:
            async with self.projects_lock:
                if project_key in self.projects:
                    logger.warning(f"Project already being monitored: {project_path}")
                    return True
                
                # Create project state
                project_state = ProjectSyncState(
                    project_path=project_path,
                    collection_name=collection_name,
                    debounce_ms=debounce_ms,
                    custom_extensions=custom_extensions
                )
                
                # Create lifecycle manager
                project_state.lifecycle_manager = EntityLifecycleManager(
                    storage_client=self.storage_client,
                    collection_name=collection_name,
                    project_path=project_path
                )
                
                # Create file system watcher
                project_state.watcher = ProjectFileSystemWatcher(
                    project_path=project_path,
                    event_queue=self.event_queue,
                    debounce_ms=debounce_ms,
                    custom_extensions=custom_extensions,
                    event_callback=self._on_file_event
                )
                
                # Add to projects
                self.projects[project_key] = project_state
                
                logger.info(f"Added project for synchronization: {project_path}")
                
                # Start monitoring if requested and engine is running
                if start_monitoring and self.is_running:
                    return await self._start_project_monitoring(project_state)
                
                return True
                
        except Exception as e:
            error_msg = f"Error adding project {project_path}: {e}"
            logger.error(error_msg)
            self.metrics.last_error_message = error_msg
            self.metrics.last_error_time = datetime.now()
            return False
    
    async def remove_project(self, project_path: Path) -> bool:
        """
        Remove a project from synchronization monitoring.
        
        Args:
            project_path: Path of the project to remove
            
        Returns:
            True if project was removed successfully
        """
        project_key = str(Path(project_path).resolve())
        
        try:
            async with self.projects_lock:
                project_state = self.projects.get(project_key)
                if not project_state:
                    logger.warning(f"Project not found for removal: {project_path}")
                    return True
                
                # Stop monitoring with timeout protection
                if project_state.watcher and project_state.is_monitoring:
                    try:
                        await asyncio.wait_for(project_state.watcher.stop_monitoring(), timeout=10.0)
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout stopping watcher for {project_path}")
                    except Exception as e:
                        logger.warning(f"Error stopping watcher for {project_path}: {e}")
                
                # Remove from projects
                del self.projects[project_key]
                
                logger.info(f"Removed project from synchronization: {project_path}")
                return True
                
        except Exception as e:
            error_msg = f"Error removing project {project_path}: {e}"
            logger.error(error_msg)
            self.metrics.last_error_message = error_msg
            self.metrics.last_error_time = datetime.now()
            return False
    
    async def _start_project_monitoring(self, project_state: ProjectSyncState) -> bool:
        """
        Start monitoring for a specific project.
        
        Args:
            project_state: Project state to start monitoring for
            
        Returns:
            True if monitoring started successfully
        """
        try:
            if not project_state.watcher:
                raise Exception("Project watcher not initialized")
            
            # Start file system monitoring
            success = await project_state.watcher.start_monitoring()
            if not success:
                raise Exception("Failed to start file system watcher")
            
            # Update state
            project_state.is_monitoring = True
            project_state.start_time = datetime.now()
            
            logger.info(f"Started monitoring project: {project_state.project_path}")
            return True
            
        except Exception as e:
            error_msg = f"Error starting project monitoring for {project_state.project_path}: {e}"
            logger.error(error_msg)
            self.metrics.last_error_message = error_msg
            self.metrics.last_error_time = datetime.now()
            return False
    
    def _on_file_event(self, event: FileSystemEvent) -> None:
        """
        Callback for file system events.
        
        Args:
            event: File system event that occurred
        """
        # Update metrics based on event type
        if event.event_type == EventType.CREATED:
            self.metrics.files_created += 1
        elif event.event_type == EventType.MODIFIED:
            self.metrics.files_modified += 1
        elif event.event_type == EventType.DELETED:
            self.metrics.files_deleted += 1
        elif event.event_type == EventType.MOVED:
            self.metrics.files_moved += 1
        
        logger.debug(f"File event received: {event}")
    
    async def _event_processing_worker(self, worker_name: str) -> None:
        """
        Background worker for processing file system events.
        
        Args:
            worker_name: Name of the worker for logging
        """
        logger.info(f"Started event processing worker: {worker_name}")
        
        while self.is_running:
            try:
                # Wait for shutdown or get event batch
                shutdown_task = asyncio.create_task(self.shutdown_event.wait())
                batch_task = asyncio.create_task(self.event_queue.create_batch(timeout=1.0))
                
                done, pending = await asyncio.wait(
                    [shutdown_task, batch_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                
                # Check if shutdown was requested
                if shutdown_task in done:
                    break
                
                # Process event batch if available
                if batch_task in done:
                    batch = await batch_task
                    if batch:
                        await self._process_event_batch(batch.events, worker_name)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in event processing worker {worker_name}: {e}")
                self.metrics.consecutive_errors += 1
                self.metrics.last_error_message = str(e)
                self.metrics.last_error_time = datetime.now()
                
                # Exponential backoff for consecutive errors
                if self.metrics.consecutive_errors > 3:
                    backoff_time = min(2 ** (self.metrics.consecutive_errors - 3), 30)
                    await asyncio.sleep(backoff_time)
        
        logger.info(f"Stopped event processing worker: {worker_name}")
    
    async def _process_event_batch(self, events: List[FileSystemEvent], worker_name: str) -> None:
        """
        Process a batch of file system events.
        
        Args:
            events: List of events to process
            worker_name: Name of the processing worker
        """
        if not events:
            return
        
        start_time = datetime.now()
        processed_count = 0
        failed_count = 0
        
        logger.debug(f"Processing batch of {len(events)} events in {worker_name}")
        
        for event in events:
            try:
                # Find the project for this event
                project_state = await self._find_project_for_event(event)
                if not project_state or not project_state.lifecycle_manager:
                    logger.warning(f"No project found for event: {event}")
                    continue
                
                # Process the event using the lifecycle manager
                result = await self._process_single_event(event, project_state)
                
                if result.get('success', False):
                    processed_count += 1
                    self.metrics.consecutive_errors = 0  # Reset error counter
                    
                    # Update entity metrics
                    self.metrics.entities_created += result.get('entities_created', 0)
                    self.metrics.entities_updated += result.get('entities_added', 0)  
                    self.metrics.entities_removed += result.get('entities_removed', 0)
                else:
                    failed_count += 1
                    self.metrics.consecutive_errors += 1
                    logger.warning(f"Event processing failed: {result.get('error', 'Unknown error')}")
                
                # Update project activity
                project_state.events_processed += 1
                project_state.last_activity = datetime.now()
                
            except Exception as e:
                failed_count += 1
                self.metrics.consecutive_errors += 1
                error_msg = f"Error processing event {event}: {e}"
                logger.error(error_msg)
                self.metrics.last_error_message = error_msg
                self.metrics.last_error_time = datetime.now()
        
        # Update metrics
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        self.metrics.events_processed += processed_count
        self.metrics.events_failed += failed_count
        
        # Update processing time metrics
        if self.metrics.events_processed > 0:
            total_events = self.metrics.events_processed
            self.metrics.avg_processing_time_ms = (
                (self.metrics.avg_processing_time_ms * (total_events - processed_count) + processing_time) / total_events
            )
        self.metrics.max_processing_time_ms = max(self.metrics.max_processing_time_ms, processing_time)
        
        logger.debug(f"Processed {processed_count}/{len(events)} events successfully in {processing_time:.1f}ms")
    
    async def _find_project_for_event(self, event: FileSystemEvent) -> Optional[ProjectSyncState]:
        """
        Find the project state that should handle this event.
        
        Args:
            event: File system event
            
        Returns:
            Project state or None if not found
        """
        file_path = Path(event.file_path).resolve()
        
        async with self.projects_lock:
            # Find the project with the longest matching path prefix
            best_match = None
            longest_match = 0
            
            for project_state in self.projects.values():
                try:
                    # Check if the file is within this project
                    project_path = project_state.project_path.resolve()
                    if file_path.is_relative_to(project_path):
                        match_length = len(project_path.parts)
                        if match_length > longest_match:
                            longest_match = match_length
                            best_match = project_state
                except (ValueError, OSError):
                    # File is not relative to this project
                    continue
            
            return best_match
    
    async def _process_single_event(
        self, 
        event: FileSystemEvent, 
        project_state: ProjectSyncState
    ) -> Dict[str, Any]:
        """
        Process a single file system event.
        
        Args:
            event: File system event to process
            project_state: Project state for the event
            
        Returns:
            Dictionary with processing results
        """
        lifecycle_manager = project_state.lifecycle_manager
        if not lifecycle_manager:
            return {"success": False, "error": "No lifecycle manager available"}
        
        try:
            # Route to appropriate handler based on event type
            if event.event_type == EventType.CREATED:
                return await lifecycle_manager.handle_file_creation(event)
            elif event.event_type == EventType.MODIFIED:
                return await lifecycle_manager.handle_file_modification(event)
            elif event.event_type == EventType.DELETED:
                return await lifecycle_manager.handle_file_deletion(event)
            elif event.event_type == EventType.MOVED:
                # Handle move as delete + create
                delete_event = FileSystemEvent.create_file_deleted(event.old_path)
                create_event = FileSystemEvent.create_file_created(event.file_path)
                
                delete_result = await lifecycle_manager.handle_file_deletion(delete_event)
                create_result = await lifecycle_manager.handle_file_creation(create_event)
                
                return {
                    "operation": "file_moved",
                    "success": delete_result.get("success", False) and create_result.get("success", False),
                    "entities_removed": delete_result.get("entities_deleted", 0),
                    "entities_created": create_result.get("entities_created", 0)
                }
            else:
                return {"success": False, "error": f"Unknown event type: {event.event_type}"}
                
        except Exception as e:
            error_msg = f"Error processing {event.event_type.value} event for {event.file_path}: {e}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    async def _metrics_collection_worker(self) -> None:
        """Background worker for collecting and updating metrics."""
        logger.info("Started metrics collection worker")
        
        while self.is_running:
            try:
                await asyncio.sleep(self.metrics_update_interval)
                
                if not self.is_running:
                    break
                
                # Update queue metrics
                self.metrics.queue_size = await self.event_queue.size()
                queue_metrics = self.event_queue.get_metrics()
                self.metrics.queue_max_size_reached = queue_metrics.get("max_queue_size_reached", 0)
                
                # Update uptime
                if self.start_time:
                    self.metrics.uptime_seconds = (datetime.now() - self.start_time).total_seconds()
                    
                    # Calculate events per second
                    if self.metrics.uptime_seconds > 0:
                        self.metrics.events_per_second = self.metrics.events_processed / self.metrics.uptime_seconds
                
                # Call metrics callback if provided
                if self.metrics_callback:
                    try:
                        self.metrics_callback(self.metrics)
                    except Exception as e:
                        logger.warning(f"Error in metrics callback: {e}")
                
                logger.debug(f"Metrics update: {self.metrics.events_processed} events processed, "
                           f"{self.metrics.queue_size} in queue, {self.metrics.events_per_second:.1f} eps")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection worker: {e}")
        
        logger.info("Stopped metrics collection worker")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status information about the sync engine.
        
        Returns:
            Dictionary with status information
        """
        return {
            "is_running": self.is_running,
            "uptime_seconds": self.metrics.uptime_seconds,
            "projects_count": len(self.projects),
            "monitoring_projects": sum(1 for p in self.projects.values() if p.is_monitoring),
            "worker_count": len(self.workers),
            "queue_size": self.metrics.queue_size,
            "events_processed": self.metrics.events_processed,
            "events_failed": self.metrics.events_failed,
            "events_per_second": self.metrics.events_per_second,
            "avg_processing_time_ms": self.metrics.avg_processing_time_ms,
            "consecutive_errors": self.metrics.consecutive_errors,
            "last_error": self.metrics.last_error_message,
            "last_error_time": self.metrics.last_error_time.isoformat() if self.metrics.last_error_time else None,
            "projects": {
                str(state.project_path): {
                    "collection_name": state.collection_name,
                    "is_monitoring": state.is_monitoring,
                    "events_processed": state.events_processed,
                    "last_activity": state.last_activity.isoformat() if state.last_activity else None
                }
                for state in self.projects.values()
            }
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_monitoring()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop_monitoring()