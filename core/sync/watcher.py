"""
Project File System Watcher.

Provides cross-platform file system monitoring with debouncing, filtering,
and event queuing for the synchronization system.
"""

import asyncio
import logging
import platform
import time
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Set, Optional, Callable, Any, List, Tuple
from datetime import datetime, timedelta

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent as WatchdogEvent
from watchdog.events import FileCreatedEvent, FileModifiedEvent, FileDeletedEvent, FileMovedEvent

# Platform-specific imports for enhanced monitoring
if platform.system() == 'Darwin':
    from watchdog.observers.fsevents import FSEventsObserver

from .events import FileSystemEvent, EventType, EventPriority
from .queue import PriorityEventQueue

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WatcherConfig:
    """Centralized configuration for the file system watcher."""
    debounce_ms: int = 500
    deletion_retention_s: float = 600.0
    shutdown_timeout_s: float = 3.0
    disable_random_sample: bool = False
    macos_cleanup_delay_s: float = 0.5
    observer_retry_delay_s: float = 0.2
    
    @classmethod
    def from_env(cls) -> 'WatcherConfig':
        """Create config from environment variables."""
        return cls(
            deletion_retention_s=float(os.environ.get('WATCHER_DELETION_RETENTION', '600')),
            disable_random_sample=os.environ.get('WATCHER_DISABLE_RANDOM') == '1'
        )


class ProjectFileSystemWatcher:
    """
    Cross-platform file system watcher with debouncing and intelligent filtering.
    
    This class monitors project directories for file changes and converts them
    into FileSystemEvent objects for processing by the synchronization system.
    
    Features:
    - Cross-platform monitoring via watchdog
    - File type filtering for supported languages
    - Event debouncing to prevent processing storms
    - Graceful error recovery and reconnection
    - Background processing with async integration
    - Enhanced macOS reliability with event confirmation
    """
    
    # Supported file extensions for monitoring
    SUPPORTED_EXTENSIONS = {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.go', '.rs', '.java', 
        '.cpp', '.c', '.h', '.hpp', '.cs', '.rb', '.php', '.swift', 
        '.kt', '.scala', '.html', '.css', '.scss', '.sass', '.vue',
        '.md', '.json', '.yaml', '.yml', '.toml', '.xml'
    }
    
    # Directories to ignore
    IGNORED_DIRECTORIES = {
        'node_modules', '.git', '__pycache__', '.pytest_cache',
        '.venv', 'venv', 'build', 'dist', '.cache', 'target',
        '.next', '.nuxt', 'coverage', '.nyc_output',
        # Additional useful ignored directories
        '.svn', '.hg', '.mypy_cache', '.tox', 'env', '.env', 
        '.coverage', '.DS_Store', 'Thumbs.db'
    }
    
    def __init__(
        self,
        project_path: Path,
        event_queue: PriorityEventQueue,
        debounce_ms: int = 500,
        recursive: bool = True,
        custom_extensions: Optional[Set[str]] = None,
        event_callback: Optional[Callable[[FileSystemEvent], None]] = None
    ):
        """
        Initialize the file system watcher.
        
        Args:
            project_path: Root directory to monitor
            event_queue: Queue to send events to
            debounce_ms: Milliseconds to wait before processing events
            recursive: Whether to monitor subdirectories
            custom_extensions: Additional file extensions to monitor
            event_callback: Optional callback for immediate event notification
        """
        self.project_path = Path(project_path).resolve()
        self.event_queue = event_queue
        self._config = WatcherConfig.from_env()
        self.debounce_ms = debounce_ms if debounce_ms is not None else self._config.debounce_ms
        self.recursive = recursive
        self.event_callback = event_callback
        
        # Load configuration
        self._config = WatcherConfig.from_env()
        
        # Platform detection for optimizations
        self._platform = platform.system()
        self._is_macos = self._platform == 'Darwin'
        self._is_windows = self._platform == 'Windows'
        self._is_linux = self._platform == 'Linux'
        
        # Adjust debounce for macOS (FSEvents has higher latency)
        if self._is_macos:
            self.debounce_ms = max(debounce_ms, 1000)  # Minimum 1s on macOS
        
        # Combine supported and custom extensions
        self.watched_extensions = self.SUPPORTED_EXTENSIONS.copy()
        if custom_extensions:
            self.watched_extensions.update(custom_extensions)
        
        # Watchdog components
        self.observer: Optional[Observer] = None
        self.event_handler: Optional['SyncFileSystemEventHandler'] = None
        
        # Debouncing state
        self._pending_events: Dict[str, FileSystemEvent] = {}
        self._debounce_tasks: Dict[str, asyncio.Task] = {}
        self._debounce_lock = asyncio.Lock()
        
        # Event confirmation tracking (for reliability)
        self._event_confirmations: Dict[str, Tuple[FileSystemEvent, float]] = {}
        self._confirmation_lock = asyncio.Lock()
        self._confirmation_task: Optional[asyncio.Task] = None
        
        # File state tracking (for missed event detection)
        self._file_states: Dict[str, Dict[str, Any]] = {}
        self._state_lock = asyncio.Lock()  # Lock for atomicity
        self._state_check_task: Optional[asyncio.Task] = None
        
        # Monitoring state
        self._is_monitoring = False
        self._monitor_start_time: Optional[datetime] = None
        
        # Error tracking
        self._error_count = 0
        self._last_error: Optional[str] = None
        self._last_error_time: Optional[datetime] = None
        
        logger.info(f"Initialized ProjectFileSystemWatcher for {self.project_path}")
        logger.info(f"Platform: {self._platform}, Debounce: {self.debounce_ms}ms")
        logger.info(f"Watching extensions: {sorted(self.watched_extensions)}")
        logger.info(f"Deletion retention: {self._config.deletion_retention_s}s, Random sample: {not self._config.disable_random_sample}")
    
    async def _ensure_clean_shutdown(self) -> None:
        """Ensure any previous observer is fully cleaned up before starting a new one."""
        if self.observer:
            logger.warning("Cleaning up previous observer before starting new one")
            await self.stop_monitoring()
            # Brief delay to ensure full cleanup on macOS
            await asyncio.sleep(0.1)
    
    async def start_monitoring(self) -> bool:
        """
        Start file system monitoring.
        
        Returns:
            True if monitoring started successfully, False otherwise
        """
        if self._is_monitoring:
            logger.warning("File system monitoring is already active")
            return True
        
        # Ensure any previous observer is fully cleaned up
        await self._ensure_clean_shutdown()
        
        try:
            # Validate project path
            if not self.project_path.exists():
                raise FileNotFoundError(f"Project path does not exist: {self.project_path}")
            
            if not self.project_path.is_dir():
                raise NotADirectoryError(f"Project path is not a directory: {self.project_path}")
            
            # Create event handler
            self.event_handler = SyncFileSystemEventHandler(self)
            
            # Set the current event loop on the handler for thread-safe communication
            try:
                current_loop = asyncio.get_running_loop()
                self.event_handler.set_event_loop(current_loop)
                logger.debug(f"Set event loop {id(current_loop)} on file system watcher")
            except RuntimeError:
                # No running event loop, this is critical for async operation
                logger.error("No running event loop found when starting watcher - events will be dropped")
                return False
            
            # Create and configure observer (platform-specific)
            if self._is_macos:
                self.observer = FSEventsObserver()
            else:
                self.observer = Observer()
            
            try:
                self.observer.schedule(
                    self.event_handler,
                    str(self.project_path),
                    recursive=self.recursive
                )
            except Exception as e:
                # Handle "already scheduled" errors on macOS
                if "already scheduled" in str(e).lower():
                    logger.warning(f"Path already being watched, attempting to recover: {e}")
                    # Force cleanup and retry
                    self.observer.stop()
                    self.observer = None
                    await asyncio.sleep(self._config.observer_retry_delay_s)  # Wait for cleanup
                    
                    # Recreate observer
                    if self._is_macos:
                        self.observer = FSEventsObserver()
                    else:
                        self.observer = Observer()
                    
                    self.observer.schedule(
                        self.event_handler,
                        str(self.project_path),
                        recursive=self.recursive
                    )
                else:
                    raise
            
            # Start observer
            self.observer.start()
            
            # Start confirmation checker for reliability
            if self._is_macos:
                self._confirmation_task = asyncio.create_task(
                    self._confirmation_checker()
                )
            
            # Start state checker for missed events (on all platforms)
            self._state_check_task = asyncio.create_task(
                self._periodic_state_check()
            )
            
            self._is_monitoring = True
            self._monitor_start_time = datetime.now()
            self._error_count = 0
            
            # Initial state scan
            await self._scan_initial_state()
            
            logger.info(f"Started monitoring {self.project_path} (recursive={self.recursive})")
            return True
            
        except Exception as e:
            error_msg = f"Failed to start file system monitoring: {e}"
            logger.error(error_msg)
            self._last_error = error_msg
            self._last_error_time = datetime.now()
            self._error_count += 1
            return False
    
    async def stop_monitoring(self) -> None:
        """Stop file system monitoring and cleanup resources."""
        if not self._is_monitoring:
            return
        
        self._is_monitoring = False
        
        # Clear event loop reference in handler to prevent further events
        if self.event_handler:
            self.event_handler.set_event_loop(None)
        
        # Stop observer with enhanced cleanup for macOS
        if self.observer:
            try:
                self.observer.stop()
                self.observer.join(timeout=5.0)  # Wait up to 5 seconds
            except Exception as e:
                logger.warning(f"Error stopping observer: {e}")
            finally:
                self.observer = None
                # On macOS, add extra delay to ensure FSEvents fully releases the watch
                if self._is_macos:
                    await asyncio.sleep(self._config.macos_cleanup_delay_s)
        
        # Cancel background tasks
        tasks_to_cancel = []
        
        if self._confirmation_task and not self._confirmation_task.done():
            tasks_to_cancel.append(self._confirmation_task)
        
        if self._state_check_task and not self._state_check_task.done():
            tasks_to_cancel.append(self._state_check_task)
        
        # Cancel pending debounce tasks
        try:
            async with self._debounce_lock:
                tasks_to_cancel.extend(
                    task for task in self._debounce_tasks.values() 
                    if not task.done()
                )
                
                # Cancel all tasks
                for task in tasks_to_cancel:
                    task.cancel()
                
                # Wait for tasks to complete with timeout protection
                if tasks_to_cancel:
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*tasks_to_cancel, return_exceptions=True),
                            timeout=self._config.shutdown_timeout_s
                        )
                    except asyncio.TimeoutError:
                        logger.warning("Timeout waiting for debounce tasks to stop - forcing cleanup")
                    except Exception as e:
                        logger.warning(f"Error waiting for tasks: {e}")
                
                self._debounce_tasks.clear()
                self._pending_events.clear()
        except RuntimeError as e:
            if "Event loop is closed" in str(e):
                # Event loop is closed, just clear the data structures
                self._debounce_tasks.clear()
                self._pending_events.clear()
                logger.debug("Event loop closed during cleanup, cleared data structures")
            else:
                raise
        
        # Clear tracking data
        self._event_confirmations.clear()
        self._file_states.clear()
        
        # Clean up references
        self.event_handler = None
        self._confirmation_task = None
        self._state_check_task = None
        
        monitor_duration = None
        if self._monitor_start_time:
            monitor_duration = datetime.now() - self._monitor_start_time
        
        logger.info(f"Stopped file system monitoring (duration: {monitor_duration})")
    
    def should_monitor_file(self, file_path: Path, check_existence: bool = True) -> bool:
        """
        Check if a file should be monitored based on extension and path filtering.
        
        Args:
            file_path: Path to check
            check_existence: Whether to check if file exists (False for deletion events)
            
        Returns:
            True if file should be monitored
        """
        try:
            # For deletion events, file won't exist, so skip existence check
            if check_existence:
                # Check if file exists and is actually a file
                if not file_path.exists() or not file_path.is_file():
                    return False
            
            # Check extension
            if file_path.suffix.lower() not in self.watched_extensions:
                return False
            
            # Check if any parent directory is ignored
            for parent in file_path.parents:
                if parent.name in self.IGNORED_DIRECTORIES:
                    return False
            
            # Check if file itself is ignored
            if file_path.name.startswith('.') and file_path.suffix not in {'.py', '.js', '.ts'}:
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error checking if file should be monitored {file_path}: {e}")
            return False
    
    async def handle_watchdog_event(self, event: WatchdogEvent) -> None:
        """
        Handle a watchdog file system event with debouncing.
        
        Args:
            event: Watchdog file system event
        """
        try:
            # Convert watchdog event to our FileSystemEvent
            fs_event = self._convert_watchdog_event(event)
            if not fs_event:
                return
            
            # Track event for confirmation (macOS reliability)
            if self._is_macos and fs_event.event_type in [EventType.CREATED, EventType.DELETED, EventType.MOVED]:
                async with self._confirmation_lock:
                    if fs_event.event_type == EventType.MOVED:
                        # For MOVED, store both paths
                        self._event_confirmations[f"moved:{fs_event.old_path}>{fs_event.file_path}"] = (
                            fs_event, time.time()
                        )
                    else:
                        self._event_confirmations[str(fs_event.file_path)] = (
                            fs_event, time.time()
                        )
            
            # Apply debouncing
            await self._debounce_event(fs_event)
            
        except Exception as e:
            logger.error(f"Error handling watchdog event {event}: {e}")
            self._error_count += 1
            self._last_error = str(e)
            self._last_error_time = datetime.now()
    
    def _convert_watchdog_event(self, event: WatchdogEvent) -> Optional[FileSystemEvent]:
        """
        Convert a watchdog event to our FileSystemEvent format.
        
        Args:
            event: Watchdog file system event
            
        Returns:
            FileSystemEvent or None if event should be ignored
        """
        try:
            file_path = Path(event.src_path).resolve()
            
            # Skip directories
            if event.is_directory:
                return None
            
            # Special handling for move events
            if isinstance(event, FileMovedEvent):
                old_path = Path(event.src_path).resolve()
                new_path = Path(event.dest_path).resolve()
                # Accept if EITHER path is monitored (rename .txt → .py)
                if not any(map(lambda p: self.should_monitor_file(p, check_existence=False),
                            (old_path, new_path))):
                    return None
                return FileSystemEvent.create_file_moved(old_path, new_path)
            
            # Check if we should monitor this file
            # For deletion events, file won't exist, so don't check existence
            check_existence = not isinstance(event, FileDeletedEvent)
            if not self.should_monitor_file(file_path, check_existence=check_existence):
                return None
            
            # Convert other event types
            if isinstance(event, FileCreatedEvent):
                return FileSystemEvent.create_file_created(file_path)
            elif isinstance(event, FileModifiedEvent):
                return FileSystemEvent.create_file_modified(file_path)
            elif isinstance(event, FileDeletedEvent):
                return FileSystemEvent.create_file_deleted(file_path)
            else:
                logger.debug(f"Unknown watchdog event type: {type(event)}")
                return None
                
        except Exception as e:
            logger.warning(f"Error converting watchdog event {event}: {e}")
            return None
    
    async def _debounce_event(self, event: FileSystemEvent) -> None:
        """
        Apply debouncing to file system events to prevent processing storms.
        
        Args:
            event: File system event to debounce
        """
        async with self._debounce_lock:
            file_key = str(event.file_path)
            
            # Cancel existing debounce task for this file
            if file_key in self._debounce_tasks:
                existing_task = self._debounce_tasks[file_key]
                if not existing_task.done():
                    existing_task.cancel()
            
            # Store the latest event for this file
            self._pending_events[file_key] = event
            
            # Create new debounce task
            debounce_seconds = self.debounce_ms / 1000.0
            task = asyncio.create_task(
                self._process_debounced_event(file_key, debounce_seconds)
            )
            self._debounce_tasks[file_key] = task
    
    async def _process_debounced_event(self, file_key: str, delay_seconds: float) -> None:
        """
        Process a debounced event after the specified delay.
        
        Args:
            file_key: Unique key for the file
            delay_seconds: Delay before processing
        """
        try:
            # Wait for debounce period
            await asyncio.sleep(delay_seconds)
            
            # Get the event to process
            async with self._debounce_lock:
                event = self._pending_events.pop(file_key, None)
                self._debounce_tasks.pop(file_key, None)
            
            if not event:
                return
            
            # Remove from confirmation tracking if present
            if self._is_macos:
                async with self._confirmation_lock:
                    if event.event_type == EventType.MOVED:
                        # Remove with composite key for MOVED
                        confirmation_key = f"moved:{event.old_path}>{event.file_path}"
                        self._event_confirmations.pop(confirmation_key, None)
                    else:
                        self._event_confirmations.pop(file_key, None)
            
            # Update file state tracking
            await self._update_file_state(event)
            
            # Enqueue the event for processing
            success = await self.event_queue.enqueue(event)
            if not success:
                logger.warning(f"Failed to enqueue debounced event: {event}")
                return
            
            # Call optional callback
            if self.event_callback:
                try:
                    self.event_callback(event)
                except Exception as e:
                    logger.warning(f"Error in event callback: {e}")
            
            logger.debug(f"Processed debounced event: {event}")
            
        except asyncio.CancelledError:
            # Task was cancelled, clean up
            async with self._debounce_lock:
                self._pending_events.pop(file_key, None)
                self._debounce_tasks.pop(file_key, None)
        except Exception as e:
            logger.error(f"Error processing debounced event for {file_key}: {e}")
    
    async def _scan_initial_state(self) -> None:
        """Scan initial file states for change detection."""
        try:
            for file_path in self._iter_project_files():
                await self._capture_file_state(file_path)
        except Exception as e:
            logger.warning(f"Error scanning initial state: {e}")
    
    async def _capture_file_state(self, file_path: Path) -> None:
        """Capture current state of a file."""
        try:
            if file_path.exists() and file_path.is_file():
                stat = file_path.stat()
                self._file_states[str(file_path)] = {
                    'mtime': stat.st_mtime,
                    'size': stat.st_size,
                    'exists': True
                }
            else:
                self._file_states[str(file_path)] = {
                    'exists': False
                }
        except Exception as e:
            logger.debug(f"Error capturing file state for {file_path}: {e}")
    
    async def _update_file_state(self, event: FileSystemEvent) -> None:
        """Update file state based on event."""
        async with self._state_lock:
            if event.event_type == EventType.DELETED:
                file_key = str(event.file_path)
                self._file_states[file_key] = {
                    'exists': False,
                    'deleted_at': time.time()
                }
            elif event.event_type == EventType.MOVED:
                # Mark the old one as deleted
                old_key = str(event.old_path)
                self._file_states[old_key] = {
                    'exists': False,
                    'deleted_at': time.time()
                }
                # Capture the new state
                await self._capture_file_state(event.file_path)
            else:
                await self._capture_file_state(event.file_path)
    
    def _iter_project_files(self) -> List[Path]:
        """Iterate through all monitored files in the project."""
        files = []
        try:
            for file_path in self.project_path.rglob('*'):
                if self.should_monitor_file(file_path):
                    files.append(file_path)
        except Exception as e:
            logger.warning(f"Error iterating project files: {e}")
        return files
    
    async def _confirmation_checker(self) -> None:
        """Background task to confirm events on macOS."""
        check_interval = 2.0  # Check every 2 seconds
        confirmation_timeout = 5.0  # Retry after 5 seconds
        
        while self._is_monitoring:
            try:
                await asyncio.sleep(check_interval)
                
                current_time = time.time()
                events_to_retry = []
                
                async with self._confirmation_lock:
                    # Find events that need confirmation
                    for file_key, (event, event_time) in list(self._event_confirmations.items()):
                        if current_time - event_time > confirmation_timeout:
                            # Event wasn't confirmed, retry it
                            events_to_retry.append(event)
                            del self._event_confirmations[file_key]
                
                # Retry unconfirmed events
                for event in events_to_retry:
                    logger.debug(f"Retrying unconfirmed event: {event}")
                    # MODIFICATION: MOVED events are now also retried
                    await self._debounce_event(event)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Error in confirmation checker: {e}")
    
    async def _periodic_state_check(self) -> None:
        """Periodically check for missed file changes."""
        check_interval = 10.0 if self._is_macos else 30.0
        purge_interval = 300.0  # Purge every 5 minutes
        last_purge_time = time.time()
        
        while self._is_monitoring:
            try:
                await asyncio.sleep(check_interval)
                
                # Periodic purge of deleted entries
                current_time = time.time()
                if current_time - last_purge_time > purge_interval:
                    cutoff_time = current_time - self._config.deletion_retention_s
                    
                    # Atomic purge under lock
                    async with self._state_lock:
                        keys_to_delete = [
                            k for k, v in self._file_states.items()
                            if not v.get('exists', True) and v.get('deleted_at', current_time) < cutoff_time
                        ]
                        for key in keys_to_delete:
                            del self._file_states[key]
                    
                    last_purge_time = current_time
                    if keys_to_delete:
                        logger.debug(f"Purged {len(keys_to_delete)} old deleted entries")
                
                # Check a sample of files for changes
                files_to_check = self._iter_project_files()
                
                if self._config.disable_random_sample:
                    # Test mode: check all files
                    sample_files = files_to_check[:100]  # Limit to 100 even in test mode
                else:
                    # Normal mode: random sample
                    sample_size = min(100, len(files_to_check))
                    if sample_size > 0:
                        import random
                        sample_files = random.sample(files_to_check, sample_size)
                    else:
                        sample_files = []
                
                for file_path in sample_files:
                    await self._check_file_for_changes(file_path)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Error in periodic state check: {e}")
    
    async def _check_file_for_changes(self, file_path: Path) -> None:
        """Check if a file has changed since last check."""
        file_key = str(file_path)
        
        try:
            current_exists = file_path.exists() and file_path.is_file()
            old_state = self._file_states.get(file_key, {})
            
            if current_exists:
                stat = file_path.stat()
                current_mtime = stat.st_mtime
                current_size = stat.st_size
                
                if not old_state.get('exists', False):
                    # File was created
                    event = FileSystemEvent.create_file_created(file_path)
                    await self._debounce_event(event)
                elif (abs(current_mtime - old_state.get('mtime', 0)) > 0.01 or
                      current_size != old_state.get('size', -1)):
                    # File was modified
                    event = FileSystemEvent.create_file_modified(file_path)
                    await self._debounce_event(event)
                
                # Update state
                self._file_states[file_key] = {
                    'mtime': current_mtime,
                    'size': current_size,
                    'exists': True
                }
            else:
                if old_state.get('exists', False):
                    # File was deleted
                    event = FileSystemEvent.create_file_deleted(file_path)
                    await self._debounce_event(event)
                    self._file_states[file_key] = {'exists': False}
                    
        except Exception as e:
            logger.debug(f"Error checking file for changes {file_path}: {e}")
    
    @property
    def is_monitoring(self) -> bool:
        """Check if file system monitoring is active."""
        return self._is_monitoring
    
    @property
    def monitoring_duration(self) -> Optional[timedelta]:
        """Get duration of current monitoring session."""
        if not self._monitor_start_time:
            return None
        return datetime.now() - self._monitor_start_time
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status information.
        
        Returns:
            Dictionary with status information
        """
        return {
            "is_monitoring": self._is_monitoring,
            "project_path": str(self.project_path),
            "platform": self._platform,
            "recursive": self.recursive,
            "debounce_ms": self.debounce_ms,
            "watched_extensions": sorted(self.watched_extensions),
            "supported_extensions_count": len(self.watched_extensions),
            "ignored_directories_count": len(self.IGNORED_DIRECTORIES),
            "monitoring_duration": str(self.monitoring_duration) if self.monitoring_duration else None,
            "pending_events": len(self._pending_events),
            "debounce_tasks": len(self._debounce_tasks),
            "tracked_files": len(self._file_states),
            "error_count": self._error_count,
            "last_error": self._last_error,
            "last_error_time": self._last_error_time.isoformat() if self._last_error_time else None
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_monitoring()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop_monitoring()


class SyncFileSystemEventHandler(FileSystemEventHandler):
    """
    Watchdog event handler that forwards events to ProjectFileSystemWatcher.
    
    This class bridges the synchronous watchdog API with our asynchronous
    event processing system.
    """
    
    def __init__(self, watcher: ProjectFileSystemWatcher):
        """
        Initialize the event handler.
        
        Args:
            watcher: ProjectFileSystemWatcher instance to forward events to
        """
        super().__init__()
        self.watcher = watcher
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
    
    def set_event_loop(self, loop: Optional[asyncio.AbstractEventLoop]) -> None:
        """
        Set the event loop to use for scheduling async tasks.
        
        Args:
            loop: The asyncio event loop to use, or None to clear
        """
        self._event_loop = loop
    
    def on_any_event(self, event: WatchdogEvent) -> None:
        """
        Handle any file system event.
        
        Args:
            event: Watchdog file system event
        """
        try:
            # Watchdog runs in a separate thread, so we need to use call_soon_threadsafe
            # to schedule the coroutine on the main event loop
            if self._event_loop and not self._event_loop.is_closed():
                try:
                    # Use call_soon_threadsafe to schedule from watchdog thread to asyncio thread
                    self._event_loop.call_soon_threadsafe(
                        lambda: asyncio.create_task(self.watcher.handle_watchdog_event(event))
                    )
                except RuntimeError as e:
                    # Event loop might be closing or closed
                    if "closed" not in str(e).lower():
                        self.logger.error(f"Failed to schedule event on loop: {e}")
                    # Don't spam warnings if event loop is closing
                    return
            else:
                # No event loop available, drop the event with less verbose logging
                self.logger.debug(f"No event loop available, dropping event: {event}")
                
        except Exception as e:
            self.logger.error(f"Error in watchdog event handler: {e}")
    
    def on_created(self, event: FileCreatedEvent) -> None:
        """Handle file creation events."""
        self.on_any_event(event)
    
    def on_modified(self, event: FileModifiedEvent) -> None:
        """Handle file modification events."""
        self.on_any_event(event)
    
    def on_deleted(self, event: FileDeletedEvent) -> None:
        """Handle file deletion events."""
        self.on_any_event(event)
    
    def on_moved(self, event: FileMovedEvent) -> None:
        """Handle file move/rename events."""
        self.on_any_event(event)