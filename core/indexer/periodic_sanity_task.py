"""
Periodic sanity task scheduler for automated delta-scan operations.

This module provides asyncio-based scheduling for periodic delta scans to maintain
collection integrity and detect missed changes over time.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
from enum import Enum

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of periodic sanity task."""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class TaskConfig:
    """Configuration for periodic sanity tasks."""
    # Timing configuration
    interval_minutes: int = 10  # Run every 10 minutes by default
    initial_delay_minutes: int = 5  # Wait 5 minutes before first run
    max_retry_attempts: int = 3
    retry_delay_seconds: int = 30
    
    # Delta scan configuration
    force_full_scan: bool = False
    tolerance_sec: float = 1.0
    max_execution_time_minutes: int = 30
    
    # Error handling
    continue_on_error: bool = True
    max_consecutive_failures: int = 5
    
    @classmethod
    def from_env(cls, prefix: str = "SANITY_TASK_") -> 'TaskConfig':
        """Create config from environment variables."""
        import os
        return cls(
            interval_minutes=int(os.environ.get(f'{prefix}INTERVAL_MINUTES', '10')),
            initial_delay_minutes=int(os.environ.get(f'{prefix}INITIAL_DELAY_MINUTES', '5')),
            max_retry_attempts=int(os.environ.get(f'{prefix}MAX_RETRIES', '3')),
            retry_delay_seconds=int(os.environ.get(f'{prefix}RETRY_DELAY', '30')),
            force_full_scan=os.environ.get(f'{prefix}FORCE_FULL_SCAN', 'false').lower() == 'true',
            tolerance_sec=float(os.environ.get(f'{prefix}TOLERANCE_SEC', '1.0')),
            max_execution_time_minutes=int(os.environ.get(f'{prefix}MAX_EXECUTION_MINUTES', '30')),
            continue_on_error=os.environ.get(f'{prefix}CONTINUE_ON_ERROR', 'true').lower() == 'true',
            max_consecutive_failures=int(os.environ.get(f'{prefix}MAX_CONSECUTIVE_FAILURES', '5'))
        )


@dataclass
class TaskMetrics:
    """Metrics for periodic sanity task execution."""
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    consecutive_failures: int = 0
    last_run_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    last_failure_time: Optional[datetime] = None
    last_execution_duration_seconds: float = 0.0
    average_execution_time_seconds: float = 0.0
    total_execution_time_seconds: float = 0.0
    
    def update_success(self, execution_time: float) -> None:
        """Update metrics for successful run."""
        self.total_runs += 1
        self.successful_runs += 1
        self.consecutive_failures = 0
        self.last_run_time = datetime.now()
        self.last_success_time = datetime.now()
        self.last_execution_duration_seconds = execution_time
        self.total_execution_time_seconds += execution_time
        self.average_execution_time_seconds = self.total_execution_time_seconds / self.total_runs
    
    def update_failure(self, execution_time: float) -> None:
        """Update metrics for failed run."""
        self.total_runs += 1
        self.failed_runs += 1
        self.consecutive_failures += 1
        self.last_run_time = datetime.now()
        self.last_failure_time = datetime.now()
        self.last_execution_duration_seconds = execution_time
        self.total_execution_time_seconds += execution_time
        self.average_execution_time_seconds = self.total_execution_time_seconds / self.total_runs
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_runs == 0:
            return 0.0
        return (self.successful_runs / self.total_runs) * 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_runs": self.total_runs,
            "successful_runs": self.successful_runs,
            "failed_runs": self.failed_runs,
            "consecutive_failures": self.consecutive_failures,
            "success_rate_percent": self.success_rate,
            "last_run_time": self.last_run_time.isoformat() if self.last_run_time else None,
            "last_success_time": self.last_success_time.isoformat() if self.last_success_time else None,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_execution_duration_seconds": self.last_execution_duration_seconds,
            "average_execution_time_seconds": self.average_execution_time_seconds,
            "total_execution_time_seconds": self.total_execution_time_seconds
        }


class PeriodicSanityTask:
    """
    Asyncio-based periodic sanity task scheduler for delta-scan operations.
    
    This class manages automated execution of delta scans at configurable intervals
    to maintain collection integrity and detect missed changes.
    """
    
    def __init__(
        self,
        indexer,  # HybridIndexer instance
        project_path: Path,
        collection_name: str,
        config: Optional[TaskConfig] = None,
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None
    ):
        """
        Initialize periodic sanity task.
        
        Args:
            indexer: HybridIndexer instance for delta scan execution
            project_path: Project directory to scan
            collection_name: Target collection name
            config: Task configuration (uses defaults if None)
            progress_callback: Optional progress callback for delta scans
        """
        self.indexer = indexer
        self.project_path = project_path
        self.collection_name = collection_name
        self.config = config or TaskConfig()
        self.progress_callback = progress_callback
        
        # State management
        self.status = TaskStatus.STOPPED
        self._task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        self._pause_event = asyncio.Event()
        
        # Metrics and error tracking
        self.metrics = TaskMetrics()
        self._last_error: Optional[str] = None
        self._start_time: Optional[datetime] = None
        
        # Lifecycle lock for thread-safe operations
        self._lifecycle_lock = asyncio.Lock()
        
        logger.info(
            f"Initialized periodic sanity task for {project_path} -> {collection_name} "
            f"(interval: {self.config.interval_minutes}min, initial_delay: {self.config.initial_delay_minutes}min)"
        )
    
    async def start(self) -> bool:
        """
        Start the periodic sanity task.
        
        Returns:
            True if started successfully, False otherwise
        """
        async with self._lifecycle_lock:
            if self.status != TaskStatus.STOPPED:
                logger.warning(f"Periodic sanity task is already {self.status.value}")
                return False
            
            try:
                # Clear shutdown and set pause events for clean state
                self._shutdown_event.clear()
                self._pause_event.set()  # Start unpaused
                
                # Create and start the background task
                self._task = asyncio.create_task(self._run_periodic_task())
                self.status = TaskStatus.RUNNING
                self._start_time = datetime.now()
                
                logger.info(f"Started periodic sanity task for {self.collection_name}")
                return True
                
            except Exception as e:
                error_msg = f"Failed to start periodic sanity task: {e}"
                logger.error(error_msg)
                self._last_error = error_msg
                self.status = TaskStatus.ERROR
                return False
    
    async def stop(self) -> None:
        """Stop the periodic sanity task and cleanup resources."""
        async with self._lifecycle_lock:
            if self.status == TaskStatus.STOPPED:
                logger.debug("Periodic sanity task is already stopped")
                return
            
            logger.info("Stopping periodic sanity task...")
            
            # Signal shutdown
            self._shutdown_event.set()
            self.status = TaskStatus.STOPPED
            
            # Cancel and wait for background task
            if self._task and not self._task.done():
                self._task.cancel()
                try:
                    await asyncio.wait_for(self._task, timeout=5.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    logger.debug("Background task cancelled or timed out during shutdown")
                except Exception as e:
                    logger.warning(f"Error during task shutdown: {e}")
            
            self._task = None
            logger.info("Periodic sanity task stopped")
    
    async def pause(self) -> None:
        """Pause the periodic sanity task."""
        if self.status == TaskStatus.RUNNING:
            self._pause_event.clear()
            self.status = TaskStatus.PAUSED
            logger.info("Periodic sanity task paused")
    
    async def resume(self) -> None:
        """Resume the periodic sanity task."""
        if self.status == TaskStatus.PAUSED:
            self._pause_event.set()
            self.status = TaskStatus.RUNNING
            logger.info("Periodic sanity task resumed")
    
    async def trigger_immediate_run(self) -> Dict[str, Any]:
        """
        Trigger an immediate delta scan outside the regular schedule.
        
        Returns:
            Dictionary with execution results
        """
        if self.status == TaskStatus.STOPPED:
            return {"success": False, "error": "Task is not running"}
        
        logger.info("Triggering immediate sanity check...")
        
        start_time = time.perf_counter()
        
        try:
            # Execute delta scan with timeout
            result = await asyncio.wait_for(
                self._execute_delta_scan(),
                timeout=self.config.max_execution_time_minutes * 60
            )
            
            execution_time = time.perf_counter() - start_time
            
            # Check if delta scan was actually successful
            if result.get("success", False):
                # Update success metrics
                self.metrics.update_success(execution_time)
                
                logger.info(f"Immediate sanity check completed in {execution_time:.2f}s")
                
                return {
                    "success": True,
                    "execution_time_seconds": execution_time,
                    "result": result
                }
            else:
                # Delta scan returned failure result
                error_msg = result.get("error_message", "Delta scan failed")
                logger.error(f"Immediate sanity check failed: {error_msg}")
                
                # Update failure metrics
                self.metrics.update_failure(execution_time)
                
                return {
                    "success": False,
                    "error": error_msg,
                    "execution_time_seconds": execution_time,
                    "result": result
                }
            
        except asyncio.TimeoutError:
            execution_time = time.perf_counter() - start_time
            error_msg = f"Immediate sanity check timed out after {self.config.max_execution_time_minutes} minutes"
            logger.error(error_msg)
            
            # Update failure metrics
            self.metrics.update_failure(execution_time)
            
            return {"success": False, "error": error_msg}
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            error_msg = f"Immediate sanity check failed: {e}"
            logger.error(error_msg)
            
            # Update failure metrics
            self.metrics.update_failure(execution_time)
            
            return {"success": False, "error": error_msg}
    
    def get_status(self) -> Dict[str, Any]:
        """Get current task status and metrics."""
        next_run_time = None
        if self.status == TaskStatus.RUNNING and self._start_time:
            # Calculate next run time based on interval
            time_since_start = datetime.now() - self._start_time
            initial_delay = timedelta(minutes=self.config.initial_delay_minutes)
            interval = timedelta(minutes=self.config.interval_minutes)
            
            if time_since_start < initial_delay:
                # Still in initial delay period
                next_run_time = self._start_time + initial_delay
            else:
                # Calculate next interval run
                elapsed_since_initial = time_since_start - initial_delay
                intervals_completed = int(elapsed_since_initial.total_seconds() // interval.total_seconds())
                next_run_time = self._start_time + initial_delay + (interval * (intervals_completed + 1))
        
        return {
            "status": self.status.value,
            "project_path": str(self.project_path),
            "collection_name": self.collection_name,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "next_run_time": next_run_time.isoformat() if next_run_time else None,
            "last_error": self._last_error,
            "config": {
                "interval_minutes": self.config.interval_minutes,
                "initial_delay_minutes": self.config.initial_delay_minutes,
                "max_retry_attempts": self.config.max_retry_attempts,
                "force_full_scan": self.config.force_full_scan,
                "max_execution_time_minutes": self.config.max_execution_time_minutes,
                "continue_on_error": self.config.continue_on_error,
                "max_consecutive_failures": self.config.max_consecutive_failures
            },
            "metrics": self.metrics.to_dict()
        }
    
    async def _run_periodic_task(self) -> None:
        """Main background task loop for periodic execution."""
        try:
            # Initial delay before first run
            logger.info(f"Waiting {self.config.initial_delay_minutes} minutes before first sanity check...")
            await asyncio.wait_for(
                self._shutdown_event.wait(),
                timeout=self.config.initial_delay_minutes * 60
            )
            # If we reach here, shutdown was requested during initial delay
            return
            
        except asyncio.TimeoutError:
            # Initial delay completed, start periodic execution
            pass
        
        while not self._shutdown_event.is_set():
            try:
                # Wait for resume if paused
                await self._pause_event.wait()
                
                # Check for shutdown during pause
                if self._shutdown_event.is_set():
                    break
                
                # Execute delta scan with retries
                await self._execute_with_retries()
                
                # Check consecutive failure limit
                if (self.metrics.consecutive_failures >= self.config.max_consecutive_failures and
                    not self.config.continue_on_error):
                    error_msg = f"Stopping task after {self.metrics.consecutive_failures} consecutive failures"
                    logger.error(error_msg)
                    self._last_error = error_msg
                    self.status = TaskStatus.ERROR
                    break
                
                # Wait for next interval
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.config.interval_minutes * 60
                    )
                    # If we reach here, shutdown was requested
                    break
                except asyncio.TimeoutError:
                    # Interval completed, continue to next iteration
                    continue
                    
            except asyncio.CancelledError:
                logger.debug("Periodic task cancelled")
                break
            except Exception as e:
                error_msg = f"Unexpected error in periodic task: {e}"
                logger.error(error_msg)
                self._last_error = error_msg
                
                if not self.config.continue_on_error:
                    self.status = TaskStatus.ERROR
                    break
                
                # Wait before retrying on unexpected errors
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.config.retry_delay_seconds
                    )
                    break  # Shutdown requested
                except asyncio.TimeoutError:
                    continue  # Retry after delay
    
    async def _execute_with_retries(self) -> None:
        """Execute delta scan with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_retry_attempts):
            if self._shutdown_event.is_set():
                break
            
            try:
                start_time = time.perf_counter()
                
                # Execute delta scan with timeout
                result = await asyncio.wait_for(
                    self._execute_delta_scan(),
                    timeout=self.config.max_execution_time_minutes * 60
                )
                
                execution_time = time.perf_counter() - start_time
                
                # Check if delta scan was actually successful
                if result.get("success", False):
                    self.metrics.update_success(execution_time)
                    
                    logger.info(
                        f"Sanity check completed successfully in {execution_time:.2f}s "
                        f"(attempt {attempt + 1}/{self.config.max_retry_attempts})"
                    )
                    return  # Success, exit retry loop
                else:
                    # Delta scan returned failure result, treat as failure
                    error_msg = result.get("error_message", "Delta scan failed")
                    logger.warning(f"Sanity check failed: {error_msg} (attempt {attempt + 1}/{self.config.max_retry_attempts})")
                    
                    if attempt == self.config.max_retry_attempts - 1:
                        # Final attempt failed
                        self.metrics.update_failure(execution_time)
                        self._last_error = error_msg
                    # Continue to retry logic below
                
            except asyncio.TimeoutError as e:
                execution_time = time.perf_counter() - start_time
                last_exception = e
                error_msg = f"Sanity check timed out after {self.config.max_execution_time_minutes} minutes"
                logger.warning(f"{error_msg} (attempt {attempt + 1}/{self.config.max_retry_attempts})")
                
                if attempt == self.config.max_retry_attempts - 1:
                    # Final attempt failed
                    self.metrics.update_failure(execution_time)
                    self._last_error = error_msg
                    
            except Exception as e:
                execution_time = time.perf_counter() - start_time
                last_exception = e
                error_msg = f"Sanity check failed: {e}"
                logger.warning(f"{error_msg} (attempt {attempt + 1}/{self.config.max_retry_attempts})")
                
                if attempt == self.config.max_retry_attempts - 1:
                    # Final attempt failed
                    self.metrics.update_failure(execution_time)
                    self._last_error = error_msg
            
            # Wait before retry (except on last attempt)
            if attempt < self.config.max_retry_attempts - 1:
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.config.retry_delay_seconds
                    )
                    break  # Shutdown requested
                except asyncio.TimeoutError:
                    continue  # Retry after delay
        
        # If we reach here, all retries failed
        if last_exception:
            logger.error(f"All {self.config.max_retry_attempts} sanity check attempts failed")
    
    async def _execute_delta_scan(self) -> Dict[str, Any]:
        """Execute the actual delta scan operation."""
        logger.debug(f"Starting delta scan for {self.collection_name}")
        
        # Call the indexer's perform_delta_scan method
        result = await self.indexer.perform_delta_scan(
            project_path=self.project_path,
            collection_name=self.collection_name,
            progress_callback=self.progress_callback,
            force_full_scan=self.config.force_full_scan
        )
        
        logger.debug(f"Delta scan completed: {result.get('summary', {})}")
        return result


class PeriodicSanityTaskManager:
    """
    Manager for multiple periodic sanity tasks across different projects/collections.
    
    Provides centralized management of multiple periodic tasks with lifecycle control.
    """
    
    def __init__(self):
        """Initialize the task manager."""
        self.tasks: Dict[str, PeriodicSanityTask] = {}
        self._manager_lock = asyncio.Lock()
        
    async def add_task(
        self,
        task_id: str,
        indexer,
        project_path: Path,
        collection_name: str,
        config: Optional[TaskConfig] = None,
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
        auto_start: bool = True
    ) -> bool:
        """
        Add a new periodic sanity task.
        
        Args:
            task_id: Unique identifier for the task
            indexer: HybridIndexer instance
            project_path: Project directory to scan
            collection_name: Target collection name
            config: Task configuration
            progress_callback: Optional progress callback
            auto_start: Whether to start the task immediately
            
        Returns:
            True if task was added successfully, False otherwise
        """
        async with self._manager_lock:
            if task_id in self.tasks:
                logger.warning(f"Task {task_id} already exists")
                return False
            
            try:
                task = PeriodicSanityTask(
                    indexer=indexer,
                    project_path=project_path,
                    collection_name=collection_name,
                    config=config,
                    progress_callback=progress_callback
                )
                
                self.tasks[task_id] = task
                
                if auto_start:
                    success = await task.start()
                    if not success:
                        # Remove task if start failed
                        del self.tasks[task_id]
                        return False
                
                logger.info(f"Added periodic sanity task: {task_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to add task {task_id}: {e}")
                return False
    
    async def remove_task(self, task_id: str) -> bool:
        """
        Remove a periodic sanity task.
        
        Args:
            task_id: Task identifier to remove
            
        Returns:
            True if task was removed successfully, False otherwise
        """
        async with self._manager_lock:
            if task_id not in self.tasks:
                logger.warning(f"Task {task_id} not found")
                return False
            
            try:
                task = self.tasks[task_id]
                await task.stop()
                del self.tasks[task_id]
                
                logger.info(f"Removed periodic sanity task: {task_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to remove task {task_id}: {e}")
                return False
    
    async def start_all(self) -> Dict[str, bool]:
        """Start all stopped tasks."""
        results = {}
        async with self._manager_lock:
            for task_id, task in self.tasks.items():
                if task.status == TaskStatus.STOPPED:
                    results[task_id] = await task.start()
                else:
                    results[task_id] = True
        return results
    
    async def stop_all(self) -> None:
        """Stop all running tasks."""
        async with self._manager_lock:
            stop_tasks = [task.stop() for task in self.tasks.values()]
            if stop_tasks:
                await asyncio.gather(*stop_tasks, return_exceptions=True)
    
    def get_task(self, task_id: str) -> Optional[PeriodicSanityTask]:
        """Get a specific task by ID."""
        return self.tasks.get(task_id)
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all managed tasks."""
        return {
            task_id: task.get_status()
            for task_id, task in self.tasks.items()
        }
    
    def get_running_count(self) -> int:
        """Get count of currently running tasks."""
        return sum(1 for task in self.tasks.values() if task.status == TaskStatus.RUNNING)
    
    def get_total_count(self) -> int:
        """Get total count of managed tasks."""
        return len(self.tasks)