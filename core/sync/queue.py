"""
Priority Event Queue.

Provides asynchronous event processing with priority handling, batch
processing, and size limits for efficient synchronization.
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Set
import heapq
from dataclasses import dataclass

from .events import FileSystemEvent, EventType, EventPriority, EventBatch

logger = logging.getLogger(__name__)


@dataclass
class QueueMetrics:
    """Metrics for monitoring queue performance"""
    total_events_enqueued: int = 0
    total_events_dequeued: int = 0
    total_events_expired: int = 0
    total_batches_created: int = 0
    current_queue_size: int = 0
    events_by_priority: Dict[EventPriority, int] = None
    events_by_type: Dict[EventType, int] = None
    avg_wait_time_seconds: float = 0.0
    max_queue_size_reached: int = 0
    
    def __post_init__(self):
        if self.events_by_priority is None:
            self.events_by_priority = defaultdict(int)
        if self.events_by_type is None:
            self.events_by_type = defaultdict(int)


class PriorityEventQueue:
    """
    Asynchronous priority queue for file system events with intelligent batching.
    
    Features:
    - Priority-based processing (deletions first, then modifications, then creations)
    - Size limits to prevent memory overflow
    - Event expiration to handle stale events
    - Batch processing for efficiency
    - Comprehensive metrics and monitoring
    - Thread-safe operations with asyncio
    """
    
    def __init__(
        self,
        max_queue_size: int = 1000,
        max_batch_size: int = 10,
        max_event_age_minutes: int = 60,
        batch_timeout_seconds: float = 1.0
    ):
        """
        Initialize the priority event queue.
        
        Args:
            max_queue_size: Maximum number of events in queue
            max_batch_size: Maximum events per batch
            max_event_age_minutes: Maximum age before event expires
            batch_timeout_seconds: Maximum time to wait for batch completion
        """
        self.max_queue_size = max_queue_size
        self.max_batch_size = max_batch_size
        self.max_event_age_minutes = max_event_age_minutes
        self.batch_timeout_seconds = batch_timeout_seconds
        
        # Priority queue implementation using heapq
        self._queue: List[FileSystemEvent] = []
        self._queue_lock = asyncio.Lock()
        
        # Event tracking for deduplication
        self._pending_events: Set[str] = set()  # Set of file paths
        self._event_waiters: List[asyncio.Future] = []
        
        # Metrics and monitoring
        self.metrics = QueueMetrics()
        self._start_time = datetime.now()
        
        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info(f"Initialized PriorityEventQueue with max_size={max_queue_size}, max_batch_size={max_batch_size}")
    
    @property
    def is_active(self) -> bool:
        """Check if the queue is currently active"""
        return self._running
    
    async def start(self) -> None:
        """Start the queue background tasks"""
        if self._running:
            return
        
        self._running = True
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        logger.info("Started PriorityEventQueue background tasks")
    
    async def stop(self) -> None:
        """Stop the queue and cleanup resources"""
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Wake up any waiting consumers
        for waiter in self._event_waiters:
            if not waiter.done():
                waiter.cancel()
        
        self._event_waiters.clear()
        logger.info("Stopped PriorityEventQueue")
    
    async def enqueue(self, event: FileSystemEvent) -> bool:
        """
        Add an event to the priority queue.
        
        Args:
            event: The file system event to enqueue
            
        Returns:
            True if event was enqueued, False if queue is full or event is duplicate
        """
        if not self._running:
            return False
            
        async with self._queue_lock:
            # Check queue size limit
            if len(self._queue) >= self.max_queue_size:
                logger.warning(f"Queue full ({len(self._queue)} events), dropping event: {event}")
                return False
            
            # Check for duplicate events (same file path)
            file_key = str(event.file_path)
            if file_key in self._pending_events:
                logger.debug(f"Dropping duplicate event for {event.file_path}")
                return False
            
            # Add to queue and tracking
            heapq.heappush(self._queue, event)
            self._pending_events.add(file_key)
            
            # Update metrics
            self.metrics.total_events_enqueued += 1
            self.metrics.current_queue_size = len(self._queue)
            self.metrics.events_by_priority[event.priority] += 1
            self.metrics.events_by_type[event.event_type] += 1
            self.metrics.max_queue_size_reached = max(
                self.metrics.max_queue_size_reached, 
                len(self._queue)
            )
            
            # Wake up any waiting consumers
            if self._event_waiters:
                waiter = self._event_waiters.pop(0)
                if not waiter.done():
                    waiter.set_result(None)
            
            logger.debug(f"Enqueued event: {event} (queue size: {len(self._queue)})")
            return True
    
    async def dequeue(self, timeout: Optional[float] = None) -> Optional[FileSystemEvent]:
        """
        Remove and return the highest priority event from the queue.
        
        Args:
            timeout: Maximum seconds to wait for an event
            
        Returns:
            The highest priority event, or None if timeout/empty
        """
        async with self._queue_lock:
            # Try to get an event immediately
            event = self._pop_valid_event()
            if event:
                return event
        
        # No event available, wait for one
        if timeout is None or timeout > 0:
            try:
                waiter = asyncio.Future()
                self._event_waiters.append(waiter)
                
                if timeout:
                    await asyncio.wait_for(waiter, timeout=timeout)
                else:
                    await waiter
                
                # Try again after waiting
                async with self._queue_lock:
                    return self._pop_valid_event()
                    
            except asyncio.TimeoutError:
                # Remove waiter if it timed out
                if waiter in self._event_waiters:
                    self._event_waiters.remove(waiter)
                return None
            except asyncio.CancelledError:
                return None
        
        return None
    
    async def batch_dequeue(self, max_size: Optional[int] = None) -> List[FileSystemEvent]:
        """
        Remove and return a batch of events for efficient processing.
        
        Args:
            max_size: Maximum events in batch (defaults to queue max_batch_size)
            
        Returns:
            List of events in priority order
        """
        if max_size is None:
            max_size = self.max_batch_size
        
        batch_events = []
        
        async with self._queue_lock:
            # Collect up to max_size events
            while len(batch_events) < max_size:
                event = self._pop_valid_event()
                if not event:
                    break
                batch_events.append(event)
            
            # Update metrics
            if batch_events:
                self.metrics.total_batches_created += 1
        
        logger.debug(f"Dequeued batch of {len(batch_events)} events")
        return batch_events
    
    async def create_batch(self, timeout: Optional[float] = None) -> Optional[EventBatch]:
        """
        Create an EventBatch with optimal batching strategy.
        
        Args:
            timeout: Maximum seconds to wait for batch completion
            
        Returns:
            EventBatch with events, or None if timeout/empty
        """
        # Start with any immediate events
        events = await self.batch_dequeue(self.max_batch_size)
        
        if not events:
            # Wait for at least one event
            first_event = await self.dequeue(timeout)
            if not first_event:
                return None
            events.append(first_event)
        
        # If we have critical priority events, process immediately
        if any(event.should_process_immediately for event in events):
            batch = EventBatch(events=events, max_size=self.max_batch_size)
            logger.debug(f"Created immediate batch with {len(events)} events (contains critical)")
            return batch
        
        # For non-critical events, try to fill batch within timeout
        if timeout and timeout > 0:
            start_time = datetime.now()
            end_time = start_time + timedelta(seconds=timeout)
            
            while len(events) < self.max_batch_size and datetime.now() < end_time:
                remaining_time = (end_time - datetime.now()).total_seconds()
                if remaining_time <= 0:
                    break
                
                additional_event = await self.dequeue(timeout=remaining_time)
                if additional_event:
                    events.append(additional_event)
                else:
                    break  # Timeout or no more events
        
        if events:
            batch = EventBatch(events=events, max_size=self.max_batch_size)
            logger.debug(f"Created batch with {len(events)} events")
            return batch
        
        return None
    
    def _pop_valid_event(self) -> Optional[FileSystemEvent]:
        """
        Pop the highest priority valid event from the queue.
        
        Must be called with _queue_lock held.
        """
        while self._queue:
            event = heapq.heappop(self._queue)
            
            # Remove from pending tracking
            file_key = str(event.file_path)
            self._pending_events.discard(file_key)
            
            # Check if event is expired
            if event.is_expired(self.max_event_age_minutes):
                logger.debug(f"Expired event dropped: {event}")
                self.metrics.total_events_expired += 1
                continue
            
            # Update metrics
            self.metrics.total_events_dequeued += 1
            self.metrics.current_queue_size = len(self._queue)
            
            # Calculate wait time
            wait_time = event.age_seconds
            if self.metrics.total_events_dequeued > 0:
                # Update rolling average wait time
                total_events = self.metrics.total_events_dequeued
                self.metrics.avg_wait_time_seconds = (
                    (self.metrics.avg_wait_time_seconds * (total_events - 1) + wait_time) / total_events
                )
            
            return event
        
        return None
    
    async def _periodic_cleanup(self) -> None:
        """Background task to cleanup expired events and update metrics"""
        while self._running:
            try:
                await asyncio.sleep(30)  # Cleanup every 30 seconds
                
                async with self._queue_lock:
                    # Remove expired events
                    valid_events = []
                    expired_count = 0
                    
                    for event in self._queue:
                        if event.is_expired(self.max_event_age_minutes):
                            expired_count += 1
                            file_key = str(event.file_path)
                            self._pending_events.discard(file_key)
                        else:
                            valid_events.append(event)
                    
                    if expired_count > 0:
                        # Rebuild heap with valid events
                        self._queue = valid_events
                        heapq.heapify(self._queue)
                        self.metrics.total_events_expired += expired_count
                        self.metrics.current_queue_size = len(self._queue)
                        logger.info(f"Cleaned up {expired_count} expired events")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
    
    async def size(self) -> int:
        """Get current queue size"""
        async with self._queue_lock:
            return len(self._queue)
    
    async def is_empty(self) -> bool:
        """Check if queue is empty"""
        return await self.size() == 0
    
    async def clear(self) -> int:
        """Clear all events from queue and return count cleared"""
        async with self._queue_lock:
            count = len(self._queue)
            self._queue.clear()
            self._pending_events.clear()
            self.metrics.current_queue_size = 0
            logger.info(f"Cleared {count} events from queue")
            return count
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive queue metrics"""
        uptime = (datetime.now() - self._start_time).total_seconds()
        
        return {
            "current_size": self.metrics.current_queue_size,
            "max_size_reached": self.metrics.max_queue_size_reached,
            "events_enqueued": self.metrics.total_events_enqueued,
            "events_dequeued": self.metrics.total_events_dequeued,
            "events_expired": self.metrics.total_events_expired,
            "batches_created": self.metrics.total_batches_created,
            "events_by_priority": dict(self.metrics.events_by_priority),
            "events_by_type": dict(self.metrics.events_by_type),
            "avg_wait_time_seconds": self.metrics.avg_wait_time_seconds,
            "uptime_seconds": uptime,
            "events_per_second": self.metrics.total_events_enqueued / max(uptime, 1),
            "queue_utilization": self.metrics.current_queue_size / self.max_queue_size
        }
    
    def __len__(self) -> int:
        """Get current queue size (sync version)"""
        return len(self._queue)
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()