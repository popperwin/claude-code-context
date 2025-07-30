"""
Tests for PriorityEventQueue behavior and batch processing validation.

Validates priority ordering, async operations, batch creation,
and queue size management for file system events.
"""

import pytest
import asyncio
from datetime import datetime
from pathlib import Path

from core.sync.events import FileSystemEvent, EventType, EventPriority, EventBatch
from core.sync.queue import PriorityEventQueue


class TestPriorityEventQueue:
    """Test suite for priority event queue functionality."""
    
    @pytest.fixture
    def event_queue(self):
        """Create a priority event queue for testing."""
        return PriorityEventQueue(max_queue_size=100, max_batch_size=10)
    
    def create_test_event(
        self,
        event_type: EventType,
        file_path: str = "/test/file.py",
        old_path: str = None
    ) -> FileSystemEvent:
        """Create a test file system event."""
        file_path_obj = Path(file_path)
        old_path_obj = Path(old_path) if old_path else None
        
        if event_type == EventType.CREATED:
            return FileSystemEvent.create_file_created(file_path_obj)
        elif event_type == EventType.MODIFIED:
            return FileSystemEvent.create_file_modified(file_path_obj)
        elif event_type == EventType.DELETED:
            return FileSystemEvent.create_file_deleted(file_path_obj)
        elif event_type == EventType.MOVED:
            if old_path_obj is None:
                raise ValueError("old_path is required for MOVED events")
            return FileSystemEvent.create_file_moved(old_path_obj, file_path_obj)
        else:
            raise ValueError(f"Unknown event type: {event_type}")
    
    @pytest.mark.asyncio
    async def test_basic_enqueue_dequeue(self, event_queue):
        """Test basic enqueue and dequeue operations."""
        await event_queue.start()
        
        try:
            # Create test event
            event = self.create_test_event(EventType.CREATED)
            
            # Enqueue event
            success = await event_queue.enqueue(event)
            assert success is True
            
            # Check queue size
            size = await event_queue.size()
            assert size == 1
            
            # Dequeue event
            dequeued_event = await event_queue.dequeue(timeout=1.0)
            assert dequeued_event is not None
            assert dequeued_event.event_type == EventType.CREATED
            assert dequeued_event.file_path == Path("/test/file.py")
            
            # Queue should be empty now
            size = await event_queue.size()
            assert size == 0
            
        finally:
            await event_queue.stop()
    
    @pytest.mark.asyncio
    async def test_priority_ordering(self, event_queue):
        """Test that events are dequeued in priority order."""
        await event_queue.start()
        
        try:
            # Create events with different priorities
            # (CRITICAL=1, HIGH=2, MEDIUM=3, LOW=4)
            low_event = self.create_test_event(EventType.MOVED, "/test/low.py", "/test/old_low.py")      # LOW=4
            medium_event = self.create_test_event(EventType.CREATED, "/test/med.py") # MEDIUM=3
            high_event = self.create_test_event(EventType.MODIFIED, "/test/high.py") # HIGH=2
            critical_event = self.create_test_event(EventType.DELETED, "/test/crit.py") # CRITICAL=1
            
            # Enqueue in non-priority order
            await event_queue.enqueue(low_event)
            await event_queue.enqueue(medium_event)
            await event_queue.enqueue(critical_event)
            await event_queue.enqueue(high_event)
            
            # Dequeue should return in priority order (lowest number = highest priority)
            event1 = await event_queue.dequeue(timeout=1.0)
            assert event1.event_type == EventType.DELETED  # CRITICAL=1
            
            event2 = await event_queue.dequeue(timeout=1.0)
            assert event2.event_type == EventType.MODIFIED  # HIGH=2
            
            event3 = await event_queue.dequeue(timeout=1.0)
            assert event3.event_type == EventType.CREATED  # MEDIUM=3
            
            event4 = await event_queue.dequeue(timeout=1.0)
            assert event4.event_type == EventType.MOVED  # LOW=4
            
        finally:
            await event_queue.stop()
    
    @pytest.mark.asyncio
    async def test_fifo_within_same_priority(self, event_queue):
        """Test FIFO ordering within same priority level."""
        await event_queue.start()
        
        try:
            # Create multiple events with same priority (MODIFIED = HIGH)
            event1 = self.create_test_event(EventType.MODIFIED, "/test/file1.py")
            event2 = self.create_test_event(EventType.MODIFIED, "/test/file2.py")
            event3 = self.create_test_event(EventType.MODIFIED, "/test/file3.py")
            
            # Add small delay between enqueues to ensure different timestamps
            await event_queue.enqueue(event1)
            await asyncio.sleep(0.001)
            await event_queue.enqueue(event2)
            await asyncio.sleep(0.001)
            await event_queue.enqueue(event3)
            
            # Should dequeue in FIFO order
            dequeued1 = await event_queue.dequeue(timeout=1.0)
            assert dequeued1.file_path == Path("/test/file1.py")
            
            dequeued2 = await event_queue.dequeue(timeout=1.0)
            assert dequeued2.file_path == Path("/test/file2.py")
            
            dequeued3 = await event_queue.dequeue(timeout=1.0)
            assert dequeued3.file_path == Path("/test/file3.py")
            
        finally:
            await event_queue.stop()
    
    @pytest.mark.asyncio
    async def test_batch_dequeue(self, event_queue):
        """Test batch dequeue functionality."""
        await event_queue.start()
        
        try:
            # Enqueue multiple events
            events = []
            for i in range(5):
                event = self.create_test_event(EventType.CREATED, f"/test/file{i}.py")
                events.append(event)
                await event_queue.enqueue(event)
            
            # Batch dequeue with limit
            batch = await event_queue.batch_dequeue(max_size=3)
            assert len(batch) == 3
            
            # Remaining events
            remaining_size = await event_queue.size()
            assert remaining_size == 2
            
            # Get remaining events
            remaining_batch = await event_queue.batch_dequeue(max_size=5)
            assert len(remaining_batch) == 2
            
        finally:
            await event_queue.stop()
    
    @pytest.mark.asyncio
    async def test_create_batch(self, event_queue):
        """Test batch creation with EventBatch wrapper."""
        await event_queue.start()
        
        try:
            # Enqueue multiple events with different priorities
            critical_event = self.create_test_event(EventType.DELETED, "/test/crit.py")
            high_event1 = self.create_test_event(EventType.MODIFIED, "/test/high1.py")
            high_event2 = self.create_test_event(EventType.MODIFIED, "/test/high2.py")
            
            await event_queue.enqueue(high_event1)
            await event_queue.enqueue(critical_event)
            await event_queue.enqueue(high_event2)
            
            # Create batch
            batch = await event_queue.create_batch(timeout=1.0)
            assert batch is not None
            assert isinstance(batch, EventBatch)
            
            # Should contain all events, prioritized
            assert len(batch.events) == 3
            assert batch.events[0].event_type == EventType.DELETED  # Critical first
            assert batch.events[1].event_type == EventType.MODIFIED  # High priority
            assert batch.events[2].event_type == EventType.MODIFIED  # High priority
            
            # Batch should have valid ID and timestamp
            assert batch.batch_id is not None
            assert batch.created_at is not None
            assert isinstance(batch.created_at, datetime)
            
        finally:
            await event_queue.stop()
    
    @pytest.mark.asyncio
    async def test_queue_size_limits(self, event_queue):
        """Test queue size limits and overflow handling."""
        # Create queue with small size limit
        small_queue = PriorityEventQueue(max_queue_size=3, max_batch_size=2)
        await small_queue.start()
        
        try:
            # Fill queue to capacity
            for i in range(3):
                event = self.create_test_event(EventType.CREATED, f"/test/file{i}.py")
                success = await small_queue.enqueue(event)
                assert success is True
            
            # Queue should be at capacity
            size = await small_queue.size()
            assert size == 3
            
            # Try to add one more (should fail or handle overflow)
            overflow_event = self.create_test_event(EventType.CREATED, "/test/overflow.py")
            success = await small_queue.enqueue(overflow_event)
            
            # Depending on implementation, might reject or handle overflow
            # Size should not exceed max_queue_size
            final_size = await small_queue.size()
            assert final_size <= 3
            
        finally:
            await small_queue.stop()
    
    @pytest.mark.asyncio
    async def test_dequeue_timeout(self, event_queue):
        """Test dequeue timeout behavior on empty queue."""
        await event_queue.start()
        
        try:
            # Try to dequeue from empty queue with short timeout
            start_time = asyncio.get_event_loop().time()
            result = await event_queue.dequeue(timeout=0.1)
            end_time = asyncio.get_event_loop().time()
            
            # Should return None after timeout
            assert result is None
            
            # Should have waited approximately the timeout duration
            elapsed = end_time - start_time
            assert 0.08 <= elapsed <= 0.15  # Allow some variance
            
        finally:
            await event_queue.stop()
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, event_queue):
        """Test concurrent enqueue/dequeue operations."""
        await event_queue.start()
        
        try:
            results = []
            
            async def enqueue_worker():
                """Worker that enqueues events."""
                for i in range(10):
                    event = self.create_test_event(EventType.CREATED, f"/test/worker{i}.py")
                    await event_queue.enqueue(event)
                    await asyncio.sleep(0.01)  # Small delay
            
            async def dequeue_worker():
                """Worker that dequeues events."""
                for _ in range(10):
                    event = await event_queue.dequeue(timeout=2.0)
                    if event:
                        results.append(event)
                    await asyncio.sleep(0.01)  # Small delay
            
            # Run both workers concurrently
            await asyncio.gather(enqueue_worker(), dequeue_worker())
            
            # Should have dequeued all enqueued events
            assert len(results) == 10
            
            # All results should be valid events
            for event in results:
                assert isinstance(event, FileSystemEvent)
                assert event.event_type == EventType.CREATED
            
        finally:
            await event_queue.stop()
    
    @pytest.mark.asyncio
    async def test_queue_metrics(self, event_queue):
        """Test queue metrics tracking."""
        await event_queue.start()
        
        try:
            # Enqueue some events
            for i in range(5):
                event = self.create_test_event(EventType.CREATED, f"/test/file{i}.py")
                await event_queue.enqueue(event)
            
            # Get metrics
            metrics = event_queue.get_metrics()
            
            # Should have basic metrics
            assert "events_enqueued" in metrics
            assert "events_dequeued" in metrics
            assert "current_size" in metrics
            assert "max_size_reached" in metrics
            
            # Current size should match
            assert metrics["current_size"] == 5
            assert metrics["events_enqueued"] >= 5
            
        finally:
            await event_queue.stop()
    
    @pytest.mark.asyncio
    async def test_queue_lifecycle(self, event_queue):
        """Test queue start/stop lifecycle."""
        # Queue should start not active
        assert not event_queue.is_active
        
        # Start queue
        await event_queue.start()
        assert event_queue.is_active
        
        # Should be able to enqueue/dequeue
        event = self.create_test_event(EventType.CREATED)
        success = await event_queue.enqueue(event)
        assert success is True
        
        # Stop queue
        await event_queue.stop()
        assert not event_queue.is_active
        
        # Should not be able to enqueue after stop
        event2 = self.create_test_event(EventType.MODIFIED)
        success = await event_queue.enqueue(event2)
        assert success is False
    
    @pytest.mark.asyncio
    async def test_batch_size_limits(self, event_queue):
        """Test batch size limiting in batch operations."""
        await event_queue.start()
        
        try:
            # Enqueue more events than max batch size
            for i in range(15):
                event = self.create_test_event(EventType.CREATED, f"/test/file{i}.py")
                await event_queue.enqueue(event)
            
            # Create batch should respect max_batch_size
            batch = await event_queue.create_batch(timeout=1.0)
            assert batch is not None
            assert len(batch.events) <= event_queue.max_batch_size
            
            # Remaining events should still be in queue
            remaining_size = await event_queue.size()
            assert remaining_size == 15 - len(batch.events)
            
        finally:
            await event_queue.stop()


class TestEventBatch:
    """Test suite for EventBatch functionality."""
    
    def test_event_batch_creation(self):
        """Test EventBatch creation and properties."""
        events = [
            FileSystemEvent.create_file_created(Path("/test/file1.py")),
            FileSystemEvent.create_file_modified(Path("/test/file2.py")),
            FileSystemEvent.create_file_deleted(Path("/test/file3.py"))
        ]
        
        batch = EventBatch(events=events)
        
        # Should have valid properties
        assert batch.batch_id is not None
        assert len(batch.batch_id) > 0
        assert batch.created_at is not None
        assert isinstance(batch.created_at, datetime)
        assert len(batch.events) == 3
        
        # Events should be in the batch
        assert batch.events[0].event_type == EventType.CREATED
        assert batch.events[1].event_type == EventType.MODIFIED
        assert batch.events[2].event_type == EventType.DELETED
    
    def test_event_batch_priority_stats(self):
        """Test EventBatch priority statistics."""
        events = [
            FileSystemEvent.create_file_deleted(Path("/test/file1.py")),  # CRITICAL
            FileSystemEvent.create_file_modified(Path("/test/file2.py")), # HIGH
            FileSystemEvent.create_file_modified(Path("/test/file3.py")), # HIGH
            FileSystemEvent.create_file_created(Path("/test/file4.py")),  # MEDIUM
        ]
        
        batch = EventBatch(events=events)
        priority_stats = batch.get_priority_stats()
        
        # Should have correct priority counts
        assert priority_stats[EventPriority.CRITICAL] == 1
        assert priority_stats[EventPriority.HIGH] == 2
        assert priority_stats[EventPriority.MEDIUM] == 1
        assert priority_stats[EventPriority.LOW] == 0
    
    def test_event_batch_file_paths(self):
        """Test EventBatch file path extraction."""
        events = [
            FileSystemEvent.create_file_created(Path("/test/file1.py")),
            FileSystemEvent.create_file_modified(Path("/test/file2.js")),
            FileSystemEvent.create_file_deleted(Path("/other/file3.go"))
        ]
        
        batch = EventBatch(events=events)
        file_paths = batch.get_file_paths()
        
        # Should return all unique file paths
        expected_paths = {
            Path("/test/file1.py"),
            Path("/test/file2.js"),
            Path("/other/file3.go")
        }
        assert file_paths == expected_paths
    
    def test_empty_event_batch(self):
        """Test EventBatch with no events."""
        batch = EventBatch(events=[])
        
        # Should handle empty batch gracefully
        assert len(batch.events) == 0
        assert batch.batch_id is not None
        assert batch.created_at is not None
        
        priority_stats = batch.get_priority_stats()
        assert all(count == 0 for count in priority_stats.values())
        
        file_paths = batch.get_file_paths()
        assert len(file_paths) == 0


class TestPriorityEventQueueEdgeCases:
    """Test edge cases and error conditions for priority event queue."""
    
    @pytest.mark.asyncio
    async def test_double_start(self):
        """Test starting queue that's already started."""
        queue = PriorityEventQueue()
        
        # Start once
        await queue.start()
        assert queue.is_active
        
        # Start again (should handle gracefully)
        await queue.start()
        assert queue.is_active
        
        await queue.stop()
    
    @pytest.mark.asyncio
    async def test_double_stop(self):
        """Test stopping queue that's already stopped."""
        queue = PriorityEventQueue()
        
        await queue.start()
        await queue.stop()
        assert not queue.is_active
        
        # Stop again (should handle gracefully)
        await queue.stop()
        assert not queue.is_active
    
    @pytest.mark.asyncio
    async def test_operations_on_stopped_queue(self):
        """Test operations on stopped queue."""
        queue = PriorityEventQueue()
        
        # Operations on never-started queue
        event = FileSystemEvent.create_file_created(Path("/test/file.py"))
        
        success = await queue.enqueue(event)
        assert success is False
        
        result = await queue.dequeue(timeout=0.1)
        assert result is None
        
        batch = await queue.batch_dequeue(max_size=5)
        assert len(batch) == 0
        
        size = await queue.size()
        assert size == 0