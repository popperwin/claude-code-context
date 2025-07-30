"""
File System Event Models.

Defines event types, priorities, and data structures for file system
monitoring and synchronization.
"""

from datetime import datetime
from enum import Enum, IntEnum
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
import uuid


class EventType(Enum):
    """Types of file system events that trigger synchronization"""
    CREATED = "created"     # New file created
    MODIFIED = "modified"   # Existing file modified 
    DELETED = "deleted"     # File deleted
    MOVED = "moved"         # File moved/renamed
    
    
class EventPriority(IntEnum):
    """
    Priority levels for event processing.
    
    Lower numeric values = higher priority.
    This ensures deletions are processed immediately to prevent stale data.
    """
    CRITICAL = 1   # File deletions (immediate processing)
    HIGH = 2       # File modifications (batched but urgent)
    MEDIUM = 3     # File creation (can be queued)
    LOW = 4        # File moves/renames (can be delayed)


class FileSystemEvent(BaseModel):
    """
    Represents a file system event for synchronization processing.
    
    This model captures all information needed to process file system
    changes and update the corresponding Qdrant collection.
    """
    
    # Event identification
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType
    
    # File information
    file_path: Path
    old_path: Optional[Path] = None  # For move events
    
    # Timing and priority
    timestamp: datetime = Field(default_factory=datetime.now)
    priority: EventPriority
    
    # Processing state
    processed: bool = False
    processing_started_at: Optional[datetime] = None
    processing_completed_at: Optional[datetime] = None
    
    # Error handling
    retry_count: int = 0
    max_retries: int = 3
    last_error: Optional[str] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('file_path')
    @classmethod
    def validate_file_path(cls, v: Path) -> Path:
        """Ensure file path is absolute"""
        if not v.is_absolute():
            raise ValueError('File path must be absolute')
        return v
    
    @field_validator('old_path')
    @classmethod  
    def validate_old_path(cls, v: Optional[Path]) -> Optional[Path]:
        """Ensure old path is absolute if provided"""
        if v is not None and not v.is_absolute():
            raise ValueError('Old path must be absolute if provided')
        return v
    
    @classmethod
    def create_file_created(cls, file_path: Path, **kwargs) -> 'FileSystemEvent':
        """Create a file creation event"""
        return cls(
            event_type=EventType.CREATED,
            file_path=file_path,
            priority=EventPriority.MEDIUM,
            **kwargs
        )
    
    @classmethod
    def create_file_modified(cls, file_path: Path, **kwargs) -> 'FileSystemEvent':
        """Create a file modification event"""
        return cls(
            event_type=EventType.MODIFIED,
            file_path=file_path,
            priority=EventPriority.HIGH,
            **kwargs
        )
    
    @classmethod
    def create_file_deleted(cls, file_path: Path, **kwargs) -> 'FileSystemEvent':
        """Create a file deletion event"""
        return cls(
            event_type=EventType.DELETED,
            file_path=file_path,
            priority=EventPriority.CRITICAL,
            **kwargs
        )
    
    @classmethod
    def create_file_moved(
        cls, 
        old_path: Path, 
        new_path: Path, 
        **kwargs
    ) -> 'FileSystemEvent':
        """Create a file move/rename event"""
        return cls(
            event_type=EventType.MOVED,
            file_path=new_path,
            old_path=old_path,
            priority=EventPriority.LOW,
            **kwargs
        )
    
    def mark_processing_started(self) -> None:
        """Mark event as started processing"""
        self.processing_started_at = datetime.now()
    
    def mark_processing_completed(self) -> None:
        """Mark event as completed processing"""
        self.processing_completed_at = datetime.now()
        self.processed = True
    
    def mark_processing_failed(self, error: str) -> None:
        """Mark event processing as failed with error"""
        self.retry_count += 1
        self.last_error = error
        self.processing_completed_at = datetime.now()
    
    def can_retry(self) -> bool:
        """Check if event can be retried"""
        return self.retry_count < self.max_retries
    
    def is_expired(self, max_age_minutes: int = 60) -> bool:
        """Check if event is too old to process"""
        age = datetime.now() - self.timestamp
        return age.total_seconds() > (max_age_minutes * 60)
    
    @property
    def processing_duration(self) -> Optional[float]:
        """Get processing duration in seconds"""
        if self.processing_started_at and self.processing_completed_at:
            delta = self.processing_completed_at - self.processing_started_at
            return delta.total_seconds()
        return None
    
    @property
    def age_seconds(self) -> float:
        """Get event age in seconds"""
        return (datetime.now() - self.timestamp).total_seconds()
    
    @property
    def should_process_immediately(self) -> bool:
        """Check if event should be processed immediately"""
        return self.priority == EventPriority.CRITICAL
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "file_path": str(self.file_path),
            "old_path": str(self.old_path) if self.old_path else None,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.value,
            "processed": self.processed,
            "retry_count": self.retry_count,
            "age_seconds": self.age_seconds,
            "processing_duration": self.processing_duration,
            "last_error": self.last_error
        }
    
    def __str__(self) -> str:
        """String representation for logging"""
        old_part = f" (from {self.old_path})" if self.old_path else ""
        return f"{self.event_type.value.upper()}: {self.file_path}{old_part} [P{self.priority}]"
    
    def __lt__(self, other: 'FileSystemEvent') -> bool:
        """Compare events for priority queue ordering (lower priority value = higher priority)"""
        if not isinstance(other, FileSystemEvent):
            return NotImplemented
        
        # Primary sort by priority (lower number = higher priority)
        if self.priority != other.priority:
            return self.priority < other.priority
        
        # Secondary sort by timestamp (older events first within same priority)
        return self.timestamp < other.timestamp


class EventBatch(BaseModel):
    """
    A batch of events for efficient processing.
    
    Groups events by type and priority for optimized processing
    while maintaining individual event tracking.
    """
    
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    events: list[FileSystemEvent] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    max_size: int = 10
    
    def add_event(self, event: FileSystemEvent) -> bool:
        """
        Add event to batch if there's space.
        
        Returns:
            True if event was added, False if batch is full
        """
        if len(self.events) >= self.max_size:
            return False
        
        self.events.append(event)
        return True
    
    def is_full(self) -> bool:
        """Check if batch is at capacity"""
        return len(self.events) >= self.max_size
    
    def get_events_by_type(self, event_type: EventType) -> list[FileSystemEvent]:
        """Get all events of a specific type from the batch"""
        return [event for event in self.events if event.event_type == event_type]
    
    def get_highest_priority(self) -> EventPriority:
        """Get the highest priority event in the batch"""
        if not self.events:
            return EventPriority.LOW
        return min(event.priority for event in self.events)
    
    @property
    def event_count(self) -> int:
        """Number of events in batch"""
        return len(self.events)
    
    @property
    def file_paths(self) -> set[Path]:
        """Get unique file paths in batch"""
        return {event.file_path for event in self.events}
    
    def get_priority_stats(self) -> Dict[EventPriority, int]:
        """Get statistics of event priorities in the batch."""
        stats = {priority: 0 for priority in EventPriority}
        for event in self.events:
            stats[event.priority] += 1
        return stats
    
    def get_file_paths(self) -> set[Path]:
        """Get unique file paths in batch."""
        return self.file_paths
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            "batch_id": self.batch_id,
            "event_count": self.event_count,
            "created_at": self.created_at.isoformat(),
            "highest_priority": self.get_highest_priority().value,
            "event_types": [event.event_type.value for event in self.events],
            "file_paths": [str(path) for path in self.file_paths]
        }