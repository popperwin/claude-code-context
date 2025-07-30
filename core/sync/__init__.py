"""
Real-time Project-Collection Synchronization System.

This module provides real-time synchronization between project filesystems
and Qdrant collections, ensuring perfect consistency and eliminating stale data.

Key Components:
- DeterministicEntityId: Stable entity identification for atomic operations
- FileSystemEvent: Event models for file system operations  
- PriorityEventQueue: Asynchronous event processing with priorities
- ProjectFileSystemWatcher: Cross-platform file monitoring with debouncing
- EntityLifecycleManager: Entity creation, update, and deletion operations
- ProjectCollectionSyncEngine: Central coordinator for synchronization
- CollectionConsistencyValidator: Validation and automatic repair

This system replaces incremental indexing for real-time projects, providing
immediate consistency without manual synchronization or restart requirements.
"""

from .deterministic import DeterministicEntityId
from .events import FileSystemEvent, EventType, EventPriority, EventBatch
from .queue import PriorityEventQueue
from .watcher import ProjectFileSystemWatcher
from .lifecycle import EntityLifecycleManager
from .engine import ProjectCollectionSyncEngine
from .validator import CollectionConsistencyValidator

__all__ = [
    "DeterministicEntityId",
    "FileSystemEvent", 
    "EventType",
    "EventPriority",
    "EventBatch",
    "PriorityEventQueue",
    "ProjectFileSystemWatcher",
    "EntityLifecycleManager",
    "ProjectCollectionSyncEngine",
    "CollectionConsistencyValidator",
]

__version__ = "1.0.0"