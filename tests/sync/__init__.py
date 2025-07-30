"""
Test suite for the real-time synchronization system.

This package contains comprehensive tests for all synchronization components:
- DeterministicEntityId generation and consistency
- FileSystemEvent models and priority handling 
- PriorityEventQueue behavior and batch processing
- ProjectFileSystemWatcher file monitoring and debouncing
- EntityLifecycleManager entity operations and atomic replacement
- ProjectCollectionSyncEngine integration and coordination
- CollectionConsistencyValidator validation and repair mechanisms

These tests ensure the synchronization system provides reliable real-time
consistency between project filesystems and Qdrant collections.
"""