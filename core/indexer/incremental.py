"""
Incremental indexing with intelligent file change detection.

This module provides efficient incremental updates by tracking file modifications,
content hashes, and indexing state to avoid unnecessary reprocessing.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, TYPE_CHECKING
from dataclasses import dataclass, asdict
import aiofiles

if TYPE_CHECKING:
    from ..storage.client import HybridQdrantClient

logger = logging.getLogger(__name__)


@dataclass
class FileIndexState:
    """State tracking for indexed files"""
    file_path: str
    file_hash: str
    file_size: int
    last_modified: float  # Unix timestamp
    last_indexed: datetime
    entity_count: int
    relation_count: int
    collection_name: str
    indexing_success: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        # Convert datetime to ISO string
        data['last_indexed'] = self.last_indexed.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileIndexState':
        """Create from dictionary (JSON deserialization)"""
        # Convert ISO string back to datetime
        data['last_indexed'] = datetime.fromisoformat(data['last_indexed'])
        return cls(**data)


class FileChangeDetector:
    """
    Efficient file change detection using multiple strategies.
    
    Strategies:
    1. File modification time comparison
    2. File size comparison  
    3. Content hash comparison (for definitive change detection)
    4. Existence checks (for deleted files)
    """
    
    def __init__(self):
        self._hash_cache: Dict[str, str] = {}
        self._stat_cache: Dict[str, os.stat_result] = {}
    
    async def compute_file_hash(self, file_path: Path) -> str:
        """
        Compute SHA256 hash of file content efficiently.
        
        Args:
            file_path: Path to file
            
        Returns:
            Hexadecimal hash string
        """
        file_key = str(file_path)
        
        # Check if we need to recompute hash
        try:
            stat = file_path.stat()
            cached_stat = self._stat_cache.get(file_key)
            
            if (cached_stat and 
                cached_stat.st_mtime == stat.st_mtime and
                cached_stat.st_size == stat.st_size and
                file_key in self._hash_cache):
                return self._hash_cache[file_key]
            
            # Update stat cache
            self._stat_cache[file_key] = stat
            
        except (OSError, IOError) as e:
            logger.warning(f"Cannot stat file {file_path}: {e}")
            return ""
        
        # Compute hash
        hasher = hashlib.sha256()
        try:
            async with aiofiles.open(file_path, 'rb') as f:
                # Read in chunks for memory efficiency
                while chunk := await f.read(8192):
                    hasher.update(chunk)
            
            file_hash = hasher.hexdigest()
            self._hash_cache[file_key] = file_hash
            return file_hash
            
        except (OSError, IOError) as e:
            logger.warning(f"Cannot read file {file_path}: {e}")
            return ""
    
    def is_file_modified(
        self, 
        file_path: Path, 
        last_state: FileIndexState
    ) -> bool:
        """
        Check if file has been modified since last indexing.
        
        Args:
            file_path: Path to check
            last_state: Previous indexing state
            
        Returns:
            True if file has been modified
        """
        try:
            stat = file_path.stat()
            
            # Quick checks first
            if stat.st_mtime > last_state.last_modified:
                return True
            
            if stat.st_size != last_state.file_size:
                return True
            
            # If quick checks pass, file is likely unchanged
            return False
            
        except (OSError, IOError):
            # File doesn't exist or can't be accessed
            return True
    
    async def is_file_content_changed(
        self,
        file_path: Path,
        last_state: FileIndexState
    ) -> bool:
        """
        Definitive content change check using hash comparison.
        
        Args:
            file_path: Path to check
            last_state: Previous indexing state
            
        Returns:
            True if file content has changed
        """
        current_hash = await self.compute_file_hash(file_path)
        return current_hash != last_state.file_hash
    
    def clear_cache(self) -> None:
        """Clear internal caches"""
        self._hash_cache.clear()
        self._stat_cache.clear()


class IncrementalIndexer:
    """
    Manages incremental indexing state and change detection.
    
    Features:
    - Persistent state storage per collection
    - Efficient change detection strategies
    - Batch file processing for performance
    - Graceful handling of missing/corrupted state
    """
    
    def __init__(self, state_dir: Optional[Path] = None):
        """
        Initialize incremental indexer.
        
        Args:
            state_dir: Directory for state files (default: ~/.claude-indexer/state)
        """
        if state_dir is None:
            state_dir = Path.home() / ".claude-indexer" / "state"
        
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        self.change_detector = FileChangeDetector()
        self._loaded_states: Dict[str, Dict[str, FileIndexState]] = {}
        
        logger.info(f"Initialized IncrementalIndexer with state dir: {state_dir}")
    
    def _get_state_file_path(self, collection_name: str) -> Path:
        """Get path to state file for collection"""
        # Sanitize collection name for filename
        safe_name = "".join(c for c in collection_name if c.isalnum() or c in "-_")
        return self.state_dir / f"{safe_name}.json"
    
    async def _load_collection_state(
        self, 
        collection_name: str
    ) -> Dict[str, FileIndexState]:
        """
        Load indexing state for a collection.
        
        Args:
            collection_name: Name of collection
            
        Returns:
            Dictionary mapping file paths to their states
        """
        if collection_name in self._loaded_states:
            return self._loaded_states[collection_name]
        
        state_file = self._get_state_file_path(collection_name)
        state_data = {}
        
        if state_file.exists():
            try:
                async with aiofiles.open(state_file, 'r') as f:
                    content = await f.read()
                    raw_data = json.loads(content)
                    
                    # Convert to FileIndexState objects
                    for file_path, state_dict in raw_data.items():
                        try:
                            state_data[file_path] = FileIndexState.from_dict(state_dict)
                        except Exception as e:
                            logger.warning(
                                f"Invalid state entry for {file_path}: {e}"
                            )
                
                logger.info(
                    f"Loaded {len(state_data)} file states for collection {collection_name}"
                )
                
            except Exception as e:
                logger.warning(
                    f"Failed to load state file {state_file}: {e}. "
                    "Starting with empty state."
                )
        
        self._loaded_states[collection_name] = state_data
        return state_data
    
    async def _save_collection_state(
        self, 
        collection_name: str,
        state_data: Dict[str, FileIndexState]
    ) -> None:
        """
        Save indexing state for a collection.
        
        Args:
            collection_name: Name of collection
            state_data: State data to save
        """
        state_file = self._get_state_file_path(collection_name)
        
        try:
            # Convert to serializable format
            serializable_data = {
                file_path: state.to_dict()
                for file_path, state in state_data.items()
            }
            
            # Write atomically using temporary file
            temp_file = state_file.with_suffix('.tmp')
            async with aiofiles.open(temp_file, 'w') as f:
                await f.write(json.dumps(serializable_data, indent=2))
            
            # Atomic rename
            temp_file.rename(state_file)
            
            logger.debug(
                f"Saved {len(state_data)} file states for collection {collection_name}"
            )
            
        except Exception as e:
            logger.error(f"Failed to save state file {state_file}: {e}")
    
    async def get_changed_files(
        self,
        all_files: List[Path],
        collection_name: str
    ) -> List[Path]:
        """
        Identify files that need reindexing.
        
        Args:
            all_files: All discovered files
            collection_name: Target collection name
            
        Returns:
            List of files that need reindexing
        """
        # Load current state
        current_state = await self._load_collection_state(collection_name)
        
        changed_files = []
        
        for file_path in all_files:
            file_key = str(file_path)
            
            # Check if file was previously indexed
            if file_key not in current_state:
                # New file
                changed_files.append(file_path)
                continue
            
            last_state = current_state[file_key]
            
            # Check if file has been modified
            if self.change_detector.is_file_modified(file_path, last_state):
                # File modified
                changed_files.append(file_path)
                continue
            
            # File appears unchanged
            logger.debug(f"Skipping unchanged file: {file_path}")
        
        # Check for deleted files (files in state but not in current file list)
        current_file_set = {str(f) for f in all_files}
        deleted_files = set(current_state.keys()) - current_file_set
        
        if deleted_files:
            logger.info(f"Found {len(deleted_files)} deleted files")
            # Remove deleted files from state
            for deleted_file in deleted_files:
                del current_state[deleted_file]
            
            # Save updated state
            await self._save_collection_state(collection_name, current_state)
        
        logger.info(
            f"Incremental analysis: {len(changed_files)}/{len(all_files)} files need reindexing"
        )
        
        return changed_files
    
    async def cleanup_deleted_files(
        self,
        deleted_files: List[str],
        collection_name: str,
        storage_client: Optional['HybridQdrantClient'] = None
    ) -> Dict[str, int]:
        """
        Clean up entities for deleted files from the collection.
        
        Args:
            deleted_files: List of deleted file paths
            collection_name: Target collection name
            storage_client: HybridQdrantClient for deletion operations
            
        Returns:
            Dictionary with deletion statistics per file
        """
        deletion_stats = {}
        
        if not deleted_files or not storage_client:
            logger.debug("No files to clean up or no storage client provided")
            return deletion_stats
        
        logger.info(f"Cleaning up entities for {len(deleted_files)} deleted files")
        
        for file_path in deleted_files:
            try:
                # Delete entities for this file
                deletion_result = await storage_client.delete_points_by_file_path(
                    collection_name, file_path
                )
                
                if deletion_result.success:
                    deleted_count = deletion_result.affected_count
                    deletion_stats[file_path] = deleted_count
                    
                    logger.info(
                        f"Deleted {deleted_count} entities for file: {file_path}"
                    )
                else:
                    deletion_stats[file_path] = 0
                    logger.warning(
                        f"Failed to delete entities for file {file_path}: "
                        f"{deletion_result.error}"
                    )
                
            except Exception as e:
                deletion_stats[file_path] = 0
                logger.error(f"Error cleaning up file {file_path}: {e}")
        
        total_deleted = sum(deletion_stats.values())
        logger.info(
            f"Cleanup completed: {total_deleted} entities deleted "
            f"from {len([f for f, c in deletion_stats.items() if c > 0])} files"
        )
        
        return deletion_stats
    
    async def synchronize_collection(
        self,
        current_files: List[Path],
        collection_name: str,
        storage_client: Optional['HybridQdrantClient'] = None,
        force_full_sync: bool = False
    ) -> Dict[str, Any]:
        """
        Synchronize Qdrant collection with current file state.
        
        This method ensures collection consistency by:
        1. Identifying orphaned entities (from deleted files)
        2. Detecting stale entities (from modified files)  
        3. Cleaning up inconsistent data
        
        Args:
            current_files: List of current files in the project
            collection_name: Target collection name
            storage_client: HybridQdrantClient for operations
            force_full_sync: Force complete resync ignoring incremental state
            
        Returns:
            Dictionary with synchronization statistics
        """
        sync_stats = {
            "total_files_checked": 0,
            "orphaned_entities_found": 0,
            "orphaned_entities_cleaned": 0,
            "stale_entities_found": 0,
            "stale_entities_cleaned": 0,
            "files_requiring_reindex": [],
            "sync_duration_seconds": 0.0,
            "errors": []
        }
        
        if not storage_client:
            sync_stats["errors"].append("No storage client provided for synchronization")
            return sync_stats
        
        start_time = time.time()
        logger.info(f"Starting collection synchronization for {collection_name}")
        
        try:
            # Load current incremental state
            current_state = await self._load_collection_state(collection_name)
            current_file_set = {str(f) for f in current_files}
            sync_stats["total_files_checked"] = len(current_files)
            
            # Phase 1: Find and clean orphaned entities (from deleted files)
            orphan_stats = await self._find_and_clean_orphaned_entities(
                current_file_set, current_state, collection_name, storage_client
            )
            sync_stats.update(orphan_stats)
            
            # Phase 2: Detect stale entities (from modified files)
            staleness_stats = await self._detect_and_mark_stale_entities(
                current_files, current_state, collection_name, storage_client, force_full_sync
            )
            sync_stats.update(staleness_stats)
            
            # Phase 3: Update incremental state to reflect cleanup
            if orphan_stats.get("files_cleaned", []):
                await self._save_collection_state(collection_name, current_state)
            
        except Exception as e:
            error_msg = f"Collection synchronization failed: {e}"
            sync_stats["errors"].append(error_msg)
            logger.error(error_msg, exc_info=True)
        
        sync_stats["sync_duration_seconds"] = time.time() - start_time
        
        logger.info(
            f"Collection synchronization completed: "
            f"{sync_stats['orphaned_entities_cleaned']} orphaned entities cleaned, "
            f"{sync_stats['stale_entities_found']} stale entities detected, "
            f"{len(sync_stats['files_requiring_reindex'])} files need reindexing "
            f"in {sync_stats['sync_duration_seconds']:.2f}s"
        )
        
        return sync_stats
    
    async def _find_and_clean_orphaned_entities(
        self,
        current_file_set: Set[str],
        current_state: Dict[str, FileIndexState],
        collection_name: str,
        storage_client: 'HybridQdrantClient'
    ) -> Dict[str, Any]:
        """Find and clean entities from files that no longer exist"""
        stats = {
            "orphaned_entities_found": 0,
            "orphaned_entities_cleaned": 0,
            "files_cleaned": []
        }
        
        # Find files in state but not in current file set (deleted files)
        deleted_files = set(current_state.keys()) - current_file_set
        
        if not deleted_files:
            logger.debug("No orphaned entities found")
            return stats
        
        logger.info(f"Found {len(deleted_files)} deleted files with potential orphaned entities")
        
        # Count and clean orphaned entities
        for deleted_file in deleted_files:
            try:
                # Count entities before deletion
                entity_count = await storage_client.count_points_by_filter(
                    collection_name, {"file_path": deleted_file}
                )
                
                if entity_count > 0:
                    stats["orphaned_entities_found"] += entity_count
                    
                    # Delete orphaned entities
                    deletion_result = await storage_client.delete_points_by_file_path(
                        collection_name, deleted_file
                    )
                    
                    if deletion_result.success:
                        cleaned_count = deletion_result.affected_count
                        stats["orphaned_entities_cleaned"] += cleaned_count
                        stats["files_cleaned"].append(deleted_file)
                        
                        logger.info(f"Cleaned {cleaned_count} orphaned entities from {deleted_file}")
                    else:
                        logger.warning(f"Failed to clean orphaned entities from {deleted_file}")
                
                # Remove from incremental state
                if deleted_file in current_state:
                    del current_state[deleted_file]
                
            except Exception as e:
                logger.error(f"Error cleaning orphaned entities from {deleted_file}: {e}")
        
        return stats
    
    async def _detect_and_mark_stale_entities(
        self,
        current_files: List[Path],
        current_state: Dict[str, FileIndexState],
        collection_name: str,
        storage_client: 'HybridQdrantClient',
        force_full_sync: bool = False
    ) -> Dict[str, Any]:
        """Detect entities that may be stale due to file modifications"""
        stats = {
            "stale_entities_found": 0,
            "files_requiring_reindex": []
        }
        
        for file_path in current_files:
            file_key = str(file_path)
            
            try:
                # Check if file needs reindexing (modified or not in state)
                needs_reindex = False
                
                if file_key not in current_state:
                    # New file
                    needs_reindex = True
                    logger.debug(f"New file detected: {file_path}")
                elif force_full_sync:
                    # Force reindex
                    needs_reindex = True
                    logger.debug(f"Force reindex: {file_path}")
                elif self.change_detector.is_file_modified(file_path, current_state[file_key]):
                    # Modified file
                    needs_reindex = True
                    logger.debug(f"Modified file detected: {file_path}")
                
                if needs_reindex:
                    # Count existing entities for this file
                    existing_count = await storage_client.count_points_by_filter(
                        collection_name, {"file_path": file_key}
                    )
                    
                    if existing_count > 0:
                        stats["stale_entities_found"] += existing_count
                        logger.debug(f"Found {existing_count} potentially stale entities in {file_path}")
                    
                    stats["files_requiring_reindex"].append(str(file_path))
                
            except Exception as e:
                logger.error(f"Error checking staleness for {file_path}: {e}")
        
        return stats
    
    async def get_file_entity_count(
        self,
        file_path: str,
        collection_name: str,
        storage_client: Optional['HybridQdrantClient'] = None
    ) -> int:
        """
        Get current entity count for a file from the collection.
        
        Args:
            file_path: File path to check
            collection_name: Collection to search in
            storage_client: HybridQdrantClient for count operations
            
        Returns:
            Number of entities found for the file
        """
        if not storage_client:
            logger.warning("No storage client provided for entity count")
            return 0
        
        try:
            count = await storage_client.count_points_by_filter(
                collection_name, {"file_path": file_path}
            )
            logger.debug(f"File {file_path} has {count} entities in {collection_name}")
            return count
            
        except Exception as e:
            logger.error(f"Error counting entities for file {file_path}: {e}")
            return 0
    
    async def update_file_state(
        self,
        file_path: Path,
        collection_name: str,
        entity_count: int = 0,
        relation_count: int = 0,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> None:
        """
        Update indexing state for a file after processing.
        
        Args:
            file_path: Path to file
            collection_name: Collection name
            entity_count: Number of entities extracted
            relation_count: Number of relations extracted  
            success: Whether indexing was successful
            error_message: Error message if failed
        """
        # Load current state
        current_state = await self._load_collection_state(collection_name)
        
        try:
            # Get file stats
            stat = file_path.stat()
            file_hash = await self.change_detector.compute_file_hash(file_path)
            
            # Create/update state entry
            file_state = FileIndexState(
                file_path=str(file_path),
                file_hash=file_hash,
                file_size=stat.st_size,
                last_modified=stat.st_mtime,
                last_indexed=datetime.now(),
                entity_count=entity_count,
                relation_count=relation_count,
                collection_name=collection_name,
                indexing_success=success,
                error_message=error_message
            )
            
            # Update state
            current_state[str(file_path)] = file_state
            self._loaded_states[collection_name] = current_state
            
            # Save state (async in background to avoid blocking)
            asyncio.create_task(
                self._save_collection_state(collection_name, current_state)
            )
            
        except Exception as e:
            logger.error(f"Failed to update state for {file_path}: {e}")
    
    async def get_collection_stats(
        self, 
        collection_name: str
    ) -> Dict[str, Any]:
        """
        Get statistics for a collection.
        
        Args:
            collection_name: Collection name
            
        Returns:
            Statistics dictionary
        """
        current_state = await self._load_collection_state(collection_name)
        
        if not current_state:
            return {
                "total_files": 0,
                "successful_files": 0,
                "failed_files": 0,
                "total_entities": 0,
                "total_relations": 0,
                "last_update": None
            }
        
        successful_files = sum(
            1 for state in current_state.values() 
            if state.indexing_success
        )
        failed_files = len(current_state) - successful_files
        total_entities = sum(state.entity_count for state in current_state.values())
        total_relations = sum(state.relation_count for state in current_state.values())
        
        # Find most recent update
        last_update = max(
            (state.last_indexed for state in current_state.values()),
            default=None
        )
        
        return {
            "total_files": len(current_state),
            "successful_files": successful_files,
            "failed_files": failed_files,
            "total_entities": total_entities,
            "total_relations": total_relations,
            "last_update": last_update,
            "success_rate": successful_files / len(current_state) if current_state else 0.0
        }
    
    async def reset_state(self, collection_name: str) -> None:
        """
        Reset indexing state for a collection.
        
        Args:
            collection_name: Collection to reset
        """
        state_file = self._get_state_file_path(collection_name)
        
        try:
            if state_file.exists():
                state_file.unlink()
            
            # Clear loaded state
            if collection_name in self._loaded_states:
                del self._loaded_states[collection_name]
            
            logger.info(f"Reset indexing state for collection {collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to reset state for {collection_name}: {e}")
    
    def clear_caches(self) -> None:
        """Clear all internal caches"""
        self.change_detector.clear_cache()
        self._loaded_states.clear()
        logger.debug("Cleared incremental indexer caches")