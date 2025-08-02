"""
Delta calculation logic for workspace and collection state comparison.

This module provides delta calculation functionality to determine file changes
between workspace state and indexed collection state for incremental updates.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Set, Any, Optional

logger = logging.getLogger(__name__)


def parse_timestamp_to_unix(timestamp_value: Any) -> Optional[float]:
    """
    Convert various timestamp formats to Unix timestamp for delta calculation.
    
    Args:
        timestamp_value: Can be ISO string, Unix timestamp (float), or None
        
    Returns:
        Unix timestamp as float, or None if parsing fails
    """
    if timestamp_value is None:
        return None
        
    # If already a number, return as-is (backward compatibility)
    if isinstance(timestamp_value, (int, float)):
        return float(timestamp_value)
        
    # Handle ISO format strings
    if isinstance(timestamp_value, str):
        try:
            # Handle ISO format with Z suffix
            if timestamp_value.endswith('Z'):
                timestamp_value = timestamp_value[:-1] + '+00:00'
            dt = datetime.fromisoformat(timestamp_value)
            return dt.timestamp()
        except (ValueError, AttributeError):
            logger.warning(f"Invalid timestamp format: {timestamp_value}")
            return None
    
    logger.warning(f"Unsupported timestamp type: {type(timestamp_value)}")
    return None


@dataclass
class DeltaScanResult:
    """
    Result of a delta scan operation comparing workspace and collection states.
    """
    added_files: Set[str] = field(default_factory=set)
    modified_files: Set[str] = field(default_factory=set)
    deleted_files: Set[str] = field(default_factory=set)
    unchanged_files: Set[str] = field(default_factory=set)
    scan_time: float = 0.0
    total_workspace_files: int = 0
    total_collection_entities: int = 0
    
    @property
    def total_changes(self) -> int:
        """Total number of changes detected"""
        return len(self.added_files) + len(self.modified_files) + len(self.deleted_files)
    
    @property
    def change_ratio(self) -> float:
        """Ratio of changed files to total files"""
        total_files = self.total_workspace_files
        if total_files == 0:
            return 0.0
        return self.total_changes / total_files


class DeltaCalculator:
    """
    Delta calculation engine for workspace and collection state comparison.
    
    Provides efficient algorithms to determine file changes between workspace
    and indexed collection state for incremental indexing operations.
    """
    
    def __init__(self, tolerance_sec: float = 1.0):
        """
        Initialize delta calculator with default tolerance.
        
        Args:
            tolerance_sec: Default time tolerance for modification detection
        """
        self._default_tolerance = tolerance_sec
    
    def calculate_delta(
        self,
        workspace_state: Dict[str, Any],  # WorkspaceState objects
        collection_state: Dict[str, Any],
        tolerance_sec: Optional[float] = None
    ) -> DeltaScanResult:
        """
        Calculate delta between workspace files and indexed entities.
        
        Compares file modification times with entity indexed_at timestamps to determine:
        - Added files: Exist in workspace but not in collection
        - Modified files: Files newer than their indexed entities (considering tolerance)
        - Deleted files: Files in collection but missing from workspace
        - Unchanged files: Files that haven't changed since indexing
        
        Args:
            workspace_state: File metadata from workspace scan
            collection_state: Entity metadata from collection state
            tolerance_sec: Grace period for timestamp comparison (default: class default)
            
        Returns:
            DeltaScanResult with categorized file changes
        """
        if tolerance_sec is None:
            tolerance_sec = self._default_tolerance
            
        start_time = time.perf_counter()
        
        logger.info(f"Starting delta calculation with {len(workspace_state)} workspace files")
        
        # Extract entities by file from collection state
        collection_entities = collection_state.get("entities", {})
        
        # Initialize result sets
        added_files = set()
        modified_files = set()
        deleted_files = set()
        unchanged_files = set()
        
        # Get all file paths from both sources
        workspace_files = set(workspace_state.keys())
        collection_files = set(collection_entities.keys())
        
        logger.debug(f"Workspace files: {len(workspace_files)}, Collection files: {len(collection_files)}")
        
        # Find added files: in workspace but not in collection
        added_files = workspace_files - collection_files
        logger.debug(f"Added files: {len(added_files)}")
        
        # Find deleted files: in collection but not in workspace
        deleted_files = collection_files - workspace_files
        logger.debug(f"Deleted files: {len(deleted_files)}")
        
        # Check files that exist in both for modifications
        common_files = workspace_files & collection_files
        logger.debug(f"Common files to check for modifications: {len(common_files)}")
        
        for file_path in common_files:
            try:
                workspace_file = workspace_state[file_path]
                collection_file_entities = collection_entities[file_path]
                
                # Get the most recent indexed_at timestamp from all entities in this file
                indexed_timestamps = []
                for entity in collection_file_entities:
                    indexed_at = entity.get('indexed_at')
                    if indexed_at:
                        # Convert to Unix timestamp for comparison
                        unix_timestamp = parse_timestamp_to_unix(indexed_at)
                        if unix_timestamp is not None:
                            indexed_timestamps.append(unix_timestamp)
                
                if not indexed_timestamps:
                    # No indexed_at timestamps found, treat as modified to be safe
                    logger.warning(f"No indexed_at timestamps found for {file_path}, treating as modified")
                    modified_files.add(file_path)
                    continue
                
                # Use the most recent indexed timestamp
                latest_indexed_at = max(indexed_timestamps)
                file_mtime = workspace_file.mtime
                
                # Compare timestamps with tolerance
                # File is modified if its mtime is significantly newer than indexed_at
                time_diff = file_mtime - latest_indexed_at
                
                if time_diff > tolerance_sec:
                    modified_files.add(file_path)
                    logger.debug(f"Modified: {file_path} (mtime: {file_mtime}, indexed: {latest_indexed_at}, diff: {time_diff:.3f}s)")
                else:
                    unchanged_files.add(file_path)
                    logger.debug(f"Unchanged: {file_path} (diff: {time_diff:.3f}s within tolerance)")
                    
            except Exception as e:
                logger.warning(f"Error processing {file_path} for modifications: {e}")
                # When in doubt, treat as modified to ensure data consistency
                modified_files.add(file_path)
        
        scan_time = time.perf_counter() - start_time
        
        # Create result
        result = DeltaScanResult(
            added_files=added_files,
            modified_files=modified_files,
            deleted_files=deleted_files,
            unchanged_files=unchanged_files,
            scan_time=scan_time,
            total_workspace_files=len(workspace_files),
            total_collection_entities=collection_state.get("entity_count", 0)
        )
        
        logger.info(
            f"Delta calculation completed in {scan_time:.3f}s: "
            f"{len(added_files)} added, {len(modified_files)} modified, "
            f"{len(deleted_files)} deleted, {len(unchanged_files)} unchanged files"
        )
        
        return result