"""
Workspace filesystem scanning utilities for delta-scan operations.

This module provides fast filesystem traversal using os.scandir for optimal performance
in detecting file changes and collecting workspace metadata.
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class WorkspaceState:
    """
    Represents the current state of files in a workspace.
    
    This class captures file modification times and metadata for
    delta-scan comparison operations.
    """
    file_path: str
    mtime: float  # Modification timestamp
    size: int     # File size in bytes
    is_parseable: bool = True
    
    @classmethod
    def from_file_path(cls, file_path: Path) -> Optional['WorkspaceState']:
        """
        Create WorkspaceState from a file path.
        
        Args:
            file_path: Path to the file
            
        Returns:
            WorkspaceState instance or None if file doesn't exist
        """
        try:
            if not file_path.exists():
                return None
                
            stat = file_path.stat()
            return cls(
                file_path=str(file_path),
                mtime=stat.st_mtime,
                size=stat.st_size,
                is_parseable=True  # Will be determined by parser registry
            )
        except (OSError, IOError) as e:
            logger.debug(f"Cannot stat file {file_path}: {e}")
            return None


class WorkspaceScanner:
    """
    High-performance filesystem scanner for workspace state collection.
    
    Uses os.scandir() for optimal performance on large directory trees.
    """
    
    async def fast_scan_workspace(
        self,
        project_path: Path,
        tolerance_sec: float = 1.0,
        parser_registry = None
    ) -> Dict[str, WorkspaceState]:
        """
        Fast filesystem traversal using os.scandir for optimal performance.
        
        This function performs a high-speed directory traversal using os.scandir(),
        which is significantly faster than Path.rglob() for large directory trees.
        It captures file modification times and metadata for delta-scan comparison.
        
        Args:
            project_path: Root directory to scan
            tolerance_sec: Time tolerance for modification detection (default: 1.0s)
            parser_registry: Optional parser registry for file type detection
            
        Returns:
            Dictionary mapping file paths to WorkspaceState objects
        """
        start_time = time.perf_counter()
        workspace_state = {}
        
        def scan_directory_recursive(dir_path: Path) -> None:
            """Recursive directory scanning using os.scandir for performance"""
            try:
                with os.scandir(str(dir_path)) as entries:
                    for entry in entries:
                        try:
                            # Skip hidden files and common ignore patterns
                            if entry.name.startswith('.'):
                                continue
                            
                            # Skip common build/cache directories for performance
                            if entry.name in {
                                'node_modules', '__pycache__', '.git', '.svn', '.hg',
                                'build', 'dist', '.cache', '.pytest_cache', '.mypy_cache',
                                'venv', '.venv', 'env', '.env'
                            }:
                                continue
                            
                            entry_path = Path(entry.path)
                            
                            if entry.is_dir(follow_symlinks=False):
                                # Recursively scan subdirectories
                                scan_directory_recursive(entry_path)
                                
                            elif entry.is_file(follow_symlinks=False):
                                # Check if file is parseable using the registry (if provided)
                                if parser_registry and not parser_registry.can_parse(entry_path):
                                    continue
                                
                                # Get file stats efficiently using DirEntry
                                stat_result = entry.stat(follow_symlinks=False)
                                
                                # Create workspace state
                                workspace_state[str(entry_path)] = WorkspaceState(
                                    file_path=str(entry_path),
                                    mtime=stat_result.st_mtime,
                                    size=stat_result.st_size,
                                    is_parseable=True
                                )
                                
                        except (OSError, IOError) as e:
                            logger.debug(f"Skipping entry {entry.name}: {e}")
                            continue
                            
            except (OSError, IOError, PermissionError) as e:
                logger.warning(f"Cannot scan directory {dir_path}: {e}")
        
        # Perform the recursive scan
        try:
            scan_directory_recursive(project_path)
        except Exception as e:
            logger.error(f"Error during workspace scan: {e}")
            raise
        
        scan_time = time.perf_counter() - start_time
        
        logger.info(
            f"Fast workspace scan completed: {len(workspace_state)} parseable files "
            f"found in {scan_time:.3f}s ({len(workspace_state)/scan_time:.1f} files/sec)"
        )
        
        return workspace_state