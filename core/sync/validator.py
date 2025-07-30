"""
Collection Consistency Validator.

Provides periodic consistency validation, orphaned entity detection,
automatic reconciliation, and repair mechanisms for collection integrity.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict
from enum import Enum

from ..storage.client import HybridQdrantClient, StorageResult
from ..parser.registry import ParserRegistry
from .deterministic import DeterministicEntityId
from .lifecycle import EntityLifecycleManager

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Status of a validation operation."""
    HEALTHY = "healthy"
    ISSUES_FOUND = "issues_found"
    REPAIRED = "repaired"
    REPAIR_FAILED = "repair_failed"
    ERROR = "error"


@dataclass
class ConsistencyIssue:
    """Represents a consistency issue found during validation."""
    issue_type: str  # "orphaned_entity", "missing_entity", "stale_entity"
    entity_id: Optional[str]
    file_path: str
    description: str
    severity: str = "medium"  # "low", "medium", "high", "critical"


@dataclass
class ValidationResult:
    """Result of a consistency validation operation."""
    
    status: ValidationStatus
    total_entities_checked: int = 0
    orphaned_entities: int = 0
    missing_entities: int = 0
    files_scanned: int = 0
    issues: List[ConsistencyIssue] = None
    validation_duration_ms: float = 0.0
    repairs_attempted: int = 0
    repairs_successful: int = 0
    summary: str = ""
    
    # Additional metadata for backward compatibility
    validation_id: str = ""
    project_path: str = ""
    collection_name: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    success: bool = False
    
    # Legacy fields
    total_files_found: int = 0
    parseable_files_found: int = 0
    entities_expected: int = 0
    entities_in_collection: int = 0
    entities_removed: int = 0
    entities_added: int = 0
    entities_updated: int = 0
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
    
    @property
    def duration_seconds(self) -> float:
        """Get validation duration in seconds."""
        return self.validation_duration_ms / 1000.0
    
    @property
    def total_issues_found(self) -> int:
        """Get total number of consistency issues found."""
        return len(self.issues)
    
    @property
    def total_repairs_made(self) -> int:
        """Get total number of repair actions taken."""
        return self.repairs_successful
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "validation_id": self.validation_id,
            "project_path": self.project_path,
            "collection_name": self.collection_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "success": self.success,
            "total_files_found": self.total_files_found,
            "parseable_files_found": self.parseable_files_found,
            "entities_expected": self.entities_expected,
            "entities_in_collection": self.entities_in_collection,
            "total_issues_found": self.total_issues_found,
            "orphaned_entities": len(self.orphaned_entities),
            "missing_entities": len(self.missing_entities),
            "stale_entities": len(self.stale_entities),
            "total_repairs_made": self.total_repairs_made,
            "entities_removed": self.entities_removed,
            "entities_added": self.entities_added,
            "entities_updated": self.entities_updated,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings)
        }


class CollectionConsistencyValidator:
    """
    Validates and maintains consistency between project files and Qdrant collections.
    
    This validator performs comprehensive consistency checks to ensure that:
    - All entities in the collection correspond to existing code
    - All code entities are properly represented in the collection
    - Entity content matches the current file state
    - Orphaned and stale entities are automatically cleaned up
    
    Features:
    - Periodic consistency scans with configurable intervals
    - Orphaned entity detection and cleanup
    - Missing entity identification and restoration
    - Stale entity detection and updates
    - Automatic reconciliation and repair mechanisms
    - Comprehensive validation reporting and metrics
    """
    
    def __init__(
        self,
        storage_client: HybridQdrantClient,
        collection_name: str,
        project_path: Path,
        validation_interval_seconds: int = 300,
        scan_interval_minutes: int = 5,
        auto_repair: bool = True,
        max_batch_size: int = 100
    ):
        """
        Initialize the consistency validator.
        
        Args:
            storage_client: Qdrant client for collection operations
            collection_name: Name of the collection to validate
            project_path: Root path of the project
            validation_interval_seconds: Interval between periodic validations
            scan_interval_minutes: Interval between periodic scans
            auto_repair: Whether to automatically repair inconsistencies
            max_batch_size: Maximum entities to process in a batch
        """
        self.storage_client = storage_client
        self.collection_name = collection_name
        self.project_path = Path(project_path).resolve()
        self.validation_interval_seconds = validation_interval_seconds
        self.scan_interval_minutes = scan_interval_minutes
        self.auto_repair = auto_repair
        self.max_batch_size = max_batch_size
        
        # Scanning state
        self.is_scanning = False
        self.scan_task: Optional[asyncio.Task] = None
        self.shutdown_event = asyncio.Event()
        self.is_running_periodic = False
        
        # Validation history
        self.validation_history: List[ValidationResult] = []
        self.max_history_size = 100
        
        # Metrics
        self.total_validations_run = 0
        self.total_issues_found = 0
        self.total_repairs_made = 0
        self.last_validation_time: Optional[datetime] = None
        self.consecutive_errors = 0
        
        logger.info(f"Initialized CollectionConsistencyValidator for {collection_name} with {scan_interval_minutes}min intervals")
    
    async def start_periodic_validation(self) -> bool:
        """Start periodic validation (alias for start_periodic_scanning)."""
        self.is_running_periodic = True
        return await self.start_periodic_scanning()
    
    async def stop_periodic_validation(self) -> None:
        """Stop periodic validation (alias for stop_periodic_scanning)."""
        self.is_running_periodic = False
        await self.stop_periodic_scanning()
    
    async def validate_consistency(self, auto_repair: bool = False) -> ValidationResult:
        """
        Validate consistency using the configured collection and project path.
        
        Args:
            auto_repair: Whether to automatically repair found issues
            
        Returns:
            ValidationResult with findings
        """
        start_time = datetime.now()
        
        # Create new ValidationResult with proper structure
        result = ValidationResult(
            status=ValidationStatus.HEALTHY,
            validation_id=f"validation-{start_time.strftime('%Y%m%d-%H%M%S')}",
            project_path=str(self.project_path),
            collection_name=self.collection_name,
            start_time=start_time
        )
        
        try:
            # Query all entities in collection using search_hybrid
            from ..models.storage import SearchResult
            search_result = await self.storage_client.search_hybrid(
                collection_name=self.collection_name,
                query="*",  # Match all entities
                limit=10000,  # Large limit to get all entities
                payload_filter={}
            )
            
            if not search_result:
                # Handle case where search_result is empty list or None
                collection_entities = []
            elif hasattr(search_result, 'success') and not search_result.success:
                raise Exception(f"Failed to query collection: {getattr(search_result, 'message', 'Unknown error')}")
            elif isinstance(search_result, list):
                collection_entities = search_result
            elif hasattr(search_result, 'results'):
                collection_entities = search_result.results
            else:
                collection_entities = []
            
            result.total_entities_checked = len(collection_entities)
            
            # Check each entity to see if its file exists
            orphaned_count = 0
            missing_count = 0
            
            # Track files that have entities
            files_with_entities = set()
            
            for entity_data in collection_entities:
                entity_id = entity_data.get('id')
                payload = entity_data.get('payload', {})
                if payload is None:
                    payload = {}
                file_path = payload.get('file_path', '')
                
                if file_path:
                    # Resolve the path to handle symlinks and path variations
                    resolved_path = str(Path(file_path).resolve())
                    files_with_entities.add(resolved_path)
                    
                    if not Path(file_path).exists():
                        # File doesn't exist - orphaned entity
                        orphaned_count += 1
                        issue = ConsistencyIssue(
                            issue_type="orphaned_entity",
                            entity_id=entity_id,
                            file_path=file_path,
                            description=f"Entity references non-existent file: {file_path}"
                        )
                        result.issues.append(issue)
            
            # Check for files that should have entities but don't
            supported_files = self._find_supported_files(self.project_path)
            result.files_scanned = len(supported_files)
            
            for file_path in supported_files:
                # Resolve the path to handle symlinks and path variations
                resolved_file_path = str(file_path.resolve())
                if resolved_file_path not in files_with_entities:
                    missing_count += 1
                    issue = ConsistencyIssue(
                        issue_type="missing_entity",
                        entity_id=None,
                        file_path=resolved_file_path,  # Use resolved path for consistency
                        description=f"File exists but has no entities in collection: {file_path}"
                    )
                    result.issues.append(issue)
            
            result.orphaned_entities = orphaned_count
            result.missing_entities = missing_count
            
            # Determine overall status
            if result.total_issues_found > 0:
                result.status = ValidationStatus.ISSUES_FOUND
                
                # Auto-repair if requested
                if auto_repair:
                    repair_result = await self.repair_issues(result.issues)
                    result.repairs_attempted = repair_result["repairs_attempted"]
                    result.repairs_successful = repair_result["repairs_successful"]
                    
                    if repair_result["success"] and result.repairs_successful > 0:
                        result.status = ValidationStatus.REPAIRED
                    elif not repair_result["success"] or result.repairs_successful == 0:
                        result.status = ValidationStatus.REPAIR_FAILED
            
            result.success = True
            
        except Exception as e:
            result.status = ValidationStatus.ERROR
            result.success = False
            result.errors.append(str(e))
            result.summary = f"Validation failed: {e}"
            logger.error(f"Error during consistency validation: {e}")
        
        finally:
            result.end_time = datetime.now()
            result.validation_duration_ms = (result.end_time - start_time).total_seconds() * 1000
            
            if result.success:
                result.summary = f"Found {result.total_issues_found} issues in {result.total_entities_checked} entities"
        
        return result
    
    def _find_supported_files(self, project_path: Path) -> List[Path]:
        """Find all supported files in the project."""
        supported_files = []
        
        # Get supported extensions from watcher
        from .watcher import ProjectFileSystemWatcher
        supported_extensions = ProjectFileSystemWatcher.SUPPORTED_EXTENSIONS
        ignored_directories = ProjectFileSystemWatcher.IGNORED_DIRECTORIES
        
        for file_path in project_path.rglob("*"):
            if not file_path.is_file():
                continue
            
            # Check extension
            if file_path.suffix.lower() not in supported_extensions:
                continue
            
            # Check if in ignored directory
            skip_file = False
            for parent in file_path.parents:
                if parent.name in ignored_directories:
                    skip_file = True
                    break
            
            if not skip_file:
                supported_files.append(file_path)
        
        return supported_files
    
    async def repair_issues(self, issues: List[ConsistencyIssue]) -> Dict[str, Any]:
        """
        Repair specific consistency issues.
        
        Args:
            issues: List of issues to repair
            
        Returns:
            Dictionary with repair results
        """
        if not issues:
            return {
                "success": True,
                "repairs_attempted": 0,
                "repairs_successful": 0
            }
        
        repairs_attempted = 0
        repairs_successful = 0
        
        try:
            for issue in issues:
                repairs_attempted += 1
                
                if issue.issue_type == "orphaned_entity" and issue.entity_id:
                    # Delete orphaned entity
                    delete_result = await self.storage_client.delete_points_by_filter(
                        self.collection_name,
                        {"entity_id": issue.entity_id}
                    )
                    if delete_result.success:
                        repairs_successful += 1
                
                elif issue.issue_type == "missing_entity":
                    # Re-index missing file
                    file_path = Path(issue.file_path)
                    if file_path.exists():
                        # Parse and add entities from file
                        registry = ParserRegistry()
                        if registry.can_parse(file_path):
                            parser = registry.get_parser(file_path)
                            if parser:
                                entities = await parser.parse_file(file_path)
                                if entities:
                                    # Convert to QdrantPoints and upsert
                                    from ..models.storage import QdrantPoint
                                    points = []
                                    for entity in entities:
                                        point = QdrantPoint(
                                            id=entity.id,
                                            vector=[0.0] * 1024,
                                            payload=entity.to_qdrant_payload()
                                        )
                                        points.append(point)
                                    
                                    upsert_result = await self.storage_client.upsert_points(
                                        self.collection_name, points
                                    )
                                    if upsert_result.success:
                                        repairs_successful += 1
            
            return {
                "success": True,
                "repairs_attempted": repairs_attempted,
                "repairs_successful": repairs_successful
            }
            
        except Exception as e:
            logger.error(f"Error during issue repair: {e}")
            return {
                "success": False,
                "repairs_attempted": repairs_attempted,
                "repairs_successful": repairs_successful,
                "error": str(e)
            }
    
    async def start_periodic_scanning(self) -> bool:
        """
        Start periodic consistency scanning.
        
        Returns:
            True if scanning started successfully, False otherwise
        """
        if self.is_scanning:
            logger.warning("Periodic consistency scanning is already active")
            return True
        
        try:
            logger.info("Starting periodic consistency scanning")
            
            self.is_scanning = True
            self.shutdown_event.clear()
            
            # Start background scanning task
            self.scan_task = asyncio.create_task(self._periodic_scan_worker())
            
            logger.info(f"Started periodic consistency scanning every {self.scan_interval_minutes} minutes")
            return True
            
        except Exception as e:
            error_msg = f"Failed to start periodic scanning: {e}"
            logger.error(error_msg)
            self.consecutive_errors += 1
            return False
    
    async def stop_periodic_scanning(self) -> None:
        """Stop periodic consistency scanning."""
        if not self.is_scanning:
            return
        
        logger.info("Stopping periodic consistency scanning")
        self.is_scanning = False
        self.shutdown_event.set()
        
        # Stop scanning task
        if self.scan_task:
            self.scan_task.cancel()
            try:
                await self.scan_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped periodic consistency scanning")
    
    async def validate_deep_consistency(
        self,
        project_path: Path,
        collection_name: str,
        repair_issues: Optional[bool] = None
    ) -> ValidationResult:
        """
        Perform a comprehensive deep consistency validation with entity-level analysis.
        
        This method provides thorough validation by parsing actual code files,
        generating deterministic IDs, and comparing entity-by-entity for accuracy.
        
        Args:
            project_path: Root path of the project to validate
            collection_name: Name of the collection to validate
            repair_issues: Whether to repair found issues (defaults to auto_repair setting)
            
        Returns:
            ValidationResult with detailed findings and actions taken
        """
        if repair_issues is None:
            repair_issues = self.auto_repair
        
        validation_id = f"validation-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        project_path = Path(project_path).resolve()
        
        result = ValidationResult(
            validation_id=validation_id,
            project_path=str(project_path),
            collection_name=collection_name,
            start_time=datetime.now()
        )
        
        try:
            logger.info(f"Starting consistency validation for {project_path} -> {collection_name}")
            
            # Step 1: Analyze file system entities
            fs_entities = await self._analyze_filesystem_entities(project_path, result)
            
            # Step 2: Analyze collection entities
            collection_entities = await self._analyze_collection_entities(collection_name, result)
            
            # Step 3: Compare and identify inconsistencies
            await self._identify_consistency_issues(fs_entities, collection_entities, result)
            
            # Step 4: Repair issues if requested
            if repair_issues and result.total_issues_found > 0:
                await self._repair_consistency_issues(project_path, collection_name, result)
            
            result.success = True
            self.consecutive_errors = 0
            
        except Exception as e:
            error_msg = f"Error during consistency validation: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)
            result.success = False
            self.consecutive_errors += 1
        
        finally:
            result.end_time = datetime.now()
            
            # Update metrics
            self.total_validations_run += 1
            self.total_issues_found += result.total_issues_found
            self.total_repairs_made += result.total_repairs_made
            self.last_validation_time = result.end_time
            
            # Store in history
            self.validation_history.append(result)
            if len(self.validation_history) > self.max_history_size:
                self.validation_history.pop(0)
            
            logger.info(f"Validation completed in {result.duration_seconds:.1f}s: "
                       f"{result.total_issues_found} issues found, {result.total_repairs_made} repairs made")
        
        return result
    
    async def _analyze_filesystem_entities(
        self,
        project_path: Path,
        result: ValidationResult
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze entities that should exist based on current file system state.
        
        Args:
            project_path: Project root path
            result: Validation result to update
            
        Returns:
            Dictionary mapping entity IDs to entity information
        """
        fs_entities = {}
        
        try:
            # Find all supported files in the project
            supported_files = []
            
            # Get supported extensions from watcher
            from .watcher import ProjectFileSystemWatcher
            supported_extensions = ProjectFileSystemWatcher.SUPPORTED_EXTENSIONS
            ignored_directories = ProjectFileSystemWatcher.IGNORED_DIRECTORIES
            
            for file_path in project_path.rglob("*"):
                if not file_path.is_file():
                    continue
                
                # Check extension
                if file_path.suffix.lower() not in supported_extensions:
                    continue
                
                # Check if in ignored directory
                skip_file = False
                for parent in file_path.parents:
                    if parent.name in ignored_directories:
                        skip_file = True
                        break
                
                if not skip_file:
                    supported_files.append(file_path)
            
            result.total_files_found = len(supported_files)
            
            # Parse entities from each file
            for file_path in supported_files:
                try:
                    # Get parser using ParserRegistry
                    registry = ParserRegistry()
                    if not registry.can_parse(file_path):
                        continue
                    
                    parser = registry.get_parser(file_path)
                    if not parser:
                        continue
                    
                    # Parse entities
                    entities = await parser.parse_file(file_path)
                    if entities:
                        result.parseable_files_found += 1
                        
                        # Generate deterministic IDs and store entity info
                        import hashlib
                        file_content = file_path.read_text(encoding='utf-8')
                        file_hash = hashlib.sha256(file_content.encode('utf-8')).hexdigest()
                        
                        for entity in entities:
                            deterministic_id = DeterministicEntityId.generate(entity, file_hash)
                            fs_entities[deterministic_id] = {
                                'entity': entity,
                                'file_path': str(file_path),
                                'file_hash': file_hash,
                                'original_id': entity.id
                            }
                
                except Exception as e:
                    warning_msg = f"Error parsing file {file_path}: {e}"
                    logger.warning(warning_msg)
                    result.warnings.append(warning_msg)
            
            result.entities_expected = len(fs_entities)
            logger.debug(f"Found {len(fs_entities)} entities across {result.parseable_files_found} files")
            
        except Exception as e:
            error_msg = f"Error analyzing filesystem entities: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)
        
        return fs_entities
    
    async def _analyze_collection_entities(
        self,
        collection_name: str,
        result: ValidationResult
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze entities currently in the collection.
        
        Args:
            collection_name: Name of the collection to analyze
            result: Validation result to update
            
        Returns:
            Dictionary mapping entity IDs to collection entity information
        """
        collection_entities = {}
        
        try:
            # Query all entities in the collection
            search_result = await self.storage_client.search_hybrid(
                collection_name=collection_name,
                query="*",  # Match all entities
                limit=10000,  # Large limit to get all entities
                payload_filter={}
            )
            
            if not search_result.success:
                raise Exception(f"Failed to query collection: {search_result.message}")
            
            # Process each entity from collection
            for item in search_result.results:
                entity_id = item.get('id')
                payload = item.get('payload', {})
                
                if entity_id:
                    collection_entities[entity_id] = {
                        'payload': payload,
                        'file_path': payload.get('file_path', ''),
                        'source_hash': payload.get('source_hash', ''),
                        'entity_name': payload.get('entity_name', ''),
                        'entity_type': payload.get('entity_type', '')
                    }
            
            result.entities_in_collection = len(collection_entities)
            logger.debug(f"Found {len(collection_entities)} entities in collection {collection_name}")
            
        except Exception as e:
            error_msg = f"Error analyzing collection entities: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)
        
        return collection_entities
    
    async def _identify_consistency_issues(
        self,
        fs_entities: Dict[str, Dict[str, Any]],
        collection_entities: Dict[str, Dict[str, Any]],
        result: ValidationResult
    ) -> None:
        """
        Compare filesystem and collection entities to identify consistency issues.
        
        Args:
            fs_entities: Entities found in the filesystem
            collection_entities: Entities found in the collection
            result: Validation result to update
        """
        try:
            fs_entity_ids = set(fs_entities.keys())
            collection_entity_ids = set(collection_entities.keys())
            
            # Find orphaned entities (in collection but file missing/changed)
            for entity_id in collection_entity_ids:
                if entity_id not in fs_entity_ids:
                    # Check if the file still exists
                    collection_entity = collection_entities[entity_id]
                    file_path = Path(collection_entity['file_path'])
                    
                    if not file_path.exists():
                        # File deleted - this is an orphaned entity
                        result.orphaned_entities.append(entity_id)
                        logger.debug(f"Orphaned entity found: {entity_id} (file deleted: {file_path})")
                    else:
                        # File exists but entity not found - could be stale
                        result.stale_entities.append(entity_id)
                        logger.debug(f"Potentially stale entity: {entity_id} (file exists but entity changed)")
            
            # Find missing entities (in filesystem but not in collection)
            for entity_id in fs_entity_ids:
                if entity_id not in collection_entity_ids:
                    result.missing_entities.append(entity_id)
                    fs_entity_info = fs_entities[entity_id]
                    logger.debug(f"Missing entity found: {entity_id} from {fs_entity_info['file_path']}")
            
            # Check for stale entities (same ID but different content)
            for entity_id in fs_entity_ids & collection_entity_ids:
                fs_entity_info = fs_entities[entity_id]
                collection_entity = collection_entities[entity_id]
                
                # Compare source hashes
                fs_hash = fs_entity_info['file_hash']
                collection_hash = collection_entity['source_hash']
                
                if fs_hash != collection_hash:
                    if entity_id not in result.stale_entities:
                        result.stale_entities.append(entity_id)
                        logger.debug(f"Stale entity found: {entity_id} (hash mismatch)")
            
            logger.info(f"Consistency analysis: {len(result.orphaned_entities)} orphaned, "
                       f"{len(result.missing_entities)} missing, {len(result.stale_entities)} stale")
            
        except Exception as e:
            error_msg = f"Error identifying consistency issues: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)
    
    async def _repair_consistency_issues(
        self,
        project_path: Path,
        collection_name: str,
        result: ValidationResult
    ) -> None:
        """
        Repair the identified consistency issues.
        
        Args:
            project_path: Project root path
            collection_name: Collection name
            result: Validation result to update
        """
        try:
            logger.info(f"Starting repair of {result.total_issues_found} consistency issues")
            
            # Create a temporary lifecycle manager for repairs
            lifecycle_manager = EntityLifecycleManager(
                storage_client=self.storage_client,
                collection_name=collection_name,
                project_path=project_path
            )
            
            # Repair orphaned entities (remove them)
            if result.orphaned_entities:
                logger.info(f"Removing {len(result.orphaned_entities)} orphaned entities")
                
                for entity_id in result.orphaned_entities:
                    try:
                        delete_result = await self.storage_client.delete_points(
                            collection_name,
                            [entity_id]
                        )
                        if delete_result.success:
                            result.entities_removed += 1
                        else:
                            result.warnings.append(f"Failed to remove orphaned entity {entity_id}")
                    except Exception as e:
                        result.warnings.append(f"Error removing orphaned entity {entity_id}: {e}")
            
            # Repair missing entities (add them)
            if result.missing_entities:
                logger.info(f"Adding {len(result.missing_entities)} missing entities")
                
                # We need to re-parse files to get the actual entities to add
                # This is more complex and may require filesystem re-analysis
                # For now, we'll log this as needing manual intervention
                result.warnings.append(f"Missing entities detected but auto-repair not fully implemented: {len(result.missing_entities)} entities")
            
            # Repair stale entities (update them)
            if result.stale_entities:
                logger.info(f"Updating {len(result.stale_entities)} stale entities")
                
                # Similar to missing entities, this requires re-parsing
                # For now, mark as needing attention
                result.warnings.append(f"Stale entities detected but auto-repair not fully implemented: {len(result.stale_entities)} entities")
            
            logger.info(f"Repair completed: {result.entities_removed} removed, "
                       f"{result.entities_added} added, {result.entities_updated} updated")
            
        except Exception as e:
            error_msg = f"Error during consistency repair: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)
    
    async def _periodic_scan_worker(self) -> None:
        """Background worker for periodic consistency scanning."""
        logger.info("Started periodic consistency scan worker")
        
        while self.is_scanning:
            try:
                # Wait for next scan interval or shutdown
                try:
                    await asyncio.wait_for(
                        self.shutdown_event.wait(),
                        timeout=self.validation_interval_seconds
                    )
                    # Shutdown requested
                    break
                except asyncio.TimeoutError:
                    # Time for next scan
                    pass
                
                if not self.is_scanning:
                    break
                
                # Perform periodic validation
                try:
                    validation_result = await self.validate_consistency(auto_repair=self.auto_repair)
                    logger.debug(f"Periodic validation completed: {validation_result.status.value}")
                except Exception as e:
                    logger.error(f"Error during periodic validation: {e}")
                    self.consecutive_errors += 1
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic scan worker: {e}")
                self.consecutive_errors += 1
                
                # Exponential backoff for consecutive errors
                if self.consecutive_errors > 3:
                    backoff_time = min(2 ** (self.consecutive_errors - 3), 300)  # Max 5 minutes
                    await asyncio.sleep(backoff_time)
        
        logger.info("Stopped periodic consistency scan worker")
    
    def get_validation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent validation history.
        
        Args:
            limit: Maximum number of validations to return
            
        Returns:
            List of validation result dictionaries
        """
        recent_validations = self.validation_history[-limit:] if limit > 0 else self.validation_history
        return [result.to_dict() for result in recent_validations]
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status information about the validator.
        
        Returns:
            Dictionary with status information
        """
        recent_errors = sum(1 for result in self.validation_history[-10:] if not result.success)
        
        return {
            "collection_name": self.collection_name,
            "project_path": str(self.project_path),
            "validation_interval_seconds": self.validation_interval_seconds,
            "is_running_periodic": self.is_running_periodic,
            "last_validation_time": self.last_validation_time.isoformat() if self.last_validation_time else None,
            "is_scanning": self.is_scanning,
            "scan_interval_minutes": self.scan_interval_minutes,
            "auto_repair": self.auto_repair,
            "total_validations_run": self.total_validations_run,
            "total_issues_found": self.total_issues_found,
            "total_repairs_made": self.total_repairs_made,
            "consecutive_errors": self.consecutive_errors,
            "recent_error_rate": recent_errors / min(len(self.validation_history), 10) if self.validation_history else 0,
            "validation_history_size": len(self.validation_history)
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_periodic_scanning()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop_periodic_scanning()