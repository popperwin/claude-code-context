"""
Entity Lifecycle Manager.

Handles entity creation, modification, deletion, and atomic replacement
operations with proper cascade handling and entity-to-file mapping.
"""

import asyncio
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple
from datetime import datetime
from collections import defaultdict

from ..models.entities import Entity, EntityType
from ..storage.client import HybridQdrantClient, StorageResult
from ..models.storage import QdrantPoint
from ..parser.registry import ParserRegistry
from .deterministic import DeterministicEntityId
from .events import FileSystemEvent, EventType

logger = logging.getLogger(__name__)


class EntityLifecycleManager:
    """
    Manages the complete lifecycle of entities in response to file system events.
    
    This class handles:
    - Entity creation for new files
    - Entity updates for modified files
    - Entity deletion for removed files
    - Atomic replacement operations
    - Entity-to-file mapping maintenance
    - Cascade deletion handling
    
    The manager ensures that all entity operations maintain consistency
    between the file system and the Qdrant collection.
    """
    
    def __init__(
        self,
        storage_client: HybridQdrantClient,
        collection_name: str,
        project_path: Path
    ):
        """
        Initialize the entity lifecycle manager.
        
        Args:
            storage_client: Qdrant client for entity storage
            collection_name: Name of the collection to manage
            project_path: Root path of the project being managed
        """
        self.storage_client = storage_client
        self.collection_name = collection_name
        self.project_path = Path(project_path).resolve()
        
        # Entity-to-file mapping cache for fast lookups
        self._entity_file_map: Dict[str, str] = {}  # entity_id -> file_path
        self._file_entity_map: Dict[str, Set[str]] = defaultdict(set)  # file_path -> entity_ids
        self._mapping_lock = asyncio.Lock()
        
        # Operation tracking
        self._operation_count = 0
        self._last_operation_time: Optional[datetime] = None
        self._error_count = 0
        
        logger.info(f"Initialized EntityLifecycleManager for {self.project_path}")
    
    async def handle_file_creation(self, event: FileSystemEvent) -> Dict[str, Any]:
        """
        Handle file creation by parsing and indexing new entities.
        
        Args:
            event: File creation event
            
        Returns:
            Dictionary with operation results
        """
        start_time = datetime.now()
        file_path = event.file_path
        
        try:
            logger.info(f"Handling file creation: {file_path}")
            
            # Validate file exists
            if not file_path.exists():
                raise FileNotFoundError(f"File does not exist: {file_path}")
            
            # Parse entities from the new file
            entities = await self._parse_file_entities(file_path)
            if not entities:
                logger.info(f"No entities found in new file: {file_path}")
                return {
                    "operation": "file_creation",
                    "file_path": str(file_path),
                    "entities_created": 0,
                    "success": True,
                    "duration_ms": (datetime.now() - start_time).total_seconds() * 1000
                }
            
            # Store entities in collection
            storage_result = await self._store_entities(entities)
            
            # Check if storage was successful
            if storage_result.success:
                # Update entity-file mappings
                await self._update_mappings_for_entities(entities, str(file_path))
                
                self._operation_count += 1
                self._last_operation_time = datetime.now()
                
                duration_ms = (datetime.now() - start_time).total_seconds() * 1000
                logger.info(f"Created {len(entities)} entities for {file_path} in {duration_ms:.1f}ms")
                
                return {
                    "operation": "file_creation",
                    "file_path": str(file_path),
                    "entities_created": len(entities),
                    "entities": [entity.id for entity in entities],
                    "success": True,
                    "duration_ms": duration_ms
                }
            else:
                # Storage failed
                self._error_count += 1
                duration_ms = (datetime.now() - start_time).total_seconds() * 1000
                error_msg = f"Failed to store entities for {file_path}: {getattr(storage_result, 'error', 'Unknown error')}"
                logger.error(error_msg)
                
                return {
                    "operation": "file_creation",
                    "file_path": str(file_path),
                    "entities_created": 0,
                    "success": False,
                    "error": error_msg,
                    "duration_ms": duration_ms
                }
            
        except Exception as e:
            self._error_count += 1
            error_msg = f"Error handling file creation {file_path}: {e}"
            logger.error(error_msg)
            
            return {
                "operation": "file_creation",
                "file_path": str(file_path),
                "entities_created": 0,
                "success": False,
                "error": error_msg,
                "duration_ms": (datetime.now() - start_time).total_seconds() * 1000
            }
    
    async def handle_file_modification(self, event: FileSystemEvent) -> Dict[str, Any]:
        """
        Handle file modification using atomic entity replacement.
        
        Args:
            event: File modification event
            
        Returns:
            Dictionary with operation results
        """
        start_time = datetime.now()
        file_path = event.file_path
        
        try:
            logger.info(f"Handling file modification: {file_path}")
            
            # Validate file exists
            if not file_path.exists():
                # File was deleted, handle as deletion
                return await self.handle_file_deletion(event)
            
            # Get existing entities for this file
            existing_entity_ids = await self._get_entities_for_file(str(file_path))
            
            # Parse new entities from the modified file
            new_entities = await self._parse_file_entities(file_path)
            
            # Perform atomic replacement
            replacement_result = await self.atomic_entity_replacement(
                str(file_path),
                existing_entity_ids,
                new_entities
            )
            
            self._operation_count += 1
            self._last_operation_time = datetime.now()
            
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(f"Modified entities for {file_path}: {replacement_result.get('entities_added', 0)} added, "
                       f"{replacement_result.get('entities_removed', 0)} removed in {duration_ms:.1f}ms")
            
            return {
                "operation": "file_modification",
                "file_path": str(file_path),
                **replacement_result,
                "duration_ms": duration_ms
            }
            
        except Exception as e:
            self._error_count += 1
            error_msg = f"Error handling file modification {file_path}: {e}"
            logger.error(error_msg)
            
            return {
                "operation": "file_modification",
                "file_path": str(file_path),
                "success": False,
                "error": error_msg,
                "duration_ms": (datetime.now() - start_time).total_seconds() * 1000
            }
    
    async def handle_file_deletion(self, event: FileSystemEvent) -> Dict[str, Any]:
        """
        Handle file deletion with cascade deletion of all contained entities.
        
        Args:
            event: File deletion event
            
        Returns:
            Dictionary with operation results
        """
        start_time = datetime.now()
        file_path = event.file_path
        
        try:
            logger.info(f"Handling file deletion: {file_path}")
            
            # Get all entities for this file
            entity_ids = await self._get_entities_for_file(str(file_path))
            
            if not entity_ids:
                logger.info(f"No entities found for deleted file: {file_path}")
                return {
                    "operation": "file_deletion",
                    "file_path": str(file_path),
                    "entities_deleted": 0,
                    "success": True,
                    "duration_ms": (datetime.now() - start_time).total_seconds() * 1000
                }
            
            # Delete all entities for this file
            delete_result = await self.storage_client.delete_points_by_file_path(
                self.collection_name,
                str(file_path)
            )
            
            # Update entity-file mappings
            await self._remove_mappings_for_file(str(file_path))
            
            self._operation_count += 1
            self._last_operation_time = datetime.now()
            
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(f"Deleted {len(entity_ids)} entities for {file_path} in {duration_ms:.1f}ms")
            
            return {
                "operation": "file_deletion",
                "file_path": str(file_path),
                "entities_deleted": len(entity_ids),
                "entities": list(entity_ids),
                "success": delete_result.success,
                "duration_ms": duration_ms
            }
            
        except Exception as e:
            self._error_count += 1
            error_msg = f"Error handling file deletion {file_path}: {e}"
            logger.error(error_msg)
            
            return {
                "operation": "file_deletion",
                "file_path": str(file_path),
                "entities_deleted": 0,
                "success": False,
                "error": error_msg,
                "duration_ms": (datetime.now() - start_time).total_seconds() * 1000
            }
    
    async def atomic_entity_replacement(
        self,
        file_path: str,
        old_entity_ids: Set[str],
        new_entities: List[Entity]
    ) -> Dict[str, Any]:
        """
        Perform atomic replacement of entities for a file.
        
        This operation ensures that the collection always reflects the current
        state of the file by removing old entities and adding new ones in a
        coordinated manner.
        
        Args:
            file_path: Path to the file being updated
            old_entity_ids: Set of existing entity IDs to remove
            new_entities: List of new entities to add
            
        Returns:
            Dictionary with replacement results
        """
        try:
            logger.debug(f"Atomic replacement for {file_path}: {len(old_entity_ids)} old, {len(new_entities)} new")
            
            # Generate deterministic IDs for new entities
            file_content = Path(file_path).read_text(encoding='utf-8')
            file_hash = hashlib.sha256(file_content.encode('utf-8')).hexdigest()
            
            updated_entities = []
            for entity in new_entities:
                updated_entity = DeterministicEntityId.update_entity_with_deterministic_id(
                    entity, file_hash
                )
                updated_entities.append(updated_entity)
            
            new_entity_ids = {entity.id for entity in updated_entities}
            
            # Determine what needs to be added/removed
            entities_to_remove = old_entity_ids - new_entity_ids
            entities_to_add = [e for e in updated_entities if e.id not in old_entity_ids]
            
            results = {
                "file_path": file_path,
                "entities_removed": len(entities_to_remove),
                "entities_added": len(entities_to_add),
                "entities_unchanged": len(old_entity_ids & new_entity_ids),
                "success": True
            }
            
            # Remove old entities that are no longer present
            if entities_to_remove:
                logger.debug(f"Removing {len(entities_to_remove)} entities from {file_path}")
                
                # Delete by entity IDs
                for entity_id in entities_to_remove:
                    delete_result = await self.storage_client.delete_points(
                        self.collection_name,
                        [entity_id]
                    )
                    if not delete_result.success:
                        logger.warning(f"Failed to delete entity {entity_id}: {delete_result.message}")
            
            # Add new entities
            if entities_to_add:
                logger.debug(f"Adding {len(entities_to_add)} entities to {file_path}")
                storage_result = await self._store_entities(entities_to_add)
                if not storage_result.success:
                    results["success"] = False
                    results["error"] = f"Failed to store new entities: {storage_result.message}"
            
            # Update entity-file mappings
            if results["success"]:
                await self._update_mappings_for_entities(updated_entities, file_path)
                
                # Remove mappings for deleted entities
                if entities_to_remove:
                    await self._remove_mappings_for_entities(list(entities_to_remove))
            
            return results
            
        except Exception as e:
            error_msg = f"Error in atomic entity replacement for {file_path}: {e}"
            logger.error(error_msg)
            return {
                "operation": "atomic_replacement",
                "file_path": file_path,
                "entities_removed": 0,
                "entities_added": 0,
                "entities_unchanged": 0,
                "success": False,
                "error": error_msg
            }
    
    async def _parse_file_entities(self, file_path: Path) -> List[Entity]:
        """
        Parse entities from a file using the appropriate parser.
        
        Args:
            file_path: Path to the file to parse
            
        Returns:
            List of extracted entities
        """
        try:
            # Get the appropriate parser for this file using ParserRegistry
            registry = ParserRegistry()
            if not registry.can_parse(file_path):
                logger.debug(f"No parser available for {file_path}")
                return []
            
            parser = registry.get_parser(file_path)
            if not parser:
                logger.debug(f"Failed to get parser for {file_path}")
                return []
            
            # Parse entities from the file
            entities = await parser.parse_file(file_path)
            logger.debug(f"Parsed {len(entities)} entities from {file_path}")
            
            return entities
            
        except Exception as e:
            logger.error(f"Error parsing entities from {file_path}: {e}")
            return []
    
    async def _store_entities(self, entities: List[Entity]) -> StorageResult:
        """
        Store entities in the Qdrant collection.
        
        Args:
            entities: List of entities to store
            
        Returns:
            Storage result
        """
        if not entities:
            return StorageResult.successful_insert(self.collection_name, 0, 0)
        
        try:
            # Convert entities to QdrantPoint format
            qdrant_points = []
            for entity in entities:
                qdrant_point = QdrantPoint(
                    id=entity.id,
                    vector=[0.0] * 1024,  # Placeholder vector, will be computed by storage client
                    payload=entity.to_qdrant_payload()
                )
                qdrant_points.append(qdrant_point)
            
            # Store using upsert_points
            result = await self.storage_client.upsert_points(
                self.collection_name,
                qdrant_points
            )
            
            logger.debug(f"Stored {len(entities)} entities: {result.success}")
            return result
            
        except Exception as e:
            error_msg = f"Error storing entities: {e}"
            logger.error(error_msg)
            return StorageResult.failed_operation(
                "upsert", self.collection_name, error_msg, 0
            )
    
    async def _get_entities_for_file(self, file_path: str) -> Set[str]:
        """
        Get all entity IDs associated with a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Set of entity IDs
        """
        async with self._mapping_lock:
            return self._file_entity_map.get(file_path, set()).copy()
    
    async def _update_mappings_for_entities(self, entities: List[Entity], file_path: str) -> None:
        """
        Update entity-to-file mappings for the given entities.
        
        Args:
            entities: List of entities to update mappings for
            file_path: Path to the file containing the entities
        """
        async with self._mapping_lock:
            # Clear existing mappings for this file
            if file_path in self._file_entity_map:
                old_entity_ids = self._file_entity_map[file_path]
                for entity_id in old_entity_ids:
                    self._entity_file_map.pop(entity_id, None)
            
            # Add new mappings
            entity_ids = set()
            for entity in entities:
                self._entity_file_map[entity.id] = file_path
                entity_ids.add(entity.id)
            
            self._file_entity_map[file_path] = entity_ids
    
    async def _remove_mappings_for_file(self, file_path: str) -> None:
        """
        Remove all entity-to-file mappings for a file.
        
        Args:
            file_path: Path to the file to remove mappings for
        """
        async with self._mapping_lock:
            if file_path in self._file_entity_map:
                entity_ids = self._file_entity_map[file_path]
                
                # Remove entity -> file mappings
                for entity_id in entity_ids:
                    self._entity_file_map.pop(entity_id, None)
                
                # Remove file -> entities mapping
                del self._file_entity_map[file_path]
    
    async def _remove_mappings_for_entities(self, entity_ids: List[str]) -> None:
        """
        Remove mappings for specific entity IDs.
        
        Args:
            entity_ids: List of entity IDs to remove mappings for
        """
        async with self._mapping_lock:
            for entity_id in entity_ids:
                if entity_id in self._entity_file_map:
                    file_path = self._entity_file_map[entity_id]
                    
                    # Remove from entity -> file mapping
                    del self._entity_file_map[entity_id]
                    
                    # Remove from file -> entities mapping
                    if file_path in self._file_entity_map:
                        self._file_entity_map[file_path].discard(entity_id)
                        
                        # Clean up empty file mappings
                        if not self._file_entity_map[file_path]:
                            del self._file_entity_map[file_path]
    
    async def rebuild_entity_mappings(self) -> Dict[str, Any]:
        """
        Rebuild entity-to-file mappings by querying the collection.
        
        This method is useful for recovering mappings after a restart
        or when the cache becomes inconsistent.
        
        Returns:
            Dictionary with rebuild results
        """
        start_time = datetime.now()
        
        try:
            logger.info("Rebuilding entity-to-file mappings from collection")
            
            # Query all entities in the collection
            search_result = await self.storage_client.search_hybrid(
                collection_name=self.collection_name,
                query="*",  # Match all entities
                limit=10000,  # Large limit to get all entities
                payload_filter={}
            )
            
            if not search_result.success:
                raise Exception(f"Failed to query collection: {search_result.message}")
            
            # Rebuild mappings
            async with self._mapping_lock:
                self._entity_file_map.clear()
                self._file_entity_map.clear()
                
                for result in search_result.results:
                    entity_id = result.get('id')
                    file_path = result.get('payload', {}).get('file_path')
                    
                    if entity_id and file_path:
                        self._entity_file_map[entity_id] = file_path
                        self._file_entity_map[file_path].add(entity_id)
            
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            entity_count = len(self._entity_file_map)
            file_count = len(self._file_entity_map)
            
            logger.info(f"Rebuilt mappings for {entity_count} entities across {file_count} files in {duration_ms:.1f}ms")
            
            return {
                "operation": "rebuild_mappings",
                "entities_mapped": entity_count,
                "files_mapped": file_count,
                "success": True,
                "duration_ms": duration_ms
            }
            
        except Exception as e:
            error_msg = f"Error rebuilding entity mappings: {e}"
            logger.error(error_msg)
            
            return {
                "operation": "rebuild_mappings",
                "entities_mapped": 0,
                "files_mapped": 0,
                "success": False,
                "error": error_msg,
                "duration_ms": (datetime.now() - start_time).total_seconds() * 1000
            }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status information about the lifecycle manager.
        
        Returns:
            Dictionary with status information
        """
        return {
            "collection_name": self.collection_name,
            "project_path": str(self.project_path),
            "entities_tracked": len(self._entity_file_map),
            "files_tracked": len(self._file_entity_map),
            "operation_count": self._operation_count,
            "error_count": self._error_count,
            "last_operation_time": self._last_operation_time.isoformat() if self._last_operation_time else None,
            "error_rate": self._error_count / max(self._operation_count, 1)
        }