"""
Entity Change Detection System.

Implements comprehensive entity change detection using DeterministicEntityId for
stable identification, content hashing, and comparison logic for atomic operations.
"""

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

from ..models.entities import Entity, EntityType
from ..storage.client import HybridQdrantClient
from ..sync.deterministic import DeterministicEntityId

logger = logging.getLogger(__name__)


@dataclass
class EntityChangeInfo:
    """Information about entity changes."""
    
    entity_id: str
    change_type: str  # "created", "modified", "deleted", "moved"
    old_entity: Optional[Entity] = None
    new_entity: Optional[Entity] = None
    content_changed: bool = False
    signature_changed: bool = False
    location_changed: bool = False
    metadata_changed: bool = False
    confidence: float = 1.0  # Confidence in change detection
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}
    
    @property
    def has_semantic_changes(self) -> bool:
        """Check if changes affect semantic meaning."""
        return self.content_changed or self.signature_changed
    
    @property
    def has_structural_changes(self) -> bool:
        """Check if changes affect code structure."""
        return self.location_changed or self.change_type in ["created", "deleted"]


@dataclass
class FileEntitySnapshot:
    """Snapshot of entities in a file for comparison."""
    
    file_path: str
    file_hash: str
    entity_count: int
    entities_by_id: Dict[str, Entity]
    entities_by_signature: Dict[str, Entity]
    entity_locations: Dict[str, Tuple[int, int]]  # entity_id -> (start_line, end_line)
    timestamp: datetime
    
    @classmethod
    def from_entities(cls, file_path: str, file_hash: str, entities: List[Entity]) -> 'FileEntitySnapshot':
        """Create snapshot from entity list."""
        entities_by_id = {entity.id: entity for entity in entities}
        entities_by_signature = {}
        entity_locations = {}
        
        for entity in entities:
            if entity.signature:
                entities_by_signature[entity.signature] = entity
            entity_locations[entity.id] = (
                entity.location.start_line, 
                entity.location.end_line
            )
        
        return cls(
            file_path=file_path,
            file_hash=file_hash,
            entity_count=len(entities),
            entities_by_id=entities_by_id,
            entities_by_signature=entities_by_signature,
            entity_locations=entity_locations,
            timestamp=datetime.now()
        )


class EntityChangeDetector:
    """
    Detects changes in entities using deterministic IDs and content analysis.
    
    This class provides comprehensive entity change detection for the sync system,
    using stable identification via DeterministicEntityId and content-based
    comparison to detect semantic and structural changes.
    
    Features:
    - Stable entity identification across file modifications
    - Content hash-based change detection
    - Signature and location change analysis
    - Batch processing for performance
    - Confidence scoring for change detection
    """
    
    def __init__(
        self,
        storage_client: HybridQdrantClient,
        enable_content_hashing: bool = True,
        enable_signature_tracking: bool = True,
        similarity_threshold: float = 0.85
    ):
        """
        Initialize entity change detector.
        
        Args:
            storage_client: Qdrant client for entity storage
            enable_content_hashing: Whether to use content hashes for change detection
            enable_signature_tracking: Whether to track function/method signatures
            similarity_threshold: Threshold for entity similarity (0.0-1.0)
        """
        self.storage_client = storage_client
        self.enable_content_hashing = enable_content_hashing
        self.enable_signature_tracking = enable_signature_tracking
        self.similarity_threshold = similarity_threshold
        
        # Entity snapshots cache for performance
        self._file_snapshots: Dict[str, FileEntitySnapshot] = {}
        self._snapshot_lock = asyncio.Lock()
        
        # Change detection metrics
        self._total_comparisons = 0
        self._changes_detected = 0
        self._false_positives_avoided = 0
        
        logger.info(f"Initialized EntityChangeDetector with content_hashing={enable_content_hashing}, "
                   f"signature_tracking={enable_signature_tracking}")
    
    async def detect_file_changes(
        self,
        file_path: Path,
        new_entities: List[Entity],
        collection_name: str
    ) -> List[EntityChangeInfo]:
        """
        Detect changes in entities for a specific file.
        
        Args:
            file_path: Path to the file being analyzed
            new_entities: New entities parsed from the file
            collection_name: Collection name for existing entity lookup
            
        Returns:
            List of detected entity changes
        """
        try:
            # Generate file hash for deterministic IDs
            file_content = file_path.read_text(encoding='utf-8')
            file_hash = hashlib.sha256(file_content.encode('utf-8')).hexdigest()
            
            # Update entities with deterministic IDs
            updated_entities = []
            for entity in new_entities:
                updated_entity = DeterministicEntityId.update_entity_with_deterministic_id(
                    entity, file_hash
                )
                updated_entities.append(updated_entity)
            
            # Get existing entities for this file
            existing_entities = await self._get_existing_entities_for_file(
                str(file_path), collection_name
            )
            
            # Create snapshots for comparison
            old_snapshot = FileEntitySnapshot.from_entities(
                str(file_path), file_hash, existing_entities
            )
            new_snapshot = FileEntitySnapshot.from_entities(
                str(file_path), file_hash, updated_entities
            )
            
            # Detect changes between snapshots
            changes = await self._compare_snapshots(old_snapshot, new_snapshot)
            
            # Update cached snapshot
            async with self._snapshot_lock:
                self._file_snapshots[str(file_path)] = new_snapshot
            
            # Update metrics
            self._total_comparisons += 1
            if changes:
                self._changes_detected += 1
            
            logger.debug(f"Detected {len(changes)} changes in {file_path}")
            return changes
            
        except Exception as e:
            logger.error(f"Error detecting changes in {file_path}: {e}")
            return []
    
    async def detect_batch_changes(
        self,
        file_entities: Dict[Path, List[Entity]],
        collection_name: str
    ) -> Dict[str, List[EntityChangeInfo]]:
        """
        Detect changes for multiple files in batch for performance.
        
        Args:
            file_entities: Mapping of file paths to their entities
            collection_name: Collection name for existing entity lookup
            
        Returns:
            Mapping of file paths to their detected changes
        """
        changes_by_file = {}
        
        # Process files concurrently for better performance
        tasks = []
        for file_path, entities in file_entities.items():
            task = self.detect_file_changes(file_path, entities, collection_name)
            tasks.append((str(file_path), task))
        
        # Execute batch operations
        for file_path, task in tasks:
            try:
                changes = await task
                changes_by_file[file_path] = changes
            except Exception as e:
                logger.error(f"Error processing {file_path} in batch: {e}")
                changes_by_file[file_path] = []
        
        logger.info(f"Processed batch changes for {len(file_entities)} files")
        return changes_by_file
    
    async def _get_existing_entities_for_file(
        self,
        file_path: str,
        collection_name: str
    ) -> List[Entity]:
        """
        Get existing entities for a file from the collection.
        
        Args:
            file_path: Path to the file
            collection_name: Collection name
            
        Returns:
            List of existing entities
        """
        try:
            # Search for entities in this file using payload filter
            search_result = await self.storage_client.search_payload(
                collection_name=collection_name,
                filter_conditions={
                    "file_path": {"$eq": file_path}
                },
                limit=1000  # Reasonable limit for entities per file
            )
            
            if not search_result.success:
                logger.warning(f"Failed to search existing entities for {file_path}: {search_result.message}")
                return []
            
            # Convert search results to Entity objects
            entities = []
            for result in search_result.results:
                try:
                    payload = result.get('payload', {})
                    
                    # Reconstruct Entity from payload
                    entity = self._entity_from_payload(payload)
                    if entity:
                        entities.append(entity)
                except Exception as e:
                    logger.warning(f"Error reconstructing entity from payload: {e}")
                    continue
            
            logger.debug(f"Found {len(entities)} existing entities in {file_path}")
            return entities
            
        except Exception as e:
            logger.error(f"Error getting existing entities for {file_path}: {e}")
            return []
    
    def _entity_from_payload(self, payload: Dict[str, Any]) -> Optional[Entity]:
        """
        Reconstruct Entity object from Qdrant payload.
        
        Args:
            payload: Qdrant point payload
            
        Returns:
            Reconstructed Entity or None if invalid
        """
        try:
            from ..models.entities import SourceLocation
            
            # Extract required fields
            entity_id = payload.get('id')
            name = payload.get('name')
            qualified_name = payload.get('qualified_name')
            entity_type_str = payload.get('entity_type')
            source_code = payload.get('source_code', '')
            file_path = payload.get('file_path')
            
            if not all([entity_id, name, qualified_name, entity_type_str, file_path]):
                return None
            
            # Convert entity type string to enum
            try:
                entity_type = EntityType(entity_type_str)
            except ValueError:
                logger.warning(f"Unknown entity type: {entity_type_str}")
                return None
            
            # Reconstruct location
            location = SourceLocation(
                file_path=Path(file_path),
                start_line=payload.get('start_line', 0),
                end_line=payload.get('end_line', 0),
                start_column=payload.get('start_column', 0),
                end_column=payload.get('end_column', 0),
                start_byte=payload.get('start_byte', 0),
                end_byte=payload.get('end_byte', 0)
            )
            
            # Create Entity object
            entity = Entity(
                id=entity_id,
                name=name,
                qualified_name=qualified_name,
                entity_type=entity_type,
                location=location,
                source_code=source_code,
                signature=payload.get('signature'),
                docstring=payload.get('docstring'),
                source_hash=payload.get('source_hash', ''),
                metadata=payload.get('metadata', {})
            )
            
            return entity
            
        except Exception as e:
            logger.error(f"Error reconstructing entity from payload: {e}")
            return None
    
    async def _compare_snapshots(
        self,
        old_snapshot: FileEntitySnapshot,
        new_snapshot: FileEntitySnapshot
    ) -> List[EntityChangeInfo]:
        """
        Compare two file snapshots to detect entity changes.
        
        Args:
            old_snapshot: Previous file state
            new_snapshot: Current file state
            
        Returns:
            List of detected changes
        """
        changes = []
        
        old_entity_ids = set(old_snapshot.entities_by_id.keys())
        new_entity_ids = set(new_snapshot.entities_by_id.keys())
        
        # Detect deletions
        deleted_ids = old_entity_ids - new_entity_ids
        for entity_id in deleted_ids:
            old_entity = old_snapshot.entities_by_id[entity_id]
            changes.append(EntityChangeInfo(
                entity_id=entity_id,
                change_type="deleted",
                old_entity=old_entity,
                confidence=1.0,
                details={"reason": "Entity no longer present in file"}
            ))
        
        # Detect creations
        created_ids = new_entity_ids - old_entity_ids
        for entity_id in created_ids:
            new_entity = new_snapshot.entities_by_id[entity_id]
            changes.append(EntityChangeInfo(
                entity_id=entity_id,
                change_type="created",
                new_entity=new_entity,
                confidence=1.0,
                details={"reason": "New entity in file"}
            ))
        
        # Detect modifications in existing entities
        common_ids = old_entity_ids & new_entity_ids
        for entity_id in common_ids:
            old_entity = old_snapshot.entities_by_id[entity_id]
            new_entity = new_snapshot.entities_by_id[entity_id]
            
            change_info = await self._compare_entities(old_entity, new_entity)
            if change_info:
                changes.append(change_info)
        
        # Detect potential moves/renames using signature matching
        if self.enable_signature_tracking:
            move_changes = await self._detect_entity_moves(old_snapshot, new_snapshot)
            changes.extend(move_changes)
        
        return changes
    
    async def _compare_entities(self, old_entity: Entity, new_entity: Entity) -> Optional[EntityChangeInfo]:
        """
        Compare two entities to detect specific changes.
        
        Args:
            old_entity: Previous entity state
            new_entity: Current entity state
            
        Returns:
            EntityChangeInfo if changes detected, None otherwise
        """
        if old_entity.id != new_entity.id:
            logger.warning(f"Comparing entities with different IDs: {old_entity.id} vs {new_entity.id}")
            return None
        
        changes_detected = False
        content_changed = False
        signature_changed = False
        location_changed = False
        metadata_changed = False
        
        details = {}
        
        # Check content changes
        if self.enable_content_hashing:
            if old_entity.source_hash != new_entity.source_hash:
                content_changed = True
                changes_detected = True
                details["source_hash_changed"] = {
                    "old": old_entity.source_hash,
                    "new": new_entity.source_hash
                }
        
        # Check signature changes
        if self.enable_signature_tracking:
            if old_entity.signature != new_entity.signature:
                signature_changed = True
                changes_detected = True
                details["signature_changed"] = {
                    "old": old_entity.signature,
                    "new": new_entity.signature
                }
        
        # Check location changes
        old_loc = old_entity.location
        new_loc = new_entity.location
        if (old_loc.start_line != new_loc.start_line or 
            old_loc.end_line != new_loc.end_line or
            old_loc.start_column != new_loc.start_column or
            old_loc.end_column != new_loc.end_column):
            location_changed = True
            changes_detected = True
            details["location_changed"] = {
                "old": f"{old_loc.start_line}:{old_loc.start_column}-{old_loc.end_line}:{old_loc.end_column}",
                "new": f"{new_loc.start_line}:{new_loc.start_column}-{new_loc.end_line}:{new_loc.end_column}"
            }
        
        # Check metadata changes
        if old_entity.metadata != new_entity.metadata:
            metadata_changed = True
            changes_detected = True
            details["metadata_changed"] = {
                "old_keys": set(old_entity.metadata.keys()),
                "new_keys": set(new_entity.metadata.keys())
            }
        
        if not changes_detected:
            return None
        
        return EntityChangeInfo(
            entity_id=old_entity.id,
            change_type="modified",
            old_entity=old_entity,
            new_entity=new_entity,
            content_changed=content_changed,
            signature_changed=signature_changed,
            location_changed=location_changed,
            metadata_changed=metadata_changed,
            confidence=1.0,
            details=details
        )
    
    async def _detect_entity_moves(
        self,
        old_snapshot: FileEntitySnapshot,
        new_snapshot: FileEntitySnapshot
    ) -> List[EntityChangeInfo]:
        """
        Detect entity moves/renames using signature matching.
        
        Args:
            old_snapshot: Previous file state
            new_snapshot: Current file state
            
        Returns:
            List of move/rename changes detected
        """
        if not self.enable_signature_tracking:
            return []
        
        move_changes = []
        
        # Find entities that might have been moved/renamed
        old_signatures = set(old_snapshot.entities_by_signature.keys())
        new_signatures = set(new_snapshot.entities_by_signature.keys())
        
        # Look for matching signatures with different IDs
        for signature in old_signatures & new_signatures:
            old_entity = old_snapshot.entities_by_signature[signature]
            new_entity = new_snapshot.entities_by_signature[signature]
            
            if old_entity.id != new_entity.id:
                # Potential move/rename
                move_changes.append(EntityChangeInfo(
                    entity_id=new_entity.id,
                    change_type="moved",
                    old_entity=old_entity,
                    new_entity=new_entity,
                    confidence=0.8,  # Lower confidence for move detection
                    details={
                        "reason": "Same signature, different ID",
                        "old_id": old_entity.id,
                        "new_id": new_entity.id,
                        "signature": signature
                    }
                ))
        
        return move_changes
    
    def calculate_entity_similarity(self, entity1: Entity, entity2: Entity) -> float:
        """
        Calculate similarity score between two entities.
        
        Args:
            entity1: First entity
            entity2: Second entity
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if entity1.entity_type != entity2.entity_type:
            return 0.0
        
        # Weight different comparison factors
        weights = {
            'name': 0.3,
            'signature': 0.4,
            'content': 0.2,
            'location': 0.1
        }
        
        score = 0.0
        
        # Name similarity
        if entity1.name == entity2.name:
            score += weights['name']
        elif entity1.name.lower() == entity2.name.lower():
            score += weights['name'] * 0.8
        
        # Signature similarity
        if entity1.signature and entity2.signature:
            if entity1.signature == entity2.signature:
                score += weights['signature']
            else:
                # Could implement more sophisticated signature comparison
                score += weights['signature'] * 0.5
        elif not entity1.signature and not entity2.signature:
            score += weights['signature']
        
        # Content similarity (using hash)
        if entity1.source_hash == entity2.source_hash:
            score += weights['content']
        
        # Location similarity (rough approximation)
        loc1, loc2 = entity1.location, entity2.location
        line_diff = abs(loc1.start_line - loc2.start_line)
        if line_diff == 0:
            score += weights['location']
        elif line_diff <= 5:
            score += weights['location'] * 0.5
        
        return min(score, 1.0)
    
    def get_cache_status(self) -> Dict[str, Any]:
        """
        Get current cache status and metrics.
        
        Returns:
            Dictionary with cache and performance information
        """
        return {
            "cached_file_snapshots": len(self._file_snapshots),
            "total_comparisons": self._total_comparisons,
            "changes_detected": self._changes_detected,
            "false_positives_avoided": self._false_positives_avoided,
            "change_detection_rate": (
                self._changes_detected / max(self._total_comparisons, 1)
            ),
            "deterministic_id_cache_size": DeterministicEntityId.get_cache_size(),
            "configuration": {
                "content_hashing_enabled": self.enable_content_hashing,
                "signature_tracking_enabled": self.enable_signature_tracking,
                "similarity_threshold": self.similarity_threshold
            }
        }
    
    async def clear_cache(self) -> None:
        """Clear all cached data."""
        async with self._snapshot_lock:
            self._file_snapshots.clear()
        
        # Clear deterministic ID cache as well
        DeterministicEntityId.clear_cache()
        
        logger.info("Cleared EntityChangeDetector cache")
    
    async def warmup_cache(self, file_paths: List[Path], collection_name: str) -> None:
        """
        Warm up the cache with existing entities for better performance.
        
        Args:
            file_paths: List of file paths to cache
            collection_name: Collection name for entity lookup
        """
        logger.info(f"Warming up EntityChangeDetector cache for {len(file_paths)} files")
        
        for file_path in file_paths:
            try:
                existing_entities = await self._get_existing_entities_for_file(
                    str(file_path), collection_name
                )
                
                if existing_entities:
                    file_content = file_path.read_text(encoding='utf-8')
                    file_hash = hashlib.sha256(file_content.encode('utf-8')).hexdigest()
                    
                    snapshot = FileEntitySnapshot.from_entities(
                        str(file_path), file_hash, existing_entities
                    )
                    
                    async with self._snapshot_lock:
                        self._file_snapshots[str(file_path)] = snapshot
                        
            except Exception as e:
                logger.warning(f"Error warming up cache for {file_path}: {e}")
                continue
        
        logger.info(f"Cache warmup completed: {len(self._file_snapshots)} files cached")