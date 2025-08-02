"""
Collection state management for Qdrant operations in delta-scan workflows.

This module provides efficient collection state retrieval using Qdrant scroll operations
for delta comparison and change detection.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable

logger = logging.getLogger(__name__)


class CollectionStateFetcher:
    """
    Efficient collection state retrieval using Qdrant scroll operations.
    
    Provides methods to scan Qdrant collections and extract entity metadata
    for delta-scan comparison operations.
    """
    
    def __init__(self, storage_client):
        """
        Initialize with storage client dependency.
        
        Args:
            storage_client: HybridQdrantClient instance for Qdrant operations
        """
        self._client = storage_client
    
    async def get_collection_state(
        self,
        collection_name: str,
        chunk_size: int = 1000
    ) -> Dict[str, Any]:
        """
        Get current state of entities in Qdrant collection using scroll operations.
        
        This function uses Qdrant scroll API to efficiently retrieve entity metadata
        for delta-scan comparison. It processes entities in chunks to handle large
        collections without memory issues.
        
        Args:
            collection_name: Name of the Qdrant collection to scan
            chunk_size: Number of entities to process per scroll chunk (default: 1000)
            
        Returns:
            Dictionary containing collection state with entity metadata mapped by file_path
        """
        start_time = time.perf_counter()
        
        logger.info(f"Starting collection state scan: {collection_name}")
        
        try:
            # Get collection info first
            collection_info = await self._client.get_collection_info(collection_name)
            if not collection_info:
                logger.warning(f"Collection {collection_name} does not exist")
                return {
                    "exists": False,
                    "entities": {},
                    "entity_count": 0,
                    "file_count": 0,
                    "scan_time": 0.0,
                    "error": f"Collection {collection_name} not found"
                }
            
            total_points = collection_info.get("points_count", 0)
            logger.info(f"Collection {collection_name} contains {total_points} points")
            
            # Initialize state tracking
            entities_by_file = {}  # file_path -> list of entity metadata
            entity_count = 0
            
            # Use shared scroll method to iterate through all points
            async for point in self._scroll_collection_points(collection_name, chunk_size):
                try:
                    payload = point.payload or {}
                    file_path = payload.get('file_path')
                    
                    if not file_path:
                        logger.debug(f"Entity {point.id} missing file_path in payload")
                        continue
                    
                    # Extract entity metadata for delta comparison
                    entity_metadata = {
                        'entity_id': payload.get('entity_id', str(point.id)),
                        'indexed_at': payload.get('indexed_at'),
                        'entity_type': payload.get('entity_type'),
                        'name': payload.get('entity_name', payload.get('name')),
                        'qualified_name': payload.get('qualified_name'),
                        'file_hash': payload.get('file_hash'),
                        'location': payload.get('location', {}),
                        'last_modified': payload.get('last_modified')
                    }
                    
                    # Group entities by file for efficient comparison
                    if file_path not in entities_by_file:
                        entities_by_file[file_path] = []
                    entities_by_file[file_path].append(entity_metadata)
                    
                    entity_count += 1
                    
                    # Log progress for large collections
                    if entity_count % 10000 == 0:
                        logger.info(f"Processed {entity_count} entities so far")
                    
                except Exception as e:
                    logger.warning(f"Error processing entity {point.id}: {e}")
                    continue
            
            scan_time = time.perf_counter() - start_time
            file_count = len(entities_by_file)
            
            logger.info(
                f"Collection state scan completed: {entity_count} entities "
                f"across {file_count} files in {scan_time:.3f}s "
                f"({entity_count/scan_time:.1f} entities/sec)"
            )
            
            return {
                "exists": True,
                "entities": entities_by_file,
                "entity_count": entity_count,
                "file_count": file_count,
                "scan_time": scan_time,
                "collection_info": {
                    "points_count": total_points,
                    "vectors_count": collection_info.get("vectors_count", 0),
                    "status": collection_info.get("status"),
                    "optimizer_status": collection_info.get("optimizer_status")
                }
            }
            
        except Exception as e:
            scan_time = time.perf_counter() - start_time
            error_msg = f"Error scanning collection {collection_name}: {e}"
            logger.error(error_msg)
            
            return {
                "exists": False,
                "entities": {},
                "entity_count": 0,
                "file_count": 0,
                "scan_time": scan_time,
                "error": error_msg
            }
    
    async def _scroll_collection_points(
        self,
        collection_name: str,
        chunk_size: int = 1000,
        process_callback = None
    ):
        """
        Shared method to scroll through all points in a collection.
        
        Args:
            collection_name: Name of the collection to scroll
            chunk_size: Number of points per scroll chunk
            process_callback: Optional callback function to process each point
                            Should accept (point, state) and can modify state
            
        Yields:
            Individual points from the collection
        """
        next_page_offset = None
        
        while True:
            # Scroll through entities in chunks
            scroll_result = await asyncio.to_thread(
                self._client.client.scroll,
                collection_name=collection_name,
                limit=chunk_size,
                offset=next_page_offset,
                with_payload=True,
                with_vectors=False
            )
            
            # scroll returns (points, next_page_offset)
            points, next_page_offset = scroll_result
            
            if not points:
                break
            
            # Yield each point
            for point in points:
                yield point
                
            # If no next page, we're done
            if next_page_offset is None:
                break
    
    async def _validate_stale_entities_batch(
        self,
        collection_name: str,
        entity_ids: List[str],
        cutoff_timestamp: float
    ) -> List[str]:
        """
        Validate which entities in a batch are stale (indexed before cutoff).
        
        Args:
            collection_name: Collection to check
            entity_ids: List of entity IDs to validate
            cutoff_timestamp: Unix timestamp cutoff for staleness
            
        Returns:
            List of entity IDs that are stale (indexed before cutoff)
        """
        if not entity_ids:
            return []
        
        try:
            stale_entities = []
            
            # Process in smaller batches to avoid Qdrant limits
            batch_size = 100
            for i in range(0, len(entity_ids), batch_size):
                batch = entity_ids[i:i + batch_size]
                
                # Convert entity IDs to Qdrant point IDs for retrieval
                from ..storage.utils import entity_id_to_qdrant_id
                point_ids = [entity_id_to_qdrant_id(eid) for eid in batch]
                
                # Retrieve points to check indexed_at timestamps
                retrieve_result = await asyncio.to_thread(
                    self._client.client.retrieve,
                    collection_name=collection_name,
                    ids=point_ids,
                    with_payload=True,
                    with_vectors=False
                )
                
                for point in retrieve_result:
                    if not point or not point.payload:
                        continue
                    
                    indexed_at = point.payload.get('indexed_at')
                    if not indexed_at:
                        # If no indexed_at, consider it stale
                        entity_id = point.payload.get('entity_id')
                        if entity_id and entity_id in batch:
                            stale_entities.append(entity_id)
                        continue
                    
                    # Parse timestamp and compare with cutoff
                    from .delta_calculator import parse_timestamp_to_unix
                    indexed_timestamp = parse_timestamp_to_unix(indexed_at)
                    
                    if indexed_timestamp and indexed_timestamp < cutoff_timestamp:
                        entity_id = point.payload.get('entity_id')
                        if entity_id and entity_id in batch:
                            stale_entities.append(entity_id)
            
            return stale_entities
            
        except Exception as e:
            logger.error(f"Error validating stale entities batch: {e}")
            # On error, assume all entities are stale for safety
            return entity_ids