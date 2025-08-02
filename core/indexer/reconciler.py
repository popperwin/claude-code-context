"""
Entity reconciliation operations for delta-scan workflows.

This module provides chunked delete and upsert operations for efficiently
applying delta changes to Qdrant collections with progress tracking and
comprehensive error handling.
"""

import asyncio
import hashlib
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class Reconciler:
    """
    Entity reconciliation operations for delta-scan workflows.
    
    Provides chunked delete and upsert operations for efficiently applying
    delta changes to Qdrant collections with progress tracking, error handling,
    and comprehensive metrics collection.
    """
    
    def __init__(self, storage_client, embedder):
        """
        Initialize reconciler with required dependencies.
        
        Args:
            storage_client: HybridQdrantClient for Qdrant operations
            embedder: StellaEmbedder for embedding generation
        """
        self._client = storage_client
        self._embedder = embedder
    
    async def chunked_entity_delete(
        self,
        collection_name: str,
        stale_entity_ids: List[str],
        cutoff_timestamp: float,
        chunk_size: int = 10000
    ) -> Dict[str, Any]:
        """
        Delete stale entities in chunks with validation-based coordination.
        
        This method implements chunked deletion with 10k entity batching to respect
        Qdrant limits, using validation-at-operation-time to prevent race conditions
        with real-time updates.
        
        Args:
            collection_name: Target collection name
            stale_entity_ids: List of potentially stale entity IDs
            cutoff_timestamp: Timestamp threshold for staleness validation
            chunk_size: Number of entities per deletion chunk (default 10k)
            
        Returns:
            Dictionary with deletion results and metrics
        """
        if not stale_entity_ids:
            return {
                "success": True,
                "total_entities": 0,
                "deleted_entities": 0,
                "validated_chunks": 0,
                "skipped_entities": 0,
                "processing_time_ms": 0.0,
                "errors": []
            }
        
        start_time = time.perf_counter()
        total_deleted = 0
        validated_chunks = 0
        skipped_entities = 0
        errors = []
        
        logger.info(
            f"Starting chunked deletion of {len(stale_entity_ids)} stale entities "
            f"in {collection_name} (chunk_size={chunk_size})"
        )
        
        try:
            # Check if collection exists first
            try:
                collection_info = await self._client.get_collection_info(collection_name)
                if not collection_info:
                    logger.info(f"Collection {collection_name} does not exist, no entities to delete")
                    return {
                        "success": True,
                        "total_entities": len(stale_entity_ids),
                        "deleted_entities": 0,
                        "validated_chunks": 0,
                        "skipped_entities": len(stale_entity_ids),
                        "processing_time_ms": (time.perf_counter() - start_time) * 1000,
                        "errors": []
                    }
            except Exception as e:
                logger.warning(f"Cannot check collection {collection_name}: {e}, proceeding with deletion attempt")
            
            # Process entities in chunks
            chunks = self._chunk_list(stale_entity_ids, chunk_size)
            
            for chunk_idx, chunk in enumerate(chunks):
                try:
                    # CRITICAL: Validate staleness at operation time to prevent race conditions
                    validated_ids = await self._validate_stale_entities_batch(
                        collection_name, chunk, cutoff_timestamp
                    )
                    
                    if not validated_ids:
                        skipped_entities += len(chunk)
                        logger.debug(f"Chunk {chunk_idx + 1}: No stale entities found, skipping")
                        continue
                    
                    validated_chunks += 1
                    
                    # Convert entity IDs to point IDs for Qdrant deletion
                    # Use centralized normalization function for consistency
                    from ..storage.utils import entity_id_to_qdrant_id
                    point_ids = [entity_id_to_qdrant_id(eid) for eid in validated_ids]
                    
                    # Leverage existing chunked delete infrastructure
                    delete_result = await self._client.delete_points(
                        collection_name=collection_name,
                        point_ids=point_ids
                    )
                    
                    if delete_result.success:
                        chunk_deleted = len(validated_ids)
                        total_deleted += chunk_deleted
                        skipped_entities += len(chunk) - len(validated_ids)
                        
                        logger.debug(
                            f"Chunk {chunk_idx + 1}/{len(chunks)}: "
                            f"Deleted {chunk_deleted} entities "
                            f"(validated {len(validated_ids)}/{len(chunk)})"
                        )
                    else:
                        error_msg = f"Chunk {chunk_idx + 1} deletion failed: {delete_result.error}"
                        errors.append(error_msg)
                        logger.error(error_msg)
                        skipped_entities += len(chunk)
                
                except Exception as e:
                    error_msg = f"Error processing chunk {chunk_idx + 1}: {e}"
                    errors.append(error_msg)
                    logger.error(error_msg)
                    skipped_entities += len(chunk)
                    continue
        
        except Exception as e:
            error_msg = f"Fatal error in chunked deletion: {e}"
            errors.append(error_msg)
            logger.error(error_msg)
        
        processing_time = time.perf_counter() - start_time
        
        result = {
            "success": len(errors) == 0,
            "total_entities": len(stale_entity_ids),
            "deleted_entities": total_deleted,
            "validated_chunks": validated_chunks,
            "skipped_entities": skipped_entities,
            "processing_time_ms": processing_time * 1000,
            "errors": errors
        }
        
        logger.info(
            f"Chunked deletion complete: {total_deleted}/{len(stale_entity_ids)} entities "
            f"deleted in {processing_time:.3f}s ({validated_chunks} chunks validated)"
        )
        
        return result
    
    async def chunked_entity_upsert(
        self,
        collection_name: str,
        entities: List[Any],  # List[Entity]
        chunk_size: int = 1000,
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Upsert entities in chunks with progress tracking and comprehensive metrics.
        
        This method implements chunked upsert operations with batch processing for
        optimal performance, real-time progress tracking, and comprehensive error handling.
        Designed for delta-scan pipeline where large batches of entities need to be
        efficiently stored with embedding generation.
        
        Args:
            collection_name: Target collection name
            entities: List of entities to upsert
            chunk_size: Number of entities per upsert chunk (default 1000)
            progress_callback: Optional callback for progress tracking
            
        Returns:
            Dictionary with upsert results and comprehensive metrics
        """
        if not entities:
            return {
                "success": True,
                "total_entities": 0,
                "upserted_entities": 0,
                "processed_chunks": 0,
                "failed_entities": 0,
                "processing_time_ms": 0.0,
                "embedding_time_ms": 0.0,
                "storage_time_ms": 0.0,
                "average_time_per_entity_ms": 0.0,
                "entities_per_second": 0.0,
                "errors": []
            }

        # Apply deterministic IDs to incoming entities
        processed_entities = []
        for _ent in entities:
            try:
                file_hash = hashlib.sha256(_ent.location.file_path.read_bytes()).hexdigest()
                from ..sync.deterministic import DeterministicEntityId
                _ent = DeterministicEntityId.update_entity_with_deterministic_id(_ent, file_hash)
            except Exception as exc:
                logger.warning(f"Failed deterministic ID for {_ent.id}: {exc}")
            processed_entities.append(_ent)
        
        # Deduplicate entities by ID - keep last occurrence
        entity_dict = {}
        for _ent in processed_entities:
            entity_dict[_ent.id] = _ent
        entities = list(entity_dict.values())

        start_time = time.perf_counter()
        total_upserted = 0
        processed_chunks = 0
        failed_entities = 0
        total_embedding_time = 0.0
        total_storage_time = 0.0
        errors = []
        
        logger.info(
            f"Starting chunked upsert of {len(entities)} entities "
            f"in {collection_name} (chunk_size={chunk_size})"
        )
        
        try:
            # Process entities in chunks for optimal batch performance
            chunks = self._chunk_list(entities, chunk_size)
            
            for chunk_idx, chunk in enumerate(chunks):
                try:
                    chunk_start_time = time.perf_counter()
                    
                    # Generate embeddings for chunk
                    embedding_start_time = time.perf_counter()
                    
                    # Convert entities to searchable text
                    texts = [self._entity_to_searchable_text(entity) for entity in chunk]
                    
                    # Generate embeddings using the embedder
                    if self._embedder:
                        embedding_response = await self._embedder.embed_texts(texts)
                        embeddings = embedding_response.embeddings
                        
                        if len(embeddings) != len(chunk):
                            error_msg = (
                                f"Chunk {chunk_idx + 1}: Embedding count mismatch: "
                                f"{len(embeddings)} != {len(chunk)}"
                            )
                            errors.append(error_msg)
                            logger.error(error_msg)
                            failed_entities += len(chunk)
                            continue
                    else:
                        # Use zero embeddings if no embedder available
                        embeddings = [[0.0] * 1024 for _ in chunk]
                        logger.warning(f"Chunk {chunk_idx + 1}: No embedder available, using zero embeddings")
                    
                    embedding_time = time.perf_counter() - embedding_start_time
                    total_embedding_time += embedding_time
                    
                    # Create Qdrant points
                    storage_start_time = time.perf_counter()
                    points = []
                    
                    for entity, embedding in zip(chunk, embeddings):
                        try:
                            # Set indexed_at timestamp on entity (precise per-entity timing)
                            indexed_time = datetime.now()
                            entity = entity.model_copy(update={'indexed_at': indexed_time})
                            
                            payload = entity.to_qdrant_payload()
                            # Store original entity ID for retrieval consistency
                            payload["entity_id"] = entity.id
                            
                            # Convert to Qdrant-compatible ID
                            from ..storage.utils import entity_id_to_qdrant_id
                            qdrant_id = entity_id_to_qdrant_id(entity.id)
                            
                            from ..models.storage import QdrantPoint
                            point = QdrantPoint(
                                id=qdrant_id,
                                vector=embedding,
                                payload=payload
                            )
                            points.append(point)
                            
                        except Exception as e:
                            logger.warning(f"Failed to create point for entity {entity.id}: {e}")
                    
                    if not points:
                        error_msg = f"Chunk {chunk_idx + 1}: No valid points created"
                        errors.append(error_msg)
                        failed_entities += len(chunk)
                        continue
                    
                    # Upsert points to Qdrant
                    upsert_result = await self._client.upsert_points(collection_name, points)
                    
                    storage_time = time.perf_counter() - storage_start_time
                    total_storage_time += storage_time
                    
                    if upsert_result.success:
                        chunk_upserted = len(points)
                        total_upserted += chunk_upserted
                        failed_entities += len(chunk) - len(points)
                        processed_chunks += 1
                        
                        chunk_time = time.perf_counter() - chunk_start_time
                        
                        logger.debug(
                            f"Chunk {chunk_idx + 1}/{len(chunks)}: "
                            f"Upserted {chunk_upserted} entities "
                            f"(emb: {embedding_time:.3f}s, storage: {storage_time:.3f}s, "
                            f"total: {chunk_time:.3f}s)"
                        )
                        
                        # Progress callback with comprehensive metrics
                        if progress_callback:
                            progress_data = {
                                "phase": "chunked_upsert",
                                "current_chunk": chunk_idx + 1,
                                "total_chunks": len(chunks),
                                "chunk_entities": len(chunk),
                                "chunk_upserted": chunk_upserted,
                                "total_upserted": total_upserted,
                                "chunk_time_ms": chunk_time * 1000,
                                "embedding_time_ms": embedding_time * 1000,
                                "storage_time_ms": storage_time * 1000,
                                "entities_per_second": chunk_upserted / chunk_time if chunk_time > 0 else 0
                            }
                            progress_callback(
                                chunk_idx + 1,
                                len(chunks),
                                progress_data
                            )
                    else:
                        error_msg = f"Chunk {chunk_idx + 1} upsert failed: {upsert_result.error}"
                        errors.append(error_msg)
                        logger.error(error_msg)
                        failed_entities += len(chunk)
                
                except Exception as e:
                    error_msg = f"Error processing chunk {chunk_idx + 1}: {e}"
                    errors.append(error_msg)
                    logger.error(error_msg)
                    failed_entities += len(chunk)
                    continue
        
        except Exception as e:
            error_msg = f"Fatal error in chunked upsert: {e}"
            errors.append(error_msg)
            logger.error(error_msg)
        
        processing_time = time.perf_counter() - start_time
        
        # Calculate comprehensive metrics
        avg_time_per_entity = processing_time / len(entities) if entities else 0
        entities_per_second = total_upserted / processing_time if processing_time > 0 else 0
        
        result = {
            "success": len(errors) == 0,
            "total_entities": len(entities),
            "upserted_entities": total_upserted,
            "processed_chunks": processed_chunks,
            "failed_entities": failed_entities,
            "processing_time_ms": processing_time * 1000,
            "embedding_time_ms": total_embedding_time * 1000,
            "storage_time_ms": total_storage_time * 1000,
            "average_time_per_entity_ms": avg_time_per_entity * 1000,
            "entities_per_second": entities_per_second,
            "errors": errors
        }
        
        logger.info(
            f"Chunked upsert complete: {total_upserted}/{len(entities)} entities "
            f"upserted in {processing_time:.3f}s "
            f"({entities_per_second:.1f} entities/sec, {processed_chunks} chunks)"
        )
        
        return result
    
    def _entity_to_searchable_text(self, entity) -> str:
        """
        Convert entity to searchable text for embedding generation.
        
        This method creates a comprehensive text representation of the entity
        that captures its semantic meaning for embedding-based search.
        
        Args:
            entity: Entity to convert
            
        Returns:
            Searchable text representation optimized for embeddings
        """
        parts = []
        
        # Add entity type and name for primary identification
        if entity.entity_type:
            parts.append(f"Type: {entity.entity_type.value}")
        
        if entity.name:
            parts.append(f"Name: {entity.name}")
        
        if entity.qualified_name and entity.qualified_name != entity.name:
            parts.append(f"Qualified: {entity.qualified_name}")
        
        # Add signature for functional context
        if entity.signature:
            parts.append(f"Signature: {entity.signature}")
        
        # Add docstring for semantic understanding (truncated for performance)
        if entity.docstring:
            docstring = entity.docstring[:400]  # Increased limit for better context
            if len(entity.docstring) > 400:
                docstring += "..."
            parts.append(f"Description: {docstring}")
        
        # Add source code context (first few lines for implementation patterns)
        if entity.source_code:
            lines = entity.source_code.split('\n')[:6]  # Slightly more lines for context
            code_snippet = '\n'.join(lines)
            if len(lines) >= 6:
                code_snippet += "\n..."
            parts.append(f"Code: {code_snippet}")
        
        # Add file context for location-based search
        if entity.location and entity.location.file_path:
            file_name = Path(entity.location.file_path).name
            parts.append(f"File: {file_name}")
        
        # Add visibility information
        if entity.visibility:
            parts.append(f"Visibility: {entity.visibility.value}")
        
        # Join with separators optimized for embedding models
        return " | ".join(parts)
    
    def _chunk_list(self, items: List[Any], chunk_size: int) -> List[List[Any]]:
        """
        Split a list into chunks of specified size.
        
        Args:
            items: List to chunk
            chunk_size: Maximum size per chunk
            
        Returns:
            List of chunks
        """
        chunks = []
        for i in range(0, len(items), chunk_size):
            chunks.append(items[i:i + chunk_size])
        return chunks
    
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
            # Check if collection exists first
            try:
                collection_info = await self._client.get_collection_info(collection_name)
                if not collection_info:
                    logger.debug(f"Collection {collection_name} does not exist, treating all entities as stale")
                    return entity_ids
            except Exception as e:
                logger.warning(f"Cannot check collection {collection_name}: {e}, treating all entities as stale")
                return entity_ids
            
            stale_entities = []
            
            # Process in smaller batches to avoid Qdrant limits
            batch_size = 100
            for i in range(0, len(entity_ids), batch_size):
                batch = entity_ids[i:i + batch_size]
                
                # Convert entity IDs to Qdrant point IDs for retrieval
                from ..storage.utils import entity_id_to_qdrant_id
                point_ids = [entity_id_to_qdrant_id(eid) for eid in batch]
                
                try:
                    # Retrieve points to check indexed_at timestamps
                    retrieve_result = await asyncio.to_thread(
                        self._client.client.retrieve,
                        collection_name=collection_name,
                        ids=point_ids,
                        with_payload=True,
                        with_vectors=False
                    )
                except Exception as e:
                    logger.warning(f"Error retrieving batch {i//batch_size + 1}: {e}, treating batch as stale")
                    # On batch error, treat all entities in this batch as stale
                    stale_entities.extend(batch)
                    continue
                
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