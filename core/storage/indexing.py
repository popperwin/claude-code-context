"""
Batch indexing with progress tracking for claude-code-context.

Provides efficient indexing of entities with embeddings, progress monitoring,
and error handling for large codebases.
"""

import asyncio
import logging
import time
import hashlib
from typing import List, Dict, Any, Optional, Callable, AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from tqdm.asyncio import tqdm

from .client import HybridQdrantClient
from .schemas import CollectionType, CollectionConfig
from .utils import entity_id_to_qdrant_id
from ..models.entities import Entity
from ..models.storage import QdrantPoint, StorageResult
from ..embeddings.base import BaseEmbedder

logger = logging.getLogger(__name__)


@dataclass
class IndexingProgress:
    """Progress information for indexing operations"""
    total_entities: int
    processed_entities: int
    successful_entities: int
    failed_entities: int
    current_batch: int
    total_batches: int
    elapsed_time: float
    estimated_remaining_time: float
    entities_per_second: float
    current_operation: str = ""
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage"""
        if self.total_entities == 0:
            return 100.0
        return (self.processed_entities / self.total_entities) * 100
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.processed_entities == 0:
            return 100.0
        return (self.successful_entities / self.processed_entities) * 100


@dataclass
class IndexingResult:
    """Result of indexing operation"""
    total_entities: int
    successful_entities: int
    failed_entities: int
    total_time: float
    average_time_per_entity: float
    entities_per_second: float
    errors: List[str]
    collection_name: str
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_entities == 0:
            return 100.0
        return (self.successful_entities / self.total_entities) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "total_entities": self.total_entities,
            "successful_entities": self.successful_entities,
            "failed_entities": self.failed_entities,
            "success_rate": self.success_rate,
            "total_time_s": self.total_time,
            "average_time_per_entity_ms": self.average_time_per_entity * 1000,
            "entities_per_second": self.entities_per_second,
            "collection_name": self.collection_name,
            "errors": self.errors
        }


class BatchIndexer:
    """
    Batch indexer with progress tracking and error handling.
    
    Features:
    - Efficient batch processing with configurable sizes
    - Real-time progress tracking with estimated completion time
    - Comprehensive error handling and recovery
    - Memory-efficient streaming for large datasets
    - Performance monitoring and metrics
    """
    
    def __init__(
        self,
        client: HybridQdrantClient,
        embedder: Optional[BaseEmbedder] = None,
        batch_size: int = 100,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize batch indexer.
        
        Args:
            client: Hybrid Qdrant client
            embedder: Embedder for generating embeddings (uses client's if None)
            batch_size: Number of entities per batch
            max_retries: Maximum retry attempts for failed batches
            retry_delay: Delay between retries in seconds
        """
        self.client = client
        self.embedder = embedder or client.embedder
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Progress tracking
        self._progress_callbacks: List[Callable[[IndexingProgress], None]] = []
        
        # Performance metrics
        self._total_indexed = 0
        self._total_indexing_time = 0.0
        self._failed_batches = 0
        
        if not self.embedder:
            logger.warning("No embedder available for indexing")
        
        logger.info(f"Initialized BatchIndexer with batch_size={batch_size}")
    
    
    def add_progress_callback(
        self,
        callback: Callable[[IndexingProgress], None]
    ) -> None:
        """Add progress callback function"""
        self._progress_callbacks.append(callback)
    
    def remove_progress_callback(
        self,
        callback: Callable[[IndexingProgress], None]
    ) -> None:
        """Remove progress callback function"""
        if callback in self._progress_callbacks:
            self._progress_callbacks.remove(callback)
    
    async def index_entities(
        self,
        entities: List[Entity],
        collection_name: str,
        show_progress: bool = True,
        description: str = "Indexing entities"
    ) -> IndexingResult:
        """
        Index entities with embeddings and progress tracking.
        
        Args:
            entities: List of entities to index
            collection_name: Target collection name
            show_progress: Whether to show progress bar
            description: Description for progress display
            
        Returns:
            Indexing result with metrics
        """
        if not entities:
            return IndexingResult(
                total_entities=0,
                successful_entities=0,
                failed_entities=0,
                total_time=0.0,
                average_time_per_entity=0.0,
                entities_per_second=0.0,
                errors=[],
                collection_name=collection_name
            )
        
        start_time = time.time()
        total_entities = len(entities)
        total_batches = (total_entities + self.batch_size - 1) // self.batch_size
        
        successful_entities = 0
        failed_entities = 0
        errors = []
        
        logger.info(
            f"Starting indexing: {total_entities} entities in {total_batches} batches"
        )
        
        # Initialize progress tracking
        progress = IndexingProgress(
            total_entities=total_entities,
            processed_entities=0,
            successful_entities=0,
            failed_entities=0,
            current_batch=0,
            total_batches=total_batches,
            elapsed_time=0.0,
            estimated_remaining_time=0.0,
            entities_per_second=0.0
        )
        
        # Create progress bar if requested
        pbar = None
        if show_progress:
            pbar = tqdm(
                total=total_entities,
                desc=description,
                unit="entities",
                unit_scale=True
            )
        
        try:
            # Process entities in batches
            for batch_idx in range(total_batches):
                batch_start_idx = batch_idx * self.batch_size
                batch_end_idx = min(batch_start_idx + self.batch_size, total_entities)
                batch_entities = entities[batch_start_idx:batch_end_idx]
                
                # Update progress
                progress.current_batch = batch_idx + 1
                progress.current_operation = f"Processing batch {batch_idx + 1}/{total_batches}"
                
                # Process batch with retries
                batch_result = await self._process_batch_with_retries(
                    batch_entities, collection_name, batch_idx
                )
                
                # Update counters
                successful_entities += batch_result.successful_count
                failed_entities += batch_result.failed_count
                
                if batch_result.errors:
                    errors.extend(batch_result.errors)
                
                # Update progress tracking
                processed_count = batch_end_idx
                elapsed_time = time.time() - start_time
                
                progress.processed_entities = processed_count
                progress.successful_entities = successful_entities
                progress.failed_entities = failed_entities
                progress.elapsed_time = elapsed_time
                
                if processed_count > 0:
                    progress.entities_per_second = processed_count / elapsed_time
                    remaining_entities = total_entities - processed_count
                    progress.estimated_remaining_time = (
                        remaining_entities / progress.entities_per_second
                        if progress.entities_per_second > 0 else 0
                    )
                
                # Notify progress callbacks
                for callback in self._progress_callbacks:
                    try:
                        callback(progress)
                    except Exception as e:
                        logger.warning(f"Progress callback failed: {e}")
                
                # Update progress bar
                if pbar:
                    pbar.update(len(batch_entities))
                    pbar.set_postfix({
                        'success_rate': f"{progress.success_rate:.1f}%",
                        'speed': f"{progress.entities_per_second:.1f}/s"
                    })
                
                # Brief pause to prevent overwhelming the system
                if batch_idx < total_batches - 1:
                    await asyncio.sleep(0.01)
        
        finally:
            if pbar:
                pbar.close()
        
        # Calculate final metrics
        total_time = time.time() - start_time
        average_time_per_entity = total_time / total_entities if total_entities > 0 else 0
        entities_per_second = total_entities / total_time if total_time > 0 else 0
        
        # Update global metrics
        self._total_indexed += successful_entities
        self._total_indexing_time += total_time
        
        result = IndexingResult(
            total_entities=total_entities,
            successful_entities=successful_entities,
            failed_entities=failed_entities,
            total_time=total_time,
            average_time_per_entity=average_time_per_entity,
            entities_per_second=entities_per_second,
            errors=errors,
            collection_name=collection_name
        )
        
        logger.info(
            f"Indexing completed: {successful_entities}/{total_entities} entities "
            f"({result.success_rate:.1f}%) in {total_time:.2f}s "
            f"({entities_per_second:.1f} entities/s)"
        )
        
        return result
    
    async def _process_batch_with_retries(
        self,
        entities: List[Entity],
        collection_name: str,
        batch_idx: int
    ) -> 'BatchResult':
        """Process batch with retry logic"""
        
        @dataclass
        class BatchResult:
            successful_count: int
            failed_count: int
            errors: List[str]
        
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                result = await self._process_batch(entities, collection_name, batch_idx)
                
                # Check if batch was successful
                if result.successful_count > 0 or result.failed_count == 0:
                    return result
                
                # If batch failed but no exception, treat as retryable failure
                last_error = result.errors[0] if result.errors else "Batch processing failed"
                
                if attempt < self.max_retries:
                    logger.warning(
                        f"Batch {batch_idx} attempt {attempt + 1} failed: {last_error}. "
                        f"Retrying in {self.retry_delay}s..."
                    )
                    await asyncio.sleep(self.retry_delay)
                else:
                    logger.error(
                        f"Batch {batch_idx} failed after {self.max_retries + 1} attempts: {last_error}"
                    )
                    return result  # Return the final failed result
                
            except Exception as e:
                last_error = str(e)
                
                if attempt < self.max_retries:
                    logger.warning(
                        f"Batch {batch_idx} attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {self.retry_delay}s..."
                    )
                    await asyncio.sleep(self.retry_delay)
                else:
                    logger.error(
                        f"Batch {batch_idx} failed after {self.max_retries + 1} attempts: {e}"
                    )
        
        # All retries failed
        self._failed_batches += 1
        return BatchResult(
            successful_count=0,
            failed_count=len(entities),
            errors=[f"Batch {batch_idx} failed: {last_error}"]
        )
    
    async def _process_batch(
        self,
        entities: List[Entity],
        collection_name: str,
        batch_idx: int
    ) -> 'BatchResult':
        """Process a single batch of entities"""
        
        @dataclass
        class BatchResult:
            successful_count: int
            failed_count: int
            errors: List[str]
        
        if not entities:
            return BatchResult(0, 0, [])
        
        try:
            # Convert entities to searchable text
            texts = [self._entity_to_text(entity) for entity in entities]
            
            # Generate embeddings
            if self.embedder:
                embedding_response = await self.embedder.embed_texts(texts)
                embeddings = embedding_response.embeddings
                
                if len(embeddings) != len(entities):
                    error_msg = (
                        f"Embedding count mismatch: {len(embeddings)} != {len(entities)}"
                    )
                    logger.error(error_msg)
                    return BatchResult(0, len(entities), [error_msg])
            else:
                # Create zero embeddings if no embedder
                embeddings = [[0.0] * 1024 for _ in entities]
                logger.warning(f"No embedder available, using zero embeddings")
            
            # Create Qdrant points
            points = []
            for entity, embedding in zip(entities, embeddings):
                try:
                    # Set indexed_at timestamp on entity (precise per-entity timing)
                    indexed_time = datetime.now()
                    entity = entity.model_copy(update={'indexed_at': indexed_time})
                    
                    payload = entity.to_qdrant_payload()
                    # Store original entity ID in payload for retrieval
                    payload["entity_id"] = entity.id
                    
                    # Convert string ID to integer for Qdrant compatibility
                    # Use centralized normalization function for consistency
                    qdrant_id = entity_id_to_qdrant_id(entity.id)
                    
                    point = QdrantPoint(
                        id=qdrant_id,  # QdrantPoint now expects integer directly
                        vector=embedding,
                        payload=payload
                    )
                    points.append(point)
                    
                except Exception as e:
                    logger.warning(f"Failed to create point for entity {entity.id}: {e}")
            
            if not points:
                return BatchResult(0, len(entities), ["No valid points created"])
            
            # Upsert to Qdrant
            result = await self.client.upsert_points(collection_name, points)
            
            if result.success:
                return BatchResult(
                    successful_count=len(points),
                    failed_count=len(entities) - len(points),
                    errors=[]
                )
            else:
                return BatchResult(
                    successful_count=0,
                    failed_count=len(entities),
                    errors=[result.error or "Upsert failed"]
                )
                
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return BatchResult(
                successful_count=0,
                failed_count=len(entities),
                errors=[str(e)]
            )
    
    def _entity_to_text(self, entity: Entity) -> str:
        """
        Convert entity to searchable text for embedding generation.
        
        Args:
            entity: Entity to convert
            
        Returns:
            Searchable text representation
        """
        parts = []
        
        # Add entity type and name
        if entity.entity_type:
            parts.append(f"Type: {entity.entity_type.value}")
        
        if entity.name:
            parts.append(f"Name: {entity.name}")
        
        if entity.qualified_name and entity.qualified_name != entity.name:
            parts.append(f"Qualified: {entity.qualified_name}")
        
        # Add signature
        if entity.signature:
            parts.append(f"Signature: {entity.signature}")
        
        # Add docstring (truncated)
        if entity.docstring:
            docstring = entity.docstring[:300]  # Limit docstring length
            if len(entity.docstring) > 300:
                docstring += "..."
            parts.append(f"Description: {docstring}")
        
        # Add source code context (first few lines)
        if entity.source_code:
            lines = entity.source_code.split('\n')[:5]  # First 5 lines
            code_snippet = '\n'.join(lines)
            if len(lines) >= 5:
                code_snippet += "\n..."
            parts.append(f"Code: {code_snippet}")
        
        # Add file context
        if entity.location.file_path:
            file_name = Path(entity.location.file_path).name
            parts.append(f"File: {file_name}")
        
        # Add visibility
        if entity.visibility:
            parts.append(f"Visibility: {entity.visibility.value}")
        
        # Join with separators
        return " | ".join(parts)
    
    async def index_entities_stream(
        self,
        entity_stream: AsyncGenerator[Entity, None],
        collection_name: str,
        show_progress: bool = True,
        description: str = "Streaming indexing"
    ) -> IndexingResult:
        """
        Index entities from an async stream with progress tracking.
        
        Args:
            entity_stream: Async generator of entities
            collection_name: Target collection name
            show_progress: Whether to show progress bar
            description: Description for progress display
            
        Returns:
            Indexing result with metrics
        """
        start_time = time.time()
        batch_buffer = []
        successful_entities = 0
        failed_entities = 0
        errors = []
        total_processed = 0
        
        logger.info("Starting streaming indexing")
        
        # Progress tracking without known total
        pbar = None
        if show_progress:
            pbar = tqdm(
                desc=description,
                unit="entities",
                unit_scale=True
            )
        
        try:
            async for entity in entity_stream:
                batch_buffer.append(entity)
                
                # Process when batch is full
                if len(batch_buffer) >= self.batch_size:
                    batch_result = await self._process_batch_with_retries(
                        batch_buffer, collection_name, total_processed // self.batch_size
                    )
                    
                    successful_entities += batch_result.successful_count
                    failed_entities += batch_result.failed_count
                    errors.extend(batch_result.errors)
                    
                    total_processed += len(batch_buffer)
                    
                    if pbar:
                        pbar.update(len(batch_buffer))
                        elapsed = time.time() - start_time
                        speed = total_processed / elapsed if elapsed > 0 else 0
                        pbar.set_postfix({
                            'success_rate': f"{successful_entities/total_processed*100:.1f}%" if total_processed > 0 else "0%",
                            'speed': f"{speed:.1f}/s"
                        })
                    
                    batch_buffer = []
            
            # Process remaining entities
            if batch_buffer:
                batch_result = await self._process_batch_with_retries(
                    batch_buffer, collection_name, total_processed // self.batch_size
                )
                
                successful_entities += batch_result.successful_count
                failed_entities += batch_result.failed_count
                errors.extend(batch_result.errors)
                
                total_processed += len(batch_buffer)
                
                if pbar:
                    pbar.update(len(batch_buffer))
        
        finally:
            if pbar:
                pbar.close()
        
        # Calculate final metrics
        total_time = time.time() - start_time
        average_time_per_entity = total_time / total_processed if total_processed > 0 else 0
        entities_per_second = total_processed / total_time if total_time > 0 else 0
        
        result = IndexingResult(
            total_entities=total_processed,
            successful_entities=successful_entities,
            failed_entities=failed_entities,
            total_time=total_time,
            average_time_per_entity=average_time_per_entity,
            entities_per_second=entities_per_second,
            errors=errors,
            collection_name=collection_name
        )
        
        logger.info(
            f"Streaming indexing completed: {successful_entities}/{total_processed} entities "
            f"({result.success_rate:.1f}%) in {total_time:.2f}s"
        )
        
        return result
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get indexer performance metrics"""
        return {
            "total_indexed": self._total_indexed,
            "total_indexing_time_s": self._total_indexing_time,
            "failed_batches": self._failed_batches,
            "average_indexing_speed": (
                self._total_indexed / max(1, self._total_indexing_time)
            ),
            "batch_size": self.batch_size,
            "max_retries": self.max_retries,
            "retry_delay_s": self.retry_delay
        }