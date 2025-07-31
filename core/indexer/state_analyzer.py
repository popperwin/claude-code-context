"""
Entity-aware collection state analyzer.

Provides collection state detection focused on entity counts, metadata tracking,
and entity-level health validation for the HybridIndexer workflow.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from ..storage.client import HybridQdrantClient
from ..storage.schemas import CollectionType, CollectionManager

logger = logging.getLogger(__name__)


class CollectionState(Enum):
    """States of a collection for entity-level analysis."""
    EMPTY = "empty"                    # No entities in collection
    HEALTHY = "healthy"                # Collection has entities and is up-to-date
    STALE = "stale"                   # Collection exists but may need refresh
    INCONSISTENT = "inconsistent"      # Collection has integrity issues
    INACCESSIBLE = "inaccessible"     # Collection cannot be accessed
    UNKNOWN = "unknown"               # State could not be determined


@dataclass
class EntityMetadata:
    """Metadata about entities in a collection."""
    total_entities: int = 0
    last_update_time: Optional[datetime] = None
    oldest_entity_time: Optional[datetime] = None
    newest_entity_time: Optional[datetime] = None
    file_coverage: int = 0  # Number of unique files represented
    entity_types: Dict[str, int] = None  # Count by entity type
    average_entities_per_file: float = 0.0
    
    def __post_init__(self):
        if self.entity_types is None:
            self.entity_types = {}


@dataclass
class CollectionStateInfo:
    """Complete state information about a collection."""
    collection_name: str
    state: CollectionState
    entity_metadata: EntityMetadata
    health_score: float  # 0.0 to 1.0, higher is better
    last_analyzed: datetime
    analysis_duration_ms: float
    recommendations: List[str]
    issues_found: List[str]
    
    # Technical details
    collection_exists: bool = False
    collection_info: Optional[Dict[str, Any]] = None
    is_indexing: bool = False
    
    def __post_init__(self):
        if not self.recommendations:
            self.recommendations = []
        if not self.issues_found:
            self.issues_found = []


class CollectionStateAnalyzer:
    """
    Entity-aware collection state analyzer.
    
    Analyzes collection state based on entity counts, metadata, and health indicators
    to inform scan mode selection in the HybridIndexer workflow.
    
    Features:
    - Entity count-based state detection
    - Collection health scoring  
    - Entity metadata analysis
    - Scan mode recommendations
    - Collection consistency validation
    """
    
    def __init__(
        self,
        storage_client: HybridQdrantClient,
        staleness_threshold_hours: int = 24,
        min_health_score: float = 0.7,
        max_analysis_time_ms: float = 5000.0
    ):
        """
        Initialize collection state analyzer.
        
        Args:
            storage_client: Qdrant client for collection operations
            staleness_threshold_hours: Hours after which collection is considered stale
            min_health_score: Minimum score to consider collection healthy
            max_analysis_time_ms: Maximum time to spend on analysis
        """
        self.storage_client = storage_client
        self.staleness_threshold_hours = staleness_threshold_hours
        self.min_health_score = min_health_score
        self.max_analysis_time_ms = max_analysis_time_ms
        
        # Analysis cache to avoid repeated expensive operations
        self._analysis_cache: Dict[str, CollectionStateInfo] = {}
        self._cache_ttl_seconds = 300  # 5 minutes
        
        logger.info(f"Initialized CollectionStateAnalyzer with {staleness_threshold_hours}h staleness threshold")
    
    async def analyze_collection_state(
        self,
        collection_name: str,
        project_path: Optional[Path] = None,
        force_refresh: bool = False
    ) -> CollectionStateInfo:
        """
        Analyze the current state of a collection.
        
        Args:
            collection_name: Name of collection to analyze
            project_path: Optional project path for additional context
            force_refresh: Force refresh of cached analysis
            
        Returns:
            Complete collection state information
        """
        start_time = time.time()
        
        # Check cache first
        if not force_refresh and collection_name in self._analysis_cache:
            cached_info = self._analysis_cache[collection_name]
            cache_age = (datetime.now() - cached_info.last_analyzed).total_seconds()
            if cache_age < self._cache_ttl_seconds:
                logger.debug(f"Using cached analysis for {collection_name}")
                return cached_info
        
        logger.debug(f"Analyzing collection state: {collection_name}")
        
        try:
            # Initialize state info
            state_info = CollectionStateInfo(
                collection_name=collection_name,
                state=CollectionState.UNKNOWN,
                entity_metadata=EntityMetadata(),
                health_score=0.0,
                last_analyzed=datetime.now(),
                analysis_duration_ms=0.0,
                recommendations=[],
                issues_found=[]
            )
            
            # Step 1: Check if collection exists and get basic info
            await self._analyze_collection_existence(state_info)
            
            if not state_info.collection_exists:
                state_info.state = CollectionState.EMPTY
                state_info.recommendations.append("Perform initial full scan to populate collection")
            else:
                # Step 2: Analyze entity metadata
                await self._analyze_entity_metadata(state_info)
                
                # Step 3: Determine collection state
                await self._determine_collection_state(state_info, project_path)
                
                # Step 4: Calculate health score
                self._calculate_health_score(state_info)
                
                # Step 5: Generate recommendations
                self._generate_recommendations(state_info)
            
            # Finalize analysis
            state_info.analysis_duration_ms = (time.time() - start_time) * 1000
            
            # Cache result
            self._analysis_cache[collection_name] = state_info
            
            logger.info(
                f"Collection analysis complete: {collection_name} -> {state_info.state.value} "
                f"(health: {state_info.health_score:.2f}, entities: {state_info.entity_metadata.total_entities})"
            )
            
            return state_info
            
        except Exception as e:
            analysis_duration = (time.time() - start_time) * 1000
            logger.error(f"Error analyzing collection {collection_name}: {e}")
            
            # Return error state
            return CollectionStateInfo(
                collection_name=collection_name,
                state=CollectionState.INACCESSIBLE,
                entity_metadata=EntityMetadata(),
                health_score=0.0,
                last_analyzed=datetime.now(),
                analysis_duration_ms=analysis_duration,
                recommendations=["Collection is inaccessible - check Qdrant connection"],
                issues_found=[f"Analysis failed: {str(e)}"]
            )
    
    async def _analyze_collection_existence(self, state_info: CollectionStateInfo) -> None:
        """Analyze whether collection exists and get basic information."""
        try:
            collection_info = await self.storage_client.get_collection_info(state_info.collection_name)
            
            if collection_info:
                state_info.collection_exists = True
                state_info.collection_info = collection_info
                
                # Check if collection is currently being indexed
                optimizer_status = collection_info.get("optimizer_status", "").lower()
                state_info.is_indexing = optimizer_status in ["optimizing", "indexing"]
                
                logger.debug(f"Collection {state_info.collection_name} exists with {collection_info.get('points_count', 0)} points")
            else:
                state_info.collection_exists = False
                logger.debug(f"Collection {state_info.collection_name} does not exist")
                
        except Exception as e:
            logger.warning(f"Could not check collection existence: {e}")
            state_info.collection_exists = False
            state_info.issues_found.append(f"Collection access error: {str(e)}")
            # Mark as inaccessible rather than empty when there's a connection error
            raise e  # Re-raise to be caught by main exception handler
    
    async def _analyze_entity_metadata(self, state_info: CollectionStateInfo) -> None:
        """Analyze entity metadata in the collection."""
        if not state_info.collection_exists:
            return
        
        try:
            # Get basic collection info
            if state_info.collection_info:
                state_info.entity_metadata.total_entities = state_info.collection_info.get("points_count", 0)
            
            # If collection is empty, no need for detailed analysis
            if state_info.entity_metadata.total_entities == 0:
                return
            
            # Sample entities to get metadata (limit to avoid performance issues)
            sample_limit = min(1000, state_info.entity_metadata.total_entities)
            search_results = await self.storage_client.search_payload(
                collection_name=state_info.collection_name,
                query="*",  # Match all entities
                limit=sample_limit
            )
            
            if search_results:
                await self._extract_entity_metadata(search_results, state_info.entity_metadata)
            else:
                # If no search results but points_count > 0, this indicates an issue
                # Reset total_entities to match actual searchable entities
                if state_info.entity_metadata.total_entities > 0:
                    logger.warning(f"Collection {state_info.collection_name} reports {state_info.entity_metadata.total_entities} entities but search returned none")
                    state_info.entity_metadata.total_entities = 0
            
        except Exception as e:
            logger.warning(f"Could not analyze entity metadata: {e}")
            state_info.issues_found.append(f"Entity metadata analysis failed: {str(e)}")
    
    async def _extract_entity_metadata(
        self,
        search_results: List[Any],
        metadata: EntityMetadata
    ) -> None:
        """Extract metadata from sample of search results."""
        entity_types = {}
        files_seen = set()
        timestamps = []
        
        for result in search_results:
            # Handle different result formats from tests
            payload = None
            
            if hasattr(result, 'point') and hasattr(result.point, 'payload'):
                payload = result.point.payload
            elif isinstance(result, dict):
                if 'point' in result and 'payload' in result['point']:
                    payload = result['point']['payload']
                elif 'payload' in result:
                    payload = result['payload']
            
            if not payload:
                continue
            
            # Extract entity type
            entity_type = payload.get('entity_type', 'unknown')
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            
            # Extract file path
            file_path = payload.get('file_path')
            if file_path:
                files_seen.add(file_path)
            
            # Extract timestamps (if available)
            timestamp_str = payload.get('indexed_at') or payload.get('created_at')
            if timestamp_str:
                try:
                    # Handle both ISO format and ISO with Z
                    if timestamp_str.endswith('Z'):
                        timestamp_str = timestamp_str[:-1] + '+00:00'
                    timestamp = datetime.fromisoformat(timestamp_str)
                    # Convert to UTC naive datetime to avoid timezone comparison issues
                    if timestamp.tzinfo is not None:
                        timestamp = timestamp.replace(tzinfo=None)
                    timestamps.append(timestamp)
                except (ValueError, AttributeError):
                    pass
        
        # Update metadata
        metadata.entity_types = entity_types
        metadata.file_coverage = len(files_seen)
        
        if metadata.file_coverage > 0:
            metadata.average_entities_per_file = len(search_results) / metadata.file_coverage
        
        if timestamps:
            metadata.oldest_entity_time = min(timestamps)
            metadata.newest_entity_time = max(timestamps)
            metadata.last_update_time = metadata.newest_entity_time
    
    async def _determine_collection_state(
        self,
        state_info: CollectionStateInfo,
        project_path: Optional[Path]
    ) -> None:
        """Determine the overall state of the collection."""
        metadata = state_info.entity_metadata
        
        # Empty collection - check actual entity count from collection info
        if metadata.total_entities == 0:
            state_info.state = CollectionState.EMPTY
            return
        
        # Check for staleness
        if metadata.last_update_time:
            age_hours = (datetime.now() - metadata.last_update_time).total_seconds() / 3600
            if age_hours > self.staleness_threshold_hours:
                state_info.state = CollectionState.STALE
                state_info.issues_found.append(f"Collection is {age_hours:.1f} hours old (threshold: {self.staleness_threshold_hours}h)")
                return
        
        # Check for basic consistency issues
        inconsistency_indicators = []
        
        # Very low file coverage might indicate issues
        if metadata.file_coverage > 0 and metadata.average_entities_per_file < 1.0:
            inconsistency_indicators.append("Very low entities per file ratio")
        
        # Unbalanced entity types might indicate parsing issues
        if metadata.entity_types:
            total_entities = sum(metadata.entity_types.values())
            unknown_ratio = metadata.entity_types.get('unknown', 0) / max(total_entities, 1)
            if unknown_ratio > 0.5:
                inconsistency_indicators.append("High ratio of unknown entity types")
        
        if inconsistency_indicators:
            state_info.state = CollectionState.INCONSISTENT
            state_info.issues_found.extend(inconsistency_indicators)
        else:
            state_info.state = CollectionState.HEALTHY
    
    def _calculate_health_score(self, state_info: CollectionStateInfo) -> None:
        """Calculate a health score from 0.0 to 1.0."""
        score = 0.0
        metadata = state_info.entity_metadata
        
        # Base score for having entities (more generous)
        if metadata.total_entities > 0:
            score += 0.4
        
        # Score for recency
        if metadata.last_update_time:
            age_hours = (datetime.now() - metadata.last_update_time).total_seconds() / 3600
            recency_score = max(0, 1.0 - (age_hours / (self.staleness_threshold_hours * 2)))
            score += 0.4 * recency_score
        
        # Score for entity diversity (more generous for single types)
        if metadata.entity_types:
            type_count = len(metadata.entity_types)
            diversity_score = min(1.0, type_count / 3.0)  # Up to 3 types for full score
            score += 0.1 * diversity_score
        
        # Score for file coverage (more generous)
        if metadata.file_coverage > 0:
            # Reasonable entities per file ratio
            ratio = metadata.average_entities_per_file
            coverage_score = min(1.0, ratio / 5.0) if ratio > 0 else 0
            score += 0.1 * coverage_score
        
        # Penalties for issues
        issue_penalty = len(state_info.issues_found) * 0.1
        score = max(0.0, score - issue_penalty)
        
        # Additional penalties for collection state
        if state_info.state == CollectionState.INCONSISTENT:
            score *= 0.6  # 40% penalty for inconsistent state
        elif state_info.state == CollectionState.STALE:
            score *= 0.8  # 20% penalty for stale state
        elif state_info.state in [CollectionState.INACCESSIBLE, CollectionState.EMPTY]:
            score = 0.0  # Zero score for inaccessible or empty
        
        # Penalty for high unknown entity ratio
        if metadata.entity_types:
            total_entities = sum(metadata.entity_types.values())
            unknown_ratio = metadata.entity_types.get('unknown', 0) / max(total_entities, 1)
            if unknown_ratio > 0.5:
                score *= (1.0 - unknown_ratio * 0.5)  # Additional penalty for unknown entities
        
        state_info.health_score = min(1.0, score)
    
    def _generate_recommendations(self, state_info: CollectionStateInfo) -> None:
        """Generate actionable recommendations based on analysis."""
        metadata = state_info.entity_metadata
        
        if state_info.state == CollectionState.EMPTY:
            state_info.recommendations.append("Perform initial full scan to populate collection")
        
        elif state_info.state == CollectionState.STALE:
            state_info.recommendations.append("Perform incremental scan to update stale entities")
            if metadata.last_update_time:
                age_hours = (datetime.now() - metadata.last_update_time).total_seconds() / 3600
                if age_hours > self.staleness_threshold_hours * 3:
                    state_info.recommendations.append("Consider full rescan due to extended staleness")
        
        elif state_info.state == CollectionState.INCONSISTENT:
            state_info.recommendations.append("Run consistency validation and repair")
            if "unknown entity types" in str(state_info.issues_found).lower():
                state_info.recommendations.append("Check parser configuration for supported file types")
        
        elif state_info.state == CollectionState.HEALTHY:
            if state_info.health_score < self.min_health_score:
                state_info.recommendations.append("Consider incremental scan to improve collection health")
            else:
                state_info.recommendations.append("Collection is healthy - enable real-time sync monitoring")
        
        elif state_info.state == CollectionState.INACCESSIBLE:
            state_info.recommendations.append("Check Qdrant connection and collection accessibility")
        
        # Additional recommendations based on metadata
        if metadata.total_entities > 0:
            if metadata.file_coverage == 0:
                state_info.recommendations.append("Investigate missing file path metadata in entities")
            
            if not metadata.entity_types or len(metadata.entity_types) == 1:
                state_info.recommendations.append("Verify parser is extracting diverse entity types")
    
    async def get_entity_count(self, collection_name: str) -> int:
        """
        Get total number of entities in collection.
        
        Args:
            collection_name: Name of collection
            
        Returns:
            Number of entities in collection
        """
        try:
            collection_info = await self.storage_client.get_collection_info(collection_name)
            if collection_info:
                return collection_info.get("points_count", 0)
            return 0
            
        except Exception as e:
            logger.error(f"Failed to get entity count for {collection_name}: {e}")
            return 0
    
    async def get_last_entity_update(self, collection_name: str) -> Optional[datetime]:
        """
        Get timestamp of most recent entity update.
        
        Args:
            collection_name: Name of collection
            
        Returns:
            Datetime of last update, None if cannot be determined
        """
        try:
            # Sample recent entities to find latest timestamp
            search_results = await self.storage_client.search_payload(
                collection_name=collection_name,
                query="*",
                limit=100
            )
            
            latest_time = None
            for result in search_results:
                # Handle different result formats from tests
                payload = None
                
                if hasattr(result, 'point') and hasattr(result.point, 'payload'):
                    payload = result.point.payload
                elif isinstance(result, dict):
                    if 'point' in result and 'payload' in result['point']:
                        payload = result['point']['payload']
                    elif 'payload' in result:
                        payload = result['payload']
                
                if not payload:
                    continue
                
                timestamp_str = payload.get('indexed_at') or payload.get('created_at')
                if timestamp_str:
                    try:
                        # Handle both ISO format and ISO with Z
                        if timestamp_str.endswith('Z'):
                            timestamp_str = timestamp_str[:-1] + '+00:00'
                        timestamp = datetime.fromisoformat(timestamp_str)
                        # Convert to UTC naive datetime to avoid timezone comparison issues
                        if timestamp.tzinfo is not None:
                            timestamp = timestamp.replace(tzinfo=None)
                        if latest_time is None or timestamp > latest_time:
                            latest_time = timestamp
                    except (ValueError, AttributeError):
                        continue
            
            return latest_time
            
        except Exception as e:
            logger.error(f"Failed to get last entity update for {collection_name}: {e}")
            return None
    
    async def is_collection_healthy(
        self,
        collection_name: str,
        min_entities: int = 1,
        max_staleness_hours: Optional[int] = None
    ) -> bool:
        """
        Check if collection is in a healthy state.
        
        Args:
            collection_name: Name of collection
            min_entities: Minimum number of entities required
            max_staleness_hours: Maximum staleness in hours (uses default if None)
            
        Returns:
            True if collection is healthy
        """
        try:
            state_info = await self.analyze_collection_state(collection_name)
            
            # Check basic health indicators
            if not state_info.collection_exists:
                return False
            
            if state_info.entity_metadata.total_entities < min_entities:
                return False
            
            if state_info.state in [CollectionState.INACCESSIBLE, CollectionState.INCONSISTENT]:
                return False
            
            # Check staleness
            if max_staleness_hours is None:
                max_staleness_hours = self.staleness_threshold_hours
            
            if state_info.entity_metadata.last_update_time:
                age_hours = (datetime.now() - state_info.entity_metadata.last_update_time).total_seconds() / 3600
                if age_hours > max_staleness_hours:
                    return False
            
            return state_info.health_score >= self.min_health_score
            
        except Exception as e:
            logger.error(f"Error checking collection health for {collection_name}: {e}")
            return False
    
    def clear_analysis_cache(self, collection_name: Optional[str] = None) -> None:
        """
        Clear analysis cache.
        
        Args:
            collection_name: Specific collection to clear, or None for all
        """
        if collection_name:
            self._analysis_cache.pop(collection_name, None)
            logger.debug(f"Cleared analysis cache for {collection_name}")
        else:
            self._analysis_cache.clear()
            logger.debug("Cleared all analysis cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the analysis cache."""
        total_cached = len(self._analysis_cache)
        cache_ages = []
        
        now = datetime.now()
        for state_info in self._analysis_cache.values():
            age_seconds = (now - state_info.last_analyzed).total_seconds()
            cache_ages.append(age_seconds)
        
        return {
            "cached_collections": total_cached,
            "cache_ttl_seconds": self._cache_ttl_seconds,
            "average_age_seconds": sum(cache_ages) / max(len(cache_ages), 1),
            "oldest_cache_seconds": max(cache_ages) if cache_ages else 0,
            "cache_hit_eligible": sum(1 for age in cache_ages if age < self._cache_ttl_seconds)
        }