"""
Entity Scan Mode Selection.

Intelligent selection of entity-level scan modes based on collection state,
entity metadata, and performance characteristics for optimal indexing strategy.
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta

from .state_analyzer import CollectionStateAnalyzer, CollectionState
from ..storage.client import HybridQdrantClient

logger = logging.getLogger(__name__)


class EntityScanMode(Enum):
    """Entity-level scan modes for indexing operations."""
    FULL_RESCAN = "full_rescan"          # Complete re-indexing of all entities
    ENTITY_SYNC = "entity_sync"          # Sync changed entities only
    SYNC_ONLY = "sync_only"              # Real-time sync monitoring only
    AUTO = "auto"                        # Automatic mode selection


@dataclass
class ScanModeDecision:
    """Result of scan mode selection with reasoning."""
    selected_mode: EntityScanMode
    confidence: float  # 0.0 to 1.0
    reasoning: List[str]
    entity_count_estimate: int
    expected_duration_minutes: float
    performance_tier: str  # "fast", "medium", "slow"
    
    # Additional context
    collection_health_score: float = 0.0
    last_scan_age_hours: Optional[float] = None
    detected_issues: List[str] = None
    
    def __post_init__(self):
        if self.detected_issues is None:
            self.detected_issues = []


class EntityScanModeSelector:
    """
    Intelligent selection of entity-level scan modes.
    
    Analyzes collection state, entity metadata, and performance characteristics
    to determine the optimal scan mode for indexing operations.
    
    Features:
    - Collection health assessment for mode selection
    - Entity count and age-based recommendations
    - Performance-aware mode selection
    - Automatic fallback strategies
    - Configurable selection criteria
    """
    
    def __init__(
        self,
        storage_client: HybridQdrantClient,
        state_analyzer: Optional[CollectionStateAnalyzer] = None,
        performance_threshold_entities: int = 10000,
        staleness_threshold_hours: int = 24,
        health_threshold: float = 0.7
    ):
        """
        Initialize entity scan mode selector.
        
        Args:
            storage_client: Qdrant client for collection operations
            state_analyzer: Optional state analyzer (creates one if None)
            performance_threshold_entities: Entity count threshold for performance considerations
            staleness_threshold_hours: Hours after which collection is considered stale
            health_threshold: Minimum health score for SYNC_ONLY mode
        """
        self.storage_client = storage_client
        self.state_analyzer = state_analyzer or CollectionStateAnalyzer(
            storage_client=storage_client,
            staleness_threshold_hours=staleness_threshold_hours
        )
        self.performance_threshold_entities = performance_threshold_entities
        self.staleness_threshold_hours = staleness_threshold_hours
        self.health_threshold = health_threshold
        
        logger.info(f"Initialized EntityScanModeSelector with {performance_threshold_entities} entity threshold")
    
    async def select_scan_mode(
        self,
        collection_name: str,
        project_path: Path,
        requested_mode: str = "auto",
        force_mode: bool = False
    ) -> ScanModeDecision:
        """
        Select optimal scan mode for entity-level operations.
        
        Args:
            collection_name: Name of collection to analyze
            project_path: Project root path
            requested_mode: User-requested mode ("auto", "full_rescan", etc.)
            force_mode: Skip analysis and force the requested mode
            
        Returns:
            ScanModeDecision with selected mode and reasoning
        """
        logger.debug(f"Selecting scan mode for {collection_name} (requested: {requested_mode})")
        
        # Handle forced mode selection
        if force_mode and requested_mode != "auto":
            return self._create_forced_decision(requested_mode)
        
        # Analyze collection state
        try:
            state_info = await self.state_analyzer.analyze_collection_state(
                collection_name=collection_name,
                project_path=project_path
            )
        except Exception as e:
            logger.warning(f"Failed to analyze collection state: {e}")
            # Default to full rescan on analysis failure
            return ScanModeDecision(
                selected_mode=EntityScanMode.FULL_RESCAN,
                confidence=0.5,
                reasoning=[f"Collection analysis failed: {e}", "Defaulting to full rescan for safety"],
                entity_count_estimate=0,
                expected_duration_minutes=5.0,
                performance_tier="medium"
            )
        
        # Select mode based on analysis
        if requested_mode == "auto":
            return await self._select_automatic_mode(state_info, project_path)
        else:
            return await self._validate_requested_mode(state_info, requested_mode, project_path)
    
    async def _select_automatic_mode(
        self,
        state_info,
        project_path: Path
    ) -> ScanModeDecision:
        """Select mode automatically based on collection state analysis."""
        reasoning = []
        detected_issues = []
        
        # Decision matrix based on collection state
        if state_info.state == CollectionState.EMPTY:
            reasoning.append("Collection is empty - full rescan required")
            return self._create_decision(
                mode=EntityScanMode.FULL_RESCAN,
                confidence=1.0,
                reasoning=reasoning,
                entity_estimate=0,
                state_info=state_info
            )
        
        elif state_info.state == CollectionState.INACCESSIBLE:
            reasoning.append("Collection is inaccessible - attempting full rescan")
            detected_issues.append("Collection accessibility issues")
            return self._create_decision(
                mode=EntityScanMode.FULL_RESCAN,
                confidence=0.7,
                reasoning=reasoning,
                entity_estimate=0,
                state_info=state_info,
                issues=detected_issues
            )
        
        elif state_info.state == CollectionState.STALE:
            reasoning.append(f"Collection is stale (age: {self._get_age_description(state_info)})")
            
            # Check if very stale (needs full rescan)
            if state_info.entity_metadata.last_update_time:
                age_hours = (datetime.now() - state_info.entity_metadata.last_update_time).total_seconds() / 3600
                if age_hours > (self.staleness_threshold_hours * 3):
                    reasoning.append("Collection is very stale - full rescan recommended")
                    return self._create_decision(
                        mode=EntityScanMode.FULL_RESCAN,
                        confidence=0.9,
                        reasoning=reasoning,
                        entity_estimate=state_info.entity_metadata.total_entities,
                        state_info=state_info
                    )
            
            reasoning.append("Entity sync will update stale entities")
            return self._create_decision(
                mode=EntityScanMode.ENTITY_SYNC,
                confidence=0.8,
                reasoning=reasoning,
                entity_estimate=state_info.entity_metadata.total_entities,
                state_info=state_info
            )
        
        elif state_info.state == CollectionState.INCONSISTENT:
            reasoning.append("Collection has consistency issues")
            detected_issues.extend(state_info.issues_found)
            reasoning.append("Full rescan will resolve inconsistencies")
            return self._create_decision(
                mode=EntityScanMode.FULL_RESCAN,
                confidence=0.85,
                reasoning=reasoning,
                entity_estimate=state_info.entity_metadata.total_entities,
                state_info=state_info,
                issues=detected_issues
            )
        
        elif state_info.state == CollectionState.HEALTHY:
            reasoning.append(f"Collection is healthy (score: {state_info.health_score:.2f})")
            
            # Check if collection is very healthy and recent
            if (state_info.health_score >= self.health_threshold and 
                state_info.entity_metadata.last_update_time and
                (datetime.now() - state_info.entity_metadata.last_update_time).total_seconds() / 3600 < 1):
                
                reasoning.append("Collection is very healthy and recent - sync monitoring only")
                return self._create_decision(
                    mode=EntityScanMode.SYNC_ONLY,
                    confidence=0.9,
                    reasoning=reasoning,
                    entity_estimate=state_info.entity_metadata.total_entities,
                    state_info=state_info
                )
            else:
                reasoning.append("Entity sync will maintain healthy state")
                return self._create_decision(
                    mode=EntityScanMode.ENTITY_SYNC,
                    confidence=0.8,
                    reasoning=reasoning,
                    entity_estimate=state_info.entity_metadata.total_entities,
                    state_info=state_info
                )
        
        # Fallback to entity sync
        reasoning.append("Unknown state - defaulting to entity sync")
        return self._create_decision(
            mode=EntityScanMode.ENTITY_SYNC,
            confidence=0.6,
            reasoning=reasoning,
            entity_estimate=state_info.entity_metadata.total_entities,
            state_info=state_info
        )
    
    async def _validate_requested_mode(
        self,
        state_info,
        requested_mode: str,
        project_path: Path
    ) -> ScanModeDecision:
        """Validate and potentially override user-requested mode."""
        reasoning = []
        mode_enum = EntityScanMode(requested_mode)
        confidence = 0.8  # Default confidence for user-requested modes
        
        # Validation logic
        if mode_enum == EntityScanMode.SYNC_ONLY:
            if state_info.state == CollectionState.EMPTY:
                reasoning.append("Cannot use sync-only mode on empty collection")
                reasoning.append("Overriding to full rescan")
                return self._create_decision(
                    mode=EntityScanMode.FULL_RESCAN,
                    confidence=0.9,
                    reasoning=reasoning,
                    entity_estimate=0,
                    state_info=state_info
                )
            elif state_info.health_score < self.health_threshold:
                reasoning.append(f"Collection health too low for sync-only mode (score: {state_info.health_score:.2f})")
                reasoning.append("Recommending entity sync instead")
                return self._create_decision(
                    mode=EntityScanMode.ENTITY_SYNC,
                    confidence=0.8,
                    reasoning=reasoning,
                    entity_estimate=state_info.entity_metadata.total_entities,
                    state_info=state_info
                )
        
        elif mode_enum == EntityScanMode.ENTITY_SYNC:
            if state_info.state == CollectionState.EMPTY:
                reasoning.append("Cannot use entity sync on empty collection")
                reasoning.append("Overriding to full rescan")
                return self._create_decision(
                    mode=EntityScanMode.FULL_RESCAN,
                    confidence=0.9,
                    reasoning=reasoning,
                    entity_estimate=0,
                    state_info=state_info
                )
        
        # User-requested mode is valid
        reasoning.append(f"User-requested mode '{requested_mode}' is appropriate")
        return self._create_decision(
            mode=mode_enum,
            confidence=confidence,
            reasoning=reasoning,
            entity_estimate=state_info.entity_metadata.total_entities,
            state_info=state_info
        )
    
    def _create_decision(
        self,
        mode: EntityScanMode,
        confidence: float,
        reasoning: List[str],
        entity_estimate: int,
        state_info,
        issues: Optional[List[str]] = None
    ) -> ScanModeDecision:
        """Create a scan mode decision with performance estimates."""
        
        # Estimate duration based on mode and entity count
        if mode == EntityScanMode.FULL_RESCAN:
            # Estimate 2000 entities per minute for full parsing + embedding
            duration_minutes = max(1.0, entity_estimate / 2000.0)
            performance_tier = "slow" if entity_estimate > self.performance_threshold_entities else "medium"
        elif mode == EntityScanMode.ENTITY_SYNC:
            # Estimate faster sync operations (only changed entities)
            estimated_changed = max(1, entity_estimate // 10)  # Assume 10% change rate
            duration_minutes = max(0.5, estimated_changed / 5000.0)
            performance_tier = "medium" if estimated_changed > 1000 else "fast"
        else:  # SYNC_ONLY
            duration_minutes = 0.1  # Just setup time
            performance_tier = "fast"
        
        # Calculate last scan age if available
        last_scan_age_hours = None
        if state_info.entity_metadata.last_update_time:
            last_scan_age_hours = (datetime.now() - state_info.entity_metadata.last_update_time).total_seconds() / 3600
        
        return ScanModeDecision(
            selected_mode=mode,
            confidence=confidence,
            reasoning=reasoning,
            entity_count_estimate=entity_estimate,
            expected_duration_minutes=duration_minutes,
            performance_tier=performance_tier,
            collection_health_score=state_info.health_score,
            last_scan_age_hours=last_scan_age_hours,
            detected_issues=issues or []
        )
    
    def _create_forced_decision(self, requested_mode: str) -> ScanModeDecision:
        """Create decision for forced mode selection."""
        mode_enum = EntityScanMode(requested_mode)
        
        return ScanModeDecision(
            selected_mode=mode_enum,
            confidence=0.5,  # Lower confidence for forced modes
            reasoning=[f"Mode '{requested_mode}' forced by user", "Skipping collection analysis"],
            entity_count_estimate=0,
            expected_duration_minutes=1.0,
            performance_tier="unknown"
        )
    
    def _get_age_description(self, state_info) -> str:
        """Get human-readable description of collection age."""
        if not state_info.entity_metadata.last_update_time:
            return "unknown"
        
        age_hours = (datetime.now() - state_info.entity_metadata.last_update_time).total_seconds() / 3600
        
        if age_hours < 1:
            return f"{int(age_hours * 60)}m"
        elif age_hours < 24:
            return f"{int(age_hours)}h"
        else:
            return f"{int(age_hours / 24)}d"
    
    def get_mode_recommendations(
        self,
        collection_name: str,
        performance_priority: str = "balanced"  # "speed", "accuracy", "balanced"
    ) -> Dict[str, Any]:
        """
        Get general recommendations for scan mode selection.
        
        Args:
            collection_name: Collection name for context
            performance_priority: Priority for recommendations
            
        Returns:
            Dictionary with mode recommendations
        """
        recommendations = {
            "performance_priority": performance_priority,
            "collection_name": collection_name,
            "mode_descriptions": {
                "full_rescan": {
                    "description": "Complete re-indexing of all entities",
                    "use_cases": ["Empty collections", "Major inconsistencies", "Schema changes"],
                    "performance": "Slow but thorough",
                    "accuracy": "Highest"
                },
                "entity_sync": {
                    "description": "Sync changed entities only",
                    "use_cases": ["Regular updates", "Stale collections", "Partial changes"],
                    "performance": "Medium",
                    "accuracy": "High"
                },
                "sync_only": {
                    "description": "Real-time sync monitoring only",
                    "use_cases": ["Healthy collections", "Real-time workflows", "Maintenance mode"],
                    "performance": "Fastest",
                    "accuracy": "Depends on collection health"
                }
            }
        }
        
        if performance_priority == "speed":
            recommendations["recommended_order"] = ["sync_only", "entity_sync", "full_rescan"]
            recommendations["note"] = "Prioritizing speed over completeness"
        elif performance_priority == "accuracy":
            recommendations["recommended_order"] = ["full_rescan", "entity_sync", "sync_only"]
            recommendations["note"] = "Prioritizing accuracy over speed"
        else:  # balanced
            recommendations["recommended_order"] = ["entity_sync", "sync_only", "full_rescan"]
            recommendations["note"] = "Balanced approach for most use cases"
        
        return recommendations
    
    def get_status(self) -> Dict[str, Any]:
        """Get status information about the scan mode selector."""
        return {
            "performance_threshold_entities": self.performance_threshold_entities,
            "staleness_threshold_hours": self.staleness_threshold_hours,
            "health_threshold": self.health_threshold,
            "state_analyzer_configured": self.state_analyzer is not None,
            "supported_modes": [mode.value for mode in EntityScanMode]
        }