"""
Advanced result ranking algorithms for search results.

Provides sophisticated ranking and fusion algorithms for combining
payload and semantic search results with additional relevance signals.
"""

import logging
import math
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

from ..models.storage import SearchResult

logger = logging.getLogger(__name__)


class RankingStrategy(Enum):
    """Available ranking strategies"""
    SCORE_ONLY = "score_only"
    FRESHNESS_BOOST = "freshness_boost"
    POPULARITY_BOOST = "popularity_boost"
    QUALITY_BOOST = "quality_boost"
    HYBRID_RANKING = "hybrid_ranking"


@dataclass
class RankingConfig:
    """Configuration for result ranking"""
    strategy: RankingStrategy = RankingStrategy.HYBRID_RANKING
    freshness_weight: float = 0.1
    popularity_weight: float = 0.15
    quality_weight: float = 0.2
    diversity_threshold: float = 0.8
    max_results_per_file: int = 3
    boost_main_directories: bool = True
    boost_test_coverage: bool = True
    
    def __post_init__(self):
        """Validate configuration values"""
        if not 0 <= self.freshness_weight <= 1:
            raise ValueError("freshness_weight must be between 0 and 1")
        if not 0 <= self.popularity_weight <= 1:
            raise ValueError("popularity_weight must be between 0 and 1")
        if not 0 <= self.quality_weight <= 1:
            raise ValueError("quality_weight must be between 0 and 1")


class ResultRanker:
    """
    Advanced result ranking with multiple signals and fusion algorithms.
    
    Features:
    - Multi-signal ranking (score, freshness, popularity, quality)
    - Diversity-aware result selection
    - File-based result limiting
    - Directory importance boosting
    - Test coverage awareness
    """
    
    def __init__(self, config: Optional[RankingConfig] = None):
        """
        Initialize result ranker.
        
        Args:
            config: Ranking configuration
        """
        self.config = config or RankingConfig()
        logger.info(f"Initialized ResultRanker with strategy: {self.config.strategy.value}")
    
    def rank_results(
        self,
        results: List[SearchResult],
        query: str = "",
        context: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Rank search results using configured strategy.
        
        Args:
            results: Search results to rank
            query: Original search query for context
            context: Additional ranking context
            
        Returns:
            Ranked and potentially filtered results
        """
        if not results:
            return results
        
        context = context or {}
        
        # Apply ranking strategy
        if self.config.strategy == RankingStrategy.SCORE_ONLY:
            ranked_results = self._rank_by_score_only(results)
        elif self.config.strategy == RankingStrategy.FRESHNESS_BOOST:
            ranked_results = self._rank_with_freshness_boost(results, context)
        elif self.config.strategy == RankingStrategy.POPULARITY_BOOST:
            ranked_results = self._rank_with_popularity_boost(results, context)
        elif self.config.strategy == RankingStrategy.QUALITY_BOOST:
            ranked_results = self._rank_with_quality_boost(results, context)
        else:  # HYBRID_RANKING
            ranked_results = self._rank_hybrid(results, query, context)
        
        # Apply diversity filtering
        diverse_results = self._apply_diversity_filtering(ranked_results)
        
        # Apply file-based limiting
        limited_results = self._apply_file_limiting(diverse_results)
        
        logger.debug(
            f"Ranked {len(results)} results -> {len(limited_results)} final results "
            f"using {self.config.strategy.value}"
        )
        
        return limited_results
    
    def _rank_by_score_only(self, results: List[SearchResult]) -> List[SearchResult]:
        """Rank results by search score only"""
        return sorted(results, key=lambda r: r.score, reverse=True)
    
    def _rank_with_freshness_boost(
        self,
        results: List[SearchResult],
        context: Dict[str, Any]
    ) -> List[SearchResult]:
        """Rank results with freshness boost"""
        file_timestamps = context.get("file_timestamps", {})
        
        def freshness_score(result: SearchResult) -> float:
            base_score = result.score
            file_path = result.point.payload.get("file_path", "")
            
            if file_path in file_timestamps:
                # Boost recently modified files
                # This would need actual timestamp calculation
                freshness_boost = self.config.freshness_weight * 0.1  # Placeholder
                return base_score + freshness_boost
            
            return base_score
        
        return sorted(results, key=freshness_score, reverse=True)
    
    def _rank_with_popularity_boost(
        self,
        results: List[SearchResult],
        context: Dict[str, Any]
    ) -> List[SearchResult]:
        """Rank results with popularity boost based on usage patterns"""
        popular_entities = context.get("popular_entities", set())
        popular_files = context.get("popular_files", set())
        
        def popularity_score(result: SearchResult) -> float:
            base_score = result.score
            boost = 0.0
            
            entity_id = result.point.payload.get("entity_id", "")
            file_path = result.point.payload.get("file_path", "")
            
            if entity_id in popular_entities:
                boost += self.config.popularity_weight * 0.2
            
            if file_path in popular_files:
                boost += self.config.popularity_weight * 0.1
            
            return base_score + boost
        
        return sorted(results, key=popularity_score, reverse=True)
    
    def _rank_with_quality_boost(
        self,
        results: List[SearchResult],
        context: Dict[str, Any]
    ) -> List[SearchResult]:
        """Rank results with code quality signals"""
        def quality_score(result: SearchResult) -> float:
            base_score = result.score
            boost = 0.0
            
            # Boost well-documented entities
            docstring = result.point.payload.get("docstring", "")
            if docstring and len(docstring) > 50:
                boost += self.config.quality_weight * 0.1
            
            # Boost entities with type hints
            signature = result.point.payload.get("signature", "")
            if "->" in signature or ":" in signature:  # Type hints present
                boost += self.config.quality_weight * 0.05
            
            # Boost main source directories over tests/examples
            file_path = result.point.payload.get("file_path", "")
            if self.config.boost_main_directories and self._is_main_source_file(file_path):
                boost += self.config.quality_weight * 0.1
            
            return base_score + boost
        
        return sorted(results, key=quality_score, reverse=True)
    
    def _rank_hybrid(
        self,
        results: List[SearchResult],
        query: str,
        context: Dict[str, Any]
    ) -> List[SearchResult]:
        """Advanced hybrid ranking with multiple signals"""
        def hybrid_score(result: SearchResult) -> float:
            base_score = result.score
            total_boost = 0.0
            
            # Freshness boost
            file_path = result.point.payload.get("file_path", "")
            file_timestamps = context.get("file_timestamps", {})
            if file_path in file_timestamps:
                total_boost += self.config.freshness_weight * 0.1
            
            # Popularity boost
            entity_id = result.point.payload.get("entity_id", "")
            popular_entities = context.get("popular_entities", set())
            if entity_id in popular_entities:
                total_boost += self.config.popularity_weight * 0.2
            
            # Quality boost
            docstring = result.point.payload.get("docstring", "")
            if docstring and len(docstring) > 50:
                total_boost += self.config.quality_weight * 0.1
            
            # Directory importance boost
            if self.config.boost_main_directories and self._is_main_source_file(file_path):
                total_boost += 0.05
            
            # Query relevance boost
            entity_name = result.point.payload.get("entity_name", "")
            if query.lower() in entity_name.lower():
                total_boost += 0.1
            
            return base_score + total_boost
        
        return sorted(results, key=hybrid_score, reverse=True)
    
    def _apply_diversity_filtering(self, results: List[SearchResult]) -> List[SearchResult]:
        """Apply diversity filtering to avoid too similar results"""
        if self.config.diversity_threshold >= 1.0:
            return results
        
        diverse_results = []
        seen_signatures = set()
        
        for result in results:
            signature = result.point.payload.get("signature", "")
            entity_name = result.point.payload.get("entity_name", "")
            
            # Create a diversity key
            diversity_key = f"{entity_name}:{signature[:50]}"
            
            # Check if we've seen something too similar
            is_diverse = True
            for seen_sig in seen_signatures:
                similarity = self._calculate_signature_similarity(diversity_key, seen_sig)
                if similarity > self.config.diversity_threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_results.append(result)
                seen_signatures.add(diversity_key)
        
        return diverse_results
    
    def _apply_file_limiting(self, results: List[SearchResult]) -> List[SearchResult]:
        """Limit number of results per file to ensure diversity"""
        if self.config.max_results_per_file <= 0:
            return results
        
        file_counts = {}
        limited_results = []
        
        for result in results:
            file_path = result.point.payload.get("file_path", "")
            count = file_counts.get(file_path, 0)
            
            if count < self.config.max_results_per_file:
                limited_results.append(result)
                file_counts[file_path] = count + 1
        
        return limited_results
    
    def _is_main_source_file(self, file_path: str) -> bool:
        """Check if file is in main source directories (not test/example)"""
        if not file_path:
            return False
        
        path_lower = file_path.lower()
        test_indicators = ["test", "tests", "spec", "specs", "example", "examples", "__pycache__"]
        
        return not any(indicator in path_lower for indicator in test_indicators)
    
    def _calculate_signature_similarity(self, sig1: str, sig2: str) -> float:
        """Calculate similarity between two signatures (simple Jaccard)"""
        if not sig1 or not sig2:
            return 0.0
        
        words1 = set(sig1.lower().split())
        words2 = set(sig2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def fuse_search_results(
        self,
        payload_results: List[SearchResult],
        semantic_results: List[SearchResult],
        payload_weight: float = 0.7,
        semantic_weight: float = 0.3
    ) -> List[SearchResult]:
        """
        Fuse payload and semantic search results with weighted scoring.
        
        Args:
            payload_results: Results from payload search
            semantic_results: Results from semantic search
            payload_weight: Weight for payload results
            semantic_weight: Weight for semantic results
            
        Returns:
            Fused and ranked results
        """
        # Normalize weights
        total_weight = payload_weight + semantic_weight
        if total_weight > 0:
            payload_weight /= total_weight
            semantic_weight /= total_weight
        
        # Create entity ID to result mapping
        entity_scores = {}
        all_results = {}
        
        # Process payload results
        for result in payload_results:
            entity_id = result.point.payload.get("entity_id", "")
            if entity_id:
                weighted_score = result.score * payload_weight
                entity_scores[entity_id] = entity_scores.get(entity_id, 0) + weighted_score
                all_results[entity_id] = result
        
        # Process semantic results
        for result in semantic_results:
            entity_id = result.point.payload.get("entity_id", "")
            if entity_id:
                weighted_score = result.score * semantic_weight
                entity_scores[entity_id] = entity_scores.get(entity_id, 0) + weighted_score
                if entity_id not in all_results:
                    all_results[entity_id] = result
        
        # Create fused results with combined scores
        fused_results = []
        for entity_id, combined_score in entity_scores.items():
            if entity_id in all_results:
                result = all_results[entity_id]
                # Create new result with fused score
                fused_result = SearchResult(
                    point=result.point,
                    score=combined_score,
                    query=result.query,
                    search_type="fused",
                    rank=0,  # Will be updated after sorting
                    total_results=len(entity_scores)
                )
                fused_results.append(fused_result)
        
        # Sort by combined score and update ranks
        fused_results.sort(key=lambda r: r.score, reverse=True)
        for i, result in enumerate(fused_results):
            result.rank = i + 1
        
        logger.debug(
            f"Fused {len(payload_results)} payload + {len(semantic_results)} semantic "
            f"-> {len(fused_results)} results"
        )
        
        return fused_results
    
    def explain_ranking(self, result: SearchResult, context: Dict[str, Any] = None) -> str:
        """
        Provide explanation for why a result was ranked at its position.
        
        Args:
            result: Search result to explain
            context: Ranking context
            
        Returns:
            Human-readable ranking explanation
        """
        explanations = []
        context = context or {}
        
        # Base score
        explanations.append(f"Base score: {result.score:.3f}")
        
        # Entity information
        entity_name = result.point.payload.get("entity_name", "")
        entity_type = result.point.payload.get("entity_type", "")
        if entity_name and entity_type:
            explanations.append(f"{entity_type.title()}: {entity_name}")
        
        # File information
        file_path = result.point.payload.get("file_path", "")
        if file_path:
            explanations.append(f"File: {file_path}")
            if self._is_main_source_file(file_path):
                explanations.append("✓ Main source file")
        
        # Quality signals
        docstring = result.point.payload.get("docstring", "")
        if docstring and len(docstring) > 50:
            explanations.append("✓ Well documented")
        
        signature = result.point.payload.get("signature", "")
        if "->" in signature or ":" in signature:
            explanations.append("✓ Type hints present")
        
        # Popularity signals
        entity_id = result.point.payload.get("entity_id", "")
        popular_entities = context.get("popular_entities", set())
        if entity_id in popular_entities:
            explanations.append("✓ Popular entity")
        
        return " | ".join(explanations)