"""
Search functionality for hybrid Qdrant operations.

Provides search modes, result ranking, and query optimization.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass

from ..storage.client import HybridQdrantClient, SearchMode
from ..models.storage import SearchResult
from .query_analyzer import QueryAnalyzer
from .ranking import ResultRanker, RankingConfig

logger = logging.getLogger(__name__)


class SearchMode:
    """Search mode constants"""
    PAYLOAD_ONLY = "payload"
    SEMANTIC_ONLY = "semantic"
    HYBRID = "hybrid"
    AUTO = "auto"  # Automatically choose best mode


@dataclass
class SearchConfig:
    """Configuration for search operations"""
    mode: str = SearchMode.HYBRID
    limit: int = 50
    payload_weight: float = 0.8
    semantic_weight: float = 0.2
    min_score_threshold: float = 0.0
    include_file_types: List[str] = None
    exclude_file_types: List[str] = None
    include_entity_types: List[str] = None
    exclude_entity_types: List[str] = None
    
    def __post_init__(self):
        # Normalize weights
        total_weight = self.payload_weight + self.semantic_weight
        if total_weight > 0:
            self.payload_weight /= total_weight
            self.semantic_weight /= total_weight
        
        # Initialize lists
        if self.include_file_types is None:
            self.include_file_types = []
        if self.exclude_file_types is None:
            self.exclude_file_types = []
        if self.include_entity_types is None:
            self.include_entity_types = []
        if self.exclude_entity_types is None:
            self.exclude_entity_types = []


class HybridSearcher:
    """
    Advanced searcher with query optimization and result ranking.
    
    Features:
    - Intelligent query analysis and mode selection
    - Advanced result ranking and fusion
    - Query expansion and refinement
    - Performance optimization
    """
    
    def __init__(self, client: HybridQdrantClient, ranking_config: Optional[RankingConfig] = None):
        """
        Initialize hybrid searcher.
        
        Args:
            client: Hybrid Qdrant client
            ranking_config: Configuration for result ranking
        """
        self.client = client
        self.query_analyzer = QueryAnalyzer()
        self.ranker = ResultRanker(ranking_config)
        
        logger.info("Initialized HybridSearcher")
    
    async def search(
        self,
        collection_name: str,
        query: str,
        config: Optional[SearchConfig] = None
    ) -> List[SearchResult]:
        """
        Perform hybrid search with intelligent mode selection.
        
        Args:
            collection_name: Collection to search
            query: Search query
            config: Search configuration
            
        Returns:
            List of ranked search results
        """
        if config is None:
            config = SearchConfig()
        
        # Analyze query and optimize search mode
        if config.mode == SearchMode.AUTO:
            analysis = self.query_analyzer.analyze_query(query)
            optimized_mode = analysis.recommended_mode
            logger.debug(f"Query analysis: {analysis.detected_patterns} -> {optimized_mode}")
        else:
            optimized_mode = config.mode
        
        # Build filters from config
        filters = self._build_filters(config)
        
        # Perform search based on mode
        if optimized_mode == SearchMode.PAYLOAD_ONLY:
            results = await self.client.search_payload(
                collection_name, query, config.limit, filters
            )
        elif optimized_mode == SearchMode.SEMANTIC_ONLY:
            results = await self.client.search_semantic(
                collection_name, query, config.limit, filters
            )
        elif optimized_mode == SearchMode.HYBRID:
            results = await self.client.search_hybrid(
                collection_name, query, config.limit,
                config.payload_weight, config.semantic_weight, filters
            )
        else:
            # Auto mode - use hybrid as default
            results = await self.client.search_hybrid(
                collection_name, query, config.limit,
                config.payload_weight, config.semantic_weight, filters
            )
        
        # Apply post-processing
        results = self._post_process_results(results, config)
        
        # Apply advanced ranking
        ranked_results = self.ranker.rank_results(results, query)
        
        logger.debug(
            f"Search completed: {len(ranked_results)} results for query '{query}' "
            f"in mode '{optimized_mode}'"
        )
        
        return ranked_results
    
    
    def _build_filters(self, config: SearchConfig) -> Dict[str, Any]:
        """Build filters from search configuration"""
        filters = {}
        
        # File type filters
        if config.include_file_types:
            # Extract file extensions and build language filter
            languages = []
            for file_type in config.include_file_types:
                lang = self._file_type_to_language(file_type)
                if lang:
                    languages.append(lang)
            if languages:
                filters["language"] = languages
        
        # Entity type filters
        if config.include_entity_types:
            filters["entity_type"] = config.include_entity_types
        
        # Add exclude filters (would need more complex filter logic)
        # For now, just include positive filters
        
        return filters
    
    def _file_type_to_language(self, file_type: str) -> Optional[str]:
        """Convert file type to language identifier"""
        # Remove leading dot if present
        if file_type.startswith('.'):
            file_type = file_type[1:]
        
        type_map = {
            'py': 'python',
            'js': 'javascript',
            'ts': 'typescript',
            'jsx': 'javascript',
            'tsx': 'typescript',
            'go': 'go',
            'rs': 'rust',
            'java': 'java',
            'cpp': 'cpp',
            'c': 'c',
            'h': 'c',
            'hpp': 'cpp',
            'cs': 'csharp',
            'rb': 'ruby',
            'php': 'php',
            'swift': 'swift',
            'kt': 'kotlin',
            'scala': 'scala'
        }
        
        return type_map.get(file_type.lower())
    
    def _post_process_results(
        self,
        results: List[SearchResult],
        config: SearchConfig
    ) -> List[SearchResult]:
        """
        Post-process search results with filtering and ranking.
        
        Args:
            results: Raw search results
            config: Search configuration
            
        Returns:
            Processed and filtered results
        """
        processed_results = []
        
        for result in results:
            # Apply score threshold
            if result.score < config.min_score_threshold:
                continue
            
            # Apply file type exclusions
            if config.exclude_file_types:
                file_path = result.point.payload.get("file_path", "")
                if any(file_path.endswith(f".{ft.lstrip('.')}") 
                       for ft in config.exclude_file_types):
                    continue
            
            # Apply entity type exclusions
            if config.exclude_entity_types:
                entity_type = result.point.payload.get("entity_type", "")
                if entity_type in config.exclude_entity_types:
                    continue
            
            processed_results.append(result)
        
        return processed_results
    
    async def search_similar(
        self,
        collection_name: str,
        reference_entity_id: str,
        limit: int = 10
    ) -> List[SearchResult]:
        """
        Find entities similar to a reference entity.
        
        Args:
            collection_name: Collection to search
            reference_entity_id: ID of reference entity
            limit: Maximum results to return
            
        Returns:
            List of similar entities
        """
        try:
            # Get reference entity by ID
            # This would require implementing point retrieval in client
            # For now, return empty list
            logger.warning("Similar entity search not yet implemented")
            return []
            
        except Exception as e:
            logger.error(f"Similar search failed: {e}")
            return []
    
    async def search_by_location(
        self,
        collection_name: str,
        file_path: str,
        line_range: Optional[Tuple[int, int]] = None,
        limit: int = 50
    ) -> List[SearchResult]:
        """
        Search entities by file location.
        
        Args:
            collection_name: Collection to search
            file_path: File path to search in
            line_range: Optional line range (start, end)
            limit: Maximum results to return
            
        Returns:
            List of entities in the specified location
        """
        try:
            # Build location filters
            filters = {"file_path": file_path}
            
            if line_range:
                start_line, end_line = line_range
                # This would require range filters in the client
                # For now, just use file path filter
                pass
            
            # Use payload search for location-based queries
            results = await self.client.search_payload(
                collection_name, f"file:{file_path}", limit, filters
            )
            
            # Filter by line range if specified
            if line_range:
                start_line, end_line = line_range
                filtered_results = []
                
                for result in results:
                    entity_start = result.point.payload.get("start_line", 0)
                    entity_end = result.point.payload.get("end_line", 0)
                    
                    # Check if entity overlaps with requested range
                    if (entity_start <= end_line and entity_end >= start_line):
                        filtered_results.append(result)
                
                results = filtered_results
            
            return results
            
        except Exception as e:
            logger.error(f"Location search failed: {e}")
            return []
    
    async def search_by_signature(
        self,
        collection_name: str,
        signature_pattern: str,
        limit: int = 20
    ) -> List[SearchResult]:
        """
        Search entities by function/method signature.
        
        Args:
            collection_name: Collection to search
            signature_pattern: Signature pattern to match
            limit: Maximum results to return
            
        Returns:
            List of entities with matching signatures
        """
        try:
            # Use payload search focusing on signature field
            filters = {}
            
            results = await self.client.search_payload(
                collection_name, f"signature:{signature_pattern}", limit, filters
            )
            
            # Additional filtering based on signature content
            filtered_results = []
            for result in results:
                signature = result.point.payload.get("signature", "")
                if signature and signature_pattern.lower() in signature.lower():
                    filtered_results.append(result)
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Signature search failed: {e}")
            return []
    
    def get_search_suggestions(self, partial_query: str) -> List[str]:
        """
        Get search suggestions for partial query.
        
        Args:
            partial_query: Partial search query
            
        Returns:
            List of suggested completions
        """
        return self.query_analyzer.get_query_suggestions(partial_query)
    
    def fuse_search_results(
        self,
        payload_results: List[SearchResult],
        semantic_results: List[SearchResult],
        payload_weight: float = 0.7,
        semantic_weight: float = 0.3
    ) -> List[SearchResult]:
        """
        Fuse payload and semantic search results.
        
        Args:
            payload_results: Results from payload search
            semantic_results: Results from semantic search
            payload_weight: Weight for payload results
            semantic_weight: Weight for semantic results
            
        Returns:
            Fused and ranked results
        """
        return self.ranker.fuse_search_results(
            payload_results, semantic_results, payload_weight, semantic_weight
        )