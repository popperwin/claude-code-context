"""
Search functionality for hybrid Qdrant operations.

Provides search modes, result ranking, and query optimization.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass

from .client import HybridQdrantClient, SearchMode
from ..models.storage import SearchResult

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
    
    def __init__(self, client: HybridQdrantClient):
        """
        Initialize hybrid searcher.
        
        Args:
            client: Hybrid Qdrant client
        """
        self.client = client
        
        # Query analysis patterns
        self._code_patterns = [
            "def ", "class ", "function", "method", "import", 
            "variable", "constant", "async ", "await "
        ]
        self._exact_patterns = [
            "\"", "'", "exact:", "name:", "file:"
        ]
        
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
        optimized_mode = self._optimize_search_mode(query, config.mode)
        
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
        
        logger.debug(
            f"Search completed: {len(results)} results for query '{query}' "
            f"in mode '{optimized_mode}'"
        )
        
        return results
    
    def _optimize_search_mode(self, query: str, requested_mode: str) -> str:
        """
        Analyze query and optimize search mode.
        
        Args:
            query: Search query
            requested_mode: Originally requested mode
            
        Returns:
            Optimized search mode
        """
        if requested_mode != SearchMode.AUTO:
            return requested_mode
        
        query_lower = query.lower()
        
        # Check for exact match indicators
        if any(pattern in query for pattern in self._exact_patterns):
            return SearchMode.PAYLOAD_ONLY
        
        # Check for code-specific terms
        if any(pattern in query_lower for pattern in self._code_patterns):
            # Code-specific queries work well with hybrid
            return SearchMode.HYBRID
        
        # Short queries (1-2 words) work better with payload search
        words = query.split()
        if len(words) <= 2:
            return SearchMode.PAYLOAD_ONLY
        
        # Longer descriptive queries work well with semantic search
        if len(words) > 5:
            return SearchMode.SEMANTIC_ONLY
        
        # Default to hybrid for balanced results
        return SearchMode.HYBRID
    
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
        
        # Re-rank results if needed
        processed_results = self._rerank_results(processed_results)
        
        return processed_results
    
    def _rerank_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Advanced re-ranking of search results.
        
        Args:
            results: Search results to re-rank
            
        Returns:
            Re-ranked results
        """
        # For now, keep original ranking
        # Future enhancements could include:
        # - Boost results from recent files
        # - Boost results with better documentation
        # - Boost results from main source directories
        # - Apply machine learning ranking models
        
        return results
    
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
        suggestions = []
        
        # Add common code entity prefixes
        if partial_query:
            lower_query = partial_query.lower()
            
            # Function/method suggestions
            if any(prefix in lower_query for prefix in ["def", "func", "method"]):
                suggestions.extend([
                    f"{partial_query} async",
                    f"{partial_query} test",
                    f"{partial_query} private"
                ])
            
            # Class suggestions
            if "class" in lower_query:
                suggestions.extend([
                    f"{partial_query} abstract",
                    f"{partial_query} interface",
                    f"{partial_query} base"
                ])
            
            # File type suggestions
            if "file:" in lower_query or "." in partial_query:
                extensions = [".py", ".js", ".ts", ".go", ".rs", ".java"]
                for ext in extensions:
                    if not partial_query.endswith(ext):
                        suggestions.append(f"{partial_query}{ext}")
        
        return suggestions[:10]  # Limit to 10 suggestions