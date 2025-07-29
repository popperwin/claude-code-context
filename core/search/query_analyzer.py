"""
Query analysis and classification for intelligent search mode selection.

Provides query pattern recognition and search mode optimization.
"""

import logging
from typing import List, Optional
from dataclasses import dataclass

from .suggestions import QuerySuggestionEngine

logger = logging.getLogger(__name__)


@dataclass
class QueryAnalysis:
    """Result of query analysis"""
    original_query: str
    cleaned_query: str
    word_count: int
    has_exact_patterns: bool
    has_code_patterns: bool
    detected_patterns: List[str]
    confidence: float
    recommended_mode: str


class QueryAnalyzer:
    """
    Analyzes search queries to determine optimal search modes and patterns.
    
    Features:
    - Pattern recognition for exact matches, code terms, and natural language
    - Query length analysis
    - Search mode recommendations
    - Query cleaning and normalization
    """
    
    def __init__(self):
        """Initialize query analyzer with pattern definitions"""
        
        self.suggestion_engine = QuerySuggestionEngine()
        
        # Code-specific patterns that suggest hybrid search works well
        self._code_patterns = [
            "def ", "class ", "function", "method", "import", 
            "variable", "constant", "async ", "await ", "return",
            "if ", "for ", "while ", "try ", "except ", "finally ",
            "self.", "this.", "__init__", "__main__", "lambda ",
            "yield ", "with statement", "assert ", "raise ", "pass",
            "break", "continue", "global ", "nonlocal "
        ]
        
        # Exact match indicators that suggest payload-only search
        self._exact_patterns = [
            "\"", "'", "`", "exact:", "name:", "file:", "id:",
            "uuid:", "hash:", "path:", "line:", "column:"
        ]
        
        # File and location patterns
        self._file_patterns = [
            "file:", "path:", ".py", ".js", ".ts", ".go", ".rs",
            ".java", ".cpp", ".c", ".h", ".hpp", ".cs", ".rb",
            ".php", ".swift", ".kt", ".scala"
        ]
        
        # Natural language indicators that suggest semantic search
        self._semantic_patterns = [
            "how to", "what is", "why does", "when should", "where can",
            "which way", "explain", "describe", "find code that", "find",
            "show me", "help me", "i need", "looking for",
            "similar", "related to", "like", "resembles"
        ]
        
        logger.info("Initialized QueryAnalyzer")
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Perform comprehensive query analysis.
        
        Args:
            query: Search query to analyze
            
        Returns:
            QueryAnalysis with detected patterns and recommendations
        """
        if not query or not query.strip():
            return QueryAnalysis(
                original_query=query,
                cleaned_query="",
                word_count=0,
                has_exact_patterns=False,
                has_code_patterns=False,
                detected_patterns=[],
                confidence=0.0,
                recommended_mode="hybrid"
            )
        
        cleaned_query = self._clean_query(query)
        query_lower = cleaned_query.lower()
        words = cleaned_query.split()
        word_count = len(words)
        
        # Detect patterns
        detected_patterns = []
        has_exact_patterns = self._has_exact_patterns(query, detected_patterns)
        has_code_patterns = self._has_code_patterns(query_lower, detected_patterns)
        has_file_patterns = self._has_file_patterns(query_lower, detected_patterns)
        has_semantic_patterns = self._has_semantic_patterns(query_lower, detected_patterns)
        
        # Determine recommended mode and confidence
        recommended_mode, confidence = self._recommend_search_mode(
            query_lower, word_count, has_exact_patterns, has_code_patterns,
            has_file_patterns, has_semantic_patterns
        )
        
        return QueryAnalysis(
            original_query=query,
            cleaned_query=cleaned_query,
            word_count=word_count,
            has_exact_patterns=has_exact_patterns,
            has_code_patterns=has_code_patterns,
            detected_patterns=detected_patterns,
            confidence=confidence,
            recommended_mode=recommended_mode
        )
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize query string"""
        # Remove extra whitespace
        cleaned = " ".join(query.strip().split())
        
        # TODO: Add more cleaning logic:
        # - Remove special characters that don't affect search
        # - Normalize quotes
        # - Handle escaped characters
        
        return cleaned
    
    def _has_exact_patterns(self, query: str, detected_patterns: List[str]) -> bool:
        """Check for exact match patterns"""
        found = False
        for pattern in self._exact_patterns:
            if pattern in query:
                detected_patterns.append(f"exact:{pattern}")
                found = True
        return found
    
    def _has_code_patterns(self, query_lower: str, detected_patterns: List[str]) -> bool:
        """Check for code-specific patterns"""
        found = False
        for pattern in self._code_patterns:
            if pattern in query_lower:
                detected_patterns.append(f"code:{pattern.strip()}")
                found = True
        return found
    
    def _has_file_patterns(self, query_lower: str, detected_patterns: List[str]) -> bool:
        """Check for file/path patterns"""
        found = False
        for pattern in self._file_patterns:
            if pattern in query_lower:
                detected_patterns.append(f"file:{pattern}")
                found = True
        return found
    
    def _has_semantic_patterns(self, query_lower: str, detected_patterns: List[str]) -> bool:
        """Check for natural language/semantic patterns"""
        found = False
        for pattern in self._semantic_patterns:
            if pattern in query_lower:
                detected_patterns.append(f"semantic:{pattern}")
                found = True
        return found
    
    def _recommend_search_mode(
        self,
        query_lower: str,
        word_count: int,
        has_exact_patterns: bool,
        has_code_patterns: bool,
        has_file_patterns: bool,
        has_semantic_patterns: bool
    ) -> tuple[str, float]:
        """
        Recommend search mode based on query analysis.
        
        Returns:
            Tuple of (recommended_mode, confidence_score)
        """
        # High confidence exact match scenarios
        if has_exact_patterns:
            return "payload", 0.9
        
        # High confidence semantic scenarios
        if has_semantic_patterns and word_count >= 4:
            return "semantic", 0.8
        
        # Code-specific queries work well with hybrid
        if has_code_patterns:
            if word_count <= 2:
                return "payload", 0.7  # Very short code queries
            else:
                return "hybrid", 0.8   # Longer code descriptions
        
        # File/path queries usually work best with payload
        if has_file_patterns:
            return "payload", 0.7
        
        # Word count-based heuristics (lower confidence)
        if word_count <= 2:
            # Check if query contains only special characters (ambiguous case)
            if query_lower and all(not c.isalnum() and not c.isspace() for c in query_lower):
                return "hybrid", 0.4  # Special characters are ambiguous
            return "payload", 0.6  # Short queries
        elif word_count >= 6:
            return "semantic", 0.6  # Long descriptive queries
        else:
            return "hybrid", 0.5    # Medium queries - balanced approach
    
    def get_query_suggestions(self, partial_query: str) -> List[str]:
        """
        Generate query completion suggestions using advanced suggestion engine.
        
        Args:
            partial_query: Partial query string
            
        Returns:
            List of suggested query completions
        """
        if not partial_query:
            return []
        
        # Use the advanced suggestion engine
        suggestions = self.suggestion_engine.get_suggestions(partial_query, limit=10)
        
        # Extract just the text from SearchSuggestion objects
        return [suggestion.text for suggestion in suggestions]
    
    def explain_mode_choice(self, analysis: QueryAnalysis) -> str:
        """
        Provide human-readable explanation for mode recommendation.
        
        Args:
            analysis: Query analysis result
            
        Returns:
            Explanation string
        """
        mode = analysis.recommended_mode
        patterns = analysis.detected_patterns
        
        if mode == "payload":
            if analysis.has_exact_patterns:
                return f"Exact search recommended due to exact match patterns: {patterns}"
            elif analysis.word_count <= 2:
                return f"Keyword search recommended for short {analysis.word_count}-word query"
            else:
                return f"Keyword search recommended due to file/code patterns: {patterns}"
        
        elif mode == "semantic":
            # Check if any patterns are semantic patterns
            semantic_patterns = [p for p in patterns if p.startswith("semantic:")]
            if semantic_patterns:
                return f"Semantic search recommended due to natural language patterns: {patterns}"
            else:
                return f"Semantic search recommended for long {analysis.word_count}-word descriptive query"
        
        else:  # hybrid
            if analysis.has_code_patterns:
                return f"Hybrid search recommended for code query with patterns: {patterns}"
            else:
                return f"Hybrid search recommended for balanced {analysis.word_count}-word query"