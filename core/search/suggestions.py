"""
Query suggestion system for enhanced search experience.

Provides intelligent query completions, refinements, and search hints
based on query patterns, context, and user behavior.
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SuggestionType(Enum):
    """Types of search suggestions"""
    COMPLETION = "completion"
    REFINEMENT = "refinement"
    ALTERNATIVE = "alternative"
    CONTEXT = "context"


@dataclass
class SearchSuggestion:
    """A single search suggestion"""
    text: str
    type: SuggestionType
    confidence: float
    explanation: str = ""
    
    def __post_init__(self):
        """Validate suggestion data"""
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")


class QuerySuggestionEngine:
    """
    Advanced query suggestion engine with pattern-based completions.
    
    Features:
    - Pattern-based query completions
    - Context-aware suggestions
    - Query refinement hints
    - Alternative search strategies
    """
    
    def __init__(self):
        """Initialize suggestion engine with pattern libraries"""
        self._function_patterns = [
            "async {query}",
            "def {query}",
            "private {query}",
            "public {query}",
            "static {query}",
            "{query} method",
            "{query} function",
            "test {query}",
            "{query} handler",
            "{query} validator"
        ]
        
        self._class_patterns = [
            "abstract {query}",
            "interface {query}",
            "base {query}",
            "{query} class",
            "{query} model",
            "{query} service",
            "{query} controller",
            "{query} manager",
            "test {query}",
            "{query} exception"
        ]
        
        self._file_patterns = [
            "{query}.py",
            "{query}.js",
            "{query}.ts",
            "{query}.go",
            "{query}.rs",
            "{query}.java",
            "{query}.cpp",
            "{query}.h",
            "test_{query}",
            "{query}_test"
        ]
        
        self._semantic_patterns = [
            "how to {query}",
            "what is {query}",
            "find {query} that handles",
            "show me {query} examples",
            "explain {query} pattern",
            "{query} best practices",
            "{query} implementation",
            "debug {query} issues",
            "{query} error handling",
            "{query} performance"
        ]
        
        self._context_patterns = [
            "{query} in this file",
            "{query} similar to",
            "{query} with examples",
            "{query} and related",
            "{query} dependencies",
            "{query} usage patterns",
            "{query} documentation",
            "{query} test coverage",
            "{query} code review",
            "{query} refactoring"
        ]
        
        logger.info("Initialized QuerySuggestionEngine")
    
    def get_suggestions(
        self,
        partial_query: str,
        limit: int = 10,
        context: Optional[Dict[str, Any]] = None
    ) -> List[SearchSuggestion]:
        """
        Generate search suggestions for partial query.
        
        Args:
            partial_query: Partial search query
            limit: Maximum number of suggestions
            context: Additional context for suggestions
            
        Returns:
            List of search suggestions
        """
        if not partial_query or not partial_query.strip():
            return []
        
        context = context or {}
        suggestions = []
        
        # Generate different types of suggestions
        completions = self._generate_completions(partial_query, context)
        refinements = self._generate_refinements(partial_query, context)
        alternatives = self._generate_alternatives(partial_query, context)
        context_suggestions = self._generate_context_suggestions(partial_query, context)
        
        # Combine all suggestions
        all_suggestions = completions + refinements + alternatives + context_suggestions
        
        # Remove duplicates and sort by confidence
        unique_suggestions = self._deduplicate_suggestions(all_suggestions)
        sorted_suggestions = sorted(unique_suggestions, key=lambda s: s.confidence, reverse=True)
        
        logger.debug(f"Generated {len(sorted_suggestions)} suggestions for '{partial_query}'")
        
        return sorted_suggestions[:limit]
    
    def _generate_completions(
        self,
        partial_query: str,
        context: Dict[str, Any]
    ) -> List[SearchSuggestion]:
        """Generate query completions based on patterns"""
        suggestions = []
        query_lower = partial_query.lower().strip()
        words = query_lower.split()
        
        # Always suggest function patterns for reasonable queries
        # More flexible trigger conditions
        should_suggest_functions = (
            any(pattern in query_lower for pattern in ["def", "function", "func", "method"]) or
            len(words) <= 3  # For short queries, suggest function patterns
        )
        
        if should_suggest_functions:
            # Select most relevant function patterns
            selected_patterns = []
            if "async" not in query_lower:
                selected_patterns.append("async {query}")
            if "def" not in query_lower:
                selected_patterns.append("def {query}")
            if "test" not in query_lower and len(words) <= 2:
                selected_patterns.append("test {query}")
            selected_patterns.extend(["{query} method", "{query} function"])
            
            for pattern in selected_patterns[:5]:  # Limit to avoid too many suggestions
                completed = pattern.replace("{query}", partial_query)
                if completed != partial_query:
                    suggestions.append(SearchSuggestion(
                        text=completed,
                        type=SuggestionType.COMPLETION,
                        confidence=0.8,
                        explanation="Function pattern completion"
                    ))
        
        # Always suggest class patterns for reasonable queries
        should_suggest_classes = (
            any(pattern in query_lower for pattern in ["class", "interface", "type", "model"]) or
            len(words) <= 3  # For short queries, suggest class patterns
        )
        
        if should_suggest_classes:
            # Select most relevant class patterns
            selected_patterns = []
            if "class" not in query_lower:
                selected_patterns.append("{query} class")
            if "interface" not in query_lower:
                selected_patterns.append("interface {query}")
            selected_patterns.extend(["{query} model", "{query} service"])
            
            for pattern in selected_patterns[:4]:  # Limit to avoid too many suggestions
                completed = pattern.replace("{query}", partial_query)
                if completed != partial_query:
                    suggestions.append(SearchSuggestion(
                        text=completed,
                        type=SuggestionType.COMPLETION,
                        confidence=0.7,
                        explanation="Class pattern completion"
                    ))
        
        # File pattern completions - more flexible triggers
        should_suggest_files = (
            any(pattern in query_lower for pattern in ["file", "path", "."]) or
            "." not in partial_query  # If no extension, suggest file patterns
        )
        
        if should_suggest_files and len(words) <= 2:
            # Select most common file patterns
            selected_patterns = ["{query}.py", "{query}.js", "{query}.ts"]
            for pattern in selected_patterns:
                completed = pattern.replace("{query}", partial_query)
                if completed != partial_query:
                    suggestions.append(SearchSuggestion(
                        text=completed,
                        type=SuggestionType.COMPLETION,
                        confidence=0.6,
                        explanation="File pattern completion"
                    ))
        
        # Semantic pattern completions - trigger more often
        should_suggest_semantic = (
            any(pattern in query_lower for pattern in ["how", "what", "find", "show", "explain"]) or
            len(words) >= 2  # For multi-word queries, suggest semantic patterns
        )
        
        if should_suggest_semantic:
            # Select most relevant semantic patterns
            selected_patterns = []
            if not query_lower.startswith(("how", "what", "find", "show")):
                selected_patterns.extend(["how to {query}", "find {query}", "show me {query}"])
            selected_patterns.extend(["{query} examples", "{query} implementation"])
            
            for pattern in selected_patterns[:4]:  # Limit to avoid too many suggestions
                completed = pattern.replace("{query}", partial_query)
                if completed != partial_query:
                    suggestions.append(SearchSuggestion(
                        text=completed,
                        type=SuggestionType.COMPLETION,
                        confidence=0.9,
                        explanation="Semantic query completion"
                    ))
        
        return suggestions
    
    def _generate_refinements(
        self,
        partial_query: str,
        context: Dict[str, Any]
    ) -> List[SearchSuggestion]:
        """Generate query refinements to improve search results"""
        suggestions = []
        query_lower = partial_query.lower().strip()
        words = query_lower.split()
        
        # Add type hints for better specificity
        if len(words) >= 2 and not any(hint in query_lower for hint in ["async", "def", "class"]):
            suggestions.append(SearchSuggestion(
                text=f"def {partial_query}",
                type=SuggestionType.REFINEMENT,
                confidence=0.6,
                explanation="Add function context for better results"
            ))
            
            suggestions.append(SearchSuggestion(
                text=f"class {partial_query}",
                type=SuggestionType.REFINEMENT,
                confidence=0.6,
                explanation="Add class context for better results"
            ))
        
        # Add file type hints
        if not any(ext in query_lower for ext in [".py", ".js", ".ts", ".go", ".rs"]):
            file_types = [".py", ".js", ".ts", ".go"]
            for file_type in file_types:
                suggestions.append(SearchSuggestion(
                    text=f"{partial_query} {file_type}",
                    type=SuggestionType.REFINEMENT,
                    confidence=0.5,
                    explanation=f"Search in {file_type} files specifically"
                ))
        
        # Add semantic refinements for short queries
        if len(words) <= 2:
            semantic_additions = ["examples", "implementation", "usage", "documentation"]
            for addition in semantic_additions:
                suggestions.append(SearchSuggestion(
                    text=f"{partial_query} {addition}",
                    type=SuggestionType.REFINEMENT,
                    confidence=0.7,
                    explanation=f"Find {addition} related to your query"
                ))
        
        return suggestions
    
    def _generate_alternatives(
        self,
        partial_query: str,
        context: Dict[str, Any]
    ) -> List[SearchSuggestion]:
        """Generate alternative search strategies"""
        suggestions = []
        query_lower = partial_query.lower().strip()
        
        # If query looks like exact search, suggest semantic alternative
        if any(char in partial_query for char in ['"', "'", ":"]):
            semantic_query = partial_query.replace('"', '').replace("'", "")
            if ":" in semantic_query:
                semantic_query = semantic_query.split(":", 1)[1]
            
            suggestions.append(SearchSuggestion(
                text=f"how to use {semantic_query}",
                type=SuggestionType.ALTERNATIVE,
                confidence=0.8,
                explanation="Try semantic search for better understanding"
            ))
        
        # If query looks semantic, suggest exact alternative
        elif len(partial_query.split()) >= 4:
            key_words = [word for word in partial_query.split() 
                        if len(word) > 3 and word.lower() not in ["with", "that", "this", "have"]]
            if key_words:
                suggestions.append(SearchSuggestion(
                    text=f'"{key_words[0]}"',
                    type=SuggestionType.ALTERNATIVE,
                    confidence=0.7,
                    explanation="Try exact search for specific matches"
                ))
        
        return suggestions
    
    def _generate_context_suggestions(
        self,
        partial_query: str,
        context: Dict[str, Any]
    ) -> List[SearchSuggestion]:
        """Generate context-aware suggestions"""
        suggestions = []
        
        # Add context patterns
        for pattern in self._context_patterns:
            if "{query}" in pattern:
                contextualized = pattern.replace("{query}", partial_query)
                if contextualized != partial_query:
                    suggestions.append(SearchSuggestion(
                        text=contextualized,
                        type=SuggestionType.CONTEXT,
                        confidence=0.6,
                        explanation="Add context for broader search"
                    ))
        
        # Add file-specific suggestions if current file context is available
        current_file = context.get("current_file")
        if current_file:
            suggestions.append(SearchSuggestion(
                text=f"{partial_query} in {current_file}",
                type=SuggestionType.CONTEXT,
                confidence=0.8,
                explanation="Search in current file"
            ))
        
        # Add recent queries if available
        recent_queries = context.get("recent_queries", [])
        for recent in recent_queries[:3]:
            if self._is_query_related(partial_query, recent):
                suggestions.append(SearchSuggestion(
                    text=recent,
                    type=SuggestionType.CONTEXT,
                    confidence=0.7,
                    explanation="Recent search query"
                ))
        
        return suggestions
    
    def _simple_stem(self, word: str) -> str:
        """Apply simple stemming by removing common suffixes"""
        if len(word) <= 4:  # Don't stem very short words
            return word
            
        # Common English suffixes for programming terms
        suffixes = ['tion', 'ing', 'ed', 'er', 'ly', 'al', 'ive', 'ate', 'ize']
        
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]
        
        return word
    
    def _is_query_related(self, partial_query: str, recent_query: str) -> bool:
        """Check if partial query is semantically related to recent query."""
        # Constants
        MIN_WORD_LENGTH = 3
        MIN_STEM_LENGTH = 2
        MIN_PREFIX_LENGTH = 4
        
        query_lower = partial_query.lower().strip()
        recent_lower = recent_query.lower().strip()
        
        # Same query check
        if query_lower == recent_lower:
            return False
        
        # Fast bidirectional substring check
        if query_lower in recent_lower or recent_lower in query_lower:
            return True
        
        # Tokenize once
        query_words = [w for w in query_lower.split() if len(w) >= MIN_WORD_LENGTH]
        recent_words = [w for w in recent_lower.split() if len(w) >= MIN_WORD_LENGTH]
        
        # Early exit if no meaningful words
        if not query_words or not recent_words:
            return False
        
        # Create sets for O(1) lookups
        recent_words_set = set(recent_words)
        
        # Combined word analysis
        for q_word in query_words:
            # Direct match
            if q_word in recent_words_set:
                return True
            
            # Check against each recent word for advanced matching
            for r_word in recent_words:
                # Bidirectional substring (avoid redundant checks)
                if len(q_word) >= MIN_PREFIX_LENGTH:
                    if q_word in r_word or r_word in q_word:
                        return True
                
                # Prefix matching (combines old strategies 3 & 4)
                if (len(q_word) >= MIN_PREFIX_LENGTH and 
                    len(r_word) >= MIN_PREFIX_LENGTH):
                    common_prefix = min(len(q_word), len(r_word), MIN_PREFIX_LENGTH)
                    if q_word[:common_prefix] == r_word[:common_prefix]:
                        return True
                
                # Stemming comparison
                if len(q_word) > MIN_WORD_LENGTH and len(r_word) > MIN_WORD_LENGTH:
                    q_stem = self._simple_stem(q_word)
                    r_stem = self._simple_stem(r_word)
                    if q_stem == r_stem and len(q_stem) > MIN_STEM_LENGTH:
                        return True
        
        return False
    
    def _deduplicate_suggestions(self, suggestions: List[SearchSuggestion]) -> List[SearchSuggestion]:
        """Remove duplicate suggestions while preserving highest confidence"""
        seen_texts = {}
        
        for suggestion in suggestions:
            text_key = suggestion.text.lower().strip()
            if text_key not in seen_texts or suggestion.confidence > seen_texts[text_key].confidence:
                seen_texts[text_key] = suggestion
        
        return list(seen_texts.values())
    
    def get_query_hints(self, query: str) -> List[str]:
        """
        Get hints for improving query effectiveness.
        
        Args:
            query: Search query to analyze
            
        Returns:
            List of improvement hints
        """
        hints = []
        query_lower = query.lower().strip()
        words = query_lower.split()
        
        # Length-based hints
        if len(words) == 1:
            hints.append("Try adding more context words for better results")
        elif len(words) > 8:
            hints.append("Consider using shorter, more specific terms")
        
        # Pattern-based hints
        if not any(pattern in query_lower for pattern in ["def", "class", "function", "method"]):
            if len(words) <= 3:
                hints.append("Add 'def' or 'class' to search for specific code elements")
        
        # Quote hints
        if '"' not in query and "'" not in query and len(words) <= 2:
            hints.append("Use quotes for exact matches: \"ExactFunctionName\"")
        
        # File type hints
        if not any(ext in query_lower for ext in [".py", ".js", ".ts", ".go", ".rs"]):
            hints.append("Specify file type to narrow results: query.py")
        
        # Semantic hints
        if not any(word in query_lower for word in ["how", "what", "find", "show", "explain"]):
            if len(words) >= 3:
                hints.append("Start with 'how to' or 'find' for explanation-style results")
        
        return hints[:3]  # Limit to top 3 hints
    
    def explain_suggestion(self, suggestion: SearchSuggestion) -> str:
        """
        Provide detailed explanation for a suggestion.
        
        Args:
            suggestion: Suggestion to explain
            
        Returns:
            Detailed explanation
        """
        explanations = {
            SuggestionType.COMPLETION: "This completes your query based on common patterns",
            SuggestionType.REFINEMENT: "This refines your query to get more specific results",
            SuggestionType.ALTERNATIVE: "This offers an alternative search approach",
            SuggestionType.CONTEXT: "This adds context to broaden or focus your search"
        }
        
        base_explanation = explanations.get(suggestion.type, "Search suggestion")
        
        if suggestion.explanation:
            return f"{base_explanation}: {suggestion.explanation}"
        
        return base_explanation