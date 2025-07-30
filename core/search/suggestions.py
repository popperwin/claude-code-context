"""
Query suggestion system for enhanced search experience.

Provides intelligent query completions, refinements, and search hints
based on query patterns, context, and user behavior.
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import re
from pathlib import Path

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


@dataclass
class ContextAnalysis:
    """Analysis of current context"""
    is_test_file: bool = False
    file_extension: Optional[str] = None
    project_language: Optional[str] = None
    recent_query_patterns: Dict[str, int] = field(default_factory=dict)
    query_frequency: Dict[str, int] = field(default_factory=dict)


class QuerySuggestionEngine:
    """
    Advanced query suggestion engine with pattern-based completions.
    
    Features:
    - Pattern-based query completions
    - Context-aware suggestions
    - Query refinement hints
    - Alternative search strategies
    - Smart adaptive scoring
    - Dynamic pattern selection
    """
    
    # Configuration constants
    MAX_SUGGESTIONS_PER_TYPE = 6
    MAX_HINTS = 3
    
    # Base confidence levels (will be adjusted dynamically)
    CONFIDENCE_BASE = {
        'high': 0.9,
        'medium_high': 0.85,
        'medium': 0.75,
        'low': 0.6,
        'very_low': 0.5
    }
    
    # Word analysis thresholds
    MIN_WORD_LENGTH = 3
    MIN_STEM_LENGTH = 2
    MIN_PREFIX_LENGTH = 4
    SHORT_QUERY_THRESHOLD = 3
    LONG_QUERY_THRESHOLD = 8
    
    # Language detection patterns
    LANGUAGE_INDICATORS = {
        'python': ['.py', 'def ', 'import ', 'from ', '__init__', 'self.', 'pip', 'pytest'],
        'javascript': ['.js', '.jsx', 'function ', 'const ', 'let ', 'var ', 'npm', 'jest'],
        'typescript': ['.ts', '.tsx', 'interface ', 'type ', ': string', ': number', 'tsc'],
        'go': ['.go', 'func ', 'package ', 'import (', 'go mod', 'go test'],
        'rust': ['.rs', 'fn ', 'impl ', 'trait ', 'cargo', '#[test]'],
        'java': ['.java', 'public class', 'private ', 'extends ', 'implements ', 'junit']
    }
    
    def __init__(self):
        """Initialize suggestion engine with pattern libraries"""
        self._initialize_patterns()
        self._pattern_usage = defaultdict(int)  # Track pattern usage
        self._suggestion_acceptance = defaultdict(int)  # Track accepted suggestions
        logger.info("Initialized QuerySuggestionEngine with smart pragmatic features")
    
    def _initialize_patterns(self):
        """Initialize all pattern libraries"""
        # Language-agnostic patterns
        self._function_patterns = {
            'generic': [
                "{query} function",
                "{query} method",
                "test {query}",
                "{query} handler",
                "{query} validator"
            ],
            'python': [
                "def {query}",
                "async def {query}",
                "def test_{query}",
                "@property {query}",
                "{query}(self"
            ],
            'javascript': [
                "function {query}",
                "const {query} =",
                "async {query}",
                "{query}() {",
                "test('{query}'"
            ],
            'go': [
                "func {query}",
                "func Test{query}",
                "func (r *) {query}",
                "{query}() error",
                "type {query} func"
            ]
        }
        
        self._class_patterns = {
            'generic': [
                "{query} class",
                "{query} model",
                "{query} service",
                "{query} controller",
                "{query} manager"
            ],
            'python': [
                "class {query}",
                "class {query}(BaseModel)",
                "class Test{query}",
                "@dataclass {query}",
                "class {query}Exception"
            ],
            'javascript': [
                "class {query}",
                "class {query} extends",
                "interface {query}",
                "type {query} =",
                "const {query} = class"
            ],
            'go': [
                "type {query} struct",
                "type {query} interface",
                "type {query}Service",
                "type {query}Error",
                "type Mock{query}"
            ]
        }
        
        self._file_patterns = {
            'python': ["{query}.py", "test_{query}.py", "{query}_test.py", "__{query}__.py"],
            'javascript': ["{query}.js", "{query}.test.js", "{query}.spec.js", "{query}.jsx"],
            'typescript': ["{query}.ts", "{query}.test.ts", "{query}.spec.ts", "{query}.tsx"],
            'go': ["{query}.go", "{query}_test.go", "{query}_mock.go"],
            'rust': ["{query}.rs", "test_{query}.rs", "mod_{query}.rs"],
            'java': ["{query}.java", "{query}Test.java", "{query}Spec.java"]
        }
        
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
        
        self._context_patterns = {
            'test': [
                "test {query}",
                "{query} test cases",
                "{query} unit tests",
                "{query} mocks",
                "{query} fixtures"
            ],
            'general': [
                "{query} in this file",
                "{query} similar to",
                "{query} with examples",
                "{query} and related",
                "{query} dependencies"
            ]
        }
    
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
        
        # Analyze context for smarter suggestions
        context_analysis = self._analyze_context(partial_query, context)
        
        # Generate different types of suggestions with context awareness
        all_suggestions = []
        all_suggestions.extend(self._generate_completions(partial_query, context, context_analysis))
        all_suggestions.extend(self._generate_refinements(partial_query, context, context_analysis))
        all_suggestions.extend(self._generate_alternatives(partial_query, context, context_analysis))
        all_suggestions.extend(self._generate_context_suggestions(partial_query, context, context_analysis))
        
        # Apply smart scoring adjustments
        scored_suggestions = self._apply_smart_scoring(all_suggestions, context_analysis)
        
        # Remove duplicates and sort by confidence
        unique_suggestions = self._deduplicate_suggestions(scored_suggestions)
        sorted_suggestions = sorted(unique_suggestions, key=lambda s: s.confidence, reverse=True)
        
        logger.debug(f"Generated {len(sorted_suggestions)} suggestions for '{partial_query}' with context analysis")
        
        return sorted_suggestions[:limit]
    
    def _analyze_context(
        self, 
        partial_query: str, 
        context: Dict[str, Any]
    ) -> ContextAnalysis:
        """Analyze context to make smarter suggestions"""
        analysis = ContextAnalysis()
        
        # Analyze current file
        current_file = context.get("current_file", "")
        if current_file:
            path = Path(current_file)
            analysis.file_extension = path.suffix
            
            # Check if it's a test file
            file_lower = current_file.lower()
            analysis.is_test_file = any(
                pattern in file_lower 
                for pattern in ['test_', '_test.', 'spec.', '.test.', '/tests/', '/test/']
            )
        
        # Detect project language from query and context
        analysis.project_language = self._detect_language(partial_query, context)
        
        # Analyze recent queries for patterns
        recent_queries = context.get("recent_queries", [])
        for query in recent_queries:
            query_lower = query.lower()
            # Track query patterns
            if "def" in query_lower or "function" in query_lower:
                analysis.recent_query_patterns["function"] = analysis.recent_query_patterns.get("function", 0) + 1
            if "class" in query_lower or "type" in query_lower:
                analysis.recent_query_patterns["class"] = analysis.recent_query_patterns.get("class", 0) + 1
            if "test" in query_lower:
                analysis.recent_query_patterns["test"] = analysis.recent_query_patterns.get("test", 0) + 1
            
            # Track frequently searched terms
            for word in query_lower.split():
                if len(word) >= self.MIN_WORD_LENGTH:
                    analysis.query_frequency[word] = analysis.query_frequency.get(word, 0) + 1
        
        return analysis
    
    def _detect_language(self, query: str, context: Dict[str, Any]) -> Optional[str]:
        """Detect the primary programming language from context and query"""
        language_scores = defaultdict(int)
        
        # Check query for language indicators
        query_lower = query.lower()
        for lang, indicators in self.LANGUAGE_INDICATORS.items():
            for indicator in indicators:
                if indicator in query_lower:
                    language_scores[lang] += 2
        
        # Check file extension
        current_file = context.get("current_file", "")
        if current_file:
            ext = Path(current_file).suffix
            for lang, indicators in self.LANGUAGE_INDICATORS.items():
                if ext in indicators:
                    language_scores[lang] += 5
        
        # Check recent queries
        for recent in context.get("recent_queries", []):
            recent_lower = recent.lower()
            for lang, indicators in self.LANGUAGE_INDICATORS.items():
                for indicator in indicators:
                    if indicator in recent_lower:
                        language_scores[lang] += 1
        
        # Return language with highest score
        if language_scores:
            return max(language_scores.items(), key=lambda x: x[1])[0]
        return None
    
    def _generate_completions(
        self,
        partial_query: str,
        context: Dict[str, Any],
        context_analysis: ContextAnalysis
    ) -> List[SearchSuggestion]:
        """Generate query completions based on patterns and context"""
        suggestions = []
        query_lower = partial_query.lower().strip()
        words = query_lower.split()
        
        # Generate completions by category with language awareness
        suggestions.extend(self._generate_function_completions(
            partial_query, query_lower, words, context_analysis
        ))
        suggestions.extend(self._generate_class_completions(
            partial_query, query_lower, words, context_analysis
        ))
        suggestions.extend(self._generate_file_completions(
            partial_query, query_lower, context_analysis
        ))
        suggestions.extend(self._generate_semantic_completions(
            partial_query, query_lower, words, context_analysis
        ))
        
        return suggestions
    
    def _generate_function_completions(
        self, 
        partial_query: str, 
        query_lower: str, 
        words: List[str],
        context_analysis: ContextAnalysis
    ) -> List[SearchSuggestion]:
        """Generate function-related completions with language awareness"""
        suggestions = []
        
        # Adjust threshold based on context
        threshold = self.SHORT_QUERY_THRESHOLD
        if context_analysis.recent_query_patterns.get("function", 0) > 2:
            threshold += 1  # Be less aggressive if user searches functions often
        
        # Check if function suggestions are appropriate
        should_suggest = (
            self._contains_any(query_lower, ["def", "function", "func", "method"]) or
            len(words) <= threshold or
            (context_analysis.is_test_file and "test" in query_lower)
        )
        
        if not should_suggest:
            return suggestions
        
        # Select patterns based on detected language
        lang = context_analysis.project_language or 'generic'
        patterns = self._function_patterns.get(lang, self._function_patterns['generic'])
        
        # Add generic patterns if language-specific ones are few
        if len(patterns) < 5:
            patterns = patterns + self._function_patterns['generic']
        
        # Select and score patterns
        selected_patterns = self._select_function_patterns(query_lower, patterns, context_analysis)
        
        for pattern, score_boost in selected_patterns[:self.MAX_SUGGESTIONS_PER_TYPE]:
            completed = pattern.replace("{query}", partial_query)
            if completed != partial_query:
                base_confidence = self._calculate_adaptive_confidence(
                    'function', context_analysis, is_test_context=context_analysis.is_test_file
                )
                suggestions.append(SearchSuggestion(
                    text=completed,
                    type=SuggestionType.COMPLETION,
                    confidence=min(base_confidence + score_boost, 0.95),
                    explanation=f"Function pattern for {lang}"
                ))
        
        return suggestions
    
    def _generate_class_completions(
        self, 
        partial_query: str, 
        query_lower: str, 
        words: List[str],
        context_analysis: ContextAnalysis
    ) -> List[SearchSuggestion]:
        """Generate class-related completions with language awareness"""
        suggestions = []
        
        # Check if class suggestions are appropriate
        should_suggest = (
            self._contains_any(query_lower, ["class", "interface", "type", "model", "struct"]) or
            len(words) <= self.SHORT_QUERY_THRESHOLD or
            context_analysis.recent_query_patterns.get("class", 0) > 1
        )
        
        if not should_suggest:
            return suggestions
        
        # Select patterns based on detected language
        lang = context_analysis.project_language or 'generic'
        patterns = self._class_patterns.get(lang, self._class_patterns['generic'])
        
        # Prioritize patterns based on context
        if context_analysis.is_test_file:
            # Prioritize test-related class patterns
            patterns = sorted(patterns, key=lambda p: 'test' in p.lower(), reverse=True)
        
        for pattern in patterns[:self.MAX_SUGGESTIONS_PER_TYPE]:
            completed = pattern.replace("{query}", partial_query)
            if completed != partial_query:
                base_confidence = self._calculate_adaptive_confidence(
                    'class', context_analysis
                )
                suggestions.append(SearchSuggestion(
                    text=completed,
                    type=SuggestionType.COMPLETION,
                    confidence=base_confidence,
                    explanation=f"Class pattern for {lang}"
                ))
        
        return suggestions
    
    def _generate_file_completions(
        self, 
        partial_query: str, 
        query_lower: str,
        context_analysis: ContextAnalysis
    ) -> List[SearchSuggestion]:
        """Generate file-related completions with language awareness"""
        suggestions = []
        
        # Check if file suggestions are appropriate
        should_suggest = (
            self._contains_any(query_lower, ["file:", "path:", "."]) or
            ("file" in query_lower and ":" in query_lower) or
            any(ext in query_lower for ext in ['.py', '.js', '.ts', '.go', '.rs', '.java'])
        )
        
        if not should_suggest:
            return suggestions
        
        # Extract base query for file patterns
        base_query = self._extract_file_base_query(partial_query)
        
        # Use language-specific file patterns
        lang = context_analysis.project_language
        if lang and lang in self._file_patterns:
            patterns = self._file_patterns[lang]
        else:
            # Use all common patterns
            patterns = []
            for lang_patterns in self._file_patterns.values():
                patterns.extend(lang_patterns[:2])  # Take top 2 from each language
        
        # Prioritize test patterns if in test context
        if context_analysis.is_test_file:
            patterns = sorted(patterns, key=lambda p: 'test' in p.lower(), reverse=True)
        
        for pattern in patterns[:8]:  # Slightly more file suggestions
            completed = pattern.replace("{query}", base_query)
            if completed != base_query:
                confidence = self.CONFIDENCE_BASE['medium_high'] if "file:" in partial_query.lower() else self.CONFIDENCE_BASE['low']
                # Boost confidence for matching language
                if lang and any(ext in pattern for ext in self._file_patterns.get(lang, [])):
                    confidence += 0.1
                suggestions.append(SearchSuggestion(
                    text=completed,
                    type=SuggestionType.COMPLETION,
                    confidence=min(confidence, 0.95),
                    explanation=f"File pattern for {lang or 'project'}"
                ))
        
        return suggestions
    
    def _generate_semantic_completions(
        self, 
        partial_query: str, 
        query_lower: str, 
        words: List[str],
        context_analysis: ContextAnalysis
    ) -> List[SearchSuggestion]:
        """Generate semantic pattern completions"""
        suggestions = []
        
        # Check if semantic suggestions are appropriate
        should_suggest = (
            self._contains_any(query_lower, ["how", "what", "find", "show", "explain"]) or
            len(words) >= 2 or
            context_analysis.query_frequency.get("how", 0) > 1  # User likes semantic queries
        )
        
        if not should_suggest:
            return suggestions
        
        # Select relevant patterns
        patterns = self._select_semantic_patterns(query_lower)
        
        # Add test-specific semantic patterns if in test context
        if context_analysis.is_test_file:
            patterns.extend([
                "test {query} behavior",
                "{query} test coverage",
                "mock {query} for testing"
            ])
        
        for pattern in patterns[:self.MAX_SUGGESTIONS_PER_TYPE]:
            completed = pattern.replace("{query}", partial_query)
            if completed != partial_query:
                # Higher confidence for semantic patterns that match user behavior
                confidence = self.CONFIDENCE_BASE['high']
                if any(word in pattern for word in context_analysis.query_frequency.keys()):
                    confidence += 0.05
                suggestions.append(SearchSuggestion(
                    text=completed,
                    type=SuggestionType.COMPLETION,
                    confidence=min(confidence, 0.95),
                    explanation="Semantic query completion"
                ))
        
        return suggestions
    
    def _generate_refinements(
        self,
        partial_query: str,
        context: Dict[str, Any],
        context_analysis: ContextAnalysis
    ) -> List[SearchSuggestion]:
        """Generate query refinements with context awareness"""
        suggestions = []
        query_lower = partial_query.lower().strip()
        words = query_lower.split()
        
        # Language-specific refinements
        if context_analysis.project_language:
            suggestions.extend(self._generate_language_refinements(
                partial_query, context_analysis
            ))
        
        # Test-specific refinements
        if context_analysis.is_test_file:
            suggestions.extend(self._generate_test_refinements(partial_query))
        

        # Standard refinements
        suggestions.extend(self._generate_type_refinements(partial_query, query_lower, words))
        suggestions.extend(self._generate_file_type_refinements(partial_query, query_lower))
        suggestions.extend(self._generate_semantic_refinements(partial_query, query_lower, words))
        suggestions.extend(self._generate_code_refinements(partial_query, query_lower))
        
        return suggestions
    
    def _generate_language_refinements(
        self,
        partial_query: str,
        context_analysis: ContextAnalysis
    ) -> List[SearchSuggestion]:
        """Generate language-specific refinements"""
        suggestions = []
        lang = context_analysis.project_language
        
        if not lang:
            return suggestions
        
        # Language-specific additions
        lang_specific = {
            'python': ['decorator', 'async', 'property', 'classmethod'],
            'javascript': ['arrow function', 'callback', 'promise', 'async/await'],
            'go': ['goroutine', 'channel', 'interface', 'defer'],
            'rust': ['trait', 'lifetime', 'ownership', 'unsafe']
        }
        
        if lang in lang_specific:
            for addition in lang_specific[lang][:3]:
                suggestions.append(SearchSuggestion(
                    text=f"{partial_query} {addition}",
                    type=SuggestionType.REFINEMENT,
                    confidence=self.CONFIDENCE_BASE['medium'],
                    explanation=f"{lang.title()}-specific refinement"
                ))
        
        return suggestions
    
    def _generate_test_refinements(self, partial_query: str) -> List[SearchSuggestion]:
        """Generate test-specific refinements"""
        suggestions = []
        test_additions = ['unit test', 'integration test', 'mock', 'fixture', 'assertion']
        
        for addition in test_additions[:3]:
            suggestions.append(SearchSuggestion(
                text=f"{partial_query} {addition}",
                type=SuggestionType.REFINEMENT,
                confidence=self.CONFIDENCE_BASE['medium_high'],
                explanation="Test-specific refinement"
            ))
        
        return suggestions
    
    def _generate_type_refinements(
        self, 
        partial_query: str, 
        query_lower: str, 
        words: List[str]
    ) -> List[SearchSuggestion]:
        """Generate type-based refinements"""
        suggestions = []
        
        if len(words) >= 2 and not self._contains_any(query_lower, ["async", "def", "class"]):
            suggestions.append(SearchSuggestion(
                text=f"def {partial_query}",
                type=SuggestionType.REFINEMENT,
                confidence=self.CONFIDENCE_BASE['low'],
                explanation="Add function context for better results"
            ))
            
            suggestions.append(SearchSuggestion(
                text=f"class {partial_query}",
                type=SuggestionType.REFINEMENT,
                confidence=self.CONFIDENCE_BASE['low'],
                explanation="Add class context for better results"
            ))
        
        return suggestions

    def _generate_file_type_refinements(
        self, 
        partial_query: str, 
        query_lower: str
    ) -> List[SearchSuggestion]:
        """Generate file type refinements"""
        suggestions = []
        
        if not self._contains_any(query_lower, [".py", ".js", ".ts", ".go", ".rs"]):
            file_types = [".py", ".js", ".ts", ".go"]
            for file_type in file_types:
                suggestions.append(SearchSuggestion(
                    text=f"{partial_query} {file_type}",
                    type=SuggestionType.REFINEMENT,
                    confidence=0.5,
                    explanation=f"Search in {file_type} files specifically"
                ))
        
        return suggestions

    def _generate_semantic_refinements(
        self, 
        partial_query: str, 
        query_lower: str, 
        words: List[str]
    ) -> List[SearchSuggestion]:
        """Generate semantic refinements"""
        suggestions = []
        
        should_generate = (
            len(words) <= 2 or 
            self._contains_any(query_lower, ["how to", "what is", "find", "show"])
        )
        
        if not should_generate:
            return suggestions
        
        additions = ["examples", "implementation", "usage", "documentation", "patterns", "best practices"]
        is_semantic_query = self._contains_any(query_lower, ["how to", "what is", "find", "show"])
        
        for addition in additions:
            confidence = 0.82 if is_semantic_query else 0.7
            suggestions.append(SearchSuggestion(
                text=f"{partial_query} {addition}",
                type=SuggestionType.REFINEMENT,
                confidence=confidence,
                explanation=f"Find {addition} related to your query"
            ))
        
        return suggestions

    def _generate_code_refinements(
        self, 
        partial_query: str, 
        query_lower: str
    ) -> List[SearchSuggestion]:
        """Generate code-specific refinements"""
        suggestions = []
        
        code_terms = ["function", "class", "method", "interface", "variable", "util", "test"]
        if not self._contains_any(query_lower, code_terms):
            return suggestions
        
        refinements = ["definition", "implementation", "declaration", "usage", "test", "mock"]
        
        for refinement in refinements[:3]:
            suggestions.append(SearchSuggestion(
                text=f"{partial_query} {refinement}",
                type=SuggestionType.REFINEMENT,
                confidence=self.CONFIDENCE_BASE['medium'],
                explanation=f"Find {refinement} for {partial_query}"
            ))
        
        return suggestions
    
    def _apply_smart_scoring(
        self,
        suggestions: List[SearchSuggestion],
        context_analysis: ContextAnalysis
    ) -> List[SearchSuggestion]:
        """Apply smart scoring adjustments based on context"""
        scored_suggestions = []
        
        for suggestion in suggestions:
            # Copy suggestion to avoid modifying original
            new_suggestion = SearchSuggestion(
                text=suggestion.text,
                type=suggestion.type,
                confidence=suggestion.confidence,
                explanation=suggestion.explanation
            )
            
            # Boost suggestions that match frequently searched terms
            text_lower = suggestion.text.lower()
            frequency_boost = 0.0
            for term, count in context_analysis.query_frequency.items():
                if term in text_lower:
                    frequency_boost += min(0.02 * count, 0.1)
            
            # Boost test-related suggestions in test files
            if context_analysis.is_test_file and 'test' in text_lower:
                frequency_boost += 0.05
            
            # Apply boosts
            new_suggestion.confidence = min(
                new_suggestion.confidence + frequency_boost,
                0.99
            )
            
            scored_suggestions.append(new_suggestion)
        
        return scored_suggestions
    
    def _calculate_adaptive_confidence(
        self,
        suggestion_type: str,
        context_analysis: ContextAnalysis,
        is_test_context: bool = False
    ) -> float:
        """Calculate confidence based on context and user patterns"""
        # Base confidence
        if suggestion_type == 'function':
            base = self.CONFIDENCE_BASE['medium_high']
        elif suggestion_type == 'class':
            base = self.CONFIDENCE_BASE['medium']
        else:
            base = self.CONFIDENCE_BASE['low']
        
        # Adjust based on user patterns
        pattern_count = context_analysis.recent_query_patterns.get(suggestion_type, 0)
        if pattern_count > 3:
            base += 0.1
        elif pattern_count > 1:
            base += 0.05
        
        # Adjust for test context
        if is_test_context and suggestion_type in ['function', 'class']:
            base += 0.05
        
        return min(base, 0.95)
    
    # Keep all the helper methods from before, they're still needed
    def _contains_any(self, text: str, patterns: List[str]) -> bool:
        """Check if text contains any of the patterns"""
        return any(pattern in text for pattern in patterns)
    
    def _is_exact_search(self, query: str) -> bool:
        """Check if query is an exact search"""
        return any(char in query for char in ['"', "'", ":"])
    
    def _extract_file_base_query(self, query: str) -> str:
        """Extract base query from file search patterns"""
        if ":" in query:
            return query.split(":", 1)[1]
        return query
    
    def _select_function_patterns(
        self, 
        query_lower: str, 
        patterns: List[str],
        context_analysis: ContextAnalysis
    ) -> List[Tuple[str, float]]:
        """Select relevant function patterns with scoring"""
        scored_patterns = []
        
        for pattern in patterns:
            score = 0.0
            pattern_lower = pattern.lower()
            
            # Score based on query matching
            if "async" in query_lower and "async" in pattern_lower:
                score += 0.05
            if "test" in query_lower and "test" in pattern_lower:
                score += 0.05
            if context_analysis.is_test_file and "test" in pattern_lower:
                score += 0.03
            
            # Avoid duplicates
            if any(keyword in query_lower for keyword in pattern_lower.split()):
                score -= 0.02
            
            scored_patterns.append((pattern, score))
        
        # Sort by score
        return sorted(scored_patterns, key=lambda x: x[1], reverse=True)
    
    def _select_semantic_patterns(self, query_lower: str) -> List[str]:
        """Select relevant semantic patterns based on query"""
        selected = []
        
        if not query_lower.startswith(("how", "what", "find", "show")):
            selected.extend(["how to {query}", "find {query}", "show me {query}"])
        
        selected.extend([
            "{query} examples", 
            "{query} implementation",
            "find {query} that handles",
            "{query} similar to",
            "{query} in file",
            "{query} with pattern"
        ])
        
        return selected
    
    def _generate_alternatives(
        self,
        partial_query: str,
        context: Dict[str, Any],
        context_analysis: ContextAnalysis
    ) -> List[SearchSuggestion]:
        """Generate alternative search strategies with context awareness"""
        suggestions = []
        query_lower = partial_query.lower().strip()
        
        # Language-aware alternatives
        if context_analysis.project_language:
            suggestions.extend(self._generate_language_alternatives(
                partial_query, query_lower, context_analysis
            ))
        
        # Standard alternatives
        if self._is_exact_search(partial_query):
            suggestions.extend(self._generate_semantic_alternatives(partial_query))
        
        suggestions.extend(self._generate_code_term_alternatives(partial_query, query_lower))
        
        if len(partial_query.split()) >= 4:
            suggestions.extend(self._generate_exact_alternatives(partial_query))
        
        return suggestions
    
    def _generate_language_alternatives(
        self,
        partial_query: str,
        query_lower: str,
        context_analysis: ContextAnalysis
    ) -> List[SearchSuggestion]:
        """Generate language-specific term alternatives"""
        suggestions = []
        lang = context_analysis.project_language
        
        # Language-specific term mappings
        lang_terms = {
            'python': {
                'function': ['method', 'def', 'callable'],
                'class': ['type', 'dataclass', 'model']
            },
            'javascript': {
                'function': ['method', 'callback', 'handler'],
                'class': ['prototype', 'constructor', 'component']
            },
            'go': {
                'function': ['func', 'method', 'handler'],
                'class': ['struct', 'type', 'interface']
            }
        }
        
        if lang in lang_terms:
            for term, alternatives in lang_terms[lang].items():
                if term in query_lower:
                    for alt in alternatives[:2]:
                        alt_query = partial_query.replace(term, alt)
                        if alt_query != partial_query:
                            suggestions.append(SearchSuggestion(
                                text=alt_query,
                                type=SuggestionType.ALTERNATIVE,
                                confidence=self.CONFIDENCE_BASE['medium_high'],
                                explanation=f"{lang.title()} alternative: {alt}"
                            ))
        
        return suggestions
    
    def _generate_semantic_alternatives(self, partial_query: str) -> List[SearchSuggestion]:
        """Generate semantic alternatives for exact searches"""
        suggestions = []
        
        semantic_query = partial_query.replace('"', '').replace("'", "")
        if ":" in semantic_query:
            semantic_query = semantic_query.split(":", 1)[1]
        
        suggestions.append(SearchSuggestion(
            text=f"how to use {semantic_query}",
            type=SuggestionType.ALTERNATIVE,
            confidence=self.CONFIDENCE_BASE['medium_high'],
            explanation="Try semantic search for better understanding"
        ))
        
        return suggestions
    
    def _generate_code_term_alternatives(
        self, 
        partial_query: str, 
        query_lower: str
    ) -> List[SearchSuggestion]:
        """Generate alternatives for code terms"""
        suggestions = []
        
        code_terms = ["function", "class", "method", "interface", "util", "test"]
        if not self._contains_any(query_lower, code_terms):
            return suggestions
        
        term_alternatives = {
            "function": ["method", "procedure", "subroutine"],
            "class": ["type", "struct", "interface"],
            "method": ["function", "member function"],
            "interface": ["protocol", "abstract class", "trait"],
            "util": ["helper", "utility", "tool"],
            "test": ["spec", "unittest", "check"]
        }
        
        for term, alternatives in term_alternatives.items():
            if term not in query_lower:
                continue
                
            for alt in alternatives[:2]:
                alt_query = partial_query.replace(term, alt)
                if alt_query != partial_query:
                    suggestions.append(SearchSuggestion(
                        text=alt_query,
                        type=SuggestionType.ALTERNATIVE,
                        confidence=self.CONFIDENCE_BASE['medium'],
                        explanation=f"Alternative term: {alt} instead of {term}"
                    ))
        
        return suggestions
    
    def _generate_exact_alternatives(self, partial_query: str) -> List[SearchSuggestion]:
        """Generate exact search alternatives for semantic queries"""
        suggestions = []
        
        key_words = [
            word for word in partial_query.split() 
            if len(word) > 3 and word.lower() not in ["with", "that", "this", "have"]
        ]
        
        if key_words:
            suggestions.append(SearchSuggestion(
                text=f'"{key_words[0]}"',
                type=SuggestionType.ALTERNATIVE,
                confidence=self.CONFIDENCE_BASE['medium'],
                explanation="Try exact search for specific matches"
            ))
        
        return suggestions
    
    def _generate_context_suggestions(
        self,
        partial_query: str,
        context: Dict[str, Any],
        context_analysis: ContextAnalysis
    ) -> List[SearchSuggestion]:
        """Generate context-aware suggestions"""
        suggestions = []
        
        # Test-aware context suggestions
        if context_analysis.is_test_file:
            patterns = self._context_patterns['test']
        else:
            patterns = self._context_patterns['general']
        
        # Add language-specific context if detected
        if context_analysis.project_language:
            patterns = patterns[:3]  # Take fewer generic patterns
            patterns.append(f"{{query}} in {context_analysis.project_language}")
        
        for pattern in patterns[:5]:
            if "{query}" not in pattern:
                continue
                
            contextualized = pattern.replace("{query}", partial_query)
            if contextualized != partial_query:
                suggestions.append(SearchSuggestion(
                    text=contextualized,
                    type=SuggestionType.CONTEXT,
                    confidence=self.CONFIDENCE_BASE['low'] + 0.1,  # Slightly boost context suggestions
                    explanation="Add context for focused search"
                ))
        
        # File-specific suggestions
        current_file = context.get("current_file")
        if current_file:
            suggestions.append(SearchSuggestion(
                text=f"{partial_query} in {current_file}",
                type=SuggestionType.CONTEXT,
                confidence=self.CONFIDENCE_BASE['medium_high'],
                explanation="Search in current file"
            ))
        
        # Recent query suggestions with smart scoring
        recent_queries = context.get("recent_queries", [])
        for recent in recent_queries[:3]:
            if self._is_query_related(partial_query, recent):
                # Higher confidence for more related queries
                relatedness = self._calculate_relatedness(partial_query, recent)
                suggestions.append(SearchSuggestion(
                    text=recent,
                    type=SuggestionType.CONTEXT,
                    confidence=min(0.7 + relatedness * 0.2, 0.95),
                    explanation="Recent related search"
                ))
        
        return suggestions
    
    def _calculate_relatedness(self, query1: str, query2: str) -> float:
        """Calculate how related two queries are (0-1)"""
        q1_words = set(query1.lower().split())
        q2_words = set(query2.lower().split())
        
        if not q1_words or not q2_words:
            return 0.0
        
        # Jaccard similarity
        intersection = len(q1_words & q2_words)
        union = len(q1_words | q2_words)
        
        return intersection / union if union > 0 else 0.0
    
    def _simple_stem(self, word: str) -> str:
        """Apply simple stemming by removing common suffixes"""
        if len(word) <= 4:
            return word
            
        suffixes = ['tion', 'ing', 'ed', 'er', 'ly', 'al', 'ive', 'ate', 'ize']
        
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]
        
        return word
    
    def _is_query_related(self, partial_query: str, recent_query: str) -> bool:
        """Check if partial query is semantically related to recent query."""
        query_lower = partial_query.lower().strip()
        recent_lower = recent_query.lower().strip()
        
        # Same query check
        if query_lower == recent_lower:
            return False
        
        # Fast bidirectional substring check
        if query_lower in recent_lower or recent_lower in query_lower:
            return True
        
        # Tokenize once
        query_words = [w for w in query_lower.split() if len(w) >= self.MIN_WORD_LENGTH]
        recent_words = [w for w in recent_lower.split() if len(w) >= self.MIN_WORD_LENGTH]
        
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
                if len(q_word) >= self.MIN_PREFIX_LENGTH:
                    if q_word in r_word or r_word in q_word:
                        return True
                
                # Prefix matching (combines old strategies 3 & 4)
                if (len(q_word) >= self.MIN_PREFIX_LENGTH and 
                    len(r_word) >= self.MIN_PREFIX_LENGTH):
                    common_prefix = min(len(q_word), len(r_word), self.MIN_PREFIX_LENGTH)
                    if q_word[:common_prefix] == r_word[:common_prefix]:
                        return True
                
                # Stemming comparison
                if len(q_word) > self.MIN_WORD_LENGTH and len(r_word) > self.MIN_WORD_LENGTH:
                    q_stem = self._simple_stem(q_word)
                    r_stem = self._simple_stem(r_word)
                    if q_stem == r_stem and len(q_stem) > self.MIN_STEM_LENGTH:
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
        
        # Detect language for better hints
        fake_context = {"recent_queries": [query]}
        context_analysis = self._analyze_context(query, fake_context)
        
        # Length-based hints
        if len(words) == 1:
            hints.append("Try adding more context words for better results")
        elif len(words) > self.LONG_QUERY_THRESHOLD:
            hints.append("Consider using shorter, more specific terms")
        
        # Language-specific hints
        if context_analysis.project_language:
            lang = context_analysis.project_language
            if lang == "python" and "def" not in query_lower:
                hints.append("Use 'def' to search for Python functions")
            elif lang == "javascript" and "function" not in query_lower:
                hints.append("Use 'function' or 'const' for JavaScript code")
        else:
            # Generic pattern hints
            if not self._contains_any(query_lower, ["def", "class", "function", "method"]):
                if len(words) <= self.SHORT_QUERY_THRESHOLD:
                    hints.append("Add 'def' or 'class' to search for specific code elements")
        
        # Quote hints
        if '"' not in query and "'" not in query and len(words) <= 2:
            hints.append("Use quotes for exact matches: \"ExactFunctionName\"")
        
        # File type hints
        if not self._contains_any(query_lower, [".py", ".js", ".ts", ".go", ".rs"]):
            hints.append("Specify file type to narrow results: query.py")
        
        # Semantic hints
        if not self._contains_any(query_lower, ["how", "what", "find", "show", "explain"]):
            if len(words) >= 3:
                hints.append("Start with 'how to' or 'find' for explanation-style results")
        
        return hints[:self.MAX_HINTS]
    
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