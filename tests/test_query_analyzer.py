"""
Unit tests for QueryAnalyzer functionality.

Tests query pattern detection, mode recommendations, and suggestion generation.
"""

import pytest
from core.search.query_analyzer import QueryAnalyzer, QueryAnalysis


class TestQueryAnalyzer:
    """Test QueryAnalyzer functionality"""
    
    def setup_method(self):
        """Setup test instance"""
        self.analyzer = QueryAnalyzer()
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization"""
        assert self.analyzer is not None
        assert len(self.analyzer._code_patterns) > 0
        assert len(self.analyzer._exact_patterns) > 0
        assert len(self.analyzer._file_patterns) > 0
        assert len(self.analyzer._semantic_patterns) > 0
    
    def test_empty_query_analysis(self):
        """Test analysis of empty/None queries"""
        # Empty string
        analysis = self.analyzer.analyze_query("")
        assert analysis.original_query == ""
        assert analysis.cleaned_query == ""
        assert analysis.word_count == 0
        assert not analysis.has_exact_patterns
        assert not analysis.has_code_patterns
        assert len(analysis.detected_patterns) == 0
        assert analysis.confidence == 0.0
        assert analysis.recommended_mode == "hybrid"
        
        # Whitespace only
        analysis = self.analyzer.analyze_query("   ")
        assert analysis.cleaned_query == ""
        assert analysis.word_count == 0
        
        # None query
        analysis = self.analyzer.analyze_query(None)
        assert analysis.word_count == 0
    
    def test_exact_pattern_detection(self):
        """Test exact match pattern detection"""
        test_cases = [
            ('name:"MyClass"', True, "exact"),
            ("file:main.py", True, "exact"), 
            ("'exact string'", True, "exact"),
            ('"another exact"', True, "exact"),
            ("id:12345", True, "exact"),
            ("regular query", False, None)
        ]
        
        for query, should_detect, expected_mode in test_cases:
            analysis = self.analyzer.analyze_query(query)
            assert analysis.has_exact_patterns == should_detect
            if should_detect:
                assert analysis.recommended_mode == "payload"
                assert any("exact:" in pattern for pattern in analysis.detected_patterns)
                assert analysis.confidence >= 0.7
    
    def test_code_pattern_detection(self):
        """Test code-specific pattern detection"""
        test_cases = [
            ("def my_function", True, ["code:def", "code:function"]),
            ("class MyClass", True, ["code:class"]),
            ("async def handler", True, ["code:async", "code:def"]),
            ("import json", True, ["code:import"]),
            ("for i in range", True, ["code:for"]),
            ("if condition", True, ["code:if"]),
            ("try except block", True, ["code:try", "code:except"]),
            ("lambda x: x", True, ["code:lambda"]),
            ("yield from generator", True, ["code:yield"]),
            ("self.method()", True, ["code:self.", "code:method"]),
            ("regular text query", False, [])
        ]
        
        for query, should_detect, expected_patterns in test_cases:
            analysis = self.analyzer.analyze_query(query)
            assert analysis.has_code_patterns == should_detect
            
            if should_detect:
                for pattern in expected_patterns:
                    assert pattern in analysis.detected_patterns
    
    def test_file_pattern_detection(self):
        """Test file/path pattern detection"""
        test_cases = [
            ("file:main.py", True, ["file:file:", "file:.py"]),
            ("path:/home/user", True, ["file:path:"]),
            ("main.js function", True, ["file:.js"]),
            ("test.ts interface", True, ["file:.ts"]),
            ("config.go struct", True, ["file:.go"]),
            ("models.rs enum", True, ["file:.rs"]),
            ("App.java class", True, ["file:.java"]),
            ("utils.cpp template", True, ["file:.cpp"]),
            ("regular query", False, [])
        ]
        
        for query, should_detect, expected_patterns in test_cases:
            analysis = self.analyzer.analyze_query(query)
            file_patterns = [p for p in analysis.detected_patterns if p.startswith("file:")]
            
            if should_detect:
                assert len(file_patterns) > 0
                for pattern in expected_patterns:
                    assert pattern in analysis.detected_patterns
            else:
                assert len(file_patterns) == 0
    
    def test_semantic_pattern_detection(self):
        """Test natural language/semantic pattern detection"""
        test_cases = [
            ("how to implement authentication", True, ["semantic:how to"]),
            ("what is a closure", True, ["semantic:what is"]),
            ("find code that handles errors", True, ["semantic:find code that"]),
            ("show me all classes", True, ["semantic:show me"]),
            ("explain the algorithm", True, ["semantic:explain"]),
            ("looking for similar functions", True, ["semantic:looking for", "semantic:similar"]),
            ("help me understand", True, ["semantic:help me"]),
            ("describe the pattern", True, ["semantic:describe"]),
            ("def function_name", False, [])
        ]
        
        for query, should_detect, expected_patterns in test_cases:
            analysis = self.analyzer.analyze_query(query)
            semantic_patterns = [p for p in analysis.detected_patterns if p.startswith("semantic:")]
            
            if should_detect:
                assert len(semantic_patterns) > 0
                for pattern in expected_patterns:
                    assert pattern in analysis.detected_patterns
            else:
                assert len(semantic_patterns) == 0
    
    def test_mode_recommendation_exact_patterns(self):
        """Test mode recommendation for exact patterns"""
        queries = [
            'name:"ExactClass"',
            "file:specific.py",
            "id:abc123",
            "'quoted string'"
        ]
        
        for query in queries:
            analysis = self.analyzer.analyze_query(query)
            assert analysis.recommended_mode == "payload"
            assert analysis.confidence >= 0.8
    
    def test_mode_recommendation_semantic_patterns(self):
        """Test mode recommendation for semantic patterns"""
        queries = [
            "how to implement error handling in Python",
            "find code that processes user authentication",
            "show me functions similar to validation logic",
            "explain the database connection pattern"
        ]
        
        for query in queries:
            analysis = self.analyzer.analyze_query(query)
            assert analysis.recommended_mode == "semantic"
            assert analysis.confidence >= 0.6
    
    def test_mode_recommendation_code_patterns(self):
        """Test mode recommendation for code patterns"""
        test_cases = [
            ("def", "payload"),  # Short code query
            ("class", "payload"),  # Short code query
            ("def authentication handler", "hybrid"),  # Longer code query
            ("async function with error handling", "hybrid"),  # Longer code query
            ("import json module", "hybrid")  # Medium code query
        ]
        
        for query, expected_mode in test_cases:
            analysis = self.analyzer.analyze_query(query)
            assert analysis.recommended_mode == expected_mode
    
    def test_mode_recommendation_word_count(self):
        """Test mode recommendation based on word count"""
        test_cases = [
            ("test", "payload"),  # 1 word
            ("user login", "payload"),  # 2 words  
            ("user authentication system", "hybrid"),  # 3 words
            ("implement user authentication system logic", "hybrid"),  # 5 words
            ("find code that implements user authentication and authorization logic", "semantic")  # 10 words
        ]
        
        for query, expected_mode in test_cases:
            analysis = self.analyzer.analyze_query(query)
            assert analysis.recommended_mode == expected_mode
    
    def test_query_cleaning(self):
        """Test query cleaning and normalization"""
        test_cases = [
            ("  extra   spaces  ", "extra spaces"),
            ("\t\ntabs and newlines\n\t", "tabs and newlines"),
            ("normal query", "normal query"),
            ("", "")
        ]
        
        for dirty_query, expected_clean in test_cases:
            analysis = self.analyzer.analyze_query(dirty_query)
            assert analysis.cleaned_query == expected_clean
    
    def test_confidence_scoring(self):
        """Test confidence score calculation"""
        # High confidence cases
        high_confidence_queries = [
            'name:"ExactMatch"',  # Exact pattern
            "how to implement authentication logic in Python"  # Clear semantic
        ]
        
        for query in high_confidence_queries:
            analysis = self.analyzer.analyze_query(query)
            assert analysis.confidence >= 0.8
        
        # Medium confidence cases
        medium_confidence_queries = [
            "def user_handler",  # Code pattern
            "authentication system"  # Medium length
        ]
        
        for query in medium_confidence_queries:
            analysis = self.analyzer.analyze_query(query)
            assert 0.5 <= analysis.confidence < 0.8
        
        # Lower confidence cases
        low_confidence_queries = [
            "test query",  # Short generic
            "system process handler"  # Generic medium length
        ]
        
        for query in low_confidence_queries:
            analysis = self.analyzer.analyze_query(query)
            assert analysis.confidence <= 0.6
    
    def test_complex_pattern_combinations(self):
        """Test queries with multiple pattern types"""
        # Code + file patterns
        analysis = self.analyzer.analyze_query("def handler in main.py")
        assert analysis.has_code_patterns
        assert "file:.py" in analysis.detected_patterns
        assert "code:def" in analysis.detected_patterns
        
        # Semantic + file patterns  
        analysis = self.analyzer.analyze_query("find functions in utils.js")
        assert "file:.js" in analysis.detected_patterns
        assert any("semantic:" in p for p in analysis.detected_patterns)
        
        # Exact + code patterns (exact should win)
        analysis = self.analyzer.analyze_query('name:"def_function"')
        assert analysis.has_exact_patterns
        assert analysis.recommended_mode == "payload"


class TestQuerySuggestions:
    """Test query suggestion functionality"""
    
    def setup_method(self):
        """Setup test instance"""
        self.analyzer = QueryAnalyzer()
    
    def test_function_suggestions(self):
        """Test function-related suggestions"""
        suggestions = self.analyzer.get_query_suggestions("def my_func")
        
        assert len(suggestions) <= 10
        assert any("async" in suggestion for suggestion in suggestions)
        assert any("test" in suggestion for suggestion in suggestions)
        assert any("private" in suggestion for suggestion in suggestions)
    
    def test_class_suggestions(self):
        """Test class-related suggestions"""
        suggestions = self.analyzer.get_query_suggestions("class MyClass")
        
        assert len(suggestions) <= 10
        assert any("abstract" in suggestion for suggestion in suggestions)
        assert any("interface" in suggestion for suggestion in suggestions)
        assert any("base" in suggestion for suggestion in suggestions)
    
    def test_file_suggestions(self):
        """Test file extension suggestions"""
        suggestions = self.analyzer.get_query_suggestions("file:test")
        
        assert len(suggestions) <= 10
        extensions = [".py", ".js", ".ts", ".go", ".rs", ".java"]
        assert any(any(ext in suggestion for ext in extensions) for suggestion in suggestions)
    
    def test_semantic_suggestions(self):
        """Test semantic query suggestions"""
        suggestions = self.analyzer.get_query_suggestions("find code")
        
        assert len(suggestions) <= 10
        expected_additions = ["that handles", "similar to", "in file", "with pattern"]
        assert any(any(addition in suggestion for addition in expected_additions) 
                  for suggestion in suggestions)
    
    def test_empty_query_suggestions(self):
        """Test suggestions for empty query"""
        suggestions = self.analyzer.get_query_suggestions("")
        assert len(suggestions) == 0
        
        suggestions = self.analyzer.get_query_suggestions(None)
        assert len(suggestions) == 0
    
    def test_suggestion_limit(self):
        """Test that suggestions are limited to 10"""
        # Query that might generate many suggestions
        suggestions = self.analyzer.get_query_suggestions("def function class test file")
        assert len(suggestions) <= 10


class TestQueryAnalysisDataclass:
    """Test QueryAnalysis dataclass"""
    
    def test_query_analysis_creation(self):
        """Test QueryAnalysis object creation"""
        analysis = QueryAnalysis(
            original_query="test query",
            cleaned_query="test query",
            word_count=2,
            has_exact_patterns=False,
            has_code_patterns=True,
            detected_patterns=["code:test"],
            confidence=0.7,
            recommended_mode="hybrid"
        )
        
        assert analysis.original_query == "test query"
        assert analysis.cleaned_query == "test query"
        assert analysis.word_count == 2
        assert not analysis.has_exact_patterns
        assert analysis.has_code_patterns
        assert analysis.detected_patterns == ["code:test"]
        assert analysis.confidence == 0.7
        assert analysis.recommended_mode == "hybrid"


class TestQueryAnalyzerIntegration:
    """Test QueryAnalyzer integration scenarios"""
    
    def setup_method(self):
        """Setup test instance"""
        self.analyzer = QueryAnalyzer()
    
    def test_realistic_search_scenarios(self):
        """Test realistic search query scenarios"""
        scenarios = [
            # Developer searching for specific function
            {
                "query": "def authenticate_user",
                "expected_mode": "payload",
                "expected_patterns": ["code:def"]
            },
            # Developer looking for examples
            {
                "query": "how to handle JWT authentication in Python",
                "expected_mode": "semantic", 
                "expected_patterns": ["semantic:how to"]
            },
            # Developer searching in specific file
            {
                "query": "class UserModel in models.py",
                "expected_mode": "hybrid",
                "expected_patterns": ["code:class", "file:.py"]
            },
            # Developer looking for exact match
            {
                "query": 'name:"AuthenticationService"',
                "expected_mode": "payload",
                "expected_patterns": ["exact:\""]
            },
            # Natural language query
            {
                "query": "find functions that validate email addresses",
                "expected_mode": "semantic",
                "expected_patterns": ["semantic:find"]
            }
        ]
        
        for scenario in scenarios:
            analysis = self.analyzer.analyze_query(scenario["query"])
            
            assert analysis.recommended_mode == scenario["expected_mode"], \
                f"Query '{scenario['query']}' expected {scenario['expected_mode']}, got {analysis.recommended_mode}"
            
            for expected_pattern in scenario["expected_patterns"]:
                assert any(expected_pattern in pattern for pattern in analysis.detected_patterns), \
                    f"Expected pattern '{expected_pattern}' not found in {analysis.detected_patterns}"
    
    def test_edge_case_queries(self):
        """Test edge case queries"""
        edge_cases = [
            ("a", "payload"),  # Single character
            ("123", "payload"),  # Numbers only
            ("!@#$%", "hybrid"),  # Special characters only
            ("def", "payload"),  # Single keyword
            ("how", "payload"),  # Semantic starter but too short
            ("very long query with many words that should trigger semantic search mode", "semantic")
        ]
        
        for query, expected_mode in edge_cases:
            analysis = self.analyzer.analyze_query(query)
            assert analysis.recommended_mode == expected_mode, \
                f"Edge case '{query}' expected {expected_mode}, got {analysis.recommended_mode}"
    
    def test_mode_explanation(self):
        """Test mode choice explanation"""
        # Test exact pattern explanation
        analysis = self.analyzer.analyze_query('name:"TestClass"')
        explanation = self.analyzer.explain_mode_choice(analysis)
        assert "exact match patterns" in explanation.lower()
        
        # Test semantic pattern explanation
        analysis = self.analyzer.analyze_query("how to implement authentication")
        explanation = self.analyzer.explain_mode_choice(analysis)
        assert "natural language patterns" in explanation.lower()
        
        # Test code pattern explanation
        analysis = self.analyzer.analyze_query("def authentication handler")
        explanation = self.analyzer.explain_mode_choice(analysis)
        assert "code query" in explanation.lower() or "patterns" in explanation.lower()
        
        # Test word count explanation
        analysis = self.analyzer.analyze_query("test")
        explanation = self.analyzer.explain_mode_choice(analysis)
        assert "word" in explanation.lower()


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])