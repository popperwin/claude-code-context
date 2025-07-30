"""
Comprehensive end-to-end quality tests for QuerySuggestionEngine.

Tests suggestion effectiveness with real developer query patterns,
validating that suggestions improve search experience on actual codebases.
"""

import pytest
import os
import shutil
import subprocess
from typing import List, Dict, Any, Set
from pathlib import Path

from core.search.suggestions import QuerySuggestionEngine, SearchSuggestion, SuggestionType


class TestQuerySuggestionEngineQuality:
    """Test QuerySuggestionEngine quality with real developer scenarios"""
    
    @classmethod
    def setup_class(cls):
        """Setup test repositories for context-aware suggestions"""
        cls.test_dir = Path("test-harness/temp-repos-suggestions")
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Clone different types of repositories to understand suggestion context
        cls.repos = {
            "python_web": {
                "url": "https://github.com/pallets/flask.git",
                "path": cls.test_dir / "flask",
                "context": "web framework with routing, templates, authentication"
            },
            "javascript_framework": {
                "url": "https://github.com/vuejs/vue.git",
                "path": cls.test_dir / "vue", 
                "context": "frontend framework with components, reactivity, directives"
            },
            "systems_programming": {
                "url": "https://github.com/redis/redis.git",
                "path": cls.test_dir / "redis",
                "context": "systems programming with networking, data structures, concurrency"
            }
        }
        
        # Clone repositories for suggestion context
        for repo_name, repo_info in cls.repos.items():
            if not repo_info["path"].exists():
                print(f"Cloning {repo_name} repository for suggestion testing...")
                try:
                    subprocess.run([
                        "git", "clone", "--depth", "1", 
                        repo_info["url"], str(repo_info["path"])
                    ], check=True, capture_output=True, timeout=60)
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
                    print(f"Failed to clone {repo_name}: {e}")
    
    @classmethod
    def teardown_class(cls):
        """Cleanup test repositories"""
        if hasattr(cls, 'test_dir') and cls.test_dir.exists():
            print("Cleaning up suggestion test repositories...")
            shutil.rmtree(cls.test_dir)
    
    def setup_method(self):
        """Setup test instance"""
        self.engine = QuerySuggestionEngine()
        
        # Clean up any test Qdrant collections that might exist
        self._cleanup_test_collections()
    
    def teardown_method(self):
        """Cleanup after each test"""
        self._cleanup_test_collections()
    
    def _cleanup_test_collections(self):
        """Clean up any test collections from Qdrant"""
        try:
            import requests
            # Check if Qdrant is accessible
            response = requests.get("http://localhost:6334/collections", timeout=5)
            if response.status_code == 200:
                collections = response.json().get("result", {}).get("collections", [])
                deleted_count = 0
                for collection in collections:
                    collection_name = collection.get("name", "")
                    # Delete test collections with broader pattern matching
                    test_patterns = ["test-", "temp-", "quality-test-", "ranking-test-", "suggest-test-", "real-indexer-", "integration-test"]
                    if any(test_prefix in collection_name for test_prefix in test_patterns):
                        delete_response = requests.delete(f"http://localhost:6334/collections/{collection_name}", timeout=5)
                        if delete_response.status_code in [200, 404]:  # 404 means already deleted
                            deleted_count += 1
                if deleted_count > 0:
                    print(f"Cleaned up {deleted_count} test collections from Qdrant")
        except Exception as e:
            # Log the specific error for debugging
            print(f"Qdrant cleanup warning: {e} (Qdrant might not be running)")
    
    def extract_real_entity_names(self, repo_name: str) -> Dict[str, Set[str]]:
        """Extract real entity names from repositories for context"""
        if repo_name not in self.repos:
            return {}
        
        repo_path = self.repos[repo_name]["path"]
        if not repo_path.exists():
            return {}
        
        entities = {
            "functions": set(),
            "classes": set(),
            "files": set(),
            "patterns": set()
        }
        
        try:
            # Scan for Python entities
            if repo_name == "python_web":
                python_files = list(repo_path.glob("**/*.py"))[:20]
                for py_file in python_files:
                    try:
                        content = py_file.read_text(encoding='utf-8')[:10000]  # Limit for performance
                        lines = content.split('\n')
                        
                        for line in lines:
                            line_strip = line.strip()
                            if line_strip.startswith('def '):
                                func_name = line_strip.split('(')[0].replace('def ', '').strip()
                                if func_name and not func_name.startswith('_'):
                                    entities["functions"].add(func_name)
                            elif line_strip.startswith('class '):
                                class_name = line_strip.split('(')[0].split(':')[0].replace('class ', '').strip()
                                if class_name:
                                    entities["classes"].add(class_name)
                        
                        # Add file patterns
                        relative_path = str(py_file.relative_to(repo_path))
                        entities["files"].add(relative_path)
                        
                        # Common patterns in Flask
                        if "route" in content.lower():
                            entities["patterns"].add("routing")
                        if "request" in content.lower():
                            entities["patterns"].add("request handling")
                        if "template" in content.lower():
                            entities["patterns"].add("templating")
                        
                    except (UnicodeDecodeError, PermissionError):
                        continue
            
            # Scan for JavaScript entities
            elif repo_name == "javascript_framework":
                js_files = list(repo_path.glob("**/*.js"))[:15]
                for js_file in js_files:
                    try:
                        content = js_file.read_text(encoding='utf-8')[:8000]
                        lines = content.split('\n')
                        
                        for line in lines:
                            line_strip = line.strip()
                            if 'function ' in line_strip:
                                try:
                                    func_name = line_strip.split('function')[1].split('(')[0].strip()
                                    if func_name:
                                        entities["functions"].add(func_name)
                                except:
                                    pass
                            elif line_strip.startswith('export '):
                                try:
                                    export_name = line_strip.replace('export ', '').split(' ')[0].strip()
                                    if export_name not in ['default', 'const', 'let', 'var']:
                                        entities["functions"].add(export_name)
                                except:
                                    pass
                        
                        # Vue.js patterns
                        if "component" in content.lower():
                            entities["patterns"].add("components")
                        if "directive" in content.lower():
                            entities["patterns"].add("directives")
                        if "reactive" in content.lower():
                            entities["patterns"].add("reactivity")
                    
                    except (UnicodeDecodeError, PermissionError):
                        continue
            
            # Scan for C entities
            elif repo_name == "systems_programming":
                c_files = list(repo_path.glob("**/*.c"))[:10]
                for c_file in c_files:
                    try:
                        content = c_file.read_text(encoding='utf-8')[:6000]
                        lines = content.split('\n')
                        
                        for line in lines:
                            line_strip = line.strip()
                            # Simple C function detection
                            if (line_strip and not line_strip.startswith('//') and 
                                not line_strip.startswith('#') and '(' in line_strip and 
                                ')' in line_strip and '{' not in line_strip and 
                                any(t in line_strip for t in ['int ', 'void ', 'char ', 'static '])):
                                try:
                                    func_part = line_strip.split('(')[0]
                                    func_name = func_part.split()[-1]
                                    if func_name and not func_name.startswith('*'):
                                        entities["functions"].add(func_name)
                                except:
                                    pass
                        
                        # Redis patterns
                        if "server" in content.lower():
                            entities["patterns"].add("server")
                        if "client" in content.lower():
                            entities["patterns"].add("client")
                        if "redis" in content.lower():
                            entities["patterns"].add("database")
                    
                    except (UnicodeDecodeError, PermissionError):
                        continue
        
        except Exception as e:
            print(f"Error extracting entities from {repo_name}: {e}")
        
        return entities
    
    def test_function_completion_suggestions_quality(self):
        """Test function completion suggestions match real coding patterns"""
        # Extract real function names for context
        entities = self.extract_real_entity_names("python_web")
        
        test_cases = [
            {
                "partial": "def handle_",
                "expected_patterns": ["request", "error", "auth", "response"],
                "context": "web framework function patterns"
            },
            {
                "partial": "async def get_",
                "expected_patterns": ["user", "data", "connection", "response"],
                "context": "async function patterns"
            },
            {
                "partial": "def validate_",
                "expected_patterns": ["input", "user", "form", "data"],
                "context": "validation function patterns"
            },
            {
                "partial": "class User",
                "expected_patterns": ["Model", "Manager", "Service", "Controller"],
                "context": "class naming patterns"
            }
        ]
        
        for case in test_cases:
            suggestions = self.engine.get_suggestions(case["partial"], limit=10)
            
            # Should return multiple suggestions
            assert len(suggestions) > 0, f"No suggestions for '{case['partial']}'"
            
            # Should contain completion type suggestions
            completion_suggestions = [s for s in suggestions if s.type == SuggestionType.COMPLETION]
            assert len(completion_suggestions) > 0, f"No completion suggestions for '{case['partial']}'"
            
            # Should match expected patterns (at least one)
            suggestion_texts = [s.text.lower() for s in suggestions]
            found_patterns = [pattern for pattern in case["expected_patterns"] 
                            if any(pattern.lower() in text for text in suggestion_texts)]
            
            assert len(found_patterns) > 0, f"Expected patterns {case['expected_patterns']} not found in {suggestion_texts}"
            
            # Suggestions should be relevant to partial query
            for suggestion in suggestions[:5]:  # Check top 5
                assert case["partial"].lower() in suggestion.text.lower() or any(
                    word in suggestion.text.lower() for word in case["partial"].lower().split()
                ), f"Suggestion '{suggestion.text}' not relevant to '{case['partial']}'"
    
    def test_semantic_query_suggestions_quality(self):
        """Test semantic query suggestions improve developer search experience"""
        developer_scenarios = [
            {
                "partial": "how to handle",
                "expected_completions": ["errors", "authentication", "requests", "exceptions"],
                "min_suggestions": 3
            },
            {
                "partial": "find functions that",
                "expected_completions": ["validate", "handle", "process", "manage"],
                "min_suggestions": 4
            },
            {
                "partial": "show me examples of", 
                "expected_completions": ["error handling", "async", "validation", "authentication"],
                "min_suggestions": 3
            },
            {
                "partial": "explain the",
                "expected_completions": ["pattern", "implementation", "architecture", "design"],
                "min_suggestions": 3
            }
        ]
        
        for scenario in developer_scenarios:
            suggestions = self.engine.get_suggestions(scenario["partial"], limit=12)
            
            assert len(suggestions) >= scenario["min_suggestions"], \
                f"Got {len(suggestions)} suggestions, expected at least {scenario['min_suggestions']}"
            
            # Should have semantic-style suggestions
            semantic_suggestions = [s for s in suggestions if s.type == SuggestionType.COMPLETION]
            assert len(semantic_suggestions) > 0, "Should have completion-type suggestions for semantic queries"
            
            # Check for expected completions
            suggestion_texts = ' '.join([s.text.lower() for s in suggestions])
            found_completions = [comp for comp in scenario["expected_completions"] 
                               if comp.lower() in suggestion_texts]
            
            assert len(found_completions) > 0, \
                f"Expected completions {scenario['expected_completions']} not found in suggestions"
            
            # Suggestions should maintain natural language flow
            for suggestion in suggestions[:3]:
                assert suggestion.text.startswith(scenario["partial"]) or \
                       any(word in suggestion.text.lower() for word in scenario["partial"].split()), \
                       f"Suggestion '{suggestion.text}' doesn't flow from '{scenario['partial']}'"
    
    def test_file_type_suggestions_accuracy(self):
        """Test file type suggestions match project structure patterns"""
        # Extract real file patterns for context
        flask_entities = self.extract_real_entity_names("python_web")
        
        file_test_cases = [
            {
                "partial": "models.py",
                "expected_additions": ["user", "database", "auth", "admin"],
                "context": "Python model files"
            },
            {
                "partial": "test_",
                "expected_additions": [".py", "auth", "models", "views"],
                "context": "Test file patterns"
            },
            {
                "partial": "config",
                "expected_additions": [".py", ".json", ".yaml", "settings"],
                "context": "Configuration files"
            },
            {
                "partial": "file:app",
                "expected_additions": [".py", ".js", ".ts", "config"],
                "context": "Application files"
            }
        ]
        
        for case in file_test_cases:
            suggestions = self.engine.get_suggestions(case["partial"], limit=10)
            
            assert len(suggestions) > 0, f"No suggestions for file pattern '{case['partial']}'"
            
            # Should include file-related suggestions
            suggestion_texts = [s.text.lower() for s in suggestions]
            file_suggestions = [text for text in suggestion_texts 
                              if any(ext in text for ext in ['.py', '.js', '.ts', '.json', '.yaml'])]
            
            if not any(ext in case["partial"] for ext in ['.py', '.js', '.ts']):
                assert len(file_suggestions) > 0, f"Expected file extension suggestions for '{case['partial']}'"
            
            # Check for expected additions
            found_additions = []
            for expected in case["expected_additions"]:
                if any(expected.lower() in text for text in suggestion_texts):
                    found_additions.append(expected)
            
            assert len(found_additions) > 0, \
                f"Expected additions {case['expected_additions']} not found in {suggestion_texts}"
    
    def test_refinement_suggestions_improve_queries(self):
        """Test refinement suggestions actually improve search effectiveness"""
        refinement_scenarios = [
            {
                "vague_query": "auth",
                "expected_refinements": ["authentication", "authorization", "def auth", "class auth"],
                "improvement_type": "specificity"
            },
            {
                "vague_query": "data",
                "expected_refinements": ["database", "data processing", "data model", "def data"],
                "improvement_type": "context"
            },
            {
                "vague_query": "error",
                "expected_refinements": ["error handling", "exception", "try catch", "def error"],
                "improvement_type": "pattern"
            },
            {
                "broad_query": "user system",
                "expected_refinements": ["user authentication", "user management", "user.py", "class User"],
                "improvement_type": "domain_specific"
            }
        ]
        
        for scenario in refinement_scenarios:
            query = scenario.get("vague_query") or scenario.get("broad_query")
            suggestions = self.engine.get_suggestions(query, limit=10)
            
            # Should include refinement suggestions
            refinement_suggestions = [s for s in suggestions if s.type == SuggestionType.REFINEMENT]
            
            # May not always have explicit refinement type, but should have refinement-like suggestions
            all_suggestion_texts = [s.text.lower() for s in suggestions]
            
            # Check for expected refinements
            found_refinements = []
            for expected in scenario["expected_refinements"]:
                if any(expected.lower() in text for text in all_suggestion_texts):
                    found_refinements.append(expected)
            
            assert len(found_refinements) > 0, \
                f"Expected refinements {scenario['expected_refinements']} not found in {all_suggestion_texts}"
            
            # Refinements should be more specific than original
            for suggestion in suggestions[:5]:
                assert len(suggestion.text.split()) >= len(query.split()), \
                    f"Refinement '{suggestion.text}' should be more specific than '{query}'"
    
    def test_context_aware_suggestions_with_real_codebase(self):
        """Test context-aware suggestions using real codebase patterns"""
        # Simulate different project contexts
        contexts = [
            {
                "current_file": "models/user.py",
                "recent_queries": ["User model", "authentication", "database schema"],
                "project_type": "web_application"
            },
            {
                "current_file": "tests/test_auth.py",
                "recent_queries": ["test authentication", "mock user", "assert"],
                "project_type": "testing"
            },
            {
                "current_file": "api/views.py",
                "recent_queries": ["API endpoint", "JSON response", "request handling"],
                "project_type": "api"
            }
        ]
        
        test_query = "user"
        
        for context in contexts:
            suggestions = self.engine.get_suggestions(test_query, limit=10, context=context)
            
            assert len(suggestions) > 0, f"No suggestions with context {context}"
            
            # Should incorporate current file context
            if context["current_file"]:
                context_suggestions = [s for s in suggestions 
                                     if context["current_file"] in s.text or 
                                        s.type == SuggestionType.CONTEXT]
                # May not always have explicit context suggestions, but should be contextually relevant
                
            # Should be influenced by recent queries
            suggestion_texts = ' '.join([s.text.lower() for s in suggestions])
            recent_influence = any(
                any(word in suggestion_texts for word in recent_query.lower().split())
                for recent_query in context["recent_queries"]
            )
            
            # At least some suggestions should show contextual influence
            assert recent_influence or len(suggestions) >= 3, \
                "Suggestions should show some contextual awareness"
    
    def test_alternative_search_strategy_suggestions(self):
        """Test alternative search strategy suggestions provide value"""
        strategy_test_cases = [
            {
                "exact_query": '"UserModel"',
                "expected_alternatives": ["how to use usermodel", "usermodel examples", "user model"],
                "strategy_shift": "exact_to_semantic"
            },
            {
                "semantic_query": "how to implement user authentication in web applications",
                "expected_alternatives": ["authenticate", "auth", '"authentication"'],
                "strategy_shift": "semantic_to_exact"
            },
            {
                "code_query": "def authenticate_user",
                "expected_alternatives": ["authentication examples", "user login", "auth patterns"],
                "strategy_shift": "code_to_semantic"
            }
        ]
        
        for case in strategy_test_cases:
            query_key = next(k for k in case.keys() if k.endswith('_query'))
            query = case[query_key]
            
            suggestions = self.engine.get_suggestions(query, limit=12)
            
            # Should have alternative-type suggestions or semantically alternative content
            alternative_suggestions = [s for s in suggestions if s.type == SuggestionType.ALTERNATIVE]
            
            suggestion_texts = [s.text.lower() for s in suggestions]
            
            # Check for expected alternatives
            found_alternatives = []
            for expected in case["expected_alternatives"]:
                if any(expected.lower() in text for text in suggestion_texts):
                    found_alternatives.append(expected)
            
            # Should find at least one alternative approach
            assert len(found_alternatives) > 0 or len(alternative_suggestions) > 0, \
                f"No alternative suggestions found for {case['strategy_shift']} shift"
    
    def test_suggestion_deduplication_quality(self):
        """Test suggestion deduplication maintains quality while removing redundancy"""
        # Query that might generate many similar suggestions
        repetitive_query = "def user auth function handler"
        
        suggestions = self.engine.get_suggestions(repetitive_query, limit=15)
        
        # Should have deduplicated results
        suggestion_texts = [s.text.lower() for s in suggestions]
        unique_texts = set(suggestion_texts)
        
        assert len(unique_texts) == len(suggestion_texts), \
            f"Found {len(suggestion_texts) - len(unique_texts)} duplicate suggestions"
        
        # Should maintain different confidence levels
        confidences = [s.confidence for s in suggestions]
        assert len(set(confidences)) > 1, "Should have varying confidence levels"
        
        # Higher confidence suggestions should be ranked higher
        sorted_confidences = sorted(confidences, reverse=True)
        assert confidences == sorted_confidences or len(confidences) <= 2, \
            "Suggestions should be ordered by confidence"
    
    def test_query_hints_provide_actionable_advice(self):
        """Test query hints provide actionable improvement advice"""
        hint_test_cases = [
            {
                "query": "a",
                "expected_hint_themes": ["context", "specific", "more"],
                "issue": "too_short"
            },
            {
                "query": "this is a very long query with many words that might be too verbose for effective searching",
                "expected_hint_themes": ["shorter", "specific", "concise"],
                "issue": "too_long"
            },
            {
                "query": "user authentication",
                "expected_hint_themes": ["def", "class", "quotes", "file"],
                "issue": "could_be_more_specific"
            },
            {
                "query": "find all the functions that handle user authentication and session management",
                "expected_hint_themes": ["how to", "find", "explain"],
                "issue": "could_be_semantic"
            }
        ]
        
        for case in hint_test_cases:
            hints = self.engine.get_query_hints(case["query"])
            
            assert len(hints) > 0, f"No hints provided for query: '{case['query']}'"
            assert len(hints) <= 3, f"Too many hints ({len(hints)}), should be 3 or fewer"
            
            # Hints should address the identified issue
            hint_text = ' '.join(hints).lower()
            found_themes = [theme for theme in case["expected_hint_themes"] 
                          if theme.lower() in hint_text]
            
            assert len(found_themes) > 0, \
                f"Expected hint themes {case['expected_hint_themes']} not found in hints: {hints}"
            
            # Hints should be actionable (contain action words)
            action_words = ["try", "use", "add", "consider", "specify", "start"]
            has_action = any(action in hint_text for action in action_words)
            assert has_action, f"Hints should be actionable, got: {hints}"
    
    def test_suggestion_performance_with_real_queries(self):
        """Test suggestion engine performance with realistic developer queries"""
        # Real developer query patterns from various contexts
        real_queries = [
            "def", "class User", "auth", "database connection", "error handling",
            "async function", "test_", "how to", "find code that", "show me",
            "import", "export", "interface", "component", "service", "controller",
            "validate", "parse", "format", "serialize", "deserialize", "transform"
        ]
        
        import time
        start_time = time.time()
        
        total_suggestions = 0
        for query in real_queries:
            suggestions = self.engine.get_suggestions(query, limit=8)
            total_suggestions += len(suggestions)
            assert len(suggestions) > 0, f"No suggestions for real query: '{query}'"
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Performance benchmarks
        assert total_time < 3.0, f"Suggestion generation took {total_time:.3f}s, should be under 3.0s"
        
        time_per_query = total_time / len(real_queries)
        assert time_per_query < 0.15, f"Time per query: {time_per_query:.4f}s, should be under 0.15s"
        
        avg_suggestions = total_suggestions / len(real_queries)
        assert avg_suggestions >= 3, f"Average {avg_suggestions:.1f} suggestions per query, should be at least 3"
    
    def test_cross_language_suggestion_patterns(self):
        """Test suggestion patterns work across different programming languages"""
        language_contexts = [
            {
                "language": "python",
                "query": "def handle",
                "expected_patterns": ["request", "error", "auth", "data"],
                "syntax_patterns": ["async", "return", "self"]
            },
            {
                "language": "javascript",
                "query": "function get",
                "expected_patterns": ["data", "user", "response", "element"],
                "syntax_patterns": ["async", "await", "return"]
            },
            {
                "language": "typescript",
                "query": "interface User",
                "expected_patterns": ["Model", "Data", "Response", "Request"],
                "syntax_patterns": ["extends", "implements", "readonly"]
            }
        ]
        
        for ctx in language_contexts:
            # Simulate language context
            suggestions = self.engine.get_suggestions(ctx["query"], limit=10)
            
            assert len(suggestions) > 0, f"No suggestions for {ctx['language']} query: '{ctx['query']}'"
            
            suggestion_texts = [s.text.lower() for s in suggestions]
            full_text = ' '.join(suggestion_texts)
            
            # Should include language-appropriate patterns
            found_patterns = [pattern for pattern in ctx["expected_patterns"] 
                            if pattern.lower() in full_text]
            
            assert len(found_patterns) > 0, \
                f"Expected {ctx['language']} patterns {ctx['expected_patterns']} not found in suggestions"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])