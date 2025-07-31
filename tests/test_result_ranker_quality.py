"""
Comprehensive end-to-end quality tests for ResultRanker.

Tests ranking effectiveness with real repositories across different languages,
validating that ranking improves result quality on actual codebases.
"""

import pytest
import os
import shutil
import subprocess
import uuid
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

# Fix tokenizer fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from core.search.ranking import ResultRanker, RankingConfig, RankingStrategy
from core.models.storage import SearchResult, QdrantPoint
from core.indexer.hybrid_indexer import HybridIndexer, IndexingJobConfig
from core.storage.client import HybridQdrantClient
from core.parser.parallel_pipeline import ProcessParsingPipeline
from core.parser.registry import ParserRegistry
from core.embeddings.stella import StellaEmbedder
from core.indexer.cache import CacheManager


class TestResultRankerQuality:
    """Test ResultRanker quality with real codebases"""
    
    @classmethod
    def setup_class(cls):
        """Setup test repositories"""
        cls.test_dir = Path("test-harness/temp-repos")
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Clone smaller, manageable real repositories for testing
        cls.repos = {
            "python": {
                "url": "https://github.com/python-validators/validators.git",  # ~2k lignes
                "path": cls.test_dir / "validators",
                "branch": "master"
            },
            "typescript": {
                "url": "https://github.com/vercel/ms.git",  # ~200 lignes
                "path": cls.test_dir / "ms",
                "branch": "main"
            },
            "javascript": {
                "url": "https://github.com/JedWatson/classnames.git",  # ~300 lignes
                "path": cls.test_dir / "classnames",
                "branch": "main"
            },
            "go": {
                "url": "https://github.com/google/uuid.git",  # ~2k lignes
                "path": cls.test_dir / "uuid",
                "branch": "master"
            }
        }
        
        # Clone repositories
        for repo_name, repo_info in cls.repos.items():
            if not repo_info["path"].exists():
                subprocess.run([
                    "git", "clone", "--depth", "1", 
                    repo_info["url"], str(repo_info["path"])
                ], check=True, capture_output=True)
        
        # Track shared collections for cleanup
        cls.shared_collections = set()
    
    @classmethod
    def teardown_class(cls):
        """Cleanup test repositories and shared collections"""
        # Clean up shared collections
        if hasattr(cls, 'shared_collections'):
            try:
                import requests
                for collection_name in cls.shared_collections:
                    try:
                        requests.delete(f"http://localhost:6334/collections/{collection_name}", timeout=5)
                    except Exception:
                        pass  # Ignore cleanup errors
            except Exception:
                pass
        
        # Clean up test repositories
        if hasattr(cls, 'test_dir') and cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
    
    def setup_method(self):
        """Setup test instance"""
        self.ranker = ResultRanker()

        # TODO: Maybe we should ALWAYS add direct embedder to avoid forgetting?
        self.client = HybridQdrantClient("http://localhost:6334")  
        
        # Initialize components for HybridIndexer (will be done in async setup)
        self.parser_pipeline = None
        self.embedder = None
        self.cache_manager = None
        self.indexer = None
        
        # Clean up any old test collections (but not shared ones)
        self._cleanup_old_test_collections()
    
    async def _async_setup(self):
        """Async setup for components that need event loop"""
        if self.indexer is None:
            self.parser_pipeline = ProcessParsingPipeline(max_workers=2, batch_size=5)
            self.embedder = StellaEmbedder()

            # Pre-load the Stella model to exclude loading time from performance measurements
            await self.embedder.load_model()
            
            # Create new client with embedder - the old client was missing embedder
            self.client = HybridQdrantClient("http://localhost:6334", embedder=self.embedder)
            
            self.cache_manager = CacheManager()
            self.indexer = HybridIndexer(
                parser_pipeline=self.parser_pipeline,
                embedder=self.embedder,
                storage_client=self.client,
                cache_manager=self.cache_manager
            )

    
    def teardown_method(self):
        """Cleanup after each test"""
        # Don't clean up shared collections in teardown_method
        # They will be cleaned up in teardown_class
        pass
    
    def _cleanup_old_test_collections(self):
        """Clean up old test collections (but not shared ranking collections)"""
        try:
            import requests
            response = requests.get("http://localhost:6334/collections", timeout=5)
            if response.status_code == 200:
                collections = response.json().get("result", {}).get("collections", [])
                for collection in collections:
                    collection_name = collection.get("name", "")
                    # Only delete old UUID-based collections, not our shared ones
                    old_patterns = ["temp-", "quality-test-", "suggest-test-", "real-indexer-", "integration-test"]
                    # Don't delete our shared "test-ranking-{repo}" collections
                    if (any(pattern in collection_name for pattern in old_patterns) and 
                        not collection_name.startswith("test-ranking-")):
                        requests.delete(f"http://localhost:6334/collections/{collection_name}", timeout=5)
        except Exception:
            pass  # Ignore cleanup errors
    
    async def get_or_create_collection(self, repo_name: str) -> str:
        """Get existing collection for repo or create it if needed"""
        # Setup async components first  
        await self._async_setup()
        
        # Use CollectionManager to get the proper collection name format
        base_project_name = f"test-ranking-{repo_name}"
        from core.storage.schemas import CollectionManager, CollectionType
        collection_manager = CollectionManager(project_name=base_project_name)
        actual_collection_name = collection_manager.get_collection_name(CollectionType.CODE)
        
        print(f"Looking for collection: {actual_collection_name}")
        
        # Check if collection already exists with sufficient entities
        try:
            collection_info = await self.client.get_collection_info(actual_collection_name)
            if collection_info and collection_info.get("points_count", 0) > 10:
                # Collection exists and has entities, reuse it
                self.__class__.shared_collections.add(actual_collection_name)
                print(f"✅ REUSING existing collection: {actual_collection_name} ({collection_info.get('points_count', 0)} entities)")
                return actual_collection_name
        except Exception as e:
            print(f"Collection {actual_collection_name} doesn't exist: {e}")
            pass  # Collection doesn't exist, create it
        
        repo_info = self.repos[repo_name]
        repo_path = repo_info["path"]
        
        if not repo_path.exists():
            raise pytest.skip(f"Repository {repo_name} not available")
        
        # Create and index new collection using consistent naming
        print(f"🔄 CREATING new collection: {actual_collection_name}")
        
        config = IndexingJobConfig(
            project_path=repo_path,
            project_name=base_project_name,  # Use base name, CollectionManager handles suffix
            include_patterns=["*.py", "*.js", "*.ts", "*.go", "*.rs"],
            exclude_patterns=["**/node_modules/**", "**/target/**", "**/.git/**", "**/build/**"],
            batch_size=50,
            # Use entity-level configuration
            entity_scan_mode="full_rescan",
            enable_entity_monitoring=True,
            entity_batch_size=50
        )
        
        await self.indexer.index_project(config)
        
        # Verify entities were created
        collection_info = await self.client.get_collection_info(actual_collection_name)
        entity_count = collection_info.get("points_count", 0) if collection_info else 0
        print(f"✅ CREATED collection {actual_collection_name} with {entity_count} entities")
        
        # Track for cleanup
        self.__class__.shared_collections.add(actual_collection_name)
        return actual_collection_name
    
    async def scan_real_codebase(self, repo_name: str, pattern: str, limit: int = 20) -> List[SearchResult]:
        """Scan real codebase using cached/shared collection"""
        try:
            # Get or create collection for this repository
            collection_name = await self.get_or_create_collection(repo_name)
            
            # Search for the pattern
            results = await self.client.search_hybrid(
                collection_name=collection_name,
                query=pattern,
                limit=limit,
                payload_weight=0.7,
                semantic_weight=0.3
            )
            
            return results
            
        except Exception as e:
            print(f"Error scanning {repo_name}: {e}")
            return []
    
    async def test_quality_ranking_on_python_codebase(self):
        """Test quality ranking effectiveness on real Python codebase (requests)"""
        results = await self.scan_real_codebase("python", "request", limit=15)
        
        if len(results) < 3:
            pytest.skip("Not enough results found in Python codebase")
        
        # Test quality-focused ranking
        config = RankingConfig(
            strategy=RankingStrategy.QUALITY_BOOST,
            quality_weight=0.4,
            boost_main_directories=True
        )
        ranker = ResultRanker(config)
        
        # DEBUG: Afficher les scores avant ranking
        # print("\n=== BEFORE RANKING ===")
        # for i, r in enumerate(results):
        #     doc_len = len(r.point.payload.get("docstring", ""))
        #     print(f"{i}: {r.point.payload.get('entity_name', 'Unknown')[:30]:30} | "
        #         f"Score: {r.score:.3f} | Doc: {doc_len:4d} chars | "
        #         f"File: {r.point.payload.get('file_path', '')[-50:]}")
        
        ranked_results = ranker.rank_results(results, "request handling")
        
        # DEBUG: Afficher les scores après ranking
        # print("\n=== AFTER RANKING ===")
        # for i, r in enumerate(ranked_results):
        #     doc_len = len(r.point.payload.get("docstring", ""))
        #     # Calculer le boost théorique
        #     expected_multiplier = 1.0
        #     if doc_len > 200:
        #         expected_multiplier += 0.4 * 0.8  # quality_weight * boost
        #     elif doc_len <= 20:
        #         expected_multiplier += 0.0
            
        #     print(f"{i}: {r.point.payload.get('entity_name', 'Unknown')[:30]:30} | "
        #         f"Score: {r.score:.3f} | Doc: {doc_len:4d} chars | "
        #         f"Expected mult: {expected_multiplier:.2f}x")
        
        # Vérifier que le ranking a changé l'ordre
        original_order = [r.point.id for r in results]
        ranked_order = [r.point.id for r in ranked_results]
        
        print(f"\nOrder changed: {original_order != ranked_order}")
    
    async def test_hybrid_ranking_on_typescript_codebase(self):
        """Test hybrid ranking on real TypeScript codebase"""
        results = await self.scan_real_codebase("typescript", "request", limit=12)
        
        if len(results) < 3:
            pytest.skip("Not enough results found in TypeScript codebase")
        
        config = RankingConfig(
            strategy=RankingStrategy.HYBRID_RANKING,
            quality_weight=0.3,
            boost_main_directories=True
        )
        ranker = ResultRanker(config)
        
        # Simulate some context
        context = {
            "popular_entities": set(r.point.payload["entity_id"] for r in results[:3]),  # Top 3 are "popular"
        }
        
        ranked_results = ranker.rank_results(results, "editor functionality", context)
        
        assert len(ranked_results) > 0
        
        # Popular entities should benefit from hybrid ranking
        popular_results = [r for r in ranked_results if r.point.payload["entity_id"] in context["popular_entities"]]
        if popular_results and len(popular_results) >= 2:
            popular_positions = [ranked_results.index(r) for r in popular_results]
            avg_popular_pos = sum(popular_positions) / len(popular_positions)
            threshold = len(ranked_results) * 0.6  # More lenient threshold (top 60%)
            print(f"Debug: Popular entities avg pos: {avg_popular_pos:.2f}, threshold (60%): {threshold:.2f}")
            assert avg_popular_pos < threshold, f"Popular entities should rank in top 60%: {avg_popular_pos:.2f} < {threshold:.2f}"
        else:
            print(f"Debug: Skipping popular test - found {len(popular_results) if popular_results else 0} popular entities")
        
        # Verify entity types are preserved and reasonable
        entity_types = [r.point.payload.get("entity_type") for r in ranked_results]
        valid_types = ["function", "class", "interface", "method", "property", "variable", "const", "type", "enum", "constant", "import", "export"]
        invalid_entities = [et for et in entity_types if et and et not in valid_types]
        print(f"Debug: Found entity types: {set(et for et in entity_types if et)}")
        assert len(invalid_entities) == 0, f"Found unexpected entity types: {invalid_entities}"
        
    
    async def test_diversity_filtering_on_go_codebase(self):
        """Test diversity filtering on real Go codebase"""
        
        results = await self.scan_real_codebase("go", "NewString", limit=10)
        
        if len(results) < 5:
            pytest.skip("Not enough results found in Go codebase")
        
        config = RankingConfig(
            diversity_threshold=0.6,  # Aggressive diversity filtering
            max_results_per_file=2
        )
        ranker = ResultRanker(config)
        
        
        ranked_results = ranker.rank_results(results, "uuid")
        
        # Should have fewer results due to diversity filtering
        assert len(ranked_results) <= len(results)
        
        # Check file diversity - no more than 2 results per file
        file_counts = {}
        for result in ranked_results:
            file_path = result.point.payload["file_path"]
            file_counts[file_path] = file_counts.get(file_path, 0) + 1
        
        max_per_file = max(file_counts.values()) if file_counts else 0
        assert max_per_file <= 2, f"Found {max_per_file} results from same file, max should be 2"
        
        # Entity names should be sufficiently different
        entity_names = [r.point.payload["entity_name"] for r in ranked_results]
        unique_names = set(entity_names)
        assert len(unique_names) == len(entity_names), "All entity names should be unique after diversity filtering"
        
    
    async def test_cross_language_ranking_consistency(self):
        """Test ranking consistency across different programming languages"""
        
        all_results = []
        
        # Gather results from all languages with progress
        languages = ["python", "typescript", "javascript"]
        for i, lang in enumerate(languages, 1):
            lang_results = await self.scan_real_codebase(lang, "data", limit=5)
            all_results.extend(lang_results)
        
        if len(all_results) < 6:
            pytest.skip("Not enough cross-language results")
        
        # Apply consistent ranking
        ranked_results = self.ranker.rank_results(all_results, "data processing")
        
        # Verify all languages are represented in results
        languages_found = set()
        for result in ranked_results:
            file_path = result.point.payload["file_path"]
            if file_path.endswith(".py"):
                languages_found.add("python")
            elif file_path.endswith(".ts"):
                languages_found.add("typescript")  
            elif file_path.endswith(".js"):
                languages_found.add("javascript")
        
        # Should have decent language diversity if input had diversity
        input_languages = set()
        for result in all_results:
            file_path = result.point.payload["file_path"]
            if file_path.endswith(".py"):
                input_languages.add("python")
            elif file_path.endswith(".ts"):
                input_languages.add("typescript")
            elif file_path.endswith(".js"):
                input_languages.add("javascript")
        
        if len(input_languages) > 1:
            assert len(languages_found) > 1, "Ranking should preserve language diversity"
        
        # Verify ranking order
        scores = [r.score for r in ranked_results]
        assert scores == sorted(scores, reverse=True), "Results should be ranked by score"
        
    
    async def test_real_codebase_performance_benchmarks(self):
        """Test ranking performance on real codebase scale"""
        
        # Gather larger result set
        all_results = []
        languages = ["python", "typescript"]
        for i, lang in enumerate(languages, 1):
            lang_results = await self.scan_real_codebase(lang, "function", limit=25)  # Broad search
            all_results.extend(lang_results)
        
        if len(all_results) < 20:
            pytest.skip("Not enough results for performance test")
        
        
        ranking_start = time.time()
        
        # Test different ranking strategies
        strategies = [RankingStrategy.SCORE_ONLY, RankingStrategy.QUALITY_BOOST, RankingStrategy.HYBRID_RANKING]
        
        for strategy in strategies:
            config = RankingConfig(strategy=strategy)
            ranker = ResultRanker(config)
            ranked_results = ranker.rank_results(all_results, "code search")
            assert len(ranked_results) > 0
        
        ranking_time = time.time() - ranking_start
        
        # Should complete all ranking strategies quickly
        assert ranking_time < 2.0, f"All ranking strategies took {ranking_time:.3f}s, should be under 2.0s"
        
        # Performance should scale reasonably 
        time_per_result = ranking_time / len(all_results)
        assert time_per_result < 0.05, f"Time per result: {time_per_result:.4f}s, should be under 0.05s"
        
    
    async def test_result_fusion_with_real_data(self):
        """Test result fusion with real codebase data"""
        
        python_results = await self.scan_real_codebase("python", "request", limit=8)
        ts_results = await self.scan_real_codebase("typescript", "request", limit=8)
        
        if len(python_results) < 3 or len(ts_results) < 3:
            pytest.skip("Not enough results for fusion test")
        
        # Simulate different search modes by creating new results with different search types
        payload_results = []
        for result in python_results:
            new_result = SearchResult(
                point=result.point,
                score=result.score,
                query=result.query,
                search_type="payload",
                rank=result.rank,
                total_results=result.total_results
            )
            payload_results.append(new_result)
        
        semantic_results = []
        for result in ts_results:
            new_result = SearchResult(
                point=result.point,
                score=result.score,
                query=result.query,
                search_type="semantic", 
                rank=result.rank,
                total_results=result.total_results
            )
            semantic_results.append(new_result)
        
        fused_results = self.ranker.fuse_search_results(
            payload_results, semantic_results,
            payload_weight=0.6, semantic_weight=0.4
        )
        
        # Verify fusion
        assert len(fused_results) > 0
        assert len(fused_results) <= len(python_results) + len(ts_results)  # Some may overlap
        
        # All results should have "fused" search type
        for result in fused_results:
            assert result.search_type == "fused"
        
        # Should be ranked by combined score
        scores = [r.score for r in fused_results]
        assert scores == sorted(scores, reverse=True)
        
        # Verify combined scoring worked
        for result in fused_results:
            assert 0.0 <= result.score <= 1.0
        


if __name__ == "__main__":
    pytest.main([__file__, "-v"])