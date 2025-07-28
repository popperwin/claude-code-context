"""
Performance benchmark tests for Sprint 2 validation.

Validates specific performance targets from Definition of Done:
- Embedding generation: <500ms average (including warmup)
- Payload search: <20ms average  
- Semantic search: <100ms average
- Hybrid search: <100ms average
"""

import pytest
import asyncio
import time
import json
import statistics
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
import psutil
import os

from tests.fixtures.real_code_samples import generate_real_entities_from_project
from core.models.entities import Entity, EntityType
from core.embeddings.stella import StellaEmbedder
from core.storage.client import HybridQdrantClient
from core.storage.indexing import BatchIndexer
from core.storage.schemas import QdrantSchema, CollectionType, CollectionManager

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
async def performance_environment():
    """Setup performance testing environment"""
    qdrant_url = "http://localhost:6334"  # Test environment port
    collection_name = "performance-benchmark"
    
    embedder = StellaEmbedder()
    client = HybridQdrantClient(url=qdrant_url, embedder=embedder)
    
    try:
        # Load model
        model_loaded = await embedder.load_model()
        if not model_loaded:
            pytest.skip("Stella model not available for performance tests")
        
        # Connect to Qdrant
        connected = await client.connect()
        if not connected:
            pytest.skip("Qdrant not available for performance tests")
        
        # Create performance test collection
        collection_manager = CollectionManager("performance-test")
        config = collection_manager.create_collection_config(CollectionType.CODE)
        config.name = collection_name
        
        await client.create_collection(config, recreate=True)
        
        yield {
            "embedder": embedder,
            "client": client,
            "collection_name": collection_name
        }
        
    finally:
        try:
            await client.client.delete_collection(collection_name)
        except:
            pass
        await client.disconnect()
        await embedder.unload_model()


@pytest.fixture(scope="module") 
def performance_entities():
    """Generate entities specifically for performance testing"""
    entities = generate_real_entities_from_project()
    
    # Filter for performance testing
    perf_entities = []
    
    # Get diverse entity types with meaningful content
    for entity_type in [EntityType.FUNCTION, EntityType.CLASS, EntityType.FILE]:
        entities_of_type = [
            e for e in entities 
            if e.entity_type == entity_type and len(e.source_code) >= 50
        ]
        perf_entities.extend(entities_of_type[:20])  # 20 of each type
    
    return perf_entities[:60]  # Total of 60 entities for performance testing


class PerformanceBenchmark:
    """Performance benchmark result tracking"""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.measurements = []
        self.metadata = {}
        self.start_time = None
        self.system_info = self._get_system_info()
    
    def start(self):
        """Start timing a measurement"""
        self.start_time = time.perf_counter()
    
    def record(self, label: str = "", metadata: Dict[str, Any] = None):
        """Record a measurement"""
        if self.start_time is None:
            raise ValueError("Must call start() before record()")
        
        elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        
        measurement = {
            "label": label,
            "elapsed_ms": elapsed_ms,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        self.measurements.append(measurement)
        self.start_time = None
        return elapsed_ms
    
    def get_statistics(self) -> Dict[str, float]:
        """Get statistical summary of measurements"""
        times = [m["elapsed_ms"] for m in self.measurements]
        
        if not times:
            return {}
        
        return {
            "count": len(times),
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "min": min(times),
            "max": max(times),
            "stdev": statistics.stdev(times) if len(times) > 1 else 0,
            "p95": self._percentile(times, 0.95),
            "p99": self._percentile(times, 0.99)
        }
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile"""
        sorted_data = sorted(data)
        index = int(percentile * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for context"""
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "platform": os.sys.platform
        }
    
    def save_results(self, output_dir: Path):
        """Save benchmark results to file"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            "test_name": self.test_name,
            "timestamp": time.time(),
            "system_info": self.system_info,
            "statistics": self.get_statistics(),
            "measurements": self.measurements,
            "metadata": self.metadata
        }
        
        filename = f"{self.test_name}_{int(time.time())}.json"
        output_file = output_dir / filename
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved benchmark results to {output_file}")
        return output_file


@pytest.mark.performance
class TestEmbeddingPerformance:
    """Embedding performance validation tests"""
    
    @pytest.mark.asyncio
    async def test_single_embedding_performance_target(self, performance_environment, performance_entities):
        """
        Test single embedding performance against <500ms target.
        
        Definition of Done requirement: <500ms embedding generation (including warmup)
        """
        env = performance_environment
        embedder = env["embedder"]
        
        benchmark = PerformanceBenchmark("single_embedding_performance")
        TARGET_MS = 500
        
        logger.info(f"Testing single embedding performance (target: <{TARGET_MS}ms)")
        
        # Test with different entity types and content lengths
        test_cases = []
        
        for entity in performance_entities[:30]:  # Test 30 entities
            # Create realistic embedding text
            text = f"{entity.name} {entity.signature} {entity.docstring[:200]}"
            test_cases.append({
                "entity_type": entity.entity_type.value,
                "text_length": len(text),
                "text": text,
                "entity_name": entity.name
            })
        
        # Perform embeddings and measure performance
        embedding_times = []
        
        for i, test_case in enumerate(test_cases):
            benchmark.start()
            
            try:
                embedding = await embedder.embed_single(test_case["text"])
                elapsed_ms = benchmark.record(
                    label=f"embedding_{i}",
                    metadata=test_case
                )
                
                # Validate embedding
                assert len(embedding) == 1024, f"Wrong embedding dimension: {len(embedding)}"
                embedding_times.append(elapsed_ms)
                
            except Exception as e:
                logger.error(f"Embedding failed for {test_case['entity_name']}: {e}")
                continue
        
        # Analyze results
        stats = benchmark.get_statistics()
        logger.info(f"Embedding performance stats: {stats}")
        
        # Save benchmark results
        results_dir = Path("test-harness/test-env/integration/performance/results")
        benchmark.save_results(results_dir)
        
        # Validate performance targets
        assert stats["mean"] < TARGET_MS, \
            f"Average embedding time {stats['mean']:.2f}ms exceeds {TARGET_MS}ms target"
        
        assert stats["p95"] < TARGET_MS * 2, \
            f"95th percentile {stats['p95']:.2f}ms exceeds {TARGET_MS * 2}ms threshold"
        
        # At least 90% should be under target
        under_target_count = sum(1 for t in embedding_times if t < TARGET_MS)
        target_rate = under_target_count / len(embedding_times)
        
        assert target_rate >= 0.9, \
            f"Only {target_rate:.1%} embeddings under {TARGET_MS}ms target (need 90%)"
        
        logger.info(f"✅ Single embedding performance: {stats['mean']:.2f}ms avg, {target_rate:.1%} under {TARGET_MS}ms")
    
    @pytest.mark.asyncio
    async def test_batch_embedding_performance(self, performance_environment, performance_entities):
        """
        Test batch embedding performance for efficiency.
        """
        env = performance_environment
        embedder = env["embedder"]
        
        benchmark = PerformanceBenchmark("batch_embedding_performance")
        
        # Test different batch sizes
        batch_sizes = [1, 5, 10, 20, 32]
        
        for batch_size in batch_sizes:
            entities_batch = performance_entities[:batch_size]
            texts = [f"{e.name} {e.signature} {e.docstring[:100]}" for e in entities_batch]
            
            logger.info(f"Testing batch embedding with {batch_size} entities")
            
            benchmark.start()
            
            response = await embedder.embed_texts(texts)
            elapsed_ms = benchmark.record(
                label=f"batch_{batch_size}",
                metadata={
                    "batch_size": batch_size,
                    "embeddings_generated": len(response.embeddings)
                }
            )
            
            # Calculate per-embedding time
            per_embedding_ms = elapsed_ms / batch_size
            
            # Batch should be more efficient than individual embeddings
            # (allowing warmup time and overhead for smaller batches)
            efficiency_threshold = 600 if batch_size < 5 else 500
            
            assert per_embedding_ms < efficiency_threshold, \
                f"Batch embedding inefficient: {per_embedding_ms:.2f}ms per embedding (batch size {batch_size})"
            
            logger.info(f"Batch {batch_size}: {elapsed_ms:.2f}ms total, {per_embedding_ms:.2f}ms per embedding")
        
        # Save results
        results_dir = Path("test-harness/test-env/integration/performance/results")
        benchmark.save_results(results_dir)
        
        logger.info("✅ Batch embedding performance test passed")


@pytest.mark.performance
class TestSearchPerformance:
    """Search performance validation tests"""
    
    @pytest.mark.asyncio
    async def test_payload_search_performance_target(self, performance_environment, performance_entities):
        """
        Test payload search performance against <20ms target.
        
        Definition of Done requirement: <20ms payload search (adjusted for realistic test environment performance)
        """
        env = performance_environment
        client = env["client"]
        collection_name = env["collection_name"]
        
        # Index entities for search testing
        indexer = BatchIndexer(client, env["embedder"], batch_size=20)
        
        logger.info(f"Indexing {len(performance_entities)} entities for payload search performance test")
        
        indexing_result = await indexer.index_entities(
            performance_entities,
            collection_name,
            show_progress=False
        )
        
        assert indexing_result.success_rate > 0.8, f"Low indexing success: {indexing_result.success_rate:.2f}"
        
        # Test payload search performance
        benchmark = PerformanceBenchmark("payload_search_performance")
        TARGET_MS = 20
        
        # Test with various search terms
        search_terms = [
            "function",
            "class",
            "config",
            "client",
            "test",
            "setup",
            "load",
            "create",
            "get",
            "process"
        ]
        
        # Add specific entity names for exact matching
        entity_names = [e.name for e in performance_entities[:10] if e.entity_type == EntityType.FUNCTION]
        search_terms.extend(entity_names)
        
        payload_times = []
        
        for search_term in search_terms:
            benchmark.start()
            
            results = await client.search_payload(
                collection_name,
                search_term,
                limit=10
            )
            
            elapsed_ms = benchmark.record(
                label=f"payload_search",
                metadata={
                    "search_term": search_term,
                    "results_count": len(results)
                }
            )
            
            payload_times.append(elapsed_ms)
        
        # Analyze results
        stats = benchmark.get_statistics()
        logger.info(f"Payload search performance stats: {stats}")
        
        # Save results
        results_dir = Path("test-harness/test-env/integration/performance/results")
        benchmark.save_results(results_dir)
        
        # Validate performance targets
        assert stats["mean"] < TARGET_MS, \
            f"Average payload search time {stats['mean']:.2f}ms exceeds {TARGET_MS}ms target"
        
        assert stats["p95"] < TARGET_MS * 3, \
            f"95th percentile {stats['p95']:.2f}ms exceeds {TARGET_MS * 3}ms threshold"
        
        # At least 70% should be under target (realistic for test environment)
        under_target_count = sum(1 for t in payload_times if t < TARGET_MS)
        target_rate = under_target_count / len(payload_times)
        
        assert target_rate >= 0.7, \
            f"Only {target_rate:.1%} payload searches under {TARGET_MS}ms target (need 70%)"
        
        logger.info(f"✅ Payload search performance: {stats['mean']:.2f}ms avg, {target_rate:.1%} under {TARGET_MS}ms")
    
    @pytest.mark.asyncio
    async def test_semantic_search_performance_target(self, performance_environment, performance_entities):
        """
        Test semantic search performance against <100ms target.
        """
        env = performance_environment
        client = env["client"]
        collection_name = env["collection_name"]
        
        benchmark = PerformanceBenchmark("semantic_search_performance")
        TARGET_MS = 100  # Adjusted for realistic performance including embedding time
        
        # Test semantic search with conceptual queries
        semantic_queries = [
            "configuration management",
            "embedding generation",
            "file processing",
            "error handling",
            "data validation",
            "async operations",
            "client connection",
            "search functionality",
            "performance optimization",
            "test framework"
        ]
        
        semantic_times = []
        
        for query in semantic_queries:
            benchmark.start()
            
            results = await client.search_semantic(
                collection_name,
                query,
                limit=10
            )
            
            elapsed_ms = benchmark.record(
                label=f"semantic_search",
                metadata={
                    "query": query,
                    "results_count": len(results)
                }
            )
            
            semantic_times.append(elapsed_ms)
        
        # Analyze results
        stats = benchmark.get_statistics()
        logger.info(f"Semantic search performance stats: {stats}")
        
        # Save results
        results_dir = Path("test-harness/test-env/integration/performance/results")
        benchmark.save_results(results_dir)
        
        # Validate performance targets
        assert stats["mean"] < TARGET_MS, \
            f"Average semantic search time {stats['mean']:.2f}ms exceeds {TARGET_MS}ms target"
        
        assert stats["p95"] < TARGET_MS * 3, \
            f"95th percentile {stats['p95']:.2f}ms exceeds {TARGET_MS * 3}ms threshold"
        
        logger.info(f"✅ Semantic search performance: {stats['mean']:.2f}ms avg")
    
    @pytest.mark.asyncio 
    async def test_hybrid_search_performance_target(self, performance_environment, performance_entities):
        """
        Test hybrid search performance against <100ms target.
        """
        env = performance_environment
        client = env["client"]
        collection_name = env["collection_name"]
        
        benchmark = PerformanceBenchmark("hybrid_search_performance")
        TARGET_MS = 100
        
        # Test hybrid search with mixed queries (exact + conceptual)
        hybrid_queries = [
            "function configuration",
            "class management",
            "client connection",
            "test validation",
            "setup process",
            "error handling",
            "data processing",
            "search optimization"
        ]
        
        hybrid_times = []
        
        for query in hybrid_queries:
            benchmark.start()
            
            results = await client.search_hybrid(
                collection_name,
                query,
                limit=10,
                payload_weight=0.7,
                semantic_weight=0.3
            )
            
            elapsed_ms = benchmark.record(
                label=f"hybrid_search",
                metadata={
                    "query": query,
                    "results_count": len(results)
                }
            )
            
            hybrid_times.append(elapsed_ms)
        
        # Analyze results
        stats = benchmark.get_statistics()
        logger.info(f"Hybrid search performance stats: {stats}")
        
        # Save results
        results_dir = Path("test-harness/test-env/integration/performance/results")
        benchmark.save_results(results_dir)
        
        # Validate performance targets
        assert stats["mean"] < TARGET_MS, \
            f"Average hybrid search time {stats['mean']:.2f}ms exceeds {TARGET_MS}ms target"
        
        assert stats["p95"] < TARGET_MS * 1.5, \
            f"95th percentile {stats['p95']:.2f}ms exceeds {TARGET_MS * 1.5}ms threshold"
        
        logger.info(f"✅ Hybrid search performance: {stats['mean']:.2f}ms avg")


@pytest.mark.performance
class TestScalabilityPerformance:
    """Scalability and load performance tests"""
    
    @pytest.mark.asyncio
    async def test_indexing_throughput_performance(self, performance_environment, performance_entities):
        """
        Test indexing throughput performance.
        
        Target: Reasonable throughput for batch operations
        """
        env = performance_environment
        client = env["client"]
        indexer = BatchIndexer(client, env["embedder"], batch_size=20)
        collection_name = f"{env['collection_name']}_throughput"
        
        # Create separate collection for throughput testing
        collection_manager = CollectionManager("throughput-test")
        config = collection_manager.create_collection_config(CollectionType.CODE)
        config.name = collection_name
        await client.create_collection(config, recreate=True)
        
        try:
            benchmark = PerformanceBenchmark("indexing_throughput")
            
            # Test different batch sizes
            batch_sizes = [10, 20, 30]
            
            for batch_size in batch_sizes:
                entities_batch = performance_entities[:batch_size]
                
                logger.info(f"Testing indexing throughput with {batch_size} entities")
                
                benchmark.start()
                
                result = await indexer.index_entities(
                    entities_batch,
                    collection_name,
                    show_progress=False
                )
                
                elapsed_ms = benchmark.record(
                    label=f"indexing_batch_{batch_size}",
                    metadata={
                        "batch_size": batch_size,
                        "success_rate": result.success_rate,
                        "entities_per_second": result.entities_per_second
                    }
                )
                
                # Calculate throughput
                throughput = batch_size / (elapsed_ms / 1000)  # entities per second
                
                logger.info(f"Batch {batch_size}: {elapsed_ms:.2f}ms, {throughput:.1f} entities/sec")
                
                # Basic throughput validation (should process at least 1 entity/second)
                assert throughput >= 1.0, f"Low throughput: {throughput:.1f} entities/sec"
            
            # Save results
            results_dir = Path("test-harness/test-env/integration/performance/results")
            benchmark.save_results(results_dir)
            
            logger.info("✅ Indexing throughput performance test passed")
            
        finally:
            try:
                await client.client.delete_collection(collection_name)
            except:
                pass
    
    @pytest.mark.asyncio
    async def test_concurrent_search_performance(self, performance_environment, performance_entities):
        """
        Test performance under concurrent search load.
        """
        env = performance_environment
        client = env["client"]
        collection_name = env["collection_name"]
        
        # First ensure we have indexed data
        indexer = BatchIndexer(client, env["embedder"], batch_size=20)
        await indexer.index_entities(performance_entities, collection_name, show_progress=False)
        
        benchmark = PerformanceBenchmark("concurrent_search_performance")
        
        # Define search tasks
        search_tasks = []
        for i in range(10):  # 10 concurrent searches
            search_tasks.append(
                client.search_payload(collection_name, f"function_{i % 5}", limit=5)
            )
        
        logger.info(f"Testing {len(search_tasks)} concurrent searches")
        
        benchmark.start()
        
        # Execute concurrent searches
        results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        elapsed_ms = benchmark.record(
            label="concurrent_searches",
            metadata={
                "concurrent_count": len(search_tasks),
                "successful_searches": sum(1 for r in results if not isinstance(r, Exception))
            }
        )
        
        # Validate results
        successful_searches = sum(1 for r in results if not isinstance(r, Exception))
        success_rate = successful_searches / len(search_tasks)
        
        assert success_rate >= 0.8, f"Low concurrent search success rate: {success_rate:.1%}"
        
        # Calculate average time per search
        avg_time_per_search = elapsed_ms / len(search_tasks)
        
        logger.info(f"Concurrent searches: {elapsed_ms:.2f}ms total, {avg_time_per_search:.2f}ms avg per search")
        
        # Should not be significantly slower than individual searches
        assert avg_time_per_search < 20, f"Concurrent search too slow: {avg_time_per_search:.2f}ms avg"
        
        # Save results
        results_dir = Path("test-harness/test-env/integration/performance/results")
        benchmark.save_results(results_dir)
        
        logger.info("✅ Concurrent search performance test passed")


@pytest.mark.performance
class TestMemoryPerformance:
    """Memory usage and efficiency tests"""
    
    @pytest.mark.asyncio
    async def test_memory_usage_during_operations(self, performance_environment, performance_entities):
        """
        Test memory usage during various operations.
        """
        env = performance_environment
        client = env["client"]
        embedder = env["embedder"]
        collection_name = env["collection_name"]
        
        benchmark = PerformanceBenchmark("memory_usage")
        
        # Monitor memory usage
        def get_memory_usage():
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        
        initial_memory = get_memory_usage()
        logger.info(f"Initial memory usage: {initial_memory:.1f} MB")
        
        # Test embedding memory usage
        texts = [f"{e.name} {e.signature}" for e in performance_entities[:20]]
        
        memory_before_embedding = get_memory_usage()
        response = await embedder.embed_texts(texts)
        memory_after_embedding = get_memory_usage()
        
        embedding_memory_increase = memory_after_embedding - memory_before_embedding
        
        # Test indexing memory usage
        indexer = BatchIndexer(client, embedder, batch_size=10)
        
        memory_before_indexing = get_memory_usage()
        result = await indexer.index_entities(performance_entities[:30], collection_name, show_progress=False)
        memory_after_indexing = get_memory_usage()
        
        indexing_memory_increase = memory_after_indexing - memory_before_indexing
        
        logger.info(f"Memory usage - Embedding: +{embedding_memory_increase:.1f} MB, Indexing: +{indexing_memory_increase:.1f} MB")
        
        # Memory usage should be reasonable (less than 500MB increase for these operations)
        total_memory_increase = memory_after_indexing - initial_memory
        assert total_memory_increase < 500, f"Excessive memory usage: +{total_memory_increase:.1f} MB"
        
        # Save memory benchmark
        benchmark.metadata = {
            "initial_memory_mb": initial_memory,
            "embedding_memory_increase_mb": embedding_memory_increase,
            "indexing_memory_increase_mb": indexing_memory_increase,
            "total_memory_increase_mb": total_memory_increase
        }
        
        results_dir = Path("test-harness/test-env/integration/performance/results")
        benchmark.save_results(results_dir)
        
        logger.info(f"✅ Memory usage test passed - Total increase: {total_memory_increase:.1f} MB")


if __name__ == "__main__":
    # Run performance tests directly
    pytest.main([__file__, "-v", "-m", "performance"])