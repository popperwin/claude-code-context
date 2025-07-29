"""
Tests for ProcessPoolExecutor-based parallel parsing pipeline.

Tests the performance and correctness of multiprocessing-based
file parsing with batch processing and error handling.
"""

import pytest
import tempfile
import time
from pathlib import Path
from typing import List
from unittest.mock import Mock, patch
import multiprocessing as mp

from core.parser.parallel_pipeline import (
    ProcessParsingPipeline,
    BatchParseRequest,
    BatchParseResult,
    PipelineStats,
    _parse_file_batch,
    parse_files_parallel,
    parse_directory_parallel
)
from core.parser.base import ParseResult
from core.parser.registry import parser_registry
from core.models.entities import Entity, EntityType, SourceLocation


class TestBatchParseRequest:
    """Test BatchParseRequest data structure"""
    
    def test_batch_parse_request_creation(self):
        """Test creating a batch parse request"""
        files = [Path("test1.py"), Path("test2.py")]
        request = BatchParseRequest(
            file_paths=files,
            batch_id=1,
            registry_state={"test": "data"}
        )
        
        assert request.batch_id == 1
        assert len(request.file_paths) == 2
        assert request.registry_state == {"test": "data"}
        assert all(isinstance(p, Path) for p in request.file_paths)


class TestBatchParseResult:
    """Test BatchParseResult data structure"""
    
    def test_batch_result_success_rate(self):
        """Test success rate calculation"""
        result = BatchParseResult(
            batch_id=1,
            results=[],
            success_count=8,
            failure_count=2,
            parse_time=1.5,
            total_entities=150,
            total_relations=75
        )
        
        assert result.success_rate == 0.8  # 8 out of 10
    
    def test_batch_result_zero_files(self):
        """Test success rate with zero files"""
        result = BatchParseResult(
            batch_id=1,
            results=[],
            success_count=0,
            failure_count=0,
            parse_time=0.1,
            total_entities=0,
            total_relations=0
        )
        
        assert result.success_rate == 0.0


class TestPipelineStats:
    """Test PipelineStats calculations"""
    
    def test_pipeline_stats_calculations(self):
        """Test various statistics calculations"""
        stats = PipelineStats(
            total_files=100,
            processed_files=95,
            successful_files=90,
            failed_files=5,
            total_entities=450,
            total_relations=200,
            total_time=10.0,
            batches_processed=10,
            average_batch_time=1.0
        )
        
        assert stats.success_rate == 0.9  # 90/100
        assert stats.files_per_second == 9.5  # 95/10
        assert stats.entities_per_second == 45.0  # 450/10
    
    def test_pipeline_stats_zero_time(self):
        """Test stats with zero time"""
        stats = PipelineStats(total_time=0.0)
        
        assert stats.files_per_second == 0.0
        assert stats.entities_per_second == 0.0


class TestProcessParsingPipeline:
    """Test the main ProcessParsingPipeline class"""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization with defaults"""
        pipeline = ProcessParsingPipeline()
        
        assert pipeline.max_workers > 0
        assert pipeline.batch_size == 10
        assert pipeline.timeout == 300.0
        assert pipeline.registry is not None
    
    def test_pipeline_custom_parameters(self):
        """Test pipeline with custom parameters"""
        pipeline = ProcessParsingPipeline(
            max_workers=8,
            batch_size=5,
            timeout=60.0
        )
        
        assert pipeline.max_workers == 8
        assert pipeline.batch_size == 5
        assert pipeline.timeout == 60.0
    
    def test_create_batches(self):
        """Test file batching logic"""
        pipeline = ProcessParsingPipeline(batch_size=3)
        
        files = [Path(f"file{i}.py") for i in range(10)]
        batches = pipeline.create_batches(files)
        
        assert len(batches) == 4  # 3 + 3 + 3 + 1
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3  
        assert len(batches[2]) == 3
        assert len(batches[3]) == 1  # Last batch with remainder
    
    def test_create_batches_empty(self):
        """Test batching with empty file list"""
        pipeline = ProcessParsingPipeline()
        
        batches = pipeline.create_batches([])
        assert batches == []
    
    def test_create_batches_single_file(self):
        """Test batching with single file"""
        pipeline = ProcessParsingPipeline(batch_size=5)
        
        files = [Path("single.py")]
        batches = pipeline.create_batches(files)
        
        assert len(batches) == 1
        assert len(batches[0]) == 1
        assert batches[0][0] == Path("single.py")


class TestParseFileBatch:
    """Test the _parse_file_batch worker function"""
    
    @pytest.fixture
    def temp_python_files(self):
        """Create temporary Python files for testing"""
        files = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create valid Python files
            for i in range(3):
                file_path = temp_path / f"test{i}.py"
                file_path.write_text(f"""
def function_{i}():
    '''Function {i} docstring'''
    return {i}

class Class_{i}:
    def method_{i}(self):
        return {i * 10}
""")
                files.append(file_path)
            
            # Create invalid file
            invalid_file = temp_path / "invalid.py"
            invalid_file.write_text("def broken_syntax(:\n    pass")
            files.append(invalid_file)
            
            yield files
    
    def test_parse_file_batch_success(self, temp_python_files):
        """Test successful batch parsing"""
        request = BatchParseRequest(
            file_paths=temp_python_files[:3],  # Only valid files
            batch_id=1,
            registry_state={}
        )
        
        result = _parse_file_batch(request)
        
        assert result.batch_id == 1
        assert result.success_count == 3
        assert result.failure_count == 0
        assert len(result.results) == 3
        assert result.total_entities > 0  # Should have extracted entities
        assert result.parse_time > 0
        assert len(result.errors) == 0
    
    def test_parse_file_batch_with_errors(self, temp_python_files):
        """Test batch parsing with some errors"""
        request = BatchParseRequest(
            file_paths=temp_python_files,  # Includes invalid file
            batch_id=2,
            registry_state={}
        )
        
        result = _parse_file_batch(request)
        
        assert result.batch_id == 2
        assert result.success_count == 3  # 3 valid files
        assert result.failure_count >= 1  # At least the invalid file
        assert len(result.results) >= 3
        assert result.total_entities > 0
        assert result.parse_time > 0
        # May have errors depending on how parser handles syntax errors
    
    def test_parse_file_batch_nonexistent_files(self):
        """Test batch parsing with nonexistent files"""
        request = BatchParseRequest(
            file_paths=[Path("nonexistent1.py"), Path("nonexistent2.py")],
            batch_id=3,
            registry_state={}
        )
        
        result = _parse_file_batch(request)
        
        assert result.batch_id == 3
        assert result.success_count == 0
        assert result.failure_count == 2
        assert len(result.results) == 2  # ParseResult objects are created even for errors
        assert result.total_entities == 0
        assert len(result.errors) == 2
        
        # Verify that error results are properly marked
        for parse_result in result.results:
            assert not parse_result.success
            assert parse_result.partial_parse
            assert parse_result.error_recovery_applied
            assert len(parse_result.syntax_errors) == 1


class TestPipelineIntegration:
    """Integration tests for the complete pipeline"""
    
    @pytest.fixture
    def sample_project(self):
        """Create a sample project with multiple language files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            
            # Python files
            (project_path / "main.py").write_text("""
import utils
from models import User

def main():
    '''Main application entry point'''
    user = User("Alice")
    result = utils.process_user(user)
    return result

if __name__ == "__main__":
    main()
""")
            
            (project_path / "utils.py").write_text("""
def process_user(user):
    '''Process a user object'''
    return f"Processed: {user.name}"

def helper_function():
    return "helper"
""")
            
            (project_path / "models.py").write_text("""
class User:
    '''User model class'''
    def __init__(self, name):
        self.name = name
    
    def get_name(self):
        return self.name

class Admin(User):
    def __init__(self, name, permissions):
        super().__init__(name)
        self.permissions = permissions
""")
            
            # JavaScript file
            (project_path / "frontend.js").write_text("""
function initApp() {
    console.log("App initialized");
}

class Component {
    constructor(props) {
        this.props = props;
    }
    
    render() {
        return this.props.content;
    }
}
""")
            
            # Configuration files
            (project_path / "config.json").write_text("""
{
    "app_name": "TestApp",
    "version": "1.0.0",
    "debug": true
}
""")
            
            yield project_path
    
    def test_parse_directory_integration(self, sample_project):
        """Test parsing an entire directory"""
        pipeline = ProcessParsingPipeline(max_workers=2, batch_size=2)
        
        results, stats = pipeline.parse_directory(sample_project)
        
        # Verify results
        assert len(results) >= 4  # At least Python and JS files
        assert stats.total_files >= 4
        assert stats.successful_files > 0
        assert stats.total_entities > 0  # Should extract functions and classes
        assert stats.success_rate > 0.0
        assert stats.total_time > 0
        assert stats.files_per_second > 0
    
    def test_parse_files_integration(self, sample_project):
        """Test parsing specific files"""
        pipeline = ProcessParsingPipeline(max_workers=2, batch_size=1)
        
        # Get Python files only
        python_files = list(sample_project.glob("*.py"))
        
        results, stats = pipeline.parse_files(python_files)
        
        # Verify results
        assert len(results) == len(python_files)
        assert stats.total_files == len(python_files)
        assert stats.successful_files > 0
        assert stats.total_entities > 0
        assert stats.success_rate > 0.0
        assert stats.files_per_second > 0
    
    def test_progress_callback(self, sample_project):
        """Test progress callback functionality"""
        pipeline = ProcessParsingPipeline(max_workers=2, batch_size=1)
        
        progress_calls = []
        
        def progress_callback(processed, total, stats):
            progress_calls.append((processed, total, stats.success_rate))
        
        results, stats = pipeline.parse_directory(
            sample_project,
            progress_callback=progress_callback
        )
        
        # Verify progress callbacks were called
        assert len(progress_calls) > 0
        assert progress_calls[-1][0] == progress_calls[-1][1]  # Final call should be complete
        assert all(processed <= total for processed, total, _ in progress_calls)


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    @pytest.fixture
    def temp_files(self):
        """Create temporary files for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            files = []
            for i in range(3):
                file_path = temp_path / f"test{i}.py"
                file_path.write_text(f"def func_{i}(): return {i}")
                files.append(file_path)
            
            yield temp_path, files
    
    def test_parse_files_parallel_function(self, temp_files):
        """Test parse_files_parallel convenience function"""
        temp_path, files = temp_files
        
        results, stats = parse_files_parallel(
            files,
            max_workers=2,
            batch_size=2
        )
        
        assert len(results) == 3
        assert stats.total_files == 3
        assert stats.successful_files > 0
    
    def test_parse_directory_parallel_function(self, temp_files):
        """Test parse_directory_parallel convenience function"""
        temp_path, files = temp_files
        
        results, stats = parse_directory_parallel(
            temp_path,
            max_workers=2,
            batch_size=2
        )
        
        assert len(results) >= 3
        assert stats.total_files >= 3
        assert stats.successful_files > 0


class TestPerformanceBenchmarking:
    """Test performance benchmarking capabilities"""
    
    @pytest.fixture
    def benchmark_files(self):
        """Create files for benchmarking"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            files = []
            for i in range(5):  # Small number for fast tests
                file_path = temp_path / f"bench{i}.py"
                content = f"""
def benchmark_function_{i}():
    '''Benchmark function {i}'''
    data = [{{'key': j, 'value': j * {i}}} for j in range(10)]
    return sum(item['value'] for item in data)

class BenchmarkClass_{i}:
    def __init__(self):
        self.value = {i}
    
    def process(self):
        return self.value * 2
"""
                file_path.write_text(content)
                files.append(file_path)
            
            yield files
    
    def test_benchmark_performance(self, benchmark_files):
        """Test performance benchmarking"""
        pipeline = ProcessParsingPipeline(max_workers=2, batch_size=2)
        
        benchmark_results = pipeline.benchmark_performance(
            benchmark_files,
            iterations=2  # Small number for fast tests
        )
        
        # Verify benchmark results
        assert "test_files" in benchmark_results
        assert "iterations" in benchmark_results
        assert "workers" in benchmark_results
        assert "average_time" in benchmark_results
        assert "average_files_per_second" in benchmark_results
        assert "results" in benchmark_results
        
        assert benchmark_results["test_files"] == 5
        assert benchmark_results["iterations"] == 2
        assert benchmark_results["workers"] == 2
        assert len(benchmark_results["results"]) == 2
        assert benchmark_results["average_time"] > 0
        assert benchmark_results["average_files_per_second"] > 0
    
    def test_benchmark_empty_files(self):
        """Test benchmarking with no files"""
        pipeline = ProcessParsingPipeline()
        
        benchmark_results = pipeline.benchmark_performance([])
        
        assert "error" in benchmark_results
        assert benchmark_results["error"] == "No test files provided"


@pytest.mark.performance
class TestPerformanceComparison:
    """Performance comparison tests (marked for optional execution)"""
    
    @pytest.fixture
    def large_project(self):
        """Create a larger project for performance testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            
            # Create 20 Python files with substantial content
            for i in range(20):
                file_path = project_path / f"module_{i}.py"
                content = f"""
'''Module {i} for performance testing'''

import os
import sys
from typing import List, Dict, Optional

class DataProcessor_{i}:
    '''Data processor class {i}'''
    
    def __init__(self, config: Dict[str, any]):
        self.config = config
        self.data = []
    
    def process_data(self, input_data: List[Dict]) -> List[Dict]:
        '''Process input data'''
        results = []
        for item in input_data:
            processed = self._process_item(item)
            if processed:
                results.append(processed)
        return results
    
    def _process_item(self, item: Dict) -> Optional[Dict]:
        '''Process single item'''
        if not item.get('valid', True):
            return None
        
        return {{
            'id': item.get('id', ''),
            'value': item.get('value', 0) * 2,
            'processed_by': 'module_{i}'
        }}

def utility_function_{i}(param1: str, param2: int = {i}) -> str:
    '''Utility function {i}'''
    return f"{{param1}}_{{param2}}_{{i}}"

def async_operation_{i}(data: List[str]) -> None:
    '''Async operation {i}'''
    for item in data:
        process_async(item, callback=lambda x: print(f"Processed {{x}} in module {i}"))

# Module-level constants
CONSTANT_{i} = {i * 100}
CONFIG_{i} = {{
    'name': 'module_{i}',
    'version': '1.{i}.0',
    'enabled': True
}}
"""
                file_path.write_text(content)
            
            yield project_path
    
    def test_process_vs_thread_comparison(self, large_project):
        """Compare ProcessPoolExecutor vs ThreadPoolExecutor performance"""
        # This test is marked as performance and may be skipped in regular runs
        
        files = list(large_project.glob("*.py"))
        
        # Test ProcessPoolExecutor
        process_pipeline = ProcessParsingPipeline(max_workers=4, batch_size=5)
        start_time = time.perf_counter()
        process_results, process_stats = process_pipeline.parse_files(files)
        process_time = time.perf_counter() - start_time
        
        # Test ThreadPoolExecutor (using registry method)
        start_time = time.perf_counter()
        thread_results = parser_registry.parse_files_parallel(files, max_workers=4)
        thread_time = time.perf_counter() - start_time
        
        # Compare results
        print(f"\nPerformance Comparison:")
        print(f"ProcessPoolExecutor: {process_time:.3f}s, {process_stats.files_per_second:.1f} files/sec")
        print(f"ThreadPoolExecutor: {thread_time:.3f}s, {len(files)/thread_time:.1f} files/sec")
        print(f"Speedup: {thread_time/process_time:.2f}x")
        
        # Verify both approaches produce similar results
        assert len(process_results) == len(thread_results)
        assert len(process_results) == len(files)
        
        # ProcessPoolExecutor should be faster for CPU-intensive tasks
        # (This assertion might not always hold due to overhead, so it's informational)
        if len(files) > 10:  # Only for larger datasets
            print(f"Process pipeline efficiency: {process_stats.success_rate:.1%}")


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])