"""
ProcessPoolExecutor-based parallel parsing pipeline for better CPU utilization.

This module provides a high-performance parsing pipeline that uses multiprocessing
instead of threading to fully utilize all CPU cores for Tree-sitter parsing and
entity extraction operations.

Key Benefits:
- True parallelism without GIL limitations
- Better performance for CPU-intensive entity extraction
- Optimized for parsing hundreds/thousands of files
- Smart batching and result aggregation
"""

import logging
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable, Tuple, Union
import multiprocessing as mp

from .base import ParseResult, ParserProtocol
from .registry import ParserRegistry, parser_registry

logger = logging.getLogger(__name__)


@dataclass
class BatchParseRequest:
    """Request for parsing a batch of files in a worker process"""
    file_paths: List[Path]
    batch_id: int
    registry_state: Dict[str, Any]  # Serialized registry state
    
    def __post_init__(self):
        # Ensure paths are serializable
        self.file_paths = [Path(p) for p in self.file_paths]


@dataclass
class BatchParseResult:
    """Result from parsing a batch of files"""
    batch_id: int
    results: List[ParseResult]
    success_count: int
    failure_count: int
    parse_time: float
    total_entities: int
    total_relations: int
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate for the batch"""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0


@dataclass
class PipelineStats:
    """Statistics for the entire parsing pipeline"""
    total_files: int = 0
    processed_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    total_entities: int = 0
    total_relations: int = 0
    total_time: float = 0.0
    batches_processed: int = 0
    average_batch_time: float = 0.0
    
    @property 
    def success_rate(self) -> float:
        """Overall success rate"""
        return self.successful_files / self.total_files if self.total_files > 0 else 0.0
    
    @property
    def files_per_second(self) -> float:
        """Processing rate in files per second"""
        return self.processed_files / self.total_time if self.total_time > 0 else 0.0
    
    @property
    def entities_per_second(self) -> float:
        """Entity extraction rate"""
        return self.total_entities / self.total_time if self.total_time > 0 else 0.0


def _parse_file_batch(request: BatchParseRequest) -> BatchParseResult:
    """
    Parse a batch of files in a worker process.
    
    This function runs in a separate process and must be pickleable.
    It recreates the parser registry locally and processes files.
    """
    batch_start_time = time.perf_counter()
    
    # Import core.parser to ensure all parsers register themselves
    import core.parser
    from core.parser.registry import parser_registry as local_registry
    
    results = []
    success_count = 0
    failure_count = 0
    total_entities = 0
    total_relations = 0
    errors = []
    
    for file_path in request.file_paths:
        try:
            # Get parser for this file
            parser = local_registry.get_parser_for_file(file_path)
            if parser is None:
                failure_count += 1
                errors.append({
                    "file": str(file_path),
                    "error": "No parser available",
                    "type": "NO_PARSER"
                })
                continue
            
            # Parse the file
            result = parser.parse_file(file_path)
            results.append(result)
            
            if result.success:
                success_count += 1
                total_entities += len(result.entities)
                total_relations += len(result.relations)
            else:
                failure_count += 1
                errors.append({
                    "file": str(file_path),
                    "error": f"Parse errors: {len(result.syntax_errors)}",
                    "type": "PARSE_ERROR"
                })
                
        except Exception as e:
            failure_count += 1
            errors.append({
                "file": str(file_path),
                "error": str(e),
                "type": "EXCEPTION"
            })
            logger.error(f"Error parsing {file_path} in batch {request.batch_id}: {e}")
    
    parse_time = time.perf_counter() - batch_start_time
    
    return BatchParseResult(
        batch_id=request.batch_id,
        results=results,
        success_count=success_count,
        failure_count=failure_count,
        parse_time=parse_time,
        total_entities=total_entities,
        total_relations=total_relations,
        errors=errors
    )


class ProcessParsingPipeline:
    """
    High-performance parsing pipeline using ProcessPoolExecutor.
    
    Uses multiprocessing to achieve true parallelism for CPU-intensive
    Tree-sitter parsing and entity extraction operations.
    """
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        batch_size: int = 10,
        timeout: float = 300.0  # 5 minutes per batch
    ):
        """
        Initialize the parsing pipeline.
        
        Args:
            max_workers: Maximum number of worker processes (default: CPU count + 4)
            batch_size: Number of files per batch (default: 10)
            timeout: Timeout in seconds for each batch (default: 300)
        """
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.batch_size = batch_size
        self.timeout = timeout
        self.registry = parser_registry
        
        logger.info(f"Initialized ProcessParsingPipeline with {self.max_workers} workers, batch size {self.batch_size}")
    
    def create_batches(self, file_paths: List[Path]) -> List[List[Path]]:
        """
        Create batches of files for parallel processing.
        
        Args:
            file_paths: List of files to batch
            
        Returns:
            List of file batches
        """
        batches = []
        for i in range(0, len(file_paths), self.batch_size):
            batch = file_paths[i:i + self.batch_size]
            batches.append(batch)
        
        logger.debug(f"Created {len(batches)} batches from {len(file_paths)} files")
        return batches
    
    def parse_files(
        self,
        file_paths: List[Path],
        progress_callback: Optional[Callable[[int, int, PipelineStats], None]] = None
    ) -> Tuple[List[ParseResult], PipelineStats]:
        """
        Parse multiple files using ProcessPoolExecutor.
        
        Args:
            file_paths: List of files to parse
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (parse results, pipeline statistics)
        """
        if not file_paths:
            return [], PipelineStats()
        
        start_time = time.perf_counter()
        
        # Initialize stats
        stats = PipelineStats(total_files=len(file_paths))
        
        # Create batches
        file_batches = self.create_batches(file_paths)
        
        # Prepare batch requests
        batch_requests = []
        for i, batch_files in enumerate(file_batches):
            request = BatchParseRequest(
                file_paths=batch_files,
                batch_id=i,
                registry_state={}  # TODO: Serialize registry state if needed
            )
            batch_requests.append(request)
        
        all_results = []
        
        # Process batches in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batch parsing tasks
            future_to_batch = {
                executor.submit(_parse_file_batch, request): request
                for request in batch_requests
            }
            
            logger.info(f"Submitted {len(future_to_batch)} batches to {self.max_workers} workers")
            
            # Process completed batches
            for future in as_completed(future_to_batch, timeout=self.timeout):
                request = future_to_batch[future]
                
                try:
                    batch_result = future.result()
                    all_results.extend(batch_result.results)
                    
                    # Update stats
                    stats.processed_files += len(request.file_paths)
                    stats.successful_files += batch_result.success_count
                    stats.failed_files += batch_result.failure_count
                    stats.total_entities += batch_result.total_entities
                    stats.total_relations += batch_result.total_relations
                    stats.batches_processed += 1
                    
                    # Progress callback
                    if progress_callback:
                        progress_callback(stats.processed_files, stats.total_files, stats)
                    
                    logger.debug(
                        f"Batch {batch_result.batch_id} completed: "
                        f"{batch_result.success_count}/{len(request.file_paths)} files, "
                        f"{batch_result.total_entities} entities in {batch_result.parse_time:.2f}s"
                    )
                    
                except Exception as e:
                    # Handle batch failure
                    stats.failed_files += len(request.file_paths)
                    stats.processed_files += len(request.file_paths)
                    
                    logger.error(f"Batch {request.batch_id} failed: {e}")
                    
                    if progress_callback:
                        progress_callback(stats.processed_files, stats.total_files, stats)
        
        # Finalize stats
        stats.total_time = time.perf_counter() - start_time
        if stats.batches_processed > 0:
            stats.average_batch_time = stats.total_time / stats.batches_processed
        
        logger.info(
            f"Pipeline completed: {stats.successful_files}/{stats.total_files} files "
            f"({stats.success_rate:.1%}), {stats.total_entities} entities, "
            f"{stats.files_per_second:.1f} files/sec"
        )
        
        return all_results, stats
    
    def parse_directory(
        self,
        directory: Path,
        recursive: bool = True,
        follow_symlinks: bool = False,
        progress_callback: Optional[Callable[[int, int, PipelineStats], None]] = None
    ) -> Tuple[List[ParseResult], PipelineStats]:
        """
        Parse all supported files in a directory.
        
        Args:
            directory: Directory to parse
            recursive: Whether to search recursively
            follow_symlinks: Whether to follow symbolic links
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (parse results, pipeline statistics)
        """
        logger.info(f"Discovering files in {directory} (recursive={recursive})")
        
        # Discover files
        files = self.registry.discover_files(
            directory, 
            recursive=recursive, 
            follow_symlinks=follow_symlinks
        )
        
        if not files:
            logger.warning(f"No parseable files found in {directory}")
            return [], PipelineStats()
        
        logger.info(f"Found {len(files)} parseable files")
        
        # Parse discovered files
        return self.parse_files(files, progress_callback)
    
    def benchmark_performance(
        self,
        test_files: List[Path],
        iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Benchmark pipeline performance.
        
        Args:
            test_files: Files to use for benchmarking
            iterations: Number of iterations to run
            
        Returns:
            Performance benchmark results
        """
        if not test_files:
            return {"error": "No test files provided"}
        
        logger.info(f"Benchmarking pipeline with {len(test_files)} files, {iterations} iterations")
        
        results = []
        
        for i in range(iterations):
            logger.debug(f"Benchmark iteration {i + 1}/{iterations}")
            
            _, stats = self.parse_files(test_files.copy())
            results.append({
                "iteration": i + 1,
                "total_time": stats.total_time,
                "files_per_second": stats.files_per_second,
                "entities_per_second": stats.entities_per_second,
                "success_rate": stats.success_rate,
                "total_entities": stats.total_entities,
                "total_relations": stats.total_relations
            })
        
        # Calculate averages
        avg_time = sum(r["total_time"] for r in results) / len(results)
        avg_files_per_sec = sum(r["files_per_second"] for r in results) / len(results)
        avg_entities_per_sec = sum(r["entities_per_second"] for r in results) / len(results)
        avg_success_rate = sum(r["success_rate"] for r in results) / len(results)
        
        return {
            "test_files": len(test_files),
            "iterations": iterations,
            "workers": self.max_workers,
            "batch_size": self.batch_size,
            "average_time": avg_time,
            "average_files_per_second": avg_files_per_sec,
            "average_entities_per_second": avg_entities_per_sec,
            "average_success_rate": avg_success_rate,
            "results": results,
            "speedup_estimate": f"{avg_files_per_sec / (len(test_files) / avg_time):.1f}x over sequential"
        }


# Global pipeline instance for convenience
default_pipeline = ProcessParsingPipeline()


def parse_files_parallel(
    file_paths: List[Path],
    max_workers: Optional[int] = None,
    batch_size: int = 10,
    progress_callback: Optional[Callable[[int, int, PipelineStats], None]] = None
) -> Tuple[List[ParseResult], PipelineStats]:
    """
    Convenience function to parse files in parallel using ProcessPoolExecutor.
    
    Args:
        file_paths: List of files to parse
        max_workers: Maximum number of worker processes
        batch_size: Number of files per batch
        progress_callback: Optional callback for progress updates
        
    Returns:
        Tuple of (parse results, pipeline statistics)
    """
    pipeline = ProcessParsingPipeline(
        max_workers=max_workers,
        batch_size=batch_size
    )
    
    return pipeline.parse_files(file_paths, progress_callback)


def parse_directory_parallel(
    directory: Path,
    max_workers: Optional[int] = None,
    batch_size: int = 10,
    recursive: bool = True,
    progress_callback: Optional[Callable[[int, int, PipelineStats], None]] = None
) -> Tuple[List[ParseResult], PipelineStats]:
    """
    Convenience function to parse a directory in parallel using ProcessPoolExecutor.
    
    Args:
        directory: Directory to parse
        max_workers: Maximum number of worker processes
        batch_size: Number of files per batch
        recursive: Whether to search recursively
        progress_callback: Optional callback for progress updates
        
    Returns:
        Tuple of (parse results, pipeline statistics)
    """
    pipeline = ProcessParsingPipeline(
        max_workers=max_workers,
        batch_size=batch_size
    )
    
    return pipeline.parse_directory(
        directory,
        recursive=recursive,
        progress_callback=progress_callback
    )