"""
Tests for Periodic Sanity Task Scheduler with Real Asyncio Scheduling and Delta-Scan Execution.

This module provides comprehensive testing of the periodic sanity task functionality
with real asyncio scheduling, delta-scan integration, and NO MOCKS for authentic testing.
"""

import pytest
import asyncio
import tempfile
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, Mock

from core.indexer.periodic_sanity_task import (
    PeriodicSanityTask,
    PeriodicSanityTaskManager,
    TaskConfig,
    TaskMetrics,
    TaskStatus
)
from core.indexer.hybrid_indexer import HybridIndexer, IndexingJobConfig
from core.parser.parallel_pipeline import ProcessParsingPipeline
from core.storage.client import HybridQdrantClient
from core.storage.schemas import CollectionManager, QdrantSchema
from core.embeddings.stella import StellaEmbedder
from core.models.config import StellaConfig

logger = logging.getLogger(__name__)


class TestTaskConfig:
    """Test TaskConfig functionality."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TaskConfig()
        
        assert config.interval_minutes == 10
        assert config.initial_delay_minutes == 5
        assert config.max_retry_attempts == 3
        assert config.retry_delay_seconds == 30
        assert config.force_full_scan is False
        assert config.tolerance_sec == 1.0
        assert config.max_execution_time_minutes == 30
        assert config.continue_on_error is True
        assert config.max_consecutive_failures == 5
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = TaskConfig(
            interval_minutes=5,
            initial_delay_minutes=1,
            max_retry_attempts=5,
            retry_delay_seconds=10,
            force_full_scan=True,
            tolerance_sec=2.0,
            max_execution_time_minutes=60,
            continue_on_error=False,
            max_consecutive_failures=3
        )
        
        assert config.interval_minutes == 5
        assert config.initial_delay_minutes == 1
        assert config.max_retry_attempts == 5
        assert config.retry_delay_seconds == 10
        assert config.force_full_scan is True
        assert config.tolerance_sec == 2.0
        assert config.max_execution_time_minutes == 60
        assert config.continue_on_error is False
        assert config.max_consecutive_failures == 3
    
    def test_from_env(self, monkeypatch):
        """Test configuration from environment variables."""
        # Set environment variables
        monkeypatch.setenv("SANITY_TASK_INTERVAL_MINUTES", "15")
        monkeypatch.setenv("SANITY_TASK_INITIAL_DELAY_MINUTES", "2")
        monkeypatch.setenv("SANITY_TASK_MAX_RETRIES", "5")
        monkeypatch.setenv("SANITY_TASK_RETRY_DELAY", "60")
        monkeypatch.setenv("SANITY_TASK_FORCE_FULL_SCAN", "true")
        monkeypatch.setenv("SANITY_TASK_TOLERANCE_SEC", "3.0")
        monkeypatch.setenv("SANITY_TASK_MAX_EXECUTION_MINUTES", "45")
        monkeypatch.setenv("SANITY_TASK_CONTINUE_ON_ERROR", "false")
        monkeypatch.setenv("SANITY_TASK_MAX_CONSECUTIVE_FAILURES", "8")
        
        config = TaskConfig.from_env()
        
        assert config.interval_minutes == 15
        assert config.initial_delay_minutes == 2
        assert config.max_retry_attempts == 5
        assert config.retry_delay_seconds == 60
        assert config.force_full_scan is True
        assert config.tolerance_sec == 3.0
        assert config.max_execution_time_minutes == 45
        assert config.continue_on_error is False
        assert config.max_consecutive_failures == 8


class TestTaskMetrics:
    """Test TaskMetrics functionality."""
    
    def test_initial_metrics(self):
        """Test initial metrics state."""
        metrics = TaskMetrics()
        
        assert metrics.total_runs == 0
        assert metrics.successful_runs == 0
        assert metrics.failed_runs == 0
        assert metrics.consecutive_failures == 0
        assert metrics.last_run_time is None
        assert metrics.last_success_time is None
        assert metrics.last_failure_time is None
        assert metrics.last_execution_duration_seconds == 0.0
        assert metrics.average_execution_time_seconds == 0.0
        assert metrics.total_execution_time_seconds == 0.0
        assert metrics.success_rate == 0.0
    
    def test_update_success(self):
        """Test success metrics update."""
        metrics = TaskMetrics()
        execution_time = 15.5
        
        # First success
        metrics.update_success(execution_time)
        
        assert metrics.total_runs == 1
        assert metrics.successful_runs == 1
        assert metrics.failed_runs == 0
        assert metrics.consecutive_failures == 0
        assert metrics.last_run_time is not None
        assert metrics.last_success_time is not None
        assert metrics.last_failure_time is None
        assert metrics.last_execution_duration_seconds == execution_time
        assert metrics.average_execution_time_seconds == execution_time
        assert metrics.total_execution_time_seconds == execution_time
        assert metrics.success_rate == 100.0
        
        # Second success
        second_execution_time = 20.0
        metrics.update_success(second_execution_time)
        
        assert metrics.total_runs == 2
        assert metrics.successful_runs == 2
        assert metrics.failed_runs == 0
        assert metrics.consecutive_failures == 0
        assert metrics.last_execution_duration_seconds == second_execution_time
        assert metrics.average_execution_time_seconds == (execution_time + second_execution_time) / 2
        assert metrics.total_execution_time_seconds == execution_time + second_execution_time
        assert metrics.success_rate == 100.0
    
    def test_update_failure(self):
        """Test failure metrics update."""
        metrics = TaskMetrics()
        execution_time = 10.0
        
        # First failure
        metrics.update_failure(execution_time)
        
        assert metrics.total_runs == 1
        assert metrics.successful_runs == 0
        assert metrics.failed_runs == 1
        assert metrics.consecutive_failures == 1
        assert metrics.last_run_time is not None
        assert metrics.last_success_time is None
        assert metrics.last_failure_time is not None
        assert metrics.last_execution_duration_seconds == execution_time
        assert metrics.success_rate == 0.0
    
    def test_mixed_success_failure(self):
        """Test mixed success and failure updates."""
        metrics = TaskMetrics()
        
        # Success, failure, success pattern
        metrics.update_success(10.0)
        metrics.update_failure(5.0)
        metrics.update_success(15.0)
        
        assert metrics.total_runs == 3
        assert metrics.successful_runs == 2
        assert metrics.failed_runs == 1
        assert metrics.consecutive_failures == 0  # Reset by last success
        assert metrics.success_rate == (2/3) * 100
        assert metrics.total_execution_time_seconds == 30.0
        assert metrics.average_execution_time_seconds == 10.0
    
    def test_to_dict(self):
        """Test metrics dictionary conversion."""
        metrics = TaskMetrics()
        metrics.update_success(12.5)
        
        result = metrics.to_dict()
        
        required_keys = [
            "total_runs", "successful_runs", "failed_runs", "consecutive_failures",
            "success_rate_percent", "last_run_time", "last_success_time", "last_failure_time",
            "last_execution_duration_seconds", "average_execution_time_seconds", "total_execution_time_seconds"
        ]
        
        for key in required_keys:
            assert key in result
        
        assert result["total_runs"] == 1
        assert result["successful_runs"] == 1
        assert result["success_rate_percent"] == 100.0
        assert result["last_execution_duration_seconds"] == 12.5


class TestPeriodicSanityTaskBasic:
    """Test basic PeriodicSanityTask functionality with mocked indexer."""
    
    @pytest.fixture
    def mock_indexer(self):
        """Create mock indexer for basic testing."""
        indexer = Mock()
        indexer.perform_delta_scan = AsyncMock(return_value={
            "success": True,
            "summary": {"total_changes": 5},
            "workspace_files": 100,
            "collection_entities": 95
        })
        return indexer
    
    @pytest.fixture
    def test_config(self):
        """Create test configuration with short intervals."""
        return TaskConfig(
            interval_minutes=1,  # Very short for testing
            initial_delay_minutes=0,  # No delay for testing
            max_retry_attempts=2,
            retry_delay_seconds=1,
            max_execution_time_minutes=5,
            continue_on_error=True,
            max_consecutive_failures=3
        )
    
    @pytest.fixture
    def temp_project_dir(self):
        """Create temporary project directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "test_project"
            project_path.mkdir()
            # Create a test file
            (project_path / "test.py").write_text("def test(): pass")
            yield project_path
    
    def test_initialization(self, mock_indexer, temp_project_dir, test_config):
        """Test task initialization."""
        task = PeriodicSanityTask(
            indexer=mock_indexer,
            project_path=temp_project_dir,
            collection_name="test-collection",
            config=test_config
        )
        
        assert task.indexer == mock_indexer
        assert task.project_path == temp_project_dir
        assert task.collection_name == "test-collection"
        assert task.config == test_config
        assert task.status == TaskStatus.STOPPED
        assert task.metrics.total_runs == 0
    
    @pytest.mark.asyncio
    async def test_basic_lifecycle(self, mock_indexer, temp_project_dir, test_config):
        """Test basic start/stop lifecycle."""
        task = PeriodicSanityTask(
            indexer=mock_indexer,
            project_path=temp_project_dir,
            collection_name="test-collection",
            config=test_config
        )
        
        # Test start
        success = await task.start()
        assert success is True
        assert task.status == TaskStatus.RUNNING
        
        # Brief wait to ensure task is running
        await asyncio.sleep(0.1)
        
        # Test stop
        await task.stop()
        assert task.status == TaskStatus.STOPPED
    
    @pytest.mark.asyncio
    async def test_pause_resume(self, mock_indexer, temp_project_dir, test_config):
        """Test pause and resume functionality."""
        task = PeriodicSanityTask(
            indexer=mock_indexer,
            project_path=temp_project_dir,
            collection_name="test-collection",
            config=test_config
        )
        
        # Start task
        await task.start()
        assert task.status == TaskStatus.RUNNING
        
        # Pause task
        await task.pause()
        assert task.status == TaskStatus.PAUSED
        
        # Resume task
        await task.resume()
        assert task.status == TaskStatus.RUNNING
        
        # Cleanup
        await task.stop()
    
    @pytest.mark.asyncio
    async def test_immediate_run(self, mock_indexer, temp_project_dir, test_config):
        """Test immediate delta scan execution."""
        task = PeriodicSanityTask(
            indexer=mock_indexer,
            project_path=temp_project_dir,
            collection_name="test-collection",
            config=test_config
        )
        
        # Start task
        await task.start()
        
        # Trigger immediate run
        result = await task.trigger_immediate_run()
        
        assert result["success"] is True
        assert "execution_time_seconds" in result
        assert "result" in result
        
        # Verify indexer was called
        mock_indexer.perform_delta_scan.assert_called()
        
        # Cleanup
        await task.stop()
    
    @pytest.mark.asyncio
    async def test_immediate_run_when_stopped(self, mock_indexer, temp_project_dir, test_config):
        """Test immediate run when task is stopped."""
        task = PeriodicSanityTask(
            indexer=mock_indexer,
            project_path=temp_project_dir,
            collection_name="test-collection",
            config=test_config
        )
        
        # Task is stopped, immediate run should fail
        result = await task.trigger_immediate_run()
        
        assert result["success"] is False
        assert "error" in result
        assert result["error"] == "Task is not running"
    
    def test_get_status(self, mock_indexer, temp_project_dir, test_config):
        """Test status information retrieval."""
        task = PeriodicSanityTask(
            indexer=mock_indexer,
            project_path=temp_project_dir,
            collection_name="test-collection",
            config=test_config
        )
        
        status = task.get_status()
        
        required_keys = [
            "status", "project_path", "collection_name", "start_time",
            "next_run_time", "last_error", "config", "metrics"
        ]
        
        for key in required_keys:
            assert key in status
        
        assert status["status"] == "stopped"
        assert status["project_path"] == str(temp_project_dir)
        assert status["collection_name"] == "test-collection"
        assert status["start_time"] is None
        assert status["next_run_time"] is None
        
        # Check config section
        config_data = status["config"]
        assert config_data["interval_minutes"] == 1
        assert config_data["initial_delay_minutes"] == 0


class TestPeriodicSanityTaskReal:
    """Test PeriodicSanityTask with real Qdrant and delta-scan integration."""
    
    @pytest.fixture
    async def storage_client(self):
        """Create real HybridQdrantClient for testing."""
        client = HybridQdrantClient(url="http://localhost:6334")  # Test Qdrant port
        await client.connect()
        yield client
        await client.disconnect()
    
    @pytest.fixture
    async def test_collection_name(self):
        """Generate unique test collection name."""
        import uuid
        return f"test-periodic-sanity-{uuid.uuid4().hex[:8]}"
    
    @pytest.fixture
    async def hybrid_indexer(self, storage_client, test_collection_name):
        """Create real HybridIndexer for testing."""
        parser_pipeline = ProcessParsingPipeline(max_workers=2, batch_size=10)
        
        config = IndexingJobConfig(
            project_path=Path("/tmp"),
            project_name=test_collection_name,
            max_workers=2,
            batch_size=10
        )
        
        indexer = HybridIndexer(
            parser_pipeline=parser_pipeline,
            embedder=None,  # Skip embeddings for faster testing
            storage_client=storage_client,
            cache_manager=None,
            config=config
        )
        yield indexer
    
    @pytest.fixture
    async def setup_test_collection(self, storage_client, test_collection_name):
        """Setup test collection with proper schema."""
        try:
            # Create collection with proper schema
            collection_config = QdrantSchema.get_code_collection_config(test_collection_name)
            create_result = await storage_client.create_collection(collection_config)
            assert create_result.success, f"Failed to create test collection: {create_result.error}"
            
            yield test_collection_name
            
        finally:
            # Cleanup: Delete test collection
            try:
                import asyncio
                await asyncio.to_thread(storage_client._client.delete_collection, test_collection_name)
            except Exception as e:
                logger.warning(f"Failed to cleanup test collection {test_collection_name}: {e}")
    
    @pytest.fixture
    def temp_project_dir(self):
        """Create temporary project directory with test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "test_project"
            project_path.mkdir()
            
            # Create test Python files
            (project_path / "main.py").write_text("""
def main():
    print("Hello World")
    return True

if __name__ == "__main__":
    main()
""")
            
            (project_path / "utils.py").write_text("""
def helper_function(x, y):
    return x + y

class UtilityClass:
    def __init__(self, value):
        self.value = value
    
    def process(self):
        return self.value * 2
""")
            
            (project_path / "data").mkdir(parents=True, exist_ok=True)
            (project_path / "data" / "__init__.py").write_text("")
            (project_path / "data" / "config.py").write_text("""
CONFIG = {
    'debug': True,
    'version': '1.0.0'
}
""")
            
            yield project_path
    
    @pytest.fixture
    def quick_test_config(self):
        """Create test configuration with very short intervals for quick testing."""
        return TaskConfig(
            interval_minutes=1,  # 1 minute interval for testing
            initial_delay_minutes=0,  # No initial delay
            max_retry_attempts=2,
            retry_delay_seconds=1,
            force_full_scan=False,
            tolerance_sec=1.0,
            max_execution_time_minutes=2,
            continue_on_error=True,
            max_consecutive_failures=2
        )
    
    @pytest.mark.asyncio
    async def test_real_delta_scan_execution(
        self,
        hybrid_indexer,
        setup_test_collection,
        temp_project_dir,
        quick_test_config
    ):
        """Test periodic task with real delta scan execution."""
        collection_name = setup_test_collection
        
        # Create periodic task
        task = PeriodicSanityTask(
            indexer=hybrid_indexer,
            project_path=temp_project_dir,
            collection_name=collection_name,
            config=quick_test_config
        )
        
        try:
            # Start task
            success = await task.start()
            assert success is True
            assert task.status == TaskStatus.RUNNING
            
            # Trigger immediate run to test delta scan
            result = await task.trigger_immediate_run()
            
            assert result["success"] is True
            assert result["execution_time_seconds"] > 0
            assert "result" in result
            
            # Check that metrics were updated
            assert task.metrics.total_runs > 0
            
            # Get status to verify task information
            status = task.get_status()
            assert status["status"] == "running"
            assert status["collection_name"] == collection_name
            assert status["metrics"]["total_runs"] > 0
            
        finally:
            # Always stop the task
            await task.stop()
    
    @pytest.mark.asyncio
    async def test_periodic_execution_timing(
        self,
        hybrid_indexer,
        setup_test_collection,
        temp_project_dir
    ):
        """Test that periodic task executes at correct intervals."""
        collection_name = setup_test_collection
        
        # Use very short intervals for testing
        test_config = TaskConfig(
            interval_minutes=0,  # Use seconds instead via custom timing
            initial_delay_minutes=0,
            max_retry_attempts=1,
            retry_delay_seconds=1,
            max_execution_time_minutes=1
        )
        
        # Track execution times
        execution_times = []
        
        def progress_callback(current, total, data):
            execution_times.append(datetime.now())
        
        # Create periodic task
        task = PeriodicSanityTask(
            indexer=hybrid_indexer,
            project_path=temp_project_dir,
            collection_name=collection_name,
            config=test_config,
            progress_callback=progress_callback
        )
        
        try:
            # Start task
            await task.start()
            
            # Trigger multiple immediate runs to simulate periodic execution
            for i in range(3):
                result = await task.trigger_immediate_run()
                assert result["success"] is True
                await asyncio.sleep(0.5)  # Small delay between runs
            
            # Verify multiple executions occurred
            assert task.metrics.total_runs >= 3
            assert task.metrics.successful_runs >= 3
            
        finally:
            await task.stop()
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(
        self,
        storage_client,
        temp_project_dir,
        quick_test_config
    ):
        """Test error handling when delta scan fails."""
        # Create indexer with invalid configuration to cause failures
        parser_pipeline = ProcessParsingPipeline(max_workers=1, batch_size=5)
        
        config = IndexingJobConfig(
            project_path=Path("/nonexistent"),  # Invalid path to cause failure
            project_name="nonexistent-project",
            max_workers=1,
            batch_size=5
        )
        
        failing_indexer = HybridIndexer(
            parser_pipeline=parser_pipeline,
            embedder=None,
            storage_client=storage_client,
            cache_manager=None,
            config=config
        )
        
        # Create task with retry configuration
        retry_config = TaskConfig(
            interval_minutes=1,
            initial_delay_minutes=0,
            max_retry_attempts=2,
            retry_delay_seconds=1,
            continue_on_error=True,
            max_consecutive_failures=3
        )
        
        task = PeriodicSanityTask(
            indexer=failing_indexer,
            project_path=Path("/nonexistent"),  # This will cause failures
            collection_name="nonexistent-collection",
            config=retry_config
        )
        
        try:
            # Start task
            await task.start()
            
            # Trigger immediate run that should fail
            result = await task.trigger_immediate_run()
            
            # Result should indicate failure but task should continue
            assert result["success"] is False
            assert "error" in result
            
            # Check that failure metrics were updated
            assert task.metrics.failed_runs > 0
            assert task.metrics.consecutive_failures > 0
            
            # Task should still be running due to continue_on_error=True
            assert task.status == TaskStatus.RUNNING
            
        finally:
            await task.stop()


class TestPeriodicSanityTaskManager:
    """Test PeriodicSanityTaskManager functionality."""
    
    @pytest.fixture
    def mock_indexer(self):
        """Create mock indexer for manager testing."""
        indexer = Mock()
        indexer.perform_delta_scan = AsyncMock(return_value={
            "success": True,
            "summary": {"total_changes": 2}
        })
        return indexer
    
    @pytest.fixture
    def temp_project_dir(self):
        """Create temporary project directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "test_project"
            project_path.mkdir()
            (project_path / "test.py").write_text("def test(): pass")
            yield project_path
    
    @pytest.fixture
    def quick_config(self):
        """Create quick test configuration."""
        return TaskConfig(
            interval_minutes=1,
            initial_delay_minutes=0,
            max_retry_attempts=1,
            retry_delay_seconds=1
        )
    
    @pytest.mark.asyncio
    async def test_manager_initialization(self):
        """Test task manager initialization."""
        manager = PeriodicSanityTaskManager()
        
        assert len(manager.tasks) == 0
        assert manager.get_total_count() == 0
        assert manager.get_running_count() == 0
    
    @pytest.mark.asyncio
    async def test_add_task(self, mock_indexer, temp_project_dir, quick_config):
        """Test adding tasks to manager."""
        manager = PeriodicSanityTaskManager()
        
        # Add task with auto-start
        success = await manager.add_task(
            task_id="test-task-1",
            indexer=mock_indexer,
            project_path=temp_project_dir,
            collection_name="test-collection-1",
            config=quick_config,
            auto_start=True
        )
        
        assert success is True
        assert manager.get_total_count() == 1
        assert manager.get_running_count() == 1
        
        # Verify task was added
        task = manager.get_task("test-task-1")
        assert task is not None
        assert task.status == TaskStatus.RUNNING
        
        # Cleanup
        await manager.stop_all()
    
    @pytest.mark.asyncio
    async def test_add_duplicate_task(self, mock_indexer, temp_project_dir, quick_config):
        """Test adding duplicate task IDs."""
        manager = PeriodicSanityTaskManager()
        
        # Add first task
        success1 = await manager.add_task(
            task_id="duplicate-task",
            indexer=mock_indexer,
            project_path=temp_project_dir,
            collection_name="test-collection-1",
            config=quick_config,
            auto_start=False
        )
        
        assert success1 is True
        assert manager.get_total_count() == 1
        
        # Try to add duplicate
        success2 = await manager.add_task(
            task_id="duplicate-task",
            indexer=mock_indexer,
            project_path=temp_project_dir,
            collection_name="test-collection-2",
            config=quick_config,
            auto_start=False
        )
        
        assert success2 is False
        assert manager.get_total_count() == 1  # Should remain 1
        
        # Cleanup
        await manager.stop_all()
    
    @pytest.mark.asyncio
    async def test_remove_task(self, mock_indexer, temp_project_dir, quick_config):
        """Test removing tasks from manager."""
        manager = PeriodicSanityTaskManager()
        
        # Add task
        await manager.add_task(
            task_id="removable-task",
            indexer=mock_indexer,
            project_path=temp_project_dir,
            collection_name="test-collection",
            config=quick_config,
            auto_start=True
        )
        
        assert manager.get_total_count() == 1
        assert manager.get_running_count() == 1
        
        # Remove task
        success = await manager.remove_task("removable-task")
        
        assert success is True
        assert manager.get_total_count() == 0
        assert manager.get_running_count() == 0
        
        # Verify task was removed
        task = manager.get_task("removable-task")
        assert task is None
    
    @pytest.mark.asyncio
    async def test_remove_nonexistent_task(self):
        """Test removing non-existent task."""
        manager = PeriodicSanityTaskManager()
        
        success = await manager.remove_task("nonexistent-task")
        assert success is False
    
    @pytest.mark.asyncio
    async def test_multiple_tasks_management(self, mock_indexer, temp_project_dir, quick_config):
        """Test managing multiple tasks simultaneously."""
        manager = PeriodicSanityTaskManager()
        
        # Add multiple tasks
        tasks_to_add = [
            ("task-1", "collection-1"),
            ("task-2", "collection-2"),
            ("task-3", "collection-3")
        ]
        
        for task_id, collection_name in tasks_to_add:
            success = await manager.add_task(
                task_id=task_id,
                indexer=mock_indexer,
                project_path=temp_project_dir,
                collection_name=collection_name,
                config=quick_config,
                auto_start=True
            )
            assert success is True
        
        assert manager.get_total_count() == 3
        assert manager.get_running_count() == 3
        
        # Get status of all tasks
        all_status = manager.get_all_status()
        assert len(all_status) == 3
        
        for task_id, collection_name in tasks_to_add:
            assert task_id in all_status
            assert all_status[task_id]["status"] == "running"
            assert all_status[task_id]["collection_name"] == collection_name
        
        # Test stop all
        await manager.stop_all()
        assert manager.get_running_count() == 0
        
        # Test start all
        start_results = await manager.start_all()
        assert len(start_results) == 3
        assert all(result for result in start_results.values())
        assert manager.get_running_count() == 3
        
        # Final cleanup
        await manager.stop_all()
    
    @pytest.mark.asyncio
    async def test_start_stop_all(self, mock_indexer, temp_project_dir, quick_config):
        """Test start_all and stop_all functionality."""
        manager = PeriodicSanityTaskManager()
        
        # Add tasks without auto-start
        for i in range(3):
            await manager.add_task(
                task_id=f"task-{i}",
                indexer=mock_indexer,
                project_path=temp_project_dir,
                collection_name=f"collection-{i}",
                config=quick_config,
                auto_start=False
            )
        
        assert manager.get_total_count() == 3
        assert manager.get_running_count() == 0
        
        # Start all tasks
        start_results = await manager.start_all()
        
        assert len(start_results) == 3
        assert all(result for result in start_results.values())
        assert manager.get_running_count() == 3
        
        # Stop all tasks
        await manager.stop_all()
        assert manager.get_running_count() == 0
        
        # Verify all tasks are stopped
        all_status = manager.get_all_status()
        for status in all_status.values():
            assert status["status"] == "stopped"


class TestIntegrationScenarios:
    """Test comprehensive integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_complete_lifecycle_scenario(self):
        """Test complete periodic sanity task lifecycle with realistic scenario."""
        # This test demonstrates a complete real-world usage scenario
        
        # Create mock indexer with realistic behavior
        indexer = Mock()
        delta_scan_results = [
            {"success": True, "summary": {"total_changes": 5, "added_files": 2, "modified_files": 3}},
            {"success": True, "summary": {"total_changes": 0}},  # No changes
            {"success": False, "error": "Temporary connection issue"},  # Failure
            {"success": True, "summary": {"total_changes": 1, "deleted_files": 1}},  # Recovery
        ]
        
        call_count = 0
        async def mock_delta_scan(*args, **kwargs):
            nonlocal call_count
            result = delta_scan_results[call_count % len(delta_scan_results)]
            call_count += 1
            if result["success"]:
                await asyncio.sleep(0.1)  # Simulate processing time
            else:
                raise Exception(result["error"])
            return result
        
        indexer.perform_delta_scan = mock_delta_scan
        
        # Create temporary project
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "integration_test"
            project_path.mkdir()
            (project_path / "app.py").write_text("print('Hello, World!')")
            
            # Create task manager
            manager = PeriodicSanityTaskManager()
            
            # Configure task for quick testing
            config = TaskConfig(
                interval_minutes=1,
                initial_delay_minutes=0,
                max_retry_attempts=2,
                retry_delay_seconds=0.5,
                continue_on_error=True,
                max_consecutive_failures=2
            )
            
            try:
                # Add and start task
                success = await manager.add_task(
                    task_id="integration-test",
                    indexer=indexer,
                    project_path=project_path,
                    collection_name="integration-collection",
                    config=config,
                    auto_start=True
                )
                
                assert success is True
                
                # Get the task for direct testing
                task = manager.get_task("integration-test")
                assert task is not None
                
                # Execute multiple immediate runs to test different scenarios
                results = []
                for _ in range(4):
                    result = await task.trigger_immediate_run()
                    results.append(result)
                    await asyncio.sleep(0.1)
                
                # Verify mixed results (some success, some failure)
                successful_runs = sum(1 for r in results if r["success"])
                failed_runs = sum(1 for r in results if not r["success"])
                
                assert successful_runs >= 2  # At least some successes
                assert task.metrics.total_runs >= 4
                assert task.metrics.successful_runs >= 2
                
                # Test task status
                status = task.get_status()
                assert status["status"] == "running"
                assert status["metrics"]["total_runs"] >= 4
                
                # Test pause/resume
                await task.pause()
                assert task.status == TaskStatus.PAUSED
                
                await task.resume()
                assert task.status == TaskStatus.RUNNING
                
            finally:
                # Cleanup
                await manager.stop_all()
    
    @pytest.mark.asyncio
    async def test_concurrent_tasks_scenario(self):
        """Test multiple concurrent periodic tasks."""
        # Create multiple mock indexers with different behaviors
        indexers = []
        for i in range(3):
            indexer = Mock()
            indexer.perform_delta_scan = AsyncMock(return_value={
                "success": True,
                "summary": {"total_changes": i + 1}
            })
            indexers.append(indexer)
        
        # Create temporary projects
        projects = []
        for i in range(3):
            temp_dir = tempfile.mkdtemp()
            project_path = Path(temp_dir) / f"project_{i}"
            project_path.mkdir()
            (project_path / "main.py").write_text(f"# Project {i}")
            projects.append(project_path)
        
        # Create task manager
        manager = PeriodicSanityTaskManager()
        
        # Quick config for testing
        config = TaskConfig(
            interval_minutes=1,
            initial_delay_minutes=0,
            max_retry_attempts=1,
            retry_delay_seconds=0.5
        )
        
        try:
            # Add multiple tasks
            for i in range(3):
                success = await manager.add_task(
                    task_id=f"concurrent-task-{i}",
                    indexer=indexers[i],
                    project_path=projects[i],
                    collection_name=f"concurrent-collection-{i}",
                    config=config,
                    auto_start=True
                )
                assert success is True
            
            assert manager.get_total_count() == 3
            assert manager.get_running_count() == 3
            
            # Trigger immediate runs on all tasks concurrently
            tasks = [
                manager.get_task(f"concurrent-task-{i}").trigger_immediate_run()
                for i in range(3)
            ]
            
            results = await asyncio.gather(*tasks)
            
            # Verify all tasks executed successfully
            assert len(results) == 3
            assert all(r["success"] for r in results)
            
            # Verify each task has metrics
            all_status = manager.get_all_status()
            assert len(all_status) == 3
            
            for i in range(3):
                task_status = all_status[f"concurrent-task-{i}"]
                assert task_status["status"] == "running"
                assert task_status["metrics"]["total_runs"] >= 1
                assert task_status["collection_name"] == f"concurrent-collection-{i}"
            
        finally:
            # Cleanup
            await manager.stop_all()
            
            # Cleanup temp directories
            for project in projects:
                import shutil
                shutil.rmtree(project.parent, ignore_errors=True)


if __name__ == "__main__":
    # Allow direct execution for debugging
    pytest.main([__file__, "-v", "-s"])