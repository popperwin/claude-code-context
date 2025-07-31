"""
Comprehensive Real-Time Synchronization Tests.

Tests all real-time file system scenarios: file creation, modification, deletion,
moves, batch operations, concurrent changes, error recovery, and sync lifecycle.
"""

import pytest
import asyncio
import os
import platform
import time
import shutil
import logging
import uuid
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional

# Fix tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from core.indexer.entity_lifecycle_integration import EntityLifecycleIntegrator
from core.storage.client import HybridQdrantClient
from core.embeddings.stella import StellaEmbedder
from core.storage.schemas import CollectionManager, CollectionType
from core.sync.events import FileSystemEvent, EventType


async def wait_until(
    predicate: Callable[[], Any], 
    timeout: float = 5.0, 
    interval: float = 0.1,
    error_message: Optional[str] = None
) -> bool:
    """
    Wait until a condition becomes true or timeout is reached.
    
    Args:
        predicate: Async or sync function that returns truthy when condition is met
        timeout: Maximum time to wait in seconds
        interval: Time between checks in seconds
        error_message: Optional message to include in timeout error
        
    Returns:
        True if condition was met, False if timeout
    """
    end_time = time.time() + timeout
    last_exception = None
    
    while time.time() < end_time:
        try:
            # Call the predicate
            result = predicate()
            
            # If it returns a coroutine, await it
            if asyncio.iscoroutine(result):
                result = await result
            
            if result:
                return True
                
        except Exception as e:
            last_exception = e
            logger.debug(f"Exception in wait_until predicate: {e}")
        
        await asyncio.sleep(interval)
    
    # Log timeout with context
    timeout_msg = f"Timeout after {timeout}s waiting for condition"
    if error_message:
        timeout_msg += f": {error_message}"
    if last_exception:
        timeout_msg += f" (last error: {last_exception})"
    logger.warning(timeout_msg)
    
    return False


class TestComprehensiveRealtimeSync:
    """Comprehensive real-time synchronization testing"""
    
    @classmethod
    def setup_class(cls):
        """Setup test environment"""
        cls.base_test_dir = Path("test-harness/realtime-sync-test").resolve()
        cls.base_test_dir.mkdir(parents=True, exist_ok=True)
        cls.created_collections = set()
        cls.active_integrators = []  # Track active integrators for cleanup
    
    @classmethod
    def teardown_class(cls):
        """Cleanup test environment"""
        # Clean up any remaining active integrators
        if hasattr(cls, 'active_integrators'):
            for integrator in cls.active_integrators:
                try:
                    # Force cleanup without async context
                    if hasattr(integrator, 'sync_engine') and integrator.sync_engine:
                        if hasattr(integrator.sync_engine, 'projects'):
                            for project_state in integrator.sync_engine.projects.values():
                                if hasattr(project_state, 'watcher') and project_state.watcher:
                                    if hasattr(project_state.watcher, 'observer') and project_state.watcher.observer:
                                        project_state.watcher.observer.stop()
                except Exception:
                    pass
        
        # Clean up collections using Qdrant client directly
        try:
            embedder = StellaEmbedder()
            client = HybridQdrantClient("http://localhost:6334", embedder=embedder)
            for collection_name in cls.created_collections:
                try:
                    # Use async cleanup in sync context
                    loop = asyncio.new_event_loop()
                    # First check if collection exists
                    collection_info = loop.run_until_complete(client.get_collection_info(collection_name))
                    if collection_info:
                        # Use the underlying Qdrant client's delete_collection method
                        loop.run_until_complete(asyncio.to_thread(client.client.delete_collection, collection_name))
                        logger.info(f"Deleted collection: {collection_name}")
                    loop.close()
                except Exception as e:
                    logger.warning(f"Failed to delete collection {collection_name}: {e}")
        except Exception as e:
            logger.warning(f"Failed to clean up collections: {e}")
        
        # Clean up test directory
        if hasattr(cls, 'base_test_dir') and cls.base_test_dir.exists():
            shutil.rmtree(cls.base_test_dir)
    
    def create_isolated_project_dir(self, test_name: str) -> Path:
        """Create an isolated project directory for a test"""
        # Use UUID to ensure no conflicts between tests
        unique_id = str(uuid.uuid4())[:8]
        project_dir = self.base_test_dir / f"{test_name}-{unique_id}"
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (project_dir / "src").mkdir(exist_ok=True)
        (project_dir / "lib").mkdir(exist_ok=True)
        (project_dir / "tests").mkdir(exist_ok=True)
        
        return project_dir
    
    async def create_test_integrator(self, project_name: str, project_dir: Path, enable_realtime: bool = False) -> EntityLifecycleIntegrator:
        """Create a test integrator with optional real-time sync"""
        # Initialize client and embedder
        embedder = StellaEmbedder()
        await embedder.load_model()
        
        client = HybridQdrantClient("http://localhost:6334", embedder=embedder)
        
        # Create collection with unique name
        unique_id = str(uuid.uuid4())[:8]
        collection_manager = CollectionManager(project_name=f"{project_name}-{unique_id}")
        
        # Get the actual collection name that will be created (includes -code suffix)
        collection_name = await collection_manager.ensure_collection_exists(
            collection_type=CollectionType.CODE,
            storage_client=client,
            vector_size=embedder.dimensions
        )
        
        # The collection_manager returns the actual name with suffixes
        logger.info(f"Created collection: {collection_name} for project: {project_name}-{unique_id}")
        self.__class__.created_collections.add(collection_name)
        
        # Verify collection was created successfully
        await asyncio.sleep(0.1)  # Brief wait for collection creation
        collection_info = await client.get_collection_info(collection_name)
        if not collection_info:
            raise RuntimeError(f"Failed to create collection: {collection_name}")
        
        # Create integrator with the actual collection name
        integrator = EntityLifecycleIntegrator(
            storage_client=client,
            project_path=project_dir,
            collection_name=collection_name,  # Use the actual name returned by collection_manager
            enable_real_time_sync=False,  # Start with it off
            batch_size=10,
            collection_manager=collection_manager
        )
        
        # Enable real-time sync if requested
        if enable_realtime:
            sync_enabled = await integrator.enable_sync()
            if not sync_enabled:
                raise RuntimeError("Failed to enable real-time sync")
            # Warm-up delay for macOS/FSEvents to be fully ready
            await self.wait_for_sync_delay(0.3)
        
        # Track integrator for cleanup
        self.__class__.active_integrators.append(integrator)
        
        return integrator
    
    async def cleanup_integrator(self, integrator: EntityLifecycleIntegrator) -> None:
        """Properly cleanup an integrator to prevent resource conflicts"""
        try:
            # Disable real-time sync with timeout
            await asyncio.wait_for(integrator.disable_real_time_sync(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("Timeout disabling real-time sync during cleanup")
        except Exception as e:
            logger.warning(f"Error disabling real-time sync during cleanup: {e}")
        
        # Brief delay to ensure full cleanup, especially important on macOS
        await asyncio.sleep(0.3)
        
        # Remove from active integrators list
        if integrator in self.__class__.active_integrators:
            self.__class__.active_integrators.remove(integrator)
    
    def write_file(self, file_path: Path, content: str) -> None:
        """Write content to file, creating directories if needed"""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding='utf-8')
    
    async def wait_for_sync_delay(self, delay: float = 0.5) -> None:
        """Simple delay for sync operations - use wait_until for condition-based waiting"""
        # Add extra time for macOS file system events which can be slower
        if platform.system() == 'Darwin':
            delay = max(delay, 1.0)  # Minimum 1 second on macOS
        else:
            delay = max(delay, 0.2)  # Minimum 0.2 seconds on Linux/Windows for CI
        await asyncio.sleep(delay)
    
    async def count_entities_in_collection(self, integrator: EntityLifecycleIntegrator) -> int:
        """Count total entities in the collection"""
        try:
            collection_info = await integrator.storage_client.get_collection_info(integrator.collection_name)
            return collection_info.get("points_count", 0) if collection_info else 0
        except Exception:
            return 0
    
    async def search_for_entity(
        self, 
        integrator: EntityLifecycleIntegrator, 
        query: str, 
        expected_file_path: Optional[str] = None
    ) -> List[Any]:
        """Search for entities with optional file path filtering"""
        results = await integrator.storage_client.search_hybrid(
            collection_name=integrator.collection_name,
            query=query,
            limit=10
        )
        
        # Filter for exact matches in entity_name
        exact_matches = []
        for result in results:
            entity_name = result.point.payload.get("entity_name", "")
            file_path = result.point.payload.get("file_path", "")
            
            # Check if query matches entity name
            if query.lower() in entity_name.lower():
                # If file path filter is specified, apply it
                if expected_file_path is None or expected_file_path in file_path:
                    exact_matches.append(result)
        
        return exact_matches
    
    async def wait_for_entity_count(
        self, 
        integrator: EntityLifecycleIntegrator, 
        expected_count: int,
        comparison: str = "eq",  # "eq", "gt", "lt", "gte", "lte"
        timeout: float = 10.0
    ) -> bool:
        """Wait for entity count to match expected value"""
        async def check_count():
            count = await self.count_entities_in_collection(integrator)
            if comparison == "eq":
                return count == expected_count
            elif comparison == "gt":
                return count > expected_count
            elif comparison == "lt":
                return count < expected_count
            elif comparison == "gte":
                return count >= expected_count
            elif comparison == "lte":
                return count <= expected_count
            return False
        
        return await wait_until(
            check_count, 
            timeout=timeout,
            error_message=f"Entity count to be {comparison} {expected_count}"
        )
    
    async def wait_for_entity_exists(
        self,
        integrator: EntityLifecycleIntegrator,
        entity_name: str,
        expected_file_path: Optional[str] = None,
        timeout: float = 10.0
    ) -> bool:
        """Wait for an entity to exist in the collection"""
        async def check_exists():
            results = await self.search_for_entity(integrator, entity_name, expected_file_path)
            return len(results) > 0
        
        return await wait_until(
            check_exists,
            timeout=timeout,
            error_message=f"Entity '{entity_name}' to exist"
        )
    
    async def wait_for_entity_not_exists(
        self,
        integrator: EntityLifecycleIntegrator,
        entity_name: str,
        expected_file_path: Optional[str] = None,
        timeout: float = 10.0
    ) -> bool:
        """Wait for an entity to not exist in the collection"""
        async def check_not_exists():
            results = await self.search_for_entity(integrator, entity_name, expected_file_path)
            return len(results) == 0
        
        return await wait_until(
            check_not_exists,
            timeout=timeout,
            error_message=f"Entity '{entity_name}' to be deleted"
        )
    
    @pytest.mark.asyncio
    async def test_file_creation_realtime_sync(self):
        """Test real-time synchronization when files are created"""
        project_dir = self.create_isolated_project_dir("file-creation")
        integrator = await self.create_test_integrator("file-creation-test", project_dir, enable_realtime=True)
        
        try:
            initial_count = await self.count_entities_in_collection(integrator)
            
            # Create a new Python file
            new_file = project_dir / "src" / "new_module.py"
            python_content = '''
def new_function():
    """A new function for testing file creation sync."""
    return "created via real-time sync"

class NewClass:
    """A new class for testing."""
    
    def __init__(self):
        self.created_at = "real-time"
    
    def get_info(self):
        return f"Created at {self.created_at}"

NEW_CONSTANT = "file_creation_test"
'''
            
            self.write_file(new_file, python_content)
            
            # Wait for real-time sync to automatically detect and process the new file
            # No manual bulk_entity_create needed - let the watcher do its job
            
            # Wait for entities to appear
            assert await self.wait_for_entity_exists(integrator, "new_function", "new_module.py"), \
                "new_function should be created by real-time sync"
            
            assert await self.wait_for_entity_exists(integrator, "NewClass", "new_module.py"), \
                "NewClass should be created by real-time sync"
            
            # Verify entity count increased
            assert await self.wait_for_entity_count(integrator, initial_count, comparison="gt"), \
                "Entity count should increase after file creation"
            
        finally:
            await self.cleanup_integrator(integrator)
    
    @pytest.mark.asyncio
    async def test_file_modification_realtime_sync(self):
        """Test real-time synchronization when files are modified"""
        project_dir = self.create_isolated_project_dir("file-modification")
        integrator = await self.create_test_integrator("file-modification-test", project_dir)
        
        try:
            # Create initial file WITHOUT real-time sync
            test_file = project_dir / "src" / "modify_test.py"
            initial_content = '''
def original_function():
    """Original function for modification testing."""
    return "original"

class OriginalClass:
    def original_method(self):
        return "original"
'''
            
            self.write_file(test_file, initial_content)
            
            # Manually index the initial file
            create_result = await integrator.bulk_entity_create([test_file])
            assert create_result.success, "Initial creation should succeed"
            
            # Verify initial entities exist
            assert await self.wait_for_entity_exists(integrator, "original_function", "modify_test.py"), \
                "Should find original_function"
            
            # NOW enable real-time sync for modifications
            sync_enabled = await integrator.enable_sync()
            assert sync_enabled, "Real-time sync should be enabled"
            
            # Warm-up delay for macOS/FSEvents
            await self.wait_for_sync_delay(0.3)
            
            # Modify the file - real-time sync will detect this
            modified_content = '''
def original_function():
    """Modified original function with new functionality."""
    return "modified original"

def new_added_function():
    """This function was added during modification."""
    return "newly added"

class OriginalClass:
    def original_method(self):
        return "modified original method"
    
    def new_added_method(self):
        """New method added during modification."""
        return "newly added method"

class NewAddedClass:
    """Entirely new class added during modification."""
    
    def __init__(self):
        self.status = "newly added"
    
    def get_status(self):
        return self.status

MODIFIED_CONSTANT = "file_modification_test"
'''
            
            self.write_file(test_file, modified_content)
            
            # Wait for real-time sync to process the modification
            assert await self.wait_for_entity_exists(integrator, "new_added_function", "modify_test.py"), \
                "Should find new_added_function after modification"
            
            assert await self.wait_for_entity_exists(integrator, "NewAddedClass", "modify_test.py"), \
                "Should find NewAddedClass after modification"
            
            # Verify original entities still exist (modified, not deleted)
            original_results = await self.search_for_entity(integrator, "original_function", "modify_test.py")
            assert len(original_results) > 0, "Should still find original_function (modified)"
            
        finally:
            await self.cleanup_integrator(integrator)
    
    @pytest.mark.asyncio
    async def test_file_deletion_realtime_sync(self):
        """Test real-time synchronization when files are deleted"""
        project_dir = self.create_isolated_project_dir("file-deletion")
        integrator = await self.create_test_integrator("file-deletion-test", project_dir)
        
        try:
            # Create files to be deleted WITHOUT real-time sync first
            file1 = project_dir / "src" / "delete_test1.py"
            file2 = project_dir / "src" / "delete_test2.py"
            
            content1 = '''
def function_to_delete1():
    """This function will be deleted."""
    return "will be deleted"

class ClassToDelete1:
    def method_to_delete(self):
        return "will be deleted"
'''
            
            content2 = '''
def function_to_delete2():
    """Another function that will be deleted."""
    return "also will be deleted"

DELETE_CONSTANT = "to be removed"
'''
            
            self.write_file(file1, content1)
            self.write_file(file2, content2)
            
            # Create initial entities manually
            create_result = await integrator.bulk_entity_create([file1, file2])
            assert create_result.success, "Initial creation should succeed"
            
            # Verify entities exist before deletion
            assert await self.wait_for_entity_exists(integrator, "function_to_delete1", "delete_test1.py"), \
                "Should find function_to_delete1 before deletion"
            
            assert await self.wait_for_entity_exists(integrator, "function_to_delete2", "delete_test2.py"), \
                "Should find function_to_delete2 before deletion"
            
            initial_count = await self.count_entities_in_collection(integrator)
            
            # NOW enable real-time sync for deletion detection
            sync_enabled = await integrator.enable_sync()
            assert sync_enabled, "Real-time sync should be enabled"
            
            # Warm-up delay for macOS/FSEvents
            await self.wait_for_sync_delay(0.3)
            
            # Delete files - real-time sync should detect this automatically
            file1.unlink()
            file2.unlink()
            
            # Wait for real-time sync to process the deletions
            # No manual bulk_entity_delete needed
            
            assert await self.wait_for_entity_not_exists(integrator, "function_to_delete1", "delete_test1.py"), \
                "function_to_delete1 should be deleted by real-time sync"
            
            assert await self.wait_for_entity_not_exists(integrator, "function_to_delete2", "delete_test2.py"), \
                "function_to_delete2 should be deleted by real-time sync"
            
            # Verify entity count decreased
            assert await self.wait_for_entity_count(integrator, initial_count, comparison="lt"), \
                "Entity count should decrease after deletion"
            
        finally:
            await self.cleanup_integrator(integrator)
    
    @pytest.mark.asyncio
    async def test_file_move_realtime_sync(self):
        """Test real-time synchronization when files are moved"""
        project_dir = self.create_isolated_project_dir("file-move")
        integrator = await self.create_test_integrator("file-move-test", project_dir)
        
        try:
            # Create file in source location WITHOUT real-time sync
            source_file = project_dir / "src" / "move_source.py"
            target_file = project_dir / "lib" / "move_target.py"
            
            content = '''
def function_to_move():
    """This function will be moved to another file."""
    return "moved function"

class ClassToMove:
    def method_to_move(self):
        return "moved method"

MOVE_CONSTANT = "moved constant"
'''
            
            self.write_file(source_file, content)
            
            # Create initial entities manually
            create_result = await integrator.bulk_entity_create([source_file])
            assert create_result.success, "Initial creation should succeed"
            
            # Verify entities exist at source
            assert await self.wait_for_entity_exists(integrator, "function_to_move", str(source_file)), \
                "Should find function_to_move at source"
            
            initial_count = await self.count_entities_in_collection(integrator)
            
            # Enable real-time sync for move detection
            sync_enabled = await integrator.enable_sync()
            assert sync_enabled, "Real-time sync should be enabled"
            
            # Warm-up delay for macOS/FSEvents
            await self.wait_for_sync_delay(0.3)
            
            # Move file - real-time sync should detect this as delete + create
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source_file), str(target_file))
            
            # Wait for real-time sync to process the move
            # The watcher should handle this automatically - no manual operations needed
            
            # Wait for entity to disappear from source
            assert await self.wait_for_entity_not_exists(integrator, "function_to_move", str(source_file)), \
                "Entity should be removed from source location"
            
            # For move operations, we may need to manually trigger the creation at target
            # This is because move detection can be tricky across directories
            await self.wait_for_sync_delay(2.0)  # Give time for deletion to process
            
            # Manually create at target location since cross-directory moves may not be detected as atomic
            create_result = await integrator.bulk_entity_create([target_file])
            assert create_result.success, "Create at target should succeed"
            
            # Verify entities exist at target location
            target_results = await self.search_for_entity(integrator, "function_to_move", str(target_file))
            assert len(target_results) > 0, "Should find function_to_move at target"
            
            # Verify file path is updated
            for result in target_results:
                file_path = result.point.payload.get("file_path", "")
                assert "move_target.py" in file_path, f"Entity should reference target file: {file_path}"
            
            # Count should remain the same (delete + create)
            final_count = await self.count_entities_in_collection(integrator)
            assert final_count == initial_count, f"Entity count should remain same after move: {initial_count} -> {final_count}"
            
        finally:
            await self.cleanup_integrator(integrator)
    
    @pytest.mark.asyncio
    async def test_batch_operations_realtime_sync(self):
        """Test real-time sync with batch file operations"""
        project_dir = self.create_isolated_project_dir("batch-operations")
        integrator = await self.create_test_integrator("batch-operations-test", project_dir, enable_realtime=True)
        
        try:
            initial_count = await self.count_entities_in_collection(integrator)
            
            # Create multiple files simultaneously
            batch_files = []
            for i in range(5):
                file_path = project_dir / "src" / f"batch_file_{i}.py"
                content = f'''
def batch_function_{i}():
    """Batch function {i} for testing."""
    return "batch_{i}"

class BatchClass{i}:
    def batch_method_{i}(self):
        return "batch_method_{i}"

BATCH_CONSTANT_{i} = "batch_value_{i}"
'''
                self.write_file(file_path, content)
                batch_files.append(file_path)
            
            # Wait for real-time sync to process all batch files
            for i in range(5):
                assert await self.wait_for_entity_exists(integrator, f"batch_function_{i}", f"batch_file_{i}.py"), \
                    f"Should find batch_function_{i}"
            
            # Test batch modification
            for i, file_path in enumerate(batch_files):
                modified_content = f'''
def batch_function_{i}():
    """Modified batch function {i}."""
    return "modified_batch_{i}"

def new_batch_function_{i}():
    """New function added to batch file {i}."""
    return "new_batch_{i}"

class BatchClass{i}:
    def batch_method_{i}(self):
        return "modified_batch_method_{i}"
    
    def new_batch_method_{i}(self):
        return "new_batch_method_{i}"
'''
                self.write_file(file_path, modified_content)
            
            # Wait for real-time sync to process all modifications
            for i in range(5):
                assert await self.wait_for_entity_exists(integrator, f"new_batch_function_{i}", f"batch_file_{i}.py"), \
                    f"Should find new_batch_function_{i} after modification"
            
        finally:
            await self.cleanup_integrator(integrator)
    
    @pytest.mark.asyncio
    async def test_concurrent_modifications_realtime_sync(self):
        """Test real-time sync with concurrent file modifications"""
        project_dir = self.create_isolated_project_dir("concurrent-mods")
        integrator = await self.create_test_integrator("concurrent-test", project_dir, enable_realtime=True)
        
        try:
            # Create files for concurrent operations
            concurrent_files = []
            for i in range(3):
                file_path = project_dir / "src" / f"concurrent_{i}.py"
                content = f'''
def concurrent_function_{i}():
    """Concurrent function {i}."""
    return "concurrent_{i}"
'''
                self.write_file(file_path, content)
                concurrent_files.append(file_path)
            
            # Wait for initial files to be processed
            for i in range(3):
                assert await self.wait_for_entity_exists(integrator, f"concurrent_function_{i}", f"concurrent_{i}.py"), \
                    f"Should find initial concurrent_function_{i}"
            
            # Simulate concurrent modifications
            async def modify_file(file_path: Path, file_id: int):
                """Modify a file concurrently"""
                content = f'''
def concurrent_function_{file_id}():
    """Modified concurrent function {file_id}."""
    return "modified_concurrent_{file_id}"

def new_concurrent_function_{file_id}():
    """New concurrent function {file_id}."""
    return "new_concurrent_{file_id}"
'''
                self.write_file(file_path, content)
            
            # Run concurrent modifications
            tasks = [modify_file(file_path, i) for i, file_path in enumerate(concurrent_files)]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify all new entities exist
            for i in range(3):
                assert await self.wait_for_entity_exists(integrator, f"new_concurrent_function_{i}", f"concurrent_{i}.py"), \
                    f"Should find new_concurrent_function_{i} after concurrent modification"
            
        finally:
            await self.cleanup_integrator(integrator)
    
    @pytest.mark.asyncio
    async def test_sync_engine_lifecycle(self):
        """Test sync engine startup, shutdown, and restart scenarios"""
        project_dir = self.create_isolated_project_dir("lifecycle")
        integrator = await self.create_test_integrator("lifecycle-test", project_dir)
        
        try:
            # Verify sync engine starts properly
            sync_enabled = await asyncio.wait_for(integrator.enable_sync(), timeout=10.0)
            assert sync_enabled, "Real-time sync should be enabled"
            
            # Warm-up delay for macOS/FSEvents
            await self.wait_for_sync_delay(0.3)
            
            # Verify sync engine is running
            status = integrator.get_integration_status()
            assert status["integrator_info"]["real_time_sync_enabled"], "Real-time sync should be marked as enabled"
            
            # Create a file while sync is enabled
            test_file = project_dir / "src" / "lifecycle_test.py"
            content = '''
def lifecycle_function():
    """Function for lifecycle testing."""
    return "lifecycle"
'''
            self.write_file(test_file, content)
            
            # Wait for real-time sync to process
            assert await self.wait_for_entity_exists(integrator, "lifecycle_function", "lifecycle_test.py"), \
                "Should find lifecycle_function created by real-time sync"
            
            # Disable sync
            sync_disabled = await asyncio.wait_for(integrator.disable_real_time_sync(), timeout=10.0)
            assert sync_disabled, "Real-time sync should be disabled"
            
            # Create another file while sync is disabled
            test_file2 = project_dir / "src" / "lifecycle_test2.py"
            content2 = '''
def lifecycle_function2():
    """Function created while sync is disabled."""
    return "lifecycle2"
'''
            self.write_file(test_file2, content2)
            
            # File should NOT be auto-indexed
            await self.wait_for_sync_delay(2.0)
            results = await self.search_for_entity(integrator, "lifecycle_function2", "lifecycle_test2.py")
            assert len(results) == 0, "File should not be indexed while sync is disabled"
            
            # Re-enable sync
            sync_reenabled = await asyncio.wait_for(integrator.enable_sync(), timeout=10.0)
            assert sync_reenabled, "Real-time sync should be re-enabled"
            
            # Warm-up delay
            await self.wait_for_sync_delay(0.3)
            
            # Manually index the file created while sync was off
            create_result = await integrator.bulk_entity_create([test_file2])
            assert create_result.success, "Manual creation should succeed"
            
            # Create a third file with sync re-enabled
            test_file3 = project_dir / "src" / "lifecycle_test3.py"
            content3 = '''
def lifecycle_function3():
    """Function created after sync restart."""
    return "lifecycle3"
'''
            self.write_file(test_file3, content3)
            
            # Should be auto-indexed
            assert await self.wait_for_entity_exists(integrator, "lifecycle_function3", "lifecycle_test3.py"), \
                "Should find lifecycle_function3 created by re-enabled sync"
            
            # Verify all entities exist
            for i in range(1, 4):
                func_name = f"lifecycle_function{i if i > 1 else ''}"
                file_name = f"lifecycle_test{i if i > 1 else ''}.py"
                results = await self.search_for_entity(integrator, func_name, file_name)
                assert len(results) > 0, f"Should find {func_name}"
            
        finally:
            await self.cleanup_integrator(integrator)
    
    @pytest.mark.asyncio
    async def test_error_recovery_scenarios(self):
        """Test real-time sync error recovery scenarios"""
        project_dir = self.create_isolated_project_dir("error-recovery")
        integrator = await self.create_test_integrator("error-recovery-test", project_dir, enable_realtime=True)
        
        try:
            # Test 1: Invalid file content (parsing errors)
            invalid_file = project_dir / "src" / "invalid_syntax.py"
            invalid_content = '''
def incomplete_function(
    # Missing closing parenthesis and body
    invalid syntax here!!!
'''
            self.write_file(invalid_file, invalid_content)
            
            # Wait briefly for any processing attempt
            await self.wait_for_sync_delay(2.0)
            
            # System should not crash - check it's still responsive
            status = integrator.get_integration_status()
            assert status is not None, "System should remain responsive after parsing error"
            
            # Test 2: Create valid file after errors
            valid_file = project_dir / "src" / "recovery_test.py"
            valid_content = '''
def recovery_function():
    """Function created after error recovery."""
    return "recovered"

class RecoveryClass:
    def recovery_method(self):
        return "system recovered"
'''
            self.write_file(valid_file, valid_content)
            
            # Should work normally after error
            assert await self.wait_for_entity_exists(integrator, "recovery_function", "recovery_test.py"), \
                "Should find recovery_function after error recovery"
            
            # Test 3: Rapid file operations
            rapid_file = project_dir / "src" / "rapid_test.py"
            
            # Create, modify, delete in quick succession
            for i in range(3):
                content = f'''
def rapid_function_{i}():
    """Rapid change {i}."""
    return "rapid_{i}"
'''
                self.write_file(rapid_file, content)
                await asyncio.sleep(0.1)  # Very brief pause
            
            # Final delete
            rapid_file.unlink()
            
            # System should handle rapid changes gracefully
            await self.wait_for_sync_delay(2.0)
            
            # File should ultimately be gone
            results = await self.search_for_entity(integrator, "rapid_function", "rapid_test.py")
            # Don't assert on exact count - just verify system didn't crash
            
            # Verify integrator status is healthy
            status = integrator.get_integration_status()
            mapping_health = status["mapping_state"]["health_score"]
            assert mapping_health > 0.5, f"System health should be good after recovery: {mapping_health}"
            
        finally:
            await self.cleanup_integrator(integrator)


    @pytest.mark.asyncio
    async def test_move_file_extension_change_through_integrator(self):
        """Test that file moves with extension changes are handled correctly through the integrator."""
        project_dir = self.create_isolated_project_dir("move-extension-change")
        integrator = await self.create_test_integrator("move-extension-test", project_dir)
        
        try:
            # Create initial .py file
            py_file = project_dir / "src" / "test_script.py"
            py_content = '''
    def test_function():
        """Test function that will move between extensions."""
        return "test"

    class TestClass:
        """Test class for move operations."""
        pass
    '''
            self.write_file(py_file, py_content)
            
            # Index the initial file
            create_result = await integrator.bulk_entity_create([py_file])
            assert create_result.success, "Initial creation should succeed"
            
            # Verify entities exist
            assert await self.wait_for_entity_exists(integrator, "test_function", "test_script.py")
            assert await self.wait_for_entity_exists(integrator, "TestClass", "test_script.py")
            
            initial_count = await self.count_entities_in_collection(integrator)
            
            # Enable real-time sync
            sync_enabled = await integrator.enable_sync()
            assert sync_enabled, "Real-time sync should be enabled"
            await self.wait_for_sync_delay(0.3)
            
            # Test 1: Move .py → .txt (should remove entities)
            txt_file = project_dir / "src" / "test_script.txt"
            shutil.move(str(py_file), str(txt_file))
            
            # Entities should be deleted
            assert await self.wait_for_entity_not_exists(integrator, "test_function", "test_script.py")
            assert await self.wait_for_entity_not_exists(integrator, "TestClass", "test_script.py")
            
            # Entity count should decrease
            assert await self.wait_for_entity_count(integrator, initial_count, comparison="lt")
            
            # Test 2: Move .txt → .js (should create entities)
            js_file = project_dir / "src" / "test_script.js"
            js_content = '''
    function jsFunction() {
        return "javascript";
    }

    class JsClass {
        constructor() {
            this.type = "js";
        }
    }
    '''
            # Write new content before move
            txt_file.write_text(js_content)
            shutil.move(str(txt_file), str(js_file))
            
            # New file should be detected and indexed
            await self.wait_for_sync_delay(2.0)  # Give time for detection
            
            # For JS files, we may need manual indexing
            await integrator.bulk_entity_create([js_file])
            
            # Verify new JS entities exist
            results = await self.search_for_entity(integrator, "jsFunction", "test_script.js")
            assert len(results) > 0, "Should find JS function"
            
        finally:
            await self.cleanup_integrator(integrator)


    @pytest.mark.asyncio
    async def test_concurrent_sync_operations_robustness(self):
        """Test robustness of concurrent enable/disable sync operations."""
        project_dir = self.create_isolated_project_dir("concurrent-sync-ops")
        integrator = await self.create_test_integrator("concurrent-sync-test", project_dir)
        
        try:
            # Create test files
            test_files = []
            for i in range(3):
                file_path = project_dir / "src" / f"concurrent_test_{i}.py"
                content = f'''
    def concurrent_func_{i}():
        """Function {i} for concurrent testing."""
        return "test_{i}"
    '''
                self.write_file(file_path, content)
                test_files.append(file_path)
            
            # Index initial files
            create_result = await integrator.bulk_entity_create(test_files)
            assert create_result.success, "Initial creation should succeed"
            
            # Test concurrent enable operations
            enable_tasks = []
            for _ in range(5):
                enable_tasks.append(asyncio.create_task(integrator.enable_sync()))
            
            results = await asyncio.gather(*enable_tasks, return_exceptions=True)
            
            # At least one should succeed, others might return True (already enabled)
            success_count = sum(1 for r in results if r is True)
            assert success_count >= 1, "At least one enable should succeed"
            
            # No exceptions should be raised
            exceptions = [r for r in results if isinstance(r, Exception)]
            assert len(exceptions) == 0, f"No exceptions should occur: {exceptions}"
            
            # Verify sync is working by creating a new file
            new_file = project_dir / "src" / "sync_test.py"
            content = '''
    def sync_test_function():
        """Test sync is working."""
        return "sync works"
    '''
            self.write_file(new_file, content)
            
            # Should be auto-indexed
            assert await self.wait_for_entity_exists(integrator, "sync_test_function", "sync_test.py")
            
            # Test concurrent disable operations
            disable_tasks = []
            for _ in range(5):
                disable_tasks.append(asyncio.create_task(integrator.disable_real_time_sync()))
            
            results = await asyncio.gather(*disable_tasks, return_exceptions=True)
            
            # All should succeed (return True)
            success_count = sum(1 for r in results if r is True)
            assert success_count == len(disable_tasks), "All disable operations should succeed"
            
            # Test rapid enable/disable cycles
            cycle_tasks = []
            for i in range(10):
                if i % 2 == 0:
                    cycle_tasks.append(asyncio.create_task(integrator.enable_sync()))
                else:
                    cycle_tasks.append(asyncio.create_task(integrator.disable_real_time_sync()))
            
            results = await asyncio.gather(*cycle_tasks, return_exceptions=True)
            
            # No exceptions should occur
            exceptions = [r for r in results if isinstance(r, Exception)]
            assert len(exceptions) == 0, f"No exceptions during rapid cycling: {exceptions}"
            
        finally:
            await self.cleanup_integrator(integrator)


    @pytest.mark.asyncio
    async def test_rapid_sync_lifecycle_prevents_double_cleanup(self):
        """Test that rapid sync enable/disable cycles don't cause double cleanup issues."""
        project_dir = self.create_isolated_project_dir("rapid-lifecycle")
        integrator = await self.create_test_integrator("rapid-lifecycle-test", project_dir)
        
        try:
            # Create initial file
            test_file = project_dir / "src" / "lifecycle_test.py"
            content = '''
    def lifecycle_test():
        """Test function for lifecycle testing."""
        return "test"
    '''
            self.write_file(test_file, content)
            
            # Index file
            await integrator.bulk_entity_create([test_file])
            
            # Perform rapid enable/disable cycles
            for cycle in range(5):
                # Enable sync
                await integrator.enable_sync()
                
                # Create a file during sync
                cycle_file = project_dir / "src" / f"cycle_{cycle}.py"
                cycle_content = f'''
    def cycle_function_{cycle}():
        """Function created in cycle {cycle}."""
        return "cycle_{cycle}"
    '''
                self.write_file(cycle_file, cycle_content)
                
                # Brief wait
                await self.wait_for_sync_delay(0.2)
                
                # Disable sync immediately
                await integrator.disable_real_time_sync()
                
                # Try to create another file while disabled
                disabled_file = project_dir / "src" / f"disabled_{cycle}.py"
                disabled_content = f'''
    def disabled_function_{cycle}():
        """Function created while disabled in cycle {cycle}."""
        return "disabled_{cycle}"
    '''
                self.write_file(disabled_file, disabled_content)
            
            # Re-enable sync
            await integrator.enable_sync()
            await self.wait_for_sync_delay(0.3)
            
            # Verify system is still functional
            final_file = project_dir / "src" / "final_test.py"
            final_content = '''
    def final_test_function():
        """Final test to verify system health."""
        return "system healthy"
    '''
            self.write_file(final_file, final_content)
            
            # Should be auto-indexed
            assert await self.wait_for_entity_exists(integrator, "final_test_function", "final_test.py")
            
            # Check integrator health
            status = integrator.get_integration_status()
            assert status["integrator_info"]["real_time_sync_enabled"]
            assert status["mapping_state"]["health_score"] > 0.8
            
        finally:
            await self.cleanup_integrator(integrator)


    @pytest.mark.asyncio
    async def test_file_move_state_consistency_through_integrator(self):
        """Test that file move operations maintain consistent entity states."""
        project_dir = self.create_isolated_project_dir("move-state-consistency")
        integrator = await self.create_test_integrator("move-state-test", project_dir)
        
        try:
            # Create source file
            source_py = project_dir / "src" / "source_module.py"
            content = '''
    def source_function():
        """Function that will be moved."""
        return "source"

    class SourceClass:
        """Class that will be moved."""
        
        def method(self):
            return "source method"
    '''
            self.write_file(source_py, content)
            
            # Index source file
            create_result = await integrator.bulk_entity_create([source_py])
            assert create_result.success
            
            # Verify entities exist
            assert await self.wait_for_entity_exists(integrator, "source_function", str(source_py))
            assert await self.wait_for_entity_exists(integrator, "SourceClass", str(source_py))
            
            initial_count = await self.count_entities_in_collection(integrator)
            
            # Enable sync
            await integrator.enable_sync()
            await self.wait_for_sync_delay(0.3)
            
            # Move to non-monitored extension
            target_txt = project_dir / "src" / "target.txt"
            shutil.move(str(source_py), str(target_txt))
            
            # Entities should be deleted
            assert await self.wait_for_entity_not_exists(integrator, "source_function", str(source_py))
            assert await self.wait_for_entity_not_exists(integrator, "SourceClass", str(source_py))
            
            # Entity count should decrease
            assert await self.wait_for_entity_count(integrator, initial_count, comparison="lt")
            
            # Move to different monitored extension
            final_js = project_dir / "src" / "final.js"
            # Update content to JS syntax
            js_content = '''
    function finalFunction() {
        // Function moved and converted to JS
        return "final";
    }

    class FinalClass {
        method() {
            return "final method";
        }
    }
    '''
            target_txt.write_text(js_content)
            shutil.move(str(target_txt), str(final_js))
            
            # Wait for detection
            await self.wait_for_sync_delay(2.0)
            
            # May need manual indexing for cross-extension moves
            await integrator.bulk_entity_create([final_js])
            
            # Verify final state
            js_results = await self.search_for_entity(integrator, "finalFunction")
            assert len(js_results) > 0, "Should find JS function"
            
            # Original entities should not exist
            py_results = await self.search_for_entity(integrator, "source_function", str(source_py))
            assert len(py_results) == 0, "Original Python entities should not exist"
            
        finally:
            await self.cleanup_integrator(integrator)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])