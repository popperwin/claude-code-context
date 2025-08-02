"""
FEAT6 Integration Tests: Comprehensive Delta Scan Pipeline Testing.

Tests the complete perform_delta_scan() pipeline combining all FEAT1-5 features
with real filesystem, Qdrant, and embedding operations using the Click repository.

Features tested:
- FEAT1: Workspace scanning with os.scandir
- FEAT2: Collection state retrieval with Qdrant scroll
- FEAT3: Delta calculation logic for entity differences
- FEAT4: Chunked delete operations for large batches
- FEAT5: Batch upsert operations with progress tracking
- FEAT6: Complete orchestration pipeline

NO MOCKS - Uses real repository with controlled modifications.
"""

import pytest
import asyncio
import os
import time
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

# Fix tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from core.indexer.hybrid_indexer import HybridIndexer, IndexingJobConfig
from core.parser.parallel_pipeline import ProcessParsingPipeline
from core.storage.client import HybridQdrantClient
from core.storage.schemas import CollectionManager, QdrantSchema
from core.embeddings.stella import StellaEmbedder
from core.models.config import StellaConfig

logger = logging.getLogger(__name__)


class TestDeltaScanIntegration:
    """Comprehensive integration tests for perform_delta_scan() pipeline"""

    # Shared resources across all tests to avoid repeated heavy initialization
    _shared_embedder: Optional[StellaEmbedder] = None
    _shared_storage_client: Optional[HybridQdrantClient] = None
    _shared_parser_pipeline: Optional[ProcessParsingPipeline] = None
    
    @classmethod
    def setup_class(cls):
        """Setup test environment with real repository"""
        cls.test_dir = Path("test-harness/delta-scan-integration").resolve()
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Clone a small, manageable real repository for testing
        cls.repo_info = {
            "url": "https://github.com/python-validators/validators.git",
            "path": cls.test_dir / "validators",
            "branch": "master"  # validators uses master branch
        }
        
        cls.created_collections = set()
        cls.indexers = []  # Track indexers for cleanup

        # Clean up any stray test collections from previous runs
        cls._cleanup_old_test_collections()

        # Initialize shared heavy resources once per class
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        if cls._shared_embedder is None:
            embedder_config = StellaConfig(
                model_name="stella_en_400M_v5",
                batch_size=32,
                cache_size=100,
                cache_ttl_seconds=300
            )
            cls._shared_embedder = StellaEmbedder(embedder_config)
            model_loaded = loop.run_until_complete(cls._shared_embedder.load_model())
            if not model_loaded:
                pytest.skip("Stella model not available for testing")

        if cls._shared_storage_client is None:
            cls._shared_storage_client = HybridQdrantClient(url="http://localhost:6334")
            loop.run_until_complete(cls._shared_storage_client.connect())

        if cls._shared_parser_pipeline is None:
            cls._shared_parser_pipeline = ProcessParsingPipeline(
                max_workers=2,
                batch_size=20,
                execution_mode="thread"
            )

        loop.close()
    
    @classmethod
    def teardown_class(cls):
        """Cleanup test environment and shared resources"""
        # Delete collections created during this test run
        try:
            import requests
            for collection_name in cls.created_collections:
                try:
                    requests.delete(f"http://localhost:6334/collections/{collection_name}", timeout=5)
                except Exception:
                    pass
        except Exception:
            pass

        # Disconnect shared storage client
        if cls._shared_storage_client is not None:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(cls._shared_storage_client.disconnect())
            loop.close()
            cls._shared_storage_client = None

        # Unload shared embedder
        if cls._shared_embedder is not None:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(cls._shared_embedder.unload_model())
            loop.close()
            cls._shared_embedder = None

        # Clean up test directory
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir, ignore_errors=True)
    
    def get_or_clone_repository(self) -> Path:
        """Get repository, cloning if needed, with reuse across tests"""
        repo_path = self.repo_info["path"]
        
        if repo_path.exists() and (repo_path / ".git").exists():
            # Repository exists, pull latest changes
            try:
                subprocess.run(
                    ["git", "fetch", "origin"], 
                    cwd=repo_path, 
                    check=True, 
                    capture_output=True
                )
                subprocess.run(
                    ["git", "reset", "--hard", f"origin/{self.repo_info['branch']}"],
                    cwd=repo_path,
                    check=True,
                    capture_output=True
                )
                logger.info(f"Updated existing repository at {repo_path}")
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to update repository: {e}")
                # Fall through to use existing state
        else:
            # Clone repository
            try:
                # First try with specified branch
                result = subprocess.run([
                    "git", "clone", 
                    "--branch", self.repo_info["branch"],
                    "--depth", "1",  # Shallow clone for speed
                    self.repo_info["url"], 
                    str(repo_path)
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    # Try without branch specification (use default)
                    logger.warning(f"Failed to clone with branch {self.repo_info['branch']}, trying default branch...")
                    result = subprocess.run([
                        "git", "clone", 
                        "--depth", "1",  # Shallow clone for speed
                        self.repo_info["url"], 
                        str(repo_path)
                    ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    error_msg = f"Git clone failed: {result.stderr}"
                    logger.error(error_msg)
                    pytest.skip(f"Failed to clone repository: {error_msg}")
                
                logger.info(f"Cloned repository to {repo_path}")
            except FileNotFoundError:
                pytest.skip("Git command not found - skipping integration test")
            except Exception as e:
                pytest.skip(f"Failed to clone repository: {e}")
        
        return repo_path
    
    def create_test_modifications(self, repo_path: Path) -> Dict[str, List[Path]]:
        """Create controlled modifications to the repository for delta testing"""
        modifications = {
            "added": [],
            "modified": [],
            "deleted": []
        }
        
        # Add new files
        new_file1 = repo_path / "src" / "click" / "delta_test_module.py"
        new_file1.parent.mkdir(parents=True, exist_ok=True)
        new_file1.write_text('''
"""
Delta scan test module - newly added file.
"""

def delta_test_function():
    """Test function for delta scan testing."""
    return "delta test"

class DeltaTestClass:
    """Test class for delta scan testing."""
    
    def __init__(self):
        self.test_value = "delta"
    
    def get_value(self):
        """Get the test value."""
        return self.test_value

DELTA_CONSTANT = "added_by_delta_test"
''')
        modifications["added"].append(new_file1)
        
        new_file2 = repo_path / "delta_utils.py"
        new_file2.write_text('''
"""
Delta utilities - another new file.
"""

import os
from typing import Dict, Any

def delta_utility_function(data: Dict[str, Any]) -> str:
    """Utility function for delta testing."""
    return f"Processing: {data}"

class DeltaUtility:
    """Utility class for delta operations."""
    
    @staticmethod
    def process_data(items):
        """Process data items."""
        return [item.upper() for item in items]
''')
        modifications["added"].append(new_file2)
        
        # Modify existing repository files (exclude files we just created)
        python_files = list(repo_path.glob("**/*.py"))
        newly_added_files = {str(f) for f in modifications["added"]}
        
        if python_files:
            # Find existing files to modify (exclude ones we just added)
            for py_file in python_files[:5]:  # Check more files to find a good candidate
                if (py_file.exists() and 
                    py_file.stat().st_size < 10000 and  # Small files only
                    str(py_file) not in newly_added_files):  # Don't modify files we just created
                    
                    try:
                        original_content = py_file.read_text()
                        
                        # Add a comment and new function to the file
                        modified_content = original_content + '''

# DELTA SCAN MODIFICATION - Added by integration test
def delta_scan_added_function():
    """Function added by delta scan integration test."""
    return "delta modification"
'''
                        py_file.write_text(modified_content)
                        modifications["modified"].append(py_file)
                        logger.info(f"Modified existing file: {py_file}")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to modify {py_file}: {e}")
                        continue
        
        # Create a file to delete later
        delete_file = repo_path / "temp_delete_test.py"
        delete_file.write_text('''
"""
File that will be deleted for delta testing.
"""

def function_to_delete():
    """This function will be deleted."""
    return "will be deleted"
''')
        modifications["added"].append(delete_file)  # First add it
        
        return modifications
    
    async def create_test_indexer(self, project_path: Path, project_name: str) -> HybridIndexer:
        """Create a HybridIndexer using shared resources for efficiency."""
        cls = self.__class__

        # Ensure shared embedder is ready
        if cls._shared_embedder is None:
            embedder_config = StellaConfig(
                model_name="stella_en_400M_v5",
                batch_size=32,
                cache_size=100,
                cache_ttl_seconds=300
            )
            cls._shared_embedder = StellaEmbedder(embedder_config)
            model_loaded = await cls._shared_embedder.load_model()
            if not model_loaded:
                pytest.skip("Stella model not available for testing")

        # Ensure shared storage client is connected
        if cls._shared_storage_client is None:
            cls._shared_storage_client = HybridQdrantClient(url="http://localhost:6334")
            await cls._shared_storage_client.connect()

        # Ensure shared parser pipeline is initialized
        if cls._shared_parser_pipeline is None:
            cls._shared_parser_pipeline = ProcessParsingPipeline(
                max_workers=2,
                batch_size=20,
                execution_mode="thread"  # Faster for tests
            )

        # Create indexer config specific to this project
        config = IndexingJobConfig(
            project_path=project_path,
            project_name=project_name,
            include_patterns=["*.py"],
            exclude_patterns=[
                "__pycache__/*",
                ".git/*",
                "*.pyc",
                ".tox/*",
                "build/*",
                "dist/*"
            ],
            max_workers=2,
            batch_size=20
        )

        # Instantiate the indexer with shared components
        indexer = HybridIndexer(
            parser_pipeline=cls._shared_parser_pipeline,
            embedder=cls._shared_embedder,
            storage_client=cls._shared_storage_client,
            cache_manager=None,  # Skip cache for simplicity
            config=config
        )

        # Track for cleanup purposes
        cls.indexers.append(indexer)

        return indexer

    @classmethod
    def _cleanup_old_test_collections(cls):
        """Remove leftover collections from previous test runs to keep Qdrant clean."""
        try:
            import requests
            response = requests.get("http://localhost:6334/collections", timeout=5)
            if response.status_code == 200:
                collections = response.json().get("result", {}).get("collections", [])
                for collection in collections:
                    name = collection.get("name", "")
                    if "delta-scan" in name and "-code" in name:
                        try:
                            requests.delete(f"http://localhost:6334/collections/{name}", timeout=5)
                        except Exception:
                            pass
        except Exception:
            pass
    
    async def create_collection_for_test(self, storage_client: HybridQdrantClient, project_name: str) -> str:
        """Create test collection with unique name"""
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        collection_name = f"{project_name}-{unique_id}-code"
        
        # Create collection with proper schema
        collection_config = QdrantSchema.get_code_collection_config(collection_name)
        create_result = await storage_client.create_collection(collection_config)
        
        if not create_result.success:
            raise RuntimeError(f"Failed to create collection: {create_result.error}")
        
        # Track for cleanup
        self.__class__.created_collections.add(collection_name)
        
        return collection_name
    
    @pytest.mark.asyncio
    async def test_complete_delta_scan_pipeline_initial_scan(self):
        """Test complete delta scan pipeline with initial scan of real repository"""
        repo_path = self.get_or_clone_repository()
        indexer = await self.create_test_indexer(repo_path, "delta-scan-initial")
        
        try:
            collection_name = await self.create_collection_for_test(
                indexer.storage_client, "delta-scan-initial"
            )
            
            # Perform initial delta scan (should be full scan since collection is empty)
            logger.info("Starting initial delta scan...")
            
            start_time = time.perf_counter()
            result = await indexer.perform_delta_scan(
                project_path=repo_path,
                collection_name=collection_name,
                force_full_scan=False  # Let it decide
            )
            scan_duration = time.perf_counter() - start_time
            
            # Verify comprehensive results
            assert result["success"] is True, f"Delta scan failed: {result.get('error_message')}"
            assert result["operation_type"] == "delta_scan"
            assert result["total_duration_ms"] > 0
            
            # Verify phases were executed
            phases = result["phases"]
            assert "workspace_scan" in phases
            assert "collection_state" in phases
            assert "delta_calculation" in phases
            assert "entity_processing" in phases
            assert "upsert_operations" in phases
            assert "delete_operations" in phases
            
            # Verify workspace scanning found files
            workspace_phase = phases["workspace_scan"]
            assert workspace_phase["success"] is True
            assert workspace_phase["total_files"] > 5, "Should find Python files in validators repo"
            assert workspace_phase["scan_time_ms"] > 0
            
            # Verify collection state (should be empty initially)
            collection_phase = phases["collection_state"]
            assert collection_phase["success"] is True
            assert collection_phase["total_entities"] == 0, "Collection should be empty initially"
            
            # Verify delta calculation (should detect all files as new)
            delta_phase = phases["delta_calculation"]
            assert delta_phase["success"] is True
            assert delta_phase["files_to_add"] > 5, "Should add Python files"
            assert delta_phase["files_to_modify"] == 0, "No files to modify initially"
            assert delta_phase["files_to_delete"] == 0, "No files to delete initially"
            
            # Verify entity processing
            processing_phase = phases["entity_processing"]
            assert processing_phase["success"] is True
            assert processing_phase["total_entities"] > 10, "Should extract entities from validators"
            assert processing_phase["processing_time_ms"] > 0
            
            # Verify upsert operations
            upsert_phase = phases["upsert_operations"]
            assert upsert_phase["success"] is True
            assert upsert_phase["upserted_entities"] > 10, "Should upsert entities"
            assert upsert_phase["processed_chunks"] > 0
            assert upsert_phase["upsert_time_ms"] > 0
            
            # Verify no delete operations (initial scan)
            delete_phase = phases["delete_operations"]
            assert delete_phase["success"] is True
            assert delete_phase["deleted_entities"] == 0, "No deletions in initial scan"
            
            # Performance validation
            logger.info(f"Delta scan completed in {scan_duration:.2f} seconds")
            assert scan_duration < 300, f"Initial scan took too long: {scan_duration:.2f}s"
            
            # Verify entities were actually stored in collection
            collection_info = await indexer.storage_client.get_collection_info(collection_name)
            stored_count = collection_info.get("points_count", 0) if collection_info else 0
            assert stored_count > 10, f"Expected >10 entities stored, got {stored_count}"
            
            logger.info(f"✅ Initial delta scan: {stored_count} entities in {scan_duration:.2f}s")
            
        finally:
            # No per-test cleanup needed; shared resources are cleaned up in teardown_class
            pass
    
    @pytest.mark.asyncio
    async def test_complete_delta_scan_pipeline_with_modifications(self):
        """Test complete delta scan pipeline with repository modifications"""
        repo_path = self.get_or_clone_repository()
        indexer = await self.create_test_indexer(repo_path, "delta-scan-mods")
        
        try:
            collection_name = await self.create_collection_for_test(
                indexer.storage_client, "delta-scan-mods"
            )
            
            # Phase 1: Initial scan to populate collection
            logger.info("Phase 1: Initial population scan...")
            
            initial_result = await indexer.perform_delta_scan(
                project_path=repo_path,
                collection_name=collection_name,
                force_full_scan=True  # Force full scan initially
            )
            
            assert initial_result["success"] is True
            initial_entity_count = initial_result["phases"]["upsert_operations"]["upserted_entities"]
            assert initial_entity_count > 10, "Should have entities from initial scan"
            
            # Verify entities were actually stored in Qdrant after initial scan
            verification_state = await indexer.get_collection_state(collection_name)
            actual_stored_count = verification_state.get("entity_count", 0)
            logger.error(f"Initial scan verification: upserted={initial_entity_count}, actually_stored={actual_stored_count}")
            
            if actual_stored_count != initial_entity_count:
                logger.error(f"MISMATCH: Expected {initial_entity_count} entities, but Qdrant has {actual_stored_count}")
                logger.error(f"Collection exists: {verification_state.get('exists', False)}")
                logger.error(f"Files in verification state: {len(verification_state.get('entities', {}))}")
            
            assert actual_stored_count == initial_entity_count, \
                f"Initial storage failed: {initial_entity_count} upserted but only {actual_stored_count} stored"
            
            # Wait brief moment to ensure different timestamps
            await asyncio.sleep(1.0)
            
            # Phase 2: Make modifications to repository
            logger.info("Phase 2: Creating modifications...")
            modifications = self.create_test_modifications(repo_path)
            
            # Delete one of the added files to test deletion
            if modifications["added"]:
                delete_target = modifications["added"][-1]  # temp_delete_test.py
                if delete_target.exists():
                    delete_target.unlink()
                    modifications["deleted"].append(delete_target)
                    modifications["added"].remove(delete_target)
            
            # Phase 3: Delta scan to detect modifications
            logger.info("Phase 3: Delta scan with modifications...")
            
            start_time = time.perf_counter()
            delta_result = await indexer.perform_delta_scan(
                project_path=repo_path,
                collection_name=collection_name,
                force_full_scan=False  # Should detect changes
            )
            delta_duration = time.perf_counter() - start_time
            
            # Verify delta scan detected changes
            assert delta_result["success"] is True, f"Delta scan with mods failed: {delta_result.get('error_message')}"
            
            # Verify phases detected changes
            phases = delta_result["phases"]
            
            # Workspace scan should find all files
            workspace_phase = phases["workspace_scan"]
            assert workspace_phase["total_files"] >= initial_result["phases"]["workspace_scan"]["total_files"]
            
            # Collection state should have existing entities
            collection_phase = phases["collection_state"]
            
            # Debug: Show entity count difference and check for missing files using get_collection_state
            if collection_phase["total_entities"] < initial_entity_count:
                logger.error(f"Entity count mismatch: initial={initial_entity_count}, current={collection_phase['total_entities']}")
                
                # Use get_collection_state to get complete entity information
                try:
                    current_state = await indexer.get_collection_state(collection_name)
                    current_entities = current_state.get("entities", {})
                    
                    # Check which files are missing vs existing
                    missing_files = []
                    existing_files = []
                    file_entity_counts = {}
                    
                    for file_path, entities in current_entities.items():
                        file_exists = Path(file_path).exists()
                        entity_count = len(entities) if isinstance(entities, list) else 0
                        file_entity_counts[file_path] = entity_count
                        
                        if file_exists:
                            existing_files.append(file_path)
                        else:
                            missing_files.append(file_path)
                    
                    logger.error(f"Files analysis - Existing: {len(existing_files)}, Missing: {len(missing_files)}")
                    logger.error(f"Total entities in current_state: {current_state.get('entity_count', 0)}")
                    
                    if missing_files:
                        logger.error(f"Missing files: {missing_files[:5]}")
                        missing_entity_count = sum(file_entity_counts.get(f, 0) for f in missing_files)
                        logger.error(f"Entities in missing files: {missing_entity_count}")
                    
                    # Show sample of existing files with entity counts
                    sample_existing = list(existing_files)[:5]
                    for file_path in sample_existing:
                        count = file_entity_counts.get(file_path, 0)
                        logger.error(f"Existing: {file_path} ({count} entities)")
                        
                except Exception as e:
                    logger.error(f"Error during entity debugging: {e}")
            
            assert collection_phase["total_entities"] >= initial_entity_count, \
                f"Collection entities decreased: {initial_entity_count} -> {collection_phase['total_entities']}"
            
            # Delta calculation should detect changes
            delta_phase = phases["delta_calculation"]
            assert delta_phase["files_to_add"] >= len(modifications["added"]), \
                f"Should detect {len(modifications['added'])} new files"
            assert delta_phase["files_to_modify"] >= len(modifications["modified"]), \
                f"Should detect {len(modifications['modified'])} modified files"
            # Note: deletion detection depends on whether file was previously indexed
            
            # Verify entity processing handled changes
            processing_phase = phases["entity_processing"]
            assert processing_phase["total_entities"] > 0, "Should process entities from changes"
            
            # Verify upsert operations
            upsert_phase = phases["upsert_operations"]
            if upsert_phase["upserted_entities"] > 0:
                assert upsert_phase["processed_chunks"] > 0
                assert upsert_phase["upsert_time_ms"] > 0
            
            # Performance validation for delta scan
            logger.info(f"Delta scan with modifications completed in {delta_duration:.2f} seconds")
            assert delta_duration < 120, f"Delta scan took too long: {delta_duration:.2f}s"
            
            # Verify final entity count
            final_collection_info = await indexer.storage_client.get_collection_info(collection_name)
            final_count = final_collection_info.get("points_count", 0) if final_collection_info else 0
            
            logger.info(f"✅ Delta scan with modifications: {final_count} entities in {delta_duration:.2f}s")
            
            # Verify that delta scan was faster than initial scan (should be in most cases)
            initial_duration = initial_result["total_duration_ms"] / 1000
            if delta_duration < initial_duration:
                logger.info(f"✅ Delta scan was faster: {delta_duration:.2f}s vs {initial_duration:.2f}s")
            
        finally:
            # No per-test cleanup needed; shared resources are cleaned up in teardown_class
            pass
    
    @pytest.mark.asyncio
    async def test_delta_scan_progress_tracking(self):
        """Test delta scan progress tracking and callbacks"""
        repo_path = self.get_or_clone_repository()
        indexer = await self.create_test_indexer(repo_path, "delta-scan-progress")
        
        try:
            collection_name = await self.create_collection_for_test(
                indexer.storage_client, "delta-scan-progress"
            )
            
            # Track progress events
            progress_events = []
            
            def progress_callback(current: int, total: int, progress_data: Dict[str, Any]):
                """Track progress events"""
                event = {
                    "timestamp": time.time(),
                    "current": current,
                    "total": total,
                    "progress_data": progress_data.copy(),
                    "percentage": (current / total * 100) if total > 0 else 0
                }
                progress_events.append(event)
                logger.debug(f"Progress: {current}/{total} ({event['percentage']:.1f}%) - {progress_data.get('phase', 'unknown')}")
            
            # Perform delta scan with progress tracking
            logger.info("Testing delta scan with progress tracking...")
            
            result = await indexer.perform_delta_scan(
                project_path=repo_path,
                collection_name=collection_name,
                progress_callback=progress_callback,
                force_full_scan=True
            )
            
            # Verify scan succeeded
            assert result["success"] is True
            
            # Verify progress tracking worked
            assert len(progress_events) > 0, "Should have received progress events"
            
            # Verify progress events have expected structure
            for event in progress_events:
                assert "timestamp" in event
                assert "current" in event
                assert "total" in event
                assert "progress_data" in event
                assert "percentage" in event
                
                progress_data = event["progress_data"]
                assert "phase" in progress_data
                
                # Verify percentage is reasonable
                assert 0 <= event["percentage"] <= 100
            
            # Verify we got progress from different phases
            phases_seen = set()
            for event in progress_events:
                phase = event["progress_data"].get("phase", "unknown")
                phases_seen.add(phase)
            
            # Should see progress from multiple phases
            expected_phases = {"workspace_scan", "entity_processing", "upsert_operations"}
            phases_found = expected_phases.intersection(phases_seen)
            assert len(phases_found) >= 2, f"Should see progress from multiple phases: {phases_seen}"
            
            # Verify final progress event shows completion
            if progress_events:
                final_event = progress_events[-1]
                assert final_event["current"] == final_event["total"], \
                    "Final progress should show completion"
                assert final_event["percentage"] == 100.0, \
                    "Final progress should be 100%"
            
            logger.info(f"✅ Progress tracking: {len(progress_events)} events from phases {phases_seen}")
            
        finally:
            # No per-test cleanup needed; shared resources are cleaned up in teardown_class
            pass
    
    @pytest.mark.asyncio
    async def test_delta_scan_force_full_scan_behavior(self):
        """Test delta scan force_full_scan parameter behavior"""
        repo_path = self.get_or_clone_repository()
        indexer = await self.create_test_indexer(repo_path, "delta-scan-force")
        
        try:
            collection_name = await self.create_collection_for_test(
                indexer.storage_client, "delta-scan-force"
            )
            
            # Test 1: Force full scan on empty collection
            logger.info("Test 1: Force full scan on empty collection...")
            
            full_scan_result = await indexer.perform_delta_scan(
                project_path=repo_path,
                collection_name=collection_name,
                force_full_scan=True
            )
            
            assert full_scan_result["success"] is True
            full_scan_entities = full_scan_result["phases"]["upsert_operations"]["upserted_entities"]
            assert full_scan_entities > 10, "Full scan should process entities"
        
            
            # Test 2: Force full scan again (should reprocess everything)
            logger.info("Test 2: Force full scan on populated collection...")
            
            force_scan_result = await indexer.perform_delta_scan(
                project_path=repo_path,
                collection_name=collection_name,
                force_full_scan=True
            )
            
            assert force_scan_result["success"] is True
            force_scan_entities = force_scan_result["phases"]["upsert_operations"]["upserted_entities"]
            
            # Should process similar number of entities as first scan
            entity_ratio = force_scan_entities / full_scan_entities
            assert 0.8 <= entity_ratio <= 1.2, \
                f"Force scan should process similar entities: {force_scan_entities} vs {full_scan_entities}"
            
            logger.info(f"✅ Force scan behavior verified: {full_scan_entities} → {force_scan_entities} entities")
            
        finally:
            # No per-test cleanup needed; shared resources are cleaned up in teardown_class
            pass
    
    
    @pytest.mark.asyncio
    async def test_delta_scan_error_handling_and_recovery(self):
        """Test delta scan error handling and recovery scenarios"""
        repo_path = self.get_or_clone_repository()
        indexer = await self.create_test_indexer(repo_path, "delta-scan-errors")
        
        try:
            collection_name = await self.create_collection_for_test(
                indexer.storage_client, "delta-scan-errors"
            )
            
            # Test 1: Delta scan with invalid project path
            logger.info("Test 1: Invalid project path...")
            
            invalid_path = Path("/nonexistent/path/that/does/not/exist")
            error_result = await indexer.perform_delta_scan(
                project_path=invalid_path,
                collection_name=collection_name
            )
            
            # Should handle error gracefully
            assert error_result["success"] is False
            assert "error_message" in error_result
            assert "nonexistent" in error_result["error_message"].lower() or "not" in error_result["error_message"].lower()
            
            # Test 2: Delta scan with invalid collection name
            logger.info("Test 2: Invalid collection name...")
            
            invalid_collection = "nonexistent-collection-12345"
            invalid_result = await indexer.perform_delta_scan(
                project_path=repo_path,
                collection_name=invalid_collection
            )
            
            # Should handle gracefully - either create collection or fail appropriately
            if not invalid_result["success"]:
                assert "error_message" in invalid_result
                logger.info(f"Expected error with invalid collection: {invalid_result['error_message']}")
            else:
                # If it created the collection, that's also valid behavior
                logger.info("System auto-created collection for invalid name")
            
            # Test 3: Normal scan to verify system recovery
            logger.info("Test 3: Normal scan after errors...")
            
            recovery_result = await indexer.perform_delta_scan(
                project_path=repo_path,
                collection_name=collection_name,
                force_full_scan=True
            )
            
            # Should work normally after errors
            assert recovery_result["success"] is True
            assert recovery_result["phases"]["upsert_operations"]["upserted_entities"] > 10
            
            logger.info("✅ Error handling and recovery verified")
            
        finally:
            # No per-test cleanup needed; shared resources are cleaned up in teardown_class
            pass


if __name__ == "__main__":
    # Allow direct execution for debugging
    pytest.main([__file__, "-v", "-s"])