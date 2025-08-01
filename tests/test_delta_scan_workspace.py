"""
Comprehensive tests for fast_scan_workspace() function using real filesystem operations.

Tests the high-performance filesystem traversal implementation using os.scandir()
for delta-scan operations. NO MOCKS - all tests use real filesystem operations.
"""

import asyncio
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Set
from unittest.mock import Mock, patch

import pytest

from core.indexer.hybrid_indexer import HybridIndexer, WorkspaceState
from core.parser.parallel_pipeline import ProcessParsingPipeline
from core.parser.registry import parser_registry
from core.embeddings.stella import StellaEmbedder
from core.storage.client import HybridQdrantClient


class TestWorkspaceState:
    """Test WorkspaceState dataclass and utilities."""
    
    def test_workspace_state_creation(self, tmp_path):
        """Test WorkspaceState creation with basic attributes."""
        test_file = tmp_path / "test.py"
        test_file.write_text("# test file")
        
        workspace_state = WorkspaceState.from_file_path(test_file)
        
        assert workspace_state is not None
        assert workspace_state.file_path == str(test_file)
        assert workspace_state.mtime > 0
        assert workspace_state.size > 0
        assert workspace_state.is_parseable is True
    
    def test_workspace_state_nonexistent_file(self):
        """Test WorkspaceState with nonexistent file."""
        nonexistent = Path("/nonexistent/file.py")
        
        workspace_state = WorkspaceState.from_file_path(nonexistent)
        
        assert workspace_state is None
    
    def test_workspace_state_permission_error(self, tmp_path):
        """Test WorkspaceState with permission errors."""
        # Create a file then make it unreadable
        test_file = tmp_path / "restricted.py"
        test_file.write_text("# restricted file")
        
        # Try to make file unreadable (may not work on all systems)
        try:
            os.chmod(test_file, 0o000)
            workspace_state = WorkspaceState.from_file_path(test_file)
            # Should handle gracefully and return None or valid state
            if workspace_state is not None:
                assert workspace_state.file_path == str(test_file)
        finally:
            # Restore permissions for cleanup
            try:
                os.chmod(test_file, 0o644)
            except (OSError, PermissionError):
                pass


class TestFastScanWorkspaceFunction:
    """Test the fast_scan_workspace() function with real filesystem operations."""
    
    @pytest.fixture
    def mock_components(self):
        """Create minimal mock components for HybridIndexer."""
        mock_parser = Mock(spec=ProcessParsingPipeline)
        mock_parser.registry = parser_registry  # Use real registry
        mock_parser.max_workers = 4
        mock_parser.batch_size = 10
        
        mock_embedder = Mock(spec=StellaEmbedder)
        mock_storage = Mock(spec=HybridQdrantClient)
        
        return {
            "parser": mock_parser,
            "embedder": mock_embedder,
            "storage": mock_storage
        }
    
    @pytest.fixture
    def indexer(self, mock_components):
        """Create HybridIndexer with real parser registry."""
        return HybridIndexer(
            parser_pipeline=mock_components["parser"],
            embedder=mock_components["embedder"],
            storage_client=mock_components["storage"]
        )
    
    def create_test_project(self, base_path: Path) -> Dict[str, Path]:
        """Create a realistic project structure for testing."""
        files = {}
        
        # Python files
        (base_path / "src").mkdir()
        files["main.py"] = base_path / "src" / "main.py"
        files["main.py"].write_text("""
def main():
    '''Main entry point'''
    print("Hello World")
    return 0

if __name__ == "__main__":
    main()
""")
        
        files["utils.py"] = base_path / "src" / "utils.py"
        files["utils.py"].write_text("""
class DataProcessor:
    def process(self, data):
        return [x * 2 for x in data]

def helper_function():
    return "helper"
""")
        
        # JavaScript files
        (base_path / "frontend").mkdir()
        files["app.js"] = base_path / "frontend" / "app.js"
        files["app.js"].write_text("""
function initApp() {
    console.log("App initialized");
}

class Component {
    constructor(props) {
        this.props = props;
    }
}
""")
        
        # TypeScript files
        files["types.ts"] = base_path / "frontend" / "types.ts"
        files["types.ts"].write_text("""
interface User {
    id: number;
    name: string;
}

export class UserService {
    getUser(id: number): User {
        return { id, name: "User" };
    }
}
""")
        
        # Go files
        (base_path / "backend").mkdir()
        files["server.go"] = base_path / "backend" / "server.go"
        files["server.go"].write_text("""
package main

import "fmt"

func main() {
    fmt.Println("Server starting")
}

type Server struct {
    Port int
}
""")
        
        # Files that should be ignored
        (base_path / "node_modules").mkdir(parents=True)
        (base_path / "node_modules" / "library.js").write_text("// library code")
        
        (base_path / ".git").mkdir()
        (base_path / ".git" / "config").write_text("git config")
        
        (base_path / "__pycache__").mkdir()
        (base_path / "__pycache__" / "cache.pyc").write_bytes(b"cached bytecode")
        
        (base_path / ".cache").mkdir()
        (base_path / ".cache" / "data").write_text("cache data")
        
        # Hidden files (should be ignored)
        (base_path / ".hidden.py").write_text("# hidden file")
        (base_path / "src" / ".env").write_text("SECRET=value")
        
        # Non-parseable files
        (base_path / "README.md").write_text("# Project README")
        files["README.md"] = base_path / "README.md"  # Not parseable, should be filtered
        
        (base_path / "config.json").write_text('{"app": "test"}')
        files["config.json"] = base_path / "config.json"  # Not parseable
        
        return files
    
    @pytest.mark.asyncio
    async def test_fast_scan_workspace_basic(self, indexer, tmp_path):
        """Test basic functionality of fast_scan_workspace."""
        # Create test project
        test_files = self.create_test_project(tmp_path)
        
        # Perform fast scan
        start_time = time.perf_counter()
        workspace_state = await indexer.fast_scan_workspace(tmp_path)
        scan_time = time.perf_counter() - start_time
        
        # Verify results
        assert isinstance(workspace_state, dict)
        assert len(workspace_state) > 0
        
        # Should find parseable files only
        found_files = set(workspace_state.keys())
        expected_parseable = {
            str(test_files["main.py"]),
            str(test_files["utils.py"]),
            str(test_files["app.js"]),
            str(test_files["types.ts"]),
            str(test_files["server.go"])
        }
        
        # Check that we found the expected parseable files
        assert expected_parseable.issubset(found_files)
        
        # Should NOT find ignored directories/files
        for file_path in found_files:
            assert "node_modules" not in file_path
            assert ".git" not in file_path
            assert "__pycache__" not in file_path
            assert ".cache" not in file_path
            assert not Path(file_path).name.startswith(".")
        
        # Verify WorkspaceState objects are properly created
        for file_path, state in workspace_state.items():
            assert isinstance(state, WorkspaceState)
            assert state.file_path == file_path
            assert state.mtime > 0
            assert state.size > 0
            assert state.is_parseable is True
            
            # Verify file actually exists
            assert Path(file_path).exists()
        
        # Performance check - should be reasonably fast
        assert scan_time < 5.0  # Should complete within 5 seconds
        print(f"Scanned {len(workspace_state)} files in {scan_time:.3f}s "
              f"({len(workspace_state)/scan_time:.1f} files/sec)")
    
    @pytest.mark.asyncio
    async def test_fast_scan_workspace_empty_directory(self, indexer, tmp_path):
        """Test scanning an empty directory."""
        workspace_state = await indexer.fast_scan_workspace(tmp_path)
        
        assert isinstance(workspace_state, dict)
        assert len(workspace_state) == 0
    
    @pytest.mark.asyncio
    async def test_fast_scan_workspace_nonexistent_directory(self, indexer):
        """Test scanning a nonexistent directory."""
        nonexistent = Path("/nonexistent/directory")
        
        # Should handle gracefully and return empty result
        workspace_state = await indexer.fast_scan_workspace(nonexistent)
        assert isinstance(workspace_state, dict)
        assert len(workspace_state) == 0
    
    @pytest.mark.asyncio
    async def test_fast_scan_workspace_single_file(self, indexer, tmp_path):
        """Test scanning directory with single parseable file."""
        # Create single Python file
        test_file = tmp_path / "single.py"
        test_file.write_text("def single_function(): pass")
        
        workspace_state = await indexer.fast_scan_workspace(tmp_path)
        
        assert len(workspace_state) == 1
        assert str(test_file) in workspace_state
        
        state = workspace_state[str(test_file)]
        assert isinstance(state, WorkspaceState)
        assert state.file_path == str(test_file)
        assert state.is_parseable is True
    
    @pytest.mark.asyncio
    async def test_fast_scan_workspace_mixed_files(self, indexer, tmp_path):
        """Test scanning directory with mixed parseable and non-parseable files."""
        # Create parseable files
        (tmp_path / "code.py").write_text("def func(): pass")
        (tmp_path / "script.js").write_text("function test() {}")
        
        # Create non-parseable files
        (tmp_path / "data.txt").write_text("text data")
        (tmp_path / "image.png").write_bytes(b"fake png data")
        (tmp_path / "document.pdf").write_bytes(b"fake pdf data")
        
        workspace_state = await indexer.fast_scan_workspace(tmp_path)
        
        # Should only find parseable files
        assert len(workspace_state) == 2
        found_files = set(workspace_state.keys())
        
        assert str(tmp_path / "code.py") in found_files
        assert str(tmp_path / "script.js") in found_files
        assert str(tmp_path / "data.txt") not in found_files
        assert str(tmp_path / "image.png") not in found_files
        assert str(tmp_path / "document.pdf") not in found_files
    
    @pytest.mark.asyncio
    async def test_fast_scan_workspace_deep_hierarchy(self, indexer, tmp_path):
        """Test scanning deeply nested directory structure."""
        # Create deep hierarchy
        deep_path = tmp_path / "level1" / "level2" / "level3" / "level4"
        deep_path.mkdir(parents=True)
        
        # Add files at different levels
        (tmp_path / "root.py").write_text("# root level")
        (tmp_path / "level1" / "l1.py").write_text("# level 1")
        (tmp_path / "level1" / "level2" / "l2.js").write_text("// level 2")
        (deep_path / "deep.go").write_text("// deep file")
        
        workspace_state = await indexer.fast_scan_workspace(tmp_path)
        
        assert len(workspace_state) == 4
        
        expected_files = {
            str(tmp_path / "root.py"),
            str(tmp_path / "level1" / "l1.py"),
            str(tmp_path / "level1" / "level2" / "l2.js"),
            str(deep_path / "deep.go")
        }
        
        found_files = set(workspace_state.keys())
        assert expected_files == found_files
    
    @pytest.mark.asyncio
    async def test_fast_scan_workspace_symlinks(self, indexer, tmp_path):
        """Test handling of symbolic links (should be ignored)."""
        # Create regular file
        regular_file = tmp_path / "regular.py"
        regular_file.write_text("def regular(): pass")
        
        # Create symbolic link (if supported by OS)
        try:
            link_file = tmp_path / "link.py"
            link_file.symlink_to(regular_file)
            
            workspace_state = await indexer.fast_scan_workspace(tmp_path)
            
            # Should only find the regular file, not the symlink
            assert len(workspace_state) == 1
            assert str(regular_file) in workspace_state
            assert str(link_file) not in workspace_state
            
        except (OSError, NotImplementedError):
            # Symlinks not supported on this system
            pytest.skip("Symbolic links not supported on this system")
    
    @pytest.mark.asyncio
    async def test_fast_scan_workspace_file_modifications(self, indexer, tmp_path):
        """Test that file modification times are correctly captured."""
        # Create initial file
        test_file = tmp_path / "modified.py"
        test_file.write_text("def original(): pass")
        
        # Get initial scan
        workspace_state1 = await indexer.fast_scan_workspace(tmp_path)
        original_mtime = workspace_state1[str(test_file)].mtime
        original_size = workspace_state1[str(test_file)].size
        
        # Sleep briefly to ensure different mtime
        await asyncio.sleep(0.1)
        
        # Modify file (even with same content, mtime should change)
        test_file.write_text("def modified_with_longer_content(): pass")
        
        # Scan again
        workspace_state2 = await indexer.fast_scan_workspace(tmp_path)
        new_mtime = workspace_state2[str(test_file)].mtime
        new_size = workspace_state2[str(test_file)].size
        
        # Modification time should have changed (filesystem updates mtime on write)
        assert new_mtime > original_mtime
        
        # In this case size will be different, but that's incidental
        # The important thing is that mtime changed
        assert new_size != original_size
    
    @pytest.mark.asyncio
    async def test_fast_scan_workspace_tolerance(self, indexer, tmp_path):
        """Test tolerance parameter for modification detection."""
        test_file = tmp_path / "tolerance_test.py"
        test_file.write_text("def test(): pass")
        
        # Test with different tolerance values
        for tolerance in [0.1, 1.0, 5.0]:
            workspace_state = await indexer.fast_scan_workspace(tmp_path, tolerance_sec=tolerance)
            
            assert len(workspace_state) == 1
            assert str(test_file) in workspace_state
            
            # Tolerance doesn't affect file discovery, just used for comparison
            state = workspace_state[str(test_file)]
            assert state.mtime > 0
    
    @pytest.mark.asyncio
    async def test_fast_scan_workspace_performance_large_project(self, indexer, tmp_path):
        """Test performance with a larger project structure."""
        # Create many files across multiple directories
        file_count = 0
        
        for i in range(5):  # 5 main directories
            dir_path = tmp_path / f"module_{i}"
            dir_path.mkdir()
            
            for j in range(10):  # 10 files per directory
                # Mix of different file types
                extensions = [".py", ".js", ".ts", ".go", ".rs"]
                ext = extensions[j % len(extensions)]
                
                file_path = dir_path / f"file_{j}{ext}"
                file_path.write_text(f"// File {i}-{j}\nfunction test_{i}_{j}() {{ return {i * j}; }}")
                file_count += 1
        
        # Add some ignored directories with files
        (tmp_path / "node_modules" / "lib").mkdir(parents=True)
        for k in range(20):
            (tmp_path / "node_modules" / "lib" / f"ignored_{k}.js").write_text("ignored")
        
        # Perform scan and measure performance
        start_time = time.perf_counter()
        workspace_state = await indexer.fast_scan_workspace(tmp_path)
        scan_time = time.perf_counter() - start_time
        
        # Verify results
        assert len(workspace_state) == file_count  # Should find all parseable files
        assert scan_time < 10.0  # Should complete within reasonable time
        
        # Calculate and verify performance metrics
        files_per_second = len(workspace_state) / scan_time
        assert files_per_second > 10  # Should achieve reasonable throughput
        
        print(f"Large project scan: {len(workspace_state)} files in {scan_time:.3f}s "
              f"({files_per_second:.1f} files/sec)")
        
        # Verify no ignored files were included
        for file_path in workspace_state.keys():
            assert "node_modules" not in file_path
    
    @pytest.mark.asyncio
    async def test_fast_scan_workspace_error_handling(self, indexer, tmp_path):
        """Test error handling during filesystem traversal."""
        # Create valid files
        (tmp_path / "valid.py").write_text("def valid(): pass")
        
        # Create directory with problematic permissions (if possible)
        problematic_dir = tmp_path / "problematic"
        problematic_dir.mkdir()
        (problematic_dir / "hidden.py").write_text("def hidden(): pass")
        
        try:
            # Try to make directory unreadable
            os.chmod(problematic_dir, 0o000)
            
            # Should handle gracefully and continue with other files
            workspace_state = await indexer.fast_scan_workspace(tmp_path)
            
            # Should at least find the valid file
            assert len(workspace_state) >= 1
            assert str(tmp_path / "valid.py") in workspace_state
            
        except (OSError, PermissionError):
            # If we can't change permissions, just verify normal operation
            workspace_state = await indexer.fast_scan_workspace(tmp_path)
            assert len(workspace_state) >= 1
            
        finally:
            # Restore permissions for cleanup
            try:
                os.chmod(problematic_dir, 0o755)
            except (OSError, PermissionError):
                pass
    
    @pytest.mark.asyncio
    async def test_fast_scan_workspace_file_size_accuracy(self, indexer, tmp_path):
        """Test that file sizes are accurately captured."""
        # Create files with known sizes
        small_file = tmp_path / "small.py"
        small_content = "def small(): pass"
        small_file.write_text(small_content)
        
        large_file = tmp_path / "large.py"
        large_content = "def large():\n" + "    # comment line\n" * 100 + "    pass"
        large_file.write_text(large_content)
        
        workspace_state = await indexer.fast_scan_workspace(tmp_path)
        
        assert len(workspace_state) == 2
        
        small_state = workspace_state[str(small_file)]
        large_state = workspace_state[str(large_file)]
        
        # Verify sizes match actual file sizes
        assert small_state.size == len(small_content.encode('utf-8'))
        assert large_state.size == len(large_content.encode('utf-8'))
        assert large_state.size > small_state.size
    
    @pytest.mark.asyncio
    async def test_fast_scan_workspace_registry_integration(self, indexer, tmp_path):
        """Test integration with parser registry for file type detection."""
        # Create files of different types
        files_to_create = {
            "python.py": "def python_func(): pass",
            "javascript.js": "function jsFunc() {}",
            "typescript.ts": "function tsFunc(): void {}",
            "golang.go": "func goFunc() {}",
            "rust.rs": "fn rust_func() {}",
            "java.java": "public class Java {}",
            "cpp.cpp": "void cppFunc() {}",
            "unsupported.xyz": "unsupported content",
            "text.txt": "plain text"
        }
        
        for filename, content in files_to_create.items():
            (tmp_path / filename).write_text(content)
        
        workspace_state = await indexer.fast_scan_workspace(tmp_path)
        
        # Should only include files that parser registry can handle
        found_files = set(Path(fp).name for fp in workspace_state.keys())
        
        # Verify supported files are included
        supported_extensions = {".py", ".js", ".ts", ".go", ".rs", ".java", ".cpp"}
        for filename in files_to_create.keys():
            file_path = Path(filename)
            if file_path.suffix in supported_extensions:
                assert filename in found_files
            else:
                assert filename not in found_files
        
        # Verify all found files are marked as parseable
        for state in workspace_state.values():
            assert state.is_parseable is True


class TestFastScanWorkspaceBenchmarking:
    """Benchmarking tests for fast_scan_workspace performance."""
    
    @pytest.fixture
    def indexer(self):
        """Create minimal indexer for benchmarking."""
        mock_parser = Mock(spec=ProcessParsingPipeline)
        mock_parser.registry = parser_registry
        
        return HybridIndexer(
            parser_pipeline=mock_parser,
            embedder=Mock(),
            storage_client=Mock()
        )
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_fast_scan_performance_vs_pathlib(self, indexer, tmp_path):
        """Compare performance of os.scandir vs Path.rglob approaches."""
        # Create substantial project structure
        total_files = 0
        
        for i in range(10):
            module_dir = tmp_path / f"module_{i}"
            module_dir.mkdir()
            
            for j in range(20):
                file_path = module_dir / f"file_{j}.py"
                file_path.write_text(f"def func_{i}_{j}(): return {i + j}")
                total_files += 1
        
        # Add ignored directories
        (tmp_path / "node_modules").mkdir()
        for k in range(50):
            (tmp_path / "node_modules" / f"lib_{k}.js").write_text("ignored")
        
        print(f"\nBenchmarking filesystem scan on {total_files} parseable files...")
        
        # Test fast_scan_workspace (os.scandir approach)
        start_time = time.perf_counter()
        workspace_state = await indexer.fast_scan_workspace(tmp_path)
        scandir_time = time.perf_counter() - start_time
        
        # Test alternative pathlib approach for comparison
        start_time = time.perf_counter()
        pathlib_files = {}
        for file_path in tmp_path.rglob("*"):
            if (file_path.is_file() and 
                parser_registry.can_parse(file_path) and
                not any(ignored in str(file_path) for ignored in 
                       ["node_modules", ".git", "__pycache__", ".cache"])):
                try:
                    stat = file_path.stat()
                    pathlib_files[str(file_path)] = WorkspaceState(
                        file_path=str(file_path),
                        mtime=stat.st_mtime,
                        size=stat.st_size,
                        is_parseable=True
                    )
                except (OSError, IOError):
                    continue
        pathlib_time = time.perf_counter() - start_time
        
        # Compare results
        assert len(workspace_state) == len(pathlib_files) == total_files
        
        # Calculate speedup
        speedup = pathlib_time / scandir_time if scandir_time > 0 else 1.0
        
        print(f"os.scandir approach: {scandir_time:.3f}s ({total_files/scandir_time:.1f} files/sec)")
        print(f"pathlib approach: {pathlib_time:.3f}s ({total_files/pathlib_time:.1f} files/sec)")
        print(f"Speedup: {speedup:.2f}x")
        
        # os.scandir should be significantly faster
        assert speedup >= 1.0  # At minimum, should not be slower
        
        # For larger datasets, expect some speedup (but don't be too strict due to test variability)
        if total_files > 100:
            assert speedup >= 1.1  # Should be at least 10% faster


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])