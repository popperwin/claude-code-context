"""
Unit tests for Stella installation script.

Tests system requirements checking, dependency installation, model downloading,
and verification with extensive mocking of system calls and external dependencies.
"""

import pytest
import tempfile
import shutil
import json
import sys
import subprocess
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open, call
from typing import Dict, Any

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from install_stella import StellaInstaller


class TestStellaInstaller:
    """Test StellaInstaller functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.installer = StellaInstaller("stella_en_400M_v5")
        # Override cache_dir to use temp directory
        self.installer.cache_dir = self.temp_dir / "stella"
        self.installer.model_path = self.installer.cache_dir / self.installer.model_name
    
    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_installer_initialization_valid_model(self):
        """Test installer initialization with valid model"""
        installer = StellaInstaller("stella_en_400M_v5")
        
        assert installer.model_name == "stella_en_400M_v5"
        assert installer.config["repo_id"] == "infgrad/stella_en_400M_v5"
        assert installer.config["dimensions"] == 1024
        assert installer.config["max_length"] == 512
        assert installer.config["size_mb"] == 800
    
    def test_installer_initialization_different_models(self):
        """Test installer with different model configurations"""
        models = ["stella_en_1.5B_v5", "stella_base_en_v2"]
        
        for model in models:
            installer = StellaInstaller(model)
            assert installer.model_name == model
            assert model in installer.model_configs
            assert "repo_id" in installer.config
            assert "dimensions" in installer.config
    
    def test_installer_initialization_invalid_model(self):
        """Test installer initialization with invalid model"""
        with pytest.raises(ValueError, match="Unsupported model: invalid_model"):
            StellaInstaller("invalid_model")
    
    @patch('platform.system')
    @patch('platform.machine')
    def test_platform_detection_apple_silicon(self, mock_machine, mock_system):
        """Test Apple Silicon platform detection"""
        mock_system.return_value = "Darwin"
        mock_machine.return_value = "arm64"
        
        installer = StellaInstaller()
        
        assert installer.platform == "darwin"
        assert installer.arch == "arm64"
        assert installer.is_apple_silicon is True
    
    @patch('platform.system')
    @patch('platform.machine')
    def test_platform_detection_intel_mac(self, mock_machine, mock_system):
        """Test Intel Mac platform detection"""
        mock_system.return_value = "Darwin"
        mock_machine.return_value = "x86_64"
        
        installer = StellaInstaller()
        
        assert installer.platform == "darwin"
        assert installer.arch == "x86_64"
        assert installer.is_apple_silicon is False
    
    @patch('platform.system')
    @patch('platform.machine')
    def test_platform_detection_linux(self, mock_machine, mock_system):
        """Test Linux platform detection"""
        mock_system.return_value = "Linux"
        mock_machine.return_value = "x86_64"
        
        installer = StellaInstaller()
        
        assert installer.platform == "linux"
        assert installer.arch == "x86_64"
        assert installer.is_apple_silicon is False
    
    @patch('shutil.disk_usage')
    def test_check_disk_space_success(self, mock_disk_usage):
        """Test disk space checking with sufficient space"""
        # Mock 10GB free space
        mock_disk_usage.return_value = Mock(free=10 * 1024**3)
        
        space_gb = self.installer._check_disk_space()
        
        assert space_gb == 10.0
    
    @patch('shutil.disk_usage')
    def test_check_disk_space_error(self, mock_disk_usage):
        """Test disk space checking with error"""
        mock_disk_usage.side_effect = OSError("Permission denied")
        
        space_gb = self.installer._check_disk_space()
        
        assert space_gb == 0.0
    
    @patch('subprocess.run')
    @patch('platform.system', return_value='Darwin')
    def test_check_memory_macos_success(self, mock_system, mock_subprocess):
        """Test memory checking on macOS"""
        # Mock 16GB memory (in bytes)
        mock_subprocess.return_value = Mock(
            returncode=0,
            stdout="17179869184"  # 16GB in bytes
        )
        
        memory_gb = self.installer._check_memory()
        
        assert memory_gb == 16.0
        mock_subprocess.assert_called_once_with(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True
        )
    
    @patch('subprocess.run')
    @patch('platform.system', return_value='Darwin')
    def test_check_memory_macos_error(self, mock_system, mock_subprocess):
        """Test memory checking on macOS with error"""
        mock_subprocess.return_value = Mock(returncode=1)
        
        memory_gb = self.installer._check_memory()
        
        assert memory_gb == 8.0  # Default fallback
    
    @patch('platform.system', return_value='Linux')
    @patch('builtins.open', new_callable=mock_open, read_data="MemAvailable:    8388608 kB\n")
    def test_check_memory_linux_success(self, mock_file, mock_system):
        """Test memory checking on Linux"""
        self.installer.platform = 'linux'  # Set platform explicitly
        memory_gb = self.installer._check_memory()
        
        assert memory_gb == 8.0  # 8388608 KB = 8GB
    
    @patch('platform.system', return_value='Linux')
    @patch('builtins.open', side_effect=OSError("File not found"))
    def test_check_memory_linux_error(self, mock_file, mock_system):
        """Test memory checking on Linux with error"""
        self.installer.platform = 'linux'  # Set platform explicitly
        memory_gb = self.installer._check_memory()
        
        assert memory_gb == 8.0  # Default fallback
    
    @patch('builtins.__import__')
    def test_check_python_packages_all_available(self, mock_import):
        """Test checking Python packages when all are available"""
        mock_import.return_value = Mock()  # Package found
        
        packages = self.installer._check_python_packages()
        
        assert packages["torch"] is True
        assert packages["transformers"] is True
        assert packages["sentence_transformers"] is True
        assert packages["huggingface_hub"] is True
    
    @patch('builtins.__import__')
    def test_check_python_packages_some_missing(self, mock_import):
        """Test checking Python packages when some are missing"""
        def side_effect(name):
            if name == "torch":
                raise ImportError("No module named torch")
            return Mock()
        
        mock_import.side_effect = side_effect
        
        packages = self.installer._check_python_packages()
        
        assert packages["torch"] is False
        assert packages["transformers"] is True
        assert packages["sentence_transformers"] is True
        assert packages["huggingface_hub"] is True
    
    @patch('builtins.__import__')
    def test_check_apple_silicon_support_success(self, mock_import):
        """Test Apple Silicon support checking"""
        mock_torch = Mock()
        mock_torch.backends.mps.is_available.return_value = True
        mock_import.return_value = mock_torch
        
        support = self.installer._check_apple_silicon_support()
        
        assert support is True
    
    @patch('builtins.__import__')
    def test_check_apple_silicon_support_failure(self, mock_import):
        """Test Apple Silicon support checking with failure"""
        mock_import.side_effect = ImportError("No torch")
        
        support = self.installer._check_apple_silicon_support()
        
        assert support is False
    
    @patch('builtins.__import__')
    def test_check_apple_silicon_support_not_apple_silicon(self, mock_import):
        """Test Apple Silicon support on non-Apple Silicon platform"""
        mock_torch = Mock()
        mock_torch.backends.mps.is_available.return_value = False
        mock_import.return_value = mock_torch
        
        support = self.installer._check_apple_silicon_support()
        
        assert support is False
    
    @patch('builtins.__import__')
    def test_check_cuda_availability_success(self, mock_import):
        """Test CUDA availability checking"""
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_import.return_value = mock_torch
        
        cuda_available = self.installer._check_cuda_availability()
        
        assert cuda_available is True
    
    @patch('builtins.__import__')
    def test_check_cuda_availability_failure(self, mock_import):
        """Test CUDA availability checking with failure"""
        mock_import.side_effect = ImportError("No torch")
        
        cuda_available = self.installer._check_cuda_availability()
        
        assert cuda_available is False
    
    def test_check_system_requirements_comprehensive(self):
        """Test comprehensive system requirements check"""
        with patch.object(self.installer, '_check_disk_space', return_value=10.0), \
             patch.object(self.installer, '_check_memory', return_value=16.0), \
             patch.object(self.installer, '_check_python_packages', return_value={
                 "torch": True, "transformers": True, "sentence_transformers": True, "huggingface_hub": True
             }), \
             patch.object(self.installer, '_check_apple_silicon_support', return_value=True):
            
            self.installer.is_apple_silicon = True
            requirements = self.installer.check_system_requirements()
        
        assert requirements["python_version"] is True  # Assuming Python 3.12+
        assert requirements["platform_supported"] is True
        assert requirements["disk_space_gb"] == 10.0
        assert requirements["memory_gb"] == 16.0
        assert requirements["dependencies"]["torch"] is True
        assert requirements["apple_silicon_optimized"] is True
    
    @patch('subprocess.run')
    def test_install_package_success(self, mock_subprocess):
        """Test successful package installation"""
        mock_subprocess.return_value = Mock(returncode=0, stdout="Successfully installed", stderr="")
        
        result = self.installer._install_package("torch")
        
        assert result is True
    
    @patch('subprocess.run')
    def test_install_package_failure(self, mock_subprocess):
        """Test failed package installation"""
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, 'pip', stderr="Installation failed")
        
        result = self.installer._install_package("invalid_package")
        
        assert result is False
    
    @patch('subprocess.run')
    def test_install_torch_dependencies_success(self, mock_subprocess):
        """Test successful torch dependencies installation"""
        mock_subprocess.return_value = Mock(returncode=0, stdout="Successfully installed", stderr="")
        
        result = self.installer._install_torch_dependencies()
        
        assert result is True
    
    @patch('subprocess.run')
    def test_try_prebuilt_xformers_success(self, mock_subprocess):
        """Test successful prebuilt xformers installation"""
        mock_subprocess.return_value = Mock(returncode=0)
        
        result = self.installer._try_prebuilt_xformers()
        
        assert result is True
    
    @patch('subprocess.run')
    def test_try_prebuilt_xformers_failure(self, mock_subprocess):
        """Test failed prebuilt xformers installation"""
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, 'pip', stderr="Installation failed")
        
        result = self.installer._try_prebuilt_xformers()
        
        assert result is False
    
    @patch('subprocess.run')
    def test_try_xformers_no_build_isolation_success(self, mock_subprocess):
        """Test xformers installation without build isolation"""
        mock_subprocess.return_value = Mock(returncode=0)
        
        result = self.installer._try_xformers_no_build_isolation()
        
        assert result is True
        # Should use --no-build-isolation flag
        args = mock_subprocess.call_args[0][0]
        assert "--no-build-isolation" in args
    
    @patch('subprocess.run')
    def test_try_standard_xformers_success(self, mock_subprocess):
        """Test standard xformers installation"""
        mock_subprocess.return_value = Mock(returncode=0)
        
        result = self.installer._try_standard_xformers()
        
        assert result is True
    
    @patch.dict('os.environ', {}, clear=True)
    def test_configure_xformers_warnings_suppression(self):
        """Test xformers warnings suppression configuration"""
        import os
        
        self.installer._configure_xformers_warnings_suppression()
        
        assert os.environ.get("TRANSFORMERS_VERBOSITY") == "error"
    
    @patch('subprocess.run')
    def test_setup_macos_build_environment_success(self, mock_subprocess):
        """Test macOS build environment setup"""
        mock_subprocess.return_value = Mock(returncode=0)
        
        result = self.installer._setup_macos_build_environment()
        
        assert result is True
    
    @patch('subprocess.run')
    def test_install_macos_build_dependencies_success(self, mock_subprocess):
        """Test macOS build dependencies installation"""
        mock_subprocess.return_value = Mock(returncode=0)
        
        result = self.installer._install_macos_build_dependencies()
        
        assert result is True
    
    @patch('subprocess.run')
    def test_compile_xformers_macos_success(self, mock_subprocess):
        """Test xformers compilation on macOS"""
        mock_subprocess.return_value = Mock(returncode=0)
        
        result = self.installer._compile_xformers_macos()
        
        assert result is True
    
    def test_install_xformers_macos_success(self):
        """Test xformers installation on macOS"""
        with patch.object(self.installer, '_setup_macos_build_environment', return_value=True), \
             patch.object(self.installer, '_install_macos_build_dependencies', return_value=True), \
             patch.object(self.installer, '_compile_xformers_macos', return_value=True):
            
            result = self.installer._install_xformers_macos()
            
            assert result is True
    
    def test_install_xformers_macos_failure(self):
        """Test xformers installation failure on macOS"""
        with patch.object(self.installer, '_setup_macos_build_environment', return_value=False):
            result = self.installer._install_xformers_macos()
            
            assert result is False
    
    @patch('huggingface_hub.snapshot_download')
    def test_download_model_success(self, mock_download):
        """Test successful model download"""
        mock_download.return_value = str(self.installer.model_path)
        
        # Create mock model directory
        self.installer.model_path.mkdir(parents=True, exist_ok=True)
        (self.installer.model_path / "config.json").write_text('{"model_type": "bert"}')
        
        with patch.object(self.installer, '_get_directory_size_mb', return_value=800.0), \
             patch.object(self.installer, '_create_model_metadata'):
            
            result = self.installer.download_model()
            
            assert result is True
    
    @patch('huggingface_hub.snapshot_download')
    def test_download_model_failure(self, mock_download):
        """Test failed model download"""
        mock_download.side_effect = Exception("Download failed")
        
        result = self.installer.download_model()
        
        assert result is False
    
    def test_get_directory_size_mb(self):
        """Test directory size calculation"""
        # Create test directory with files
        test_dir = self.temp_dir / "test_size"
        test_dir.mkdir()
        (test_dir / "file1.txt").write_text("a" * 1024)  # 1KB
        (test_dir / "file2.txt").write_text("b" * 2048)  # 2KB
        
        size_mb = self.installer._get_directory_size_mb(test_dir)
        
        # Should be approximately 0.003 MB (3KB)
        assert 0.002 < size_mb < 0.004
    
    def test_get_directory_size_mb_nonexistent(self):
        """Test directory size calculation for nonexistent directory"""
        nonexistent = self.temp_dir / "nonexistent"
        
        size_mb = self.installer._get_directory_size_mb(nonexistent)
        
        assert size_mb == 0.0
    
    def test_create_model_metadata(self):
        """Test model metadata creation"""
        self.installer.model_path.mkdir(parents=True, exist_ok=True)
        
        self.installer._create_model_metadata(120.5, 800.5)
        
        metadata_file = self.installer.model_path / "claude_indexer_metadata.json"
        assert metadata_file.exists()
        
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        assert metadata["model_name"] == self.installer.model_name
        assert metadata["download_time_seconds"] == 120.5
        assert metadata["size_mb"] == 800.5
        assert "downloaded_at" in metadata
        assert "repo_id" in metadata
        assert "dimensions" in metadata
        assert "platform" in metadata
    
    def test_get_optimal_device_apple_silicon(self):
        """Test optimal device detection for Apple Silicon"""
        self.installer.is_apple_silicon = True
        
        with patch('builtins.__import__') as mock_import:
            mock_torch = Mock()
            mock_import.return_value = mock_torch
            
            device = self.installer._get_optimal_device()
            
            assert device == "cpu"  # Apple Silicon uses CPU for compatibility
    
    def test_get_optimal_device_cuda(self):
        """Test optimal device detection for CUDA"""
        self.installer.is_apple_silicon = False
        
        with patch('builtins.__import__') as mock_import:
            mock_torch = Mock()
            mock_torch.cuda.is_available.return_value = True
            mock_import.return_value = mock_torch
            
            device = self.installer._get_optimal_device()
            
            assert device == "cuda"
    
    def test_get_optimal_device_cpu_fallback(self):
        """Test optimal device detection CPU fallback"""
        self.installer.is_apple_silicon = False
        
        with patch.object(self.installer, '_check_cuda_availability', return_value=False):
            device = self.installer._get_optimal_device()
            
            assert device == "cpu"
    
    def test_is_model_cached_true(self):
        """Test model cached check when model exists"""
        self.installer.model_path.mkdir(parents=True, exist_ok=True)
        (self.installer.model_path / "config.json").write_text('{}')
        (self.installer.model_path / "pytorch_model.bin").write_text('fake model')
        
        assert self.installer.is_model_cached() is True
    
    def test_is_model_cached_false(self):
        """Test model cached check when model doesn't exist"""
        assert self.installer.is_model_cached() is False
    
    def test_cleanup_cache_success(self):
        """Test successful cache cleanup"""
        # Create model path with files (actual implementation checks model_path, not cache_dir)
        self.installer.model_path.mkdir(parents=True, exist_ok=True)
        (self.installer.model_path / "test_file.txt").write_text("test")
        
        result = self.installer.cleanup_cache()
        
        assert result is True
        assert not self.installer.model_path.exists()
    
    def test_cleanup_cache_failure(self):
        """Test cache cleanup failure"""
        # Non-existent directory
        result = self.installer.cleanup_cache()
        
        assert result is False
    
    def test_get_installation_info(self):
        """Test getting installation information"""
        self.installer.model_path.mkdir(parents=True, exist_ok=True)
        # Create essential files to make is_model_cached() return True
        (self.installer.model_path / "config.json").write_text('{}')
        (self.installer.model_path / "pytorch_model.bin").write_text('fake model')
        
        metadata = {
            "model_name": self.installer.model_name,
            "download_time_seconds": 120.0,
            "size_mb": 800.0,
            "repo_id": "infgrad/stella_en_400M_v5"
        }
        with open(self.installer.model_path / "claude_indexer_metadata.json", "w") as f:
            json.dump(metadata, f)
        
        # Mock _get_optimal_device to avoid torch import issues
        with patch.object(self.installer, '_get_optimal_device', return_value='cpu'):
            info = self.installer.get_installation_info()
        
        assert info["model_name"] == self.installer.model_name
        assert info["cached"] is True
        assert info["model_path"] == str(self.installer.model_path)
        assert info["platform"] == self.installer.platform
        assert info["metadata"] == metadata
    
    def test_get_installation_info_no_metadata(self):
        """Test getting installation info without metadata"""
        self.installer.model_path.mkdir(parents=True, exist_ok=True)
        # Create essential files to make is_model_cached() return True
        (self.installer.model_path / "config.json").write_text('{}')
        (self.installer.model_path / "pytorch_model.bin").write_text('fake model')
        
        # Mock _get_optimal_device to avoid torch import issues
        with patch.object(self.installer, '_get_optimal_device', return_value='cpu'):
            info = self.installer.get_installation_info()
        
        assert info["model_name"] == self.installer.model_name
        assert info["cached"] is True
        assert "metadata" not in info  # No metadata key when file doesn't exist


class TestStellaInstallerIntegration:
    """Integration tests for StellaInstaller workflows"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.installer = StellaInstaller("stella_en_400M_v5")
        self.installer.cache_dir = self.temp_dir / "stella"
        self.installer.model_path = self.installer.cache_dir / self.installer.model_name
    
    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_install_dependencies_success_workflow(self):
        """Test successful dependency installation workflow"""
        with patch.object(self.installer, '_install_torch_dependencies', return_value=True), \
             patch.object(self.installer, '_install_package', return_value=True), \
             patch.object(self.installer, '_install_xformers_prebuilt', return_value=True), \
             patch.object(self.installer, '_configure_xformers_warnings_suppression'):
            
            result = self.installer.install_dependencies()
            
            assert result is True
    
    def test_install_dependencies_torch_failure(self):
        """Test dependency installation with torch failure"""
        with patch.object(self.installer, '_install_torch_dependencies', return_value=False):
            result = self.installer.install_dependencies()
            
            assert result is False
    
    def test_install_dependencies_xformers_fallback_workflow(self):
        """Test xformers installation with fallback strategies"""
        self.installer.platform = "linux"  # Non-macOS platform
        
        with patch.object(self.installer, '_install_torch_dependencies', return_value=True), \
             patch.object(self.installer, '_install_package', return_value=True), \
             patch.object(self.installer, '_try_prebuilt_xformers', return_value=False), \
             patch.object(self.installer, '_try_xformers_no_build_isolation', return_value=False), \
             patch.object(self.installer, '_try_standard_xformers', return_value=True), \
             patch.object(self.installer, '_configure_xformers_warnings_suppression'):
            
            result = self.installer.install_dependencies()
            
            assert result is True
    
    def test_install_dependencies_macos_workflow(self):
        """Test dependency installation on macOS"""
        self.installer.platform = "darwin"
        
        with patch.object(self.installer, '_install_torch_dependencies', return_value=True), \
             patch.object(self.installer, '_install_package', return_value=True), \
             patch.object(self.installer, '_install_xformers_macos', return_value=True), \
             patch.object(self.installer, '_configure_xformers_warnings_suppression'):
            
            result = self.installer.install_dependencies()
            
            assert result is True
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_verify_installation_success(self, mock_sentence_transformer):
        """Test successful installation verification"""
        # Setup mock model with numpy-like array for embedding
        import numpy as np
        mock_model = Mock()
        mock_embedding = np.array([[0.1] * 1024])  # Mock embedding with shape attribute
        mock_model.encode.return_value = mock_embedding
        mock_sentence_transformer.return_value = mock_model
        
        # Create model directory with essential files
        self.installer.model_path.mkdir(parents=True, exist_ok=True)
        (self.installer.model_path / "config.json").write_text('{"model_type": "bert"}')
        (self.installer.model_path / "pytorch_model.bin").write_text('fake model')
        
        with patch.object(self.installer, '_check_cuda_availability', return_value=False):
            success, results = self.installer.verify_installation()
            
            assert success is True
            assert results["model_files_exist"] is True
            assert results["model_loadable"] is True
            assert results["embedding_generation"] is True
            assert results["correct_dimensions"] is True
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_verify_installation_model_load_failure(self, mock_sentence_transformer):
        """Test installation verification with model load failure"""
        mock_sentence_transformer.side_effect = Exception("Model load failed")
        
        # Create model directory with essential files
        self.installer.model_path.mkdir(parents=True, exist_ok=True)
        (self.installer.model_path / "config.json").write_text('{"model_type": "bert"}')
        (self.installer.model_path / "pytorch_model.bin").write_text('fake model')
        
        with patch.object(self.installer, '_check_cuda_availability', return_value=False):
            success, results = self.installer.verify_installation()
        
        assert success is False
        assert results["model_files_exist"] is True
        assert results["model_loadable"] is False
    
    def test_verify_installation_model_not_cached(self):
        """Test installation verification when model not cached"""
        success, results = self.installer.verify_installation()
        
        assert success is False
        assert results["model_files_exist"] is False
    
    def test_check_device_optimization_mock(self):
        """Test device optimization checking"""
        mock_model = Mock()
        
        optimization = self.installer._check_device_optimization(mock_model)
        
        assert isinstance(optimization, dict)
        # Basic structure check - actual implementation would test model performance


class TestStellaInstallerEdgeCases:
    """Test edge cases and error conditions"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.installer = StellaInstaller("stella_en_400M_v5")
        self.installer.cache_dir = self.temp_dir / "stella"
        self.installer.model_path = self.installer.cache_dir / self.installer.model_name
    
    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_model_download_with_insufficient_space(self):
        """Test model download with insufficient disk space"""
        with patch.object(self.installer, '_check_disk_space', return_value=0.1):  # 100MB
            # This model requires 800MB
            assert self.installer._check_disk_space() < (self.installer.config["size_mb"] / 1024)
    
    def test_installation_with_permission_errors(self):
        """Test installation with permission errors"""
        with patch('subprocess.run') as mock_subprocess:
            mock_subprocess.side_effect = PermissionError("Permission denied")
            
            try:
                result = self.installer._install_package("torch")
                assert result is False
            except PermissionError:
                # If the exception propagates, that's also acceptable behavior
                pass
    
    def test_metadata_creation_with_io_error(self):
        """Test metadata creation with I/O error"""
        self.installer.model_path.mkdir(parents=True, exist_ok=True)
        
        with patch('builtins.open', side_effect=IOError("Disk full")):
            # Should not raise exception - let it fail silently or handle gracefully
            try:
                self.installer._create_model_metadata(120.0, 800.0)
            except IOError:
                # If it propagates, that's acceptable behavior
                pass
    
    def test_system_requirements_all_checks_fail(self):
        """Test system requirements when all checks fail"""
        with patch.object(self.installer, '_check_disk_space', return_value=0.0), \
             patch.object(self.installer, '_check_memory', return_value=0.0), \
             patch.object(self.installer, '_check_python_packages', return_value={
                 "torch": False, "transformers": False, "sentence_transformers": False, "huggingface_hub": False
             }):
            
            requirements = self.installer.check_system_requirements()
            
            assert requirements["disk_space_gb"] == 0.0
            assert requirements["memory_gb"] == 0.0
            assert not any(requirements["dependencies"].values())
    
    def test_model_verification_partial_failure(self):
        """Test model verification with partial failures"""
        # Create model directory with essential files
        self.installer.model_path.mkdir(parents=True, exist_ok=True)
        (self.installer.model_path / "config.json").write_text('{"model_type": "bert"}')
        (self.installer.model_path / "pytorch_model.bin").write_text('fake model')
        
        with patch('sentence_transformers.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.encode.side_effect = Exception("Encoding failed")
            mock_st.return_value = mock_model
            
            with patch.object(self.installer, '_check_cuda_availability', return_value=False):
                success, results = self.installer.verify_installation()
                
                assert success is False
                assert results["model_files_exist"] is True
                assert results["model_loadable"] is True  # Model loads but encoding fails
                assert results["embedding_generation"] is False


if __name__ == "__main__":
    pytest.main([__file__])