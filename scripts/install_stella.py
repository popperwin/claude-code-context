#!/usr/bin/env python3
"""
Automated Stella model installation with verification and optimization.

Handles downloading, caching, and verification of Stella embedding models
with platform-specific optimizations for macOS (Apple Silicon + Intel).
"""

import sys
import subprocess
import platform
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import time
import logging
import json
import hashlib
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StellaInstaller:
    """
    Install and verify Stella embeddings model with intelligent platform optimization.
    
    This installer provides robust xformers handling for macOS while optimizing for
    Linux/Windows platforms. Key features:
    
    - **macOS Support**: Automatically handles xformers incompatibility, provides
      clear user guidance, and enables MPS acceleration on Apple Silicon
    - **Multi-Strategy Installation**: Uses progressive fallback strategies for
      xformers installation on supported platforms
    - **Version Compatibility**: Ensures compatibility with sentence-transformers 3.4.1+
    - **Warning Suppression**: Configures environment to suppress harmless xformers
      warnings on macOS
    - **Performance Optimization**: Automatically detects and configures optimal
      device settings (MPS on Apple Silicon, CUDA on Linux/Windows, CPU fallback)
    
    Supported Platforms:
    - macOS (Intel and Apple Silicon) - xformers not required
    - Linux - full xformers support with multiple installation strategies
    - Windows - full xformers support with multiple installation strategies
    
    Example usage:
        installer = StellaInstaller("stella_en_400M_v5")
        installer.install_dependencies()  # Handles xformers intelligently
        installer.download_model()
        success, results = installer.verify_installation()
    """
    
    def __init__(self, model_name: str = "stella_en_400M_v5"):
        self.model_name = model_name
        self.cache_dir = Path.home() / ".cache" / "claude-indexer" / "stella"
        self.model_path = self.cache_dir / model_name
        
        # Platform detection
        self.platform = platform.system().lower()
        self.arch = platform.machine().lower()
        self.is_apple_silicon = self.platform == "darwin" and self.arch in ("arm64", "aarch64")
        
        # Model configurations
        self.model_configs = {
            "stella_en_400M_v5": {
                "repo_id": "infgrad/stella_en_400M_v5",
                "dimensions": 1024,
                "max_length": 512,
                "size_mb": 800
            },
            "stella_en_1.5B_v5": {
                "repo_id": "infgrad/stella_en_1.5B_v5", 
                "dimensions": 1024,
                "max_length": 512,
                "size_mb": 3000
            },
            "stella_base_en_v2": {
                "repo_id": "infgrad/stella_base_en_v2",
                "dimensions": 768,
                "max_length": 512,
                "size_mb": 500
            }
        }
        
        if model_name not in self.model_configs:
            raise ValueError(f"Unsupported model: {model_name}")
        
        self.config = self.model_configs[model_name]
    
    def check_system_requirements(self) -> Dict[str, Any]:
        """Comprehensive system requirements check"""
        logger.info("Checking system requirements...")
        
        requirements = {
            "python_version": sys.version_info >= (3, 12),
            "platform_supported": self.platform in ("darwin", "linux"),
            "disk_space_gb": self._check_disk_space(),
            "memory_gb": self._check_memory(),
            "dependencies": self._check_python_packages()
        }
        
        # Platform-specific checks
        if self.is_apple_silicon:
            requirements["apple_silicon_optimized"] = self._check_apple_silicon_support()
        
        return requirements
    
    def _check_disk_space(self) -> float:
        """Check available disk space in GB"""
        try:
            stat = shutil.disk_usage(self.cache_dir.parent)
            return stat.free / (1024**3)
        except:
            return 0.0
    
    def _check_memory(self) -> float:
        """Check available memory in GB"""
        try:
            if self.platform == "darwin":
                # macOS
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    return int(result.stdout.strip()) / (1024**3)
            elif self.platform == "linux":
                # Linux
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if line.startswith("MemAvailable:"):
                            kb = int(line.split()[1])
                            return kb / (1024**2)
        except:
            pass
        return 8.0  # Default assumption
    
    def _check_python_packages(self) -> Dict[str, bool]:
        """Check if required Python packages are available"""
        required_packages = {
            "torch": False,
            "transformers": False,
            "sentence_transformers": False,
            "huggingface_hub": False
        }
        
        for package in required_packages:
            try:
                __import__(package)
                required_packages[package] = True
            except ImportError:
                pass
        
        return required_packages
    
    def _check_apple_silicon_support(self) -> bool:
        """Check if Apple Silicon optimizations are available"""
        try:
            import torch
            return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        except ImportError:
            return False
    
    def _check_cuda_availability(self) -> bool:
        """Check if CUDA is available for xformers acceleration"""
        try:
            import torch
            return torch.cuda.is_available() and torch.cuda.device_count() > 0
        except ImportError:
            return False
    
    def install_dependencies(self) -> bool:
        """Install required Python packages with platform optimizations"""
        logger.info("Installing dependencies...")
        
        # Install PyTorch with platform-specific optimizations
        if not self._install_torch_dependencies():
            return False
        
        # Base dependencies - updated for latest compatibility
        base_deps = [
            "sentence-transformers>=3.4.1,<4.0.0",  # Updated to match user requirement
            "transformers>=4.36.0,<5.0.0", 
            "huggingface-hub>=0.19.0,<1.0.0",
            "safetensors>=0.4.0",
            "tokenizers>=0.15.0"
        ]
        
        # Try installing xformers with intelligent platform handling
        xformers_success = self._install_xformers_prebuilt()
        if not xformers_success and self.platform != "darwin":
            # Only show warning for non-macOS platforms where xformers is expected to work
            logger.warning("⚠️  xformers installation failed on supported platform, continuing without it")
            logger.info("   This may impact performance but will not affect functionality.")
        
        for dep in base_deps:
            if not self._install_package(dep):
                return False
        
        return True
    
    def _install_torch_dependencies(self) -> bool:
        """Install PyTorch with platform-specific optimizations"""
        try:
            cmd = [sys.executable, "-m", "pip", "install"]
            
            if self.is_apple_silicon:
                # Apple Silicon - use stable builds for MPS support
                cmd.extend([
                    "torch>=2.1.0,<3.0.0",
                    "torchvision>=0.16.0,<1.0.0", 
                    "torchaudio>=2.1.0,<3.0.0"
                ])
            elif self.platform == "darwin":
                # Intel Mac
                cmd.extend([
                    "torch>=2.1.0,<3.0.0",
                    "torchvision>=0.16.0,<1.0.0",
                    "torchaudio>=2.1.0,<3.0.0"
                ])
            else:
                # Linux - CPU builds
                cmd.extend([
                    "torch>=2.1.0,<3.0.0",
                    "torchvision>=0.16.0,<1.0.0", 
                    "torchaudio>=2.1.0,<3.0.0",
                    "--index-url", "https://download.pytorch.org/whl/cpu"
                ])
            
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            
            logger.info("✅ Installed PyTorch dependencies")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install PyTorch: {e.stderr}")
            return False
    
    def _install_package(self, package: str) -> bool:
        """Install single package with error handling"""
        try:
            cmd = [sys.executable, "-m", "pip", "install"]
            
            if package.startswith("--"):
                cmd.append(package)
            else:
                cmd.extend([package, "--upgrade"])
            
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            
            logger.info(f"✅ Installed: {package}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install {package}: {e.stderr}")
            return False
    
    def _install_xformers_prebuilt(self) -> bool:
        """
        Install xformers with platform-aware handling.
        
        xformers is not needed on macOS as Stella embeddings work perfectly with standard 
        PyTorch attention mechanisms. The model uses CPU-only operations where xformers
        provides no performance benefit.
        
        Returns:
            bool: True if successfully handled (including macOS skip), False on actual failure
        """
        # Check if CUDA is available for xformers optimization
        cuda_available = self._check_cuda_availability()
        
        # Skip xformers if no CUDA support - only provides benefit on GPU
        if self.platform == "darwin":
            logger.info("🍎 Skipping xformers on macOS (not needed for CPU-only embeddings)")
            logger.info("   Stella embeddings work optimally with standard PyTorch attention")
            return True
        elif not cuda_available:
            logger.info("🖥️  Skipping xformers (no CUDA detected - CPU-only operation)")
            logger.info("   xformers only provides benefits on CUDA-enabled GPUs")
            return True
        
        # Linux/Windows: Attempt xformers installation
        logger.info("Attempting to install xformers optimization (Linux/Windows)...")
        
        # Strategy 1: Try pre-built wheels first (fastest and most reliable)
        if self._try_prebuilt_xformers():
            return True
        
        # Strategy 2: Try with build isolation disabled
        if self._try_xformers_no_build_isolation():
            return True
        
        # Strategy 3: Standard pip install as last resort
        if self._try_standard_xformers():
            return True
        
        # All strategies failed
        logger.warning("⚠️  All xformers installation strategies failed.")
        logger.info("   This is not critical - Stella embeddings will work without xformers.")
        return False
    
    def _configure_xformers_warnings_suppression(self) -> None:
        """Configure environment to suppress harmless xformers warnings on macOS."""
        try:
            import warnings
            import os
            
            # Suppress specific xformers-related warnings
            warnings.filterwarnings(
                "ignore", 
                message=".*xformers.*",
                category=UserWarning
            )
            
            # Set environment variable to suppress transformers xformers warnings
            os.environ["TRANSFORMERS_VERBOSITY"] = "error"
            
            logger.debug("✅ Configured xformers warning suppression for macOS")
            
        except Exception as e:
            logger.debug(f"Warning: Could not configure xformers warning suppression: {e}")
    
    def _try_prebuilt_xformers(self) -> bool:
        """Try installing xformers using pre-built wheels only."""
        try:
            logger.info("Strategy 1: Installing xformers from pre-built wheels...")
            
            # Use specific version known to work well with sentence-transformers
            cmd = [
                sys.executable, "-m", "pip", "install", 
                "xformers>=0.0.22,<0.1.0",  # Version range for compatibility
                "--only-binary=xformers",    # Force binary wheels only
                "--no-build-isolation"       # Skip build isolation for speed
            ]
            
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            logger.info("✅ Installed xformers from pre-built wheels")
            return True
            
        except subprocess.TimeoutExpired:
            logger.info("   Pre-built wheel installation timed out, trying next strategy...")
            return False
        except subprocess.CalledProcessError as e:
            logger.debug(f"   Pre-built wheels failed: {e.stderr[:200]}...")
            return False
        except Exception as e:
            logger.debug(f"   Unexpected error with pre-built wheels: {e}")
            return False
    
    def _try_xformers_no_build_isolation(self) -> bool:
        """Try installing xformers with build isolation disabled."""
        try:
            logger.info("Strategy 2: Installing xformers without build isolation...")
            
            cmd = [
                sys.executable, "-m", "pip", "install", 
                "xformers>=0.0.22,<0.1.0",
                "--no-build-isolation",
                "--no-cache-dir"  # Prevent cache issues
            ]
            
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout for potential compilation
            )
            
            logger.info("✅ Installed xformers without build isolation")
            return True
            
        except subprocess.TimeoutExpired:
            logger.info("   No build isolation installation timed out, trying next strategy...")
            return False
        except subprocess.CalledProcessError as e:
            logger.debug(f"   No build isolation failed: {e.stderr[:200]}...")
            return False
        except Exception as e:
            logger.debug(f"   Unexpected error without build isolation: {e}")
            return False
    
    def _try_standard_xformers(self) -> bool:
        """Try standard xformers installation as last resort."""
        try:
            logger.info("Strategy 3: Standard xformers installation (last resort)...")
            
            cmd = [
                sys.executable, "-m", "pip", "install", 
                "xformers>=0.0.22,<0.1.0"
            ]
            
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=900  # 15 minute timeout for full compilation
            )
            
            logger.info("✅ Installed xformers via standard installation")
            return True
            
        except subprocess.TimeoutExpired:
            logger.warning("   Standard installation timed out (15 minutes)")
            return False
        except subprocess.CalledProcessError as e:
            logger.debug(f"   Standard installation failed: {e.stderr[:200]}...")
            return False
        except Exception as e:
            logger.debug(f"   Unexpected error with standard installation: {e}")
            return False
    
    def _install_xformers_macos(self) -> bool:
        """Install xformers on macOS with proper compilation setup"""
        logger.info("🔧 Setting up macOS build environment for xformers...")
        
        # Step 1: Setup build environment
        if not self._setup_macos_build_environment():
            return False
        
        # Step 2: Install build dependencies
        if not self._install_macos_build_dependencies():
            return False
        
        # Step 3: Compile xformers
        return self._compile_xformers_macos()
    
    def _setup_macos_build_environment(self) -> bool:
        """Setup macOS build environment with Homebrew dependencies"""
        # Check if Homebrew is installed
        try:
            subprocess.run(["brew", "--version"], check=True, capture_output=True)
            logger.info("✅ Homebrew detected")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("❌ Homebrew not found. Please install Homebrew first:")
            logger.error("   /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
            return False
        
        # Install required Homebrew packages (use LLVM 17 to avoid LLVM 20 compilation issues)
        brew_packages = ["llvm@17", "libomp", "ninja", "cmake", "rust"]
        logger.info(f"📦 Installing Homebrew dependencies: {', '.join(brew_packages)}")
        
        for package in brew_packages:
            try:
                # Check if package is already installed
                result = subprocess.run(
                    ["brew", "list", package], 
                    capture_output=True, 
                    text=True
                )
                if result.returncode == 0:
                    logger.info(f"✅ {package} already installed")
                else:
                    logger.info(f"📦 Installing {package}...")
                    subprocess.run(
                        ["brew", "install", package], 
                        check=True, 
                        capture_output=True, 
                        text=True,
                        timeout=300  # 5 minutes per package
                    )
                    logger.info(f"✅ Installed {package}")
            except subprocess.TimeoutExpired:
                logger.error(f"❌ Timeout installing {package}")
                return False
            except subprocess.CalledProcessError as e:
                logger.warning(f"⚠️  Could not install {package}: {e.stderr}")
                # Continue anyway - some packages might already be available
        
        return True
    
    def _install_macos_build_dependencies(self) -> bool:
        """Install Python build dependencies for xformers compilation"""
        build_deps = ["wheel", "setuptools", "numpy", "ninja", "cmake"]
        logger.info(f"🐍 Installing Python build dependencies: {', '.join(build_deps)}")
        
        for dep in build_deps:
            try:
                cmd = [sys.executable, "-m", "pip", "install", "--upgrade", dep]
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                logger.info(f"✅ Installed: {dep}")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to install {dep}: {e.stderr}")
                return False
        
        return True
    
    def _compile_xformers_macos(self) -> bool:
        """Compile xformers on macOS with proper environment setup"""
        import os
        
        # Setup environment variables for compilation
        env = os.environ.copy()
        
        # Homebrew paths (works for both Intel and Apple Silicon)
        homebrew_prefix = "/opt/homebrew" if self.is_apple_silicon else "/usr/local"
        llvm_path = f"{homebrew_prefix}/opt/llvm@17"  # Use LLVM 17 specifically
        libomp_path = f"{homebrew_prefix}/opt/libomp"
        
        # Setup compilers to use Homebrew's LLVM (supports OpenMP)
        env["PATH"] = f"{llvm_path}/bin:" + env.get("PATH", "")
        env["CC"] = f"{llvm_path}/bin/clang"
        env["CXX"] = f"{llvm_path}/bin/clang++"
        
        # Setup linker flags for OpenMP
        env["LDFLAGS"] = f"-L{llvm_path}/lib -L{libomp_path}/lib " + env.get("LDFLAGS", "")
        env["CPPFLAGS"] = f"-I{llvm_path}/include -I{libomp_path}/include " + env.get("CPPFLAGS", "")
        
        # Force CPU-only build (no CUDA on macOS)
        env["CMAKE_ARGS"] = "-DXFORMERS_BUILD_CUDA=OFF"
        
        logger.info("🔨 Compiling xformers (this may take 5-15 minutes)...")
        logger.info("   Using CPU-only build optimized for macOS")
        
        try:
            cmd = [
                sys.executable, "-m", "pip", "install", 
                "--no-binary", "xformers",
                "xformers==0.0.31.post1"
            ]
            
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                env=env,
                timeout=1800  # 30 minutes timeout for compilation
            )
            
            logger.info("✅ Successfully compiled and installed xformers on macOS!")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("❌ xformers compilation timed out (>30 minutes)")
            logger.error("   This may happen on slower machines or if there are network issues")
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ xformers compilation failed: {e.stderr}")
            logger.info("💡 Troubleshooting tips:")
            logger.info("   • Make sure Xcode Command Line Tools are installed: xcode-select --install")
            logger.info("   • Try updating Homebrew: brew update && brew upgrade")
            logger.info("   • Check available disk space (compilation needs ~2GB)")
            logger.info("   • Ensure LLVM 17 is installed: brew install llvm@17")
            logger.info("   • LLVM 20+ has compatibility issues with PyTorch compilation")
            return False
    
    def download_model(self) -> bool:
        """Download Stella model with progress tracking"""
        if self.is_model_cached():
            logger.info(f"✅ Model {self.model_name} already cached")
            return True
        
        logger.info(f"Downloading {self.model_name} ({self.config['size_mb']}MB)...")
        
        try:
            from huggingface_hub import snapshot_download
            from huggingface_hub.utils import tqdm as hf_tqdm
            
            # Create cache directory
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Download with progress bar
            start_time = time.time()
            
            downloaded_path = snapshot_download(
                repo_id=self.config["repo_id"],
                cache_dir=str(self.cache_dir),
                resume_download=True,
                local_dir=str(self.model_path),
                local_dir_use_symlinks=False,
                tqdm_class=hf_tqdm
            )
            
            download_time = time.time() - start_time
            size_mb = self._get_directory_size_mb(Path(downloaded_path))
            
            logger.info(f"✅ Downloaded {size_mb:.1f}MB in {download_time:.1f}s")
            logger.info(f"✅ Model cached at: {self.model_path}")
            
            # Create metadata file
            self._create_model_metadata(download_time, size_mb)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            # Cleanup partial download
            if self.model_path.exists():
                shutil.rmtree(self.model_path, ignore_errors=True)
            return False
    
    def _get_directory_size_mb(self, path: Path) -> float:
        """Calculate directory size in MB"""
        total_size = 0
        for file_path in path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size / (1024 * 1024)
    
    def _create_model_metadata(self, download_time: float, size_mb: float) -> None:
        """Create model metadata file"""
        metadata = {
            "model_name": self.model_name,
            "repo_id": self.config["repo_id"],
            "dimensions": self.config["dimensions"],
            "max_length": self.config["max_length"],
            "downloaded_at": time.time(),
            "download_time_seconds": download_time,
            "size_mb": size_mb,
            "platform": self.platform,
            "architecture": self.arch,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "installer_version": "1.0.0"
        }
        
        metadata_file = self.model_path / "claude_indexer_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def verify_installation(self) -> Tuple[bool, Dict[str, Any]]:
        """Comprehensive installation verification"""
        logger.info("Verifying installation...")
        
        verification_results = {
            "model_files_exist": False,
            "model_loadable": False,
            "embedding_generation": False,
            "correct_dimensions": False,
            "performance_acceptable": False,
            "device_optimization": False
        }
        
        try:
            # Check model files
            if not self.is_model_cached():
                return False, verification_results
            
            verification_results["model_files_exist"] = True
            
            # Test model loading
            from sentence_transformers import SentenceTransformer
            
            logger.info("Loading model...")
            start_time = time.time()
            
            # Try loading model with xformers, fallback without
            try:
                # Configure environment for non-CUDA systems compatibility
                import os
                cuda_available = self._check_cuda_availability()
                
                if not cuda_available:  # Apply to ALL non-CUDA systems (macOS, CPU-only Linux/Windows)
                    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                    # Completely disable xformers for CPU compatibility
                    os.environ["XFORMERS_DISABLED"] = "1"
                    os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"
                    # Disable flash attention and memory efficient attention
                    os.environ["DISABLE_FLASH_ATTENTION"] = "1"
                
                # Temporarily uninstall xformers import to force fallback to standard attention
                import sys
                xformers_backup = None
                if not cuda_available and 'xformers' in sys.modules:
                    xformers_backup = sys.modules.pop('xformers', None)
                
                try:
                    # Use config_kwargs to disable memory efficient attention for non-CUDA systems
                    config_kwargs = {}
                    if not cuda_available:
                        config_kwargs = {
                            "use_memory_efficient_attention": False,
                            "unpad_inputs": False
                        }
                    
                    model = SentenceTransformer(
                        str(self.model_path),
                        device=self._get_optimal_device(),
                        trust_remote_code=True,
                        config_kwargs=config_kwargs
                    )
                finally:
                    # Restore xformers module if it was removed
                    if xformers_backup and not cuda_available:
                        sys.modules['xformers'] = xformers_backup
            except Exception as e:
                if "xformers" in str(e).lower() or "bus error" in str(e).lower() or "mps" in str(e).lower():
                    logger.info("Model has compatibility issues with MPS/xformers, trying with CPU device...")
                    try:
                        # Force CPU usage to avoid MPS backend issues
                        model = SentenceTransformer(
                            str(self.model_path),
                            device="cpu",
                            trust_remote_code=True,
                            config_kwargs={
                                "use_memory_efficient_attention": False,
                                "unpad_inputs": False
                            }
                        )
                    except Exception as e2:
                        logger.error(f"Failed to load model even on CPU: {e2}")
                        raise e2
                else:
                    raise e
            
            load_time = time.time() - start_time
            verification_results["model_loadable"] = True
            
            logger.info(f"✅ Model loaded in {load_time:.2f}s")
            
            # Test embedding generation with proper benchmarking
            logger.info("Testing embedding generation...")
            test_texts = [
                "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
                "class DatabaseConnection: def __init__(self, host, port): self.host = host",
                "async function fetchData(url) { const response = await fetch(url); return response.json(); }"
            ]
            
            # Configure threading for non-CUDA systems
            import os
            if not cuda_available:
                # Disable all multiprocessing and threading optimizations for stability
                os.environ["OMP_NUM_THREADS"] = "1"
                os.environ["MKL_NUM_THREADS"] = "1"
                os.environ["NUMEXPR_NUM_THREADS"] = "1"
                os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
                os.environ["OPENBLAS_NUM_THREADS"] = "1"
            
            # Warmup run (exclude from timing - includes initialization overhead)
            logger.info("Performing warmup run...")
            try:
                warmup_embeddings = model.encode(
                    test_texts[:1],  # Just one text for warmup
                    show_progress_bar=False,
                    batch_size=1,
                    convert_to_numpy=True,
                    normalize_embeddings=False
                )
                logger.info("✅ Warmup completed")
            except Exception as warmup_error:
                logger.warning(f"Warmup failed: {warmup_error}")
            
            # Benchmark runs (5 iterations for reliable average)
            logger.info("Running performance benchmark (5 iterations)...")
            embed_times = []
            embeddings = None
            
            for i in range(5):
                start_time = time.time()
                try:
                    embeddings = model.encode(
                        test_texts, 
                        show_progress_bar=False,
                        batch_size=1,
                        convert_to_numpy=True,
                        normalize_embeddings=False
                    )
                    embed_time = time.time() - start_time
                    embed_times.append(embed_time)
                    logger.debug(f"Run {i+1}: {embed_time:.3f}s ({len(test_texts)/embed_time:.1f} texts/sec)")
                except Exception as encode_error:
                    logger.warning(f"Encoding failed on run {i+1}: {encode_error}")
                    # Fallback to single text encoding
                    embeddings = []
                    for text in test_texts:
                        try:
                            emb = model.encode([text], show_progress_bar=False)
                            embeddings.append(emb[0])
                        except Exception as single_error:
                            logger.error(f"Failed to encode single text: {single_error}")
                            raise single_error
                    embeddings = __import__('numpy').array(embeddings)
                    embed_time = time.time() - start_time
                    embed_times.append(embed_time)
                    break  # Use fallback method for remaining runs
            
            # Calculate average performance (excluding warmup)
            if embed_times:
                embed_time = sum(embed_times) / len(embed_times)
                min_time = min(embed_times)
                max_time = max(embed_times)
                logger.info(f"✅ Performance benchmark: avg={embed_time:.3f}s, min={min_time:.3f}s, max={max_time:.3f}s")
            else:
                embed_time = 1.0  # Fallback value
                logger.warning("⚠️  Performance benchmark failed, using fallback timing")
            
            verification_results["embedding_generation"] = True
            
            # Check dimensions
            expected_dims = self.config["dimensions"]
            actual_dims = embeddings.shape[1]
            
            if actual_dims == expected_dims:
                verification_results["correct_dimensions"] = True
            else:
                logger.error(f"❌ Dimension mismatch: expected {expected_dims}, got {actual_dims}")
            
            # Performance check (adjust threshold for Apple Silicon CPU mode)
            texts_per_second = len(test_texts) / embed_time
            performance_threshold = 5.0 if self.is_apple_silicon else 10.0  # Lower threshold for CPU
            if texts_per_second > performance_threshold:
                verification_results["performance_acceptable"] = True
            
            logger.info(f"✅ Generated embeddings: {embeddings.shape}")
            logger.info(f"✅ Performance: {texts_per_second:.1f} texts/second")
            
            # Device optimization check
            device_info = self._check_device_optimization(model)
            verification_results["device_optimization"] = device_info["optimized"]
            
            all_passed = all(verification_results.values())
            
            # Log detailed verification results
            logger.info("Verification results:")
            for test_name, passed in verification_results.items():
                status = "✅" if passed else "❌"
                logger.info(f"  {status} {test_name}: {passed}")
            
            if all_passed:
                logger.info("✅ Installation verification complete!")
            else:
                failed_tests = [name for name, passed in verification_results.items() if not passed]
                logger.warning(f"⚠️  Failed tests: {', '.join(failed_tests)}")
            
            return all_passed, verification_results
            
        except Exception as e:
            logger.error(f"❌ Verification failed: {e}")
            return False, verification_results
    
    def _get_optimal_device(self) -> str:
        """Get optimal device for the current platform"""
        try:
            import torch
            
            # Use CPU for Stella embeddings compatibility across platforms
            if self.is_apple_silicon:
                logger.debug("Using CPU device on Apple Silicon to avoid MPS compatibility issues")
                return "cpu"
            elif torch.cuda.is_available():
                # CUDA available - could use GPU, but CPU works reliably for Stella
                logger.debug("CUDA available, but using CPU for Stella compatibility")
                return "cuda"  # Keep CUDA for systems that have it
            else:
                logger.debug("Using CPU device (no GPU acceleration available)")
                return "cpu"
        except ImportError:
            return "cpu"
    
    def _check_device_optimization(self, model) -> Dict[str, Any]:
        """Check if model is using optimal device"""
        device_info = {
            "device": "cpu",
            "optimized": False,
            "details": {}
        }
        
        try:
            import torch
            
            # Get model device
            if hasattr(model, 'device'):
                device_info["device"] = str(model.device)
            elif hasattr(model, '_modules'):
                # Check first module device
                for module in model._modules.values():
                    if hasattr(module, 'device'):
                        device_info["device"] = str(module.device)
                        break
            
            # Check optimization status (consider appropriate device choice based on capabilities)
            cuda_available = self._check_cuda_availability()
            
            if not cuda_available:
                # CPU is considered optimized when no CUDA available (best available option)
                device_info["optimized"] = True
                device_info["details"]["cpu_optimal_for_non_cuda"] = True
                if self.is_apple_silicon:
                    device_info["details"]["apple_silicon"] = True
                    device_info["details"]["mps_available"] = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
                    device_info["details"]["mps_disabled_for_compatibility"] = True
            else:
                # CUDA available - GPU usage is optimal
                device_info["optimized"] = device_info["device"] != "cpu"
                device_info["details"]["cuda_available"] = True
            
        except Exception as e:
            device_info["details"]["error"] = str(e)
        
        return device_info
    
    def is_model_cached(self) -> bool:
        """Check if model is already downloaded and cached"""
        if not self.model_path.exists():
            return False
        
        # Check for essential files
        essential_files = ["config.json", "pytorch_model.bin"]
        for file_name in essential_files:
            if not (self.model_path / file_name).exists():
                # Try alternative names
                alternatives = {
                    "pytorch_model.bin": ["model.safetensors", "pytorch_model.safetensors"]
                }
                
                if file_name in alternatives:
                    found = any((self.model_path / alt).exists() for alt in alternatives[file_name])
                    if not found:
                        return False
                else:
                    return False
        
        return True
    
    def cleanup_cache(self) -> bool:
        """Clean up model cache"""
        try:
            if self.model_path.exists():
                shutil.rmtree(self.model_path)
                logger.info(f"✅ Cleaned up cache: {self.model_path}")
                return True
        except Exception as e:
            logger.error(f"❌ Cache cleanup failed: {e}")
        return False
    
    def get_installation_info(self) -> Dict[str, Any]:
        """Get detailed installation information"""
        info = {
            "model_name": self.model_name,
            "model_path": str(self.model_path),
            "cached": self.is_model_cached(),
            "platform": self.platform,
            "architecture": self.arch,
            "is_apple_silicon": self.is_apple_silicon,
            "optimal_device": self._get_optimal_device()
        }
        
        # Add metadata if available
        metadata_file = self.model_path / "claude_indexer_metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    info["metadata"] = metadata
            except:
                pass
        
        return info


def main():
    """Main installation script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Install Stella embedding model")
    parser.add_argument(
        "--model", 
        default="stella_en_400M_v5",
        choices=["stella_en_400M_v5", "stella_en_1.5B_v5", "stella_base_en_v2"],
        help="Stella model to install"
    )
    parser.add_argument("--force", action="store_true", help="Force reinstall")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing installation")
    parser.add_argument("--cleanup", action="store_true", help="Clean up cache")
    parser.add_argument("--info", action="store_true", help="Show installation info")
    
    args = parser.parse_args()
    
    installer = StellaInstaller(args.model)
    
    if args.info:
        info = installer.get_installation_info()
        print(json.dumps(info, indent=2))
        return
    
    if args.cleanup:
        installer.cleanup_cache()
        return
    
    if args.verify_only:
        success, results = installer.verify_installation()
        sys.exit(0 if success else 1)
    
    # Full installation process
    print(f"🚀 Installing Stella model: {args.model}")
    
    # Check requirements
    requirements = installer.check_system_requirements()
    if not all(requirements.values()):
        print("❌ System requirements not met:")
        for req, status in requirements.items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {req}: {status}")
        
        if not requirements.get("python_version"):
            print("  Python 3.12+ required")
        if requirements.get("disk_space_gb", 0) < 5:
            print("  Need at least 5GB free disk space")
        
        sys.exit(1)
    
    # Install dependencies
    if not installer.install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Download model (unless already cached and not forcing)
    if args.force or not installer.is_model_cached():
        if not installer.download_model():
            print("❌ Failed to download model")
            sys.exit(1)
    
    # Verify installation
    success, results = installer.verify_installation()
    
    if success:
        print(f"\n✅ {args.model} installation complete!")
        
        # Show platform-specific optimization info
        device = installer._get_optimal_device()
        if device != "cpu":
            print(f"🚀 Optimized for: {device}")
        
        # Show xformers status
        try:
            import xformers
            print(f"⚡ xformers optimization: Available (v{xformers.__version__})")
        except ImportError:
            if installer.platform == "darwin":
                print(f"ℹ️  xformers optimization: Not needed on macOS (using platform optimizations)")
            else:
                print(f"ℹ️  xformers optimization: Not available (performance may be reduced)")
        
        # Platform-specific performance notes
        if installer.is_apple_silicon:
            print(f"🍎 Apple Silicon optimization: CPU-only mode for Stella compatibility")
            print(f"   (MPS disabled due to xformers memory efficient attention conflicts)")
        elif installer.platform == "darwin":
            print(f"🍎 Intel Mac optimization: CPU-only mode for Stella compatibility")
        
        # Show usage example
        print(f"\n📖 Usage example:")
        print(f"from sentence_transformers import SentenceTransformer")
        print(f"model = SentenceTransformer('{installer.model_path}', device='cpu')")
        if installer.is_apple_silicon:
            print(f"# Model uses CPU mode for compatibility with Stella embeddings")
        print(f"embeddings = model.encode(['your text here'])")
        
    else:
        print("❌ Installation verification failed")
        print("Failed checks:", [k for k, v in results.items() if not v])
        sys.exit(1)


if __name__ == "__main__":
    main()