"""
Enhanced Stella embedder implementation with caching and performance optimizations.

Provides production-ready embedding generation with device auto-detection,
intelligent caching, and comprehensive error handling.
"""

import asyncio
import logging
import platform
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import threading

from .base import BaseEmbedder, EmbeddingResponse
from .cache import EmbeddingCache
from ..models.config import StellaConfig

logger = logging.getLogger(__name__)


class StellaEmbedder(BaseEmbedder):
    """
    Production-ready Stella embedder with caching and optimizations.
    
    Features:
    - Device auto-detection (CUDA/MPS/CPU) with fallback
    - Intelligent caching with LRU + TTL eviction
    - Batch processing with progress tracking
    - Performance monitoring and metrics
    - Thread-safe operations
    """
    
    def __init__(self, config: Optional[StellaConfig] = None):
        """
        Initialize Stella embedder.
        
        Args:
            config: Stella configuration, uses defaults if None
        """
        self.stella_config = config or StellaConfig()
        super().__init__(self.stella_config.model_dump())
        
        # Device and model state
        self._device: Optional[str] = None
        self._model_lock = threading.RLock()
        self._loading = False
        
        # Caching
        self._cache = EmbeddingCache(
            max_size=getattr(self.stella_config, 'cache_size', 10000),
            ttl_seconds=getattr(self.stella_config, 'cache_ttl_seconds', 3600)
        )
        
        # Performance tracking
        self._total_embeddings = 0
        self._total_processing_time = 0.0
        self._cache_hits = 0
        
        logger.info(f"Initialized StellaEmbedder: {self.stella_config.model_name}")
    
    @property
    def model_name(self) -> str:
        """Get the name of the embedding model"""
        return self.stella_config.model_name
    
    @property
    def dimensions(self) -> int:
        """Get the dimensionality of the embeddings"""
        return self.stella_config.dimensions
    
    @property
    def max_sequence_length(self) -> int:
        """Get the maximum sequence length supported"""
        return self.stella_config.max_length
    
    @property
    def device(self) -> Optional[str]:
        """Get current device"""
        return self._device
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._is_loaded
    
    @property
    def cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self._cache.get_stats()
    
    async def load_model(self) -> bool:
        """
        Load Stella model with platform optimizations.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        if self.is_loaded:
            return True
        
        # Prevent concurrent loading
        with self._model_lock:
            if self.is_loaded:
                return True
            
            if self._loading:
                # Wait for other thread to finish loading
                while self._loading and not self.is_loaded:
                    await asyncio.sleep(0.1)
                return self.is_loaded
            
            self._loading = True
        
        try:
            start_time = time.time()
            
            # Determine optimal device
            self._device = self._detect_optimal_device()
            logger.info(f"Loading Stella model on device: {self._device}")
            
            # Import here to avoid startup delays
            from sentence_transformers import SentenceTransformer
            import torch
            import os
            import sys
            
            # Check CUDA availability like install_stella.py
            cuda_available = torch.cuda.is_available() and torch.cuda.device_count() > 0
            
            # Configure environment for non-CUDA systems compatibility (following install_stella.py pattern)
            if not cuda_available:  # Apply to ALL non-CUDA systems (macOS, CPU-only Linux/Windows)
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                # Completely disable xformers for CPU compatibility
                os.environ["XFORMERS_DISABLED"] = "1"
                os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"
                # Disable flash attention and memory efficient attention
                os.environ["DISABLE_FLASH_ATTENTION"] = "1"
            
            # Temporarily uninstall xformers import to force fallback to standard attention
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
                
                # Load model with proper configuration
                model_path = self._get_model_path()
                self._model = SentenceTransformer(
                    model_path,
                    device=self._device,
                    trust_remote_code=True,  # Required for Stella models
                    config_kwargs=config_kwargs
                )
                
            except Exception as e:
                if "xformers" in str(e).lower() or "bus error" in str(e).lower() or "mps" in str(e).lower():
                    # Force CPU usage to avoid MPS backend issues (following install_stella.py pattern)
                    self._model = SentenceTransformer(
                        model_path,
                        device="cpu",
                        trust_remote_code=True,
                        config_kwargs={
                            "use_memory_efficient_attention": False,
                            "unpad_inputs": False
                        }
                    )
                    self._device = "cpu"  # Update device to reflect actual usage
                else:
                    raise e
            finally:
                # Restore xformers module if it was removed
                if xformers_backup and not cuda_available:
                    sys.modules['xformers'] = xformers_backup
            
            # Configure for inference
            self._model.eval()
            
            # Apply device-specific optimizations
            await self._apply_device_optimizations()
            
            # Mark as loaded
            self._is_loaded = True
            current_time = time.time()
            self._load_time = datetime.fromtimestamp(current_time)
            
            loading_time = current_time - start_time
            logger.info(
                f"Stella model loaded successfully in {loading_time:.2f}s on {self._device}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Stella model: {e}")
            self._model = None
            self._is_loaded = False
            return False
        
        finally:
            self._loading = False
    
    async def unload_model(self) -> None:
        """Unload Stella model to free memory"""
        with self._model_lock:
            if self._model is not None:
                try:
                    # Move model to CPU to free GPU memory
                    if hasattr(self._model, 'cpu'):
                        self._model.cpu()
                    
                    # Clear CUDA cache if available
                    await self._clear_gpu_cache()
                    
                    # Delete model
                    del self._model
                    
                    logger.info("Stella model unloaded")
                    
                except Exception as e:
                    logger.warning(f"Error during model unload: {e}")
                
                finally:
                    self._model = None
                    self._is_loaded = False
                    self._device = None
    
    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using Stella model with caching.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        if not texts:
            return []
        
        # Validate and preprocess texts
        valid_texts = self._validate_texts(texts)
        
        # Check cache for batch
        embeddings, cache_miss_indices = self._cache.get_batch(
            valid_texts, self.model_name
        )
        
        # Track cache hits
        cache_hits = len(valid_texts) - len(cache_miss_indices)
        self._cache_hits += cache_hits
        
        # Generate embeddings for cache misses
        if cache_miss_indices:
            miss_texts = [valid_texts[i] for i in cache_miss_indices]
            
            try:
                # Generate embeddings with Stella
                new_embeddings = await self._generate_stella_embeddings(miss_texts)
                
                # Store in cache
                self._cache.put_batch(miss_texts, new_embeddings, self.model_name)
                
                # Fill in the missing embeddings
                for i, miss_idx in enumerate(cache_miss_indices):
                    embeddings[miss_idx] = new_embeddings[i]
                
            except Exception as e:
                logger.error(f"Stella embedding generation failed: {e}")
                raise RuntimeError(f"Embedding generation failed: {e}") from e
        
        # Update statistics
        self._total_embeddings += len(valid_texts)
        
        # Filter out None values and return
        return [emb for emb in embeddings if emb is not None]
    
    async def _generate_stella_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings directly with Stella model"""
        if not texts:
            return []
        
        try:
            import torch
            
            # Configure device-specific settings
            encode_kwargs = {
                'batch_size': self.stella_config.batch_size,
                'show_progress_bar': False,
                'convert_to_numpy': True,
                'normalize_embeddings': self.stella_config.normalize_embeddings
            }
            
            # Add device-specific optimizations
            if self._device == 'cuda' and hasattr(self.stella_config, 'use_fp16'):
                encode_kwargs['precision'] = 'fp16' if self.stella_config.use_fp16 else 'fp32'
            
            # Generate embeddings
            embeddings = self._model.encode(texts, **encode_kwargs)
            
            # Convert to list of lists
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Stella model encoding failed: {e}")
            raise
    
    def _detect_optimal_device(self) -> str:
        """Detect optimal device for the current platform"""
        if hasattr(self.stella_config, 'device') and self.stella_config.device:
            return self.stella_config.device
        
        try:
            import torch
            
            # Check for CUDA
            if torch.cuda.is_available():
                device = "cuda"
                try:
                    logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
                except (RuntimeError, AssertionError):
                    # CUDA available but not properly initialized
                    logger.warning("CUDA available but not initialized, falling back to CPU")
                    device = "cpu"
                return device
            
            # Check for Apple Silicon MPS
            if (platform.system() == "Darwin" and 
                platform.machine() in ("arm64", "aarch64") and
                hasattr(torch.backends, 'mps') and 
                torch.backends.mps.is_available()):
                device = "mps"
                logger.info("Apple Silicon MPS available")
                return device
            
            # Fallback to CPU
            device = "cpu"
            logger.info("Using CPU device")
            return device
            
        except ImportError:
            logger.warning("PyTorch not available, using CPU")
            return "cpu"
    
    def _get_model_path(self) -> str:
        """Get model path (local cache or HuggingFace)"""
        # Check if model is cached locally
        model_path = self.stella_config.model_path
        if model_path.exists() and any(model_path.iterdir()):
            logger.info(f"Using cached model: {model_path}")
            return str(model_path)
        
        # Use HuggingFace model name
        logger.info(f"Loading model from HuggingFace: {self.stella_config.model_name}")
        return self.stella_config.model_name
    
    async def _apply_device_optimizations(self) -> None:
        """Apply device-specific optimizations"""
        try:
            import torch
            
            if self._device == "cuda":
                # CUDA optimizations
                if hasattr(self.stella_config, 'use_fp16') and self.stella_config.use_fp16:
                    if hasattr(self._model, 'half'):
                        self._model.half()
                        logger.debug("Applied FP16 optimization for CUDA")
                
                # Enable cuDNN optimizations
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
                
            elif self._device == "mps":
                # Apple Silicon optimizations
                if hasattr(self._model, 'to'):
                    self._model.to('mps')
                    logger.debug("Moved model to MPS device")
            
            elif self._device == "cpu":
                # CPU optimizations
                torch.set_num_threads(min(4, torch.get_num_threads()))
                logger.debug("Applied CPU threading optimization")
                
        except Exception as e:
            logger.warning(f"Failed to apply device optimizations: {e}")
    
    async def _clear_gpu_cache(self) -> None:
        """Clear GPU cache if available"""
        try:
            import torch
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("Cleared CUDA cache")
            
            # Clear MPS cache if available
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # MPS doesn't have explicit cache clearing in older versions
                pass
                
        except Exception as e:
            logger.debug(f"GPU cache clearing failed: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed Stella model information"""
        base_info = super().get_model_info()
        
        # Add Stella-specific information
        stella_info = {
            "device": self._device,
            "cache_dir": str(self.stella_config.cache_dir),
            "use_fp16": getattr(self.stella_config, 'use_fp16', False),
            "normalize_embeddings": self.stella_config.normalize_embeddings,
            "batch_size": self.stella_config.batch_size,
            "total_embeddings_generated": self._total_embeddings,
            "cache_hit_rate": self._cache_hits / max(1, self._total_embeddings),
            "average_processing_time_ms": (
                self._total_processing_time / max(1, self._total_embeddings) * 1000
            )
        }
        
        # Add cache statistics
        cache_stats = self._cache.get_stats()
        stella_info.update({f"cache_{k}": v for k, v in cache_stats.items()})
        
        return {**base_info, **stella_info}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            "total_embeddings": self._total_embeddings,
            "total_processing_time_s": self._total_processing_time,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": self._cache_hits / max(1, self._total_embeddings),
            "average_embedding_time_ms": (
                self._total_processing_time / max(1, self._total_embeddings) * 1000
            ),
            "cache_stats": self._cache.get_stats(),
            "device": self._device,
            "model_loaded": self.is_loaded
        }
    
    def clear_cache(self) -> None:
        """Clear embedding cache"""
        self._cache.clear()
        self._cache_hits = 0
        logger.info("Embedding cache cleared")
    
    def resize_cache(self, new_size: int) -> None:
        """Resize cache to new maximum size"""
        self._cache.resize(new_size)
        logger.info(f"Cache resized to {new_size}")
    
    async def warmup(self, sample_texts: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Warm up the model with sample texts.
        
        Args:
            sample_texts: Texts to use for warmup, uses defaults if None
            
        Returns:
            Warmup performance metrics
        """
        if not self.is_loaded:
            await self.load_model()
        
        if sample_texts is None:
            sample_texts = [
                "def example_function():",
                "class ExampleClass:",
                "import example_module",
                "# Example comment"
            ]
        
        logger.info(f"Warming up Stella embedder with {len(sample_texts)} samples")
        
        start_time = time.time()
        response = await self.embed_texts(sample_texts)
        warmup_time = time.time() - start_time
        
        return {
            "warmup_time_s": warmup_time,
            "sample_count": len(sample_texts),
            "embeddings_generated": response.embedding_count,
            "average_time_per_embedding_ms": response.average_embedding_time_ms,
            "device": self._device
        }