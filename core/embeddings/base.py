"""
Base embedding interface and protocols for claude-code-context.

Defines the abstract interface that all embedding providers must implement,
with concrete implementations for Stella and future embedding models.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Protocol, runtime_checkable
from pathlib import Path
import asyncio
from dataclasses import dataclass
from datetime import datetime

from core.models.config import StellaConfig
from core.models.storage import OperationResult


@dataclass
class EmbeddingRequest:
    """Request for generating embeddings"""
    texts: List[str]
    batch_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass  
class EmbeddingResponse:
    """Response containing generated embeddings"""
    embeddings: List[List[float]]
    processing_time_ms: float
    batch_id: Optional[str] = None
    model_info: Optional[Dict[str, Any]] = None
    
    @property
    def embedding_count(self) -> int:
        """Get number of embeddings generated"""
        return len(self.embeddings)
    
    @property
    def average_embedding_time_ms(self) -> float:
        """Get average time per embedding"""
        if self.embedding_count == 0:
            return 0.0
        return self.processing_time_ms / self.embedding_count


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Protocol defining the interface for embedding providers"""
    
    @property
    def model_name(self) -> str:
        """Get the name of the embedding model"""
        ...
    
    @property
    def dimensions(self) -> int:
        """Get the dimensionality of the embeddings"""
        ...
    
    @property
    def max_sequence_length(self) -> int:
        """Get the maximum sequence length supported"""
        ...
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready"""
        ...
    
    async def load_model(self) -> bool:
        """Load the embedding model"""
        ...
    
    async def unload_model(self) -> None:
        """Unload the embedding model to free memory"""
        ...
    
    async def embed_texts(self, texts: List[str]) -> EmbeddingResponse:
        """Generate embeddings for a list of texts"""
        ...
    
    async def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        ...
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        ...


class BaseEmbedder(ABC):
    """Abstract base class for embedding providers"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._model = None
        self._is_loaded = False
        self._load_time: Optional[datetime] = None
        
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the name of the embedding model"""
        pass
    
    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Get the dimensionality of the embeddings"""
        pass
    
    @property
    @abstractmethod
    def max_sequence_length(self) -> int:
        """Get the maximum sequence length supported"""
        pass
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready"""
        return self._is_loaded and self._model is not None
    
    @abstractmethod
    async def load_model(self) -> bool:
        """Load the embedding model"""
        pass
    
    @abstractmethod
    async def unload_model(self) -> None:
        """Unload the embedding model to free memory"""
        pass
    
    @abstractmethod
    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Internal method to generate embeddings"""
        pass
    
    async def embed_texts(self, texts: List[str]) -> EmbeddingResponse:
        """Generate embeddings for a list of texts with error handling"""
        if not self.is_loaded:
            await self.load_model()
        
        if not texts:
            return EmbeddingResponse(
                embeddings=[],
                processing_time_ms=0.0
            )
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            embeddings = await self._generate_embeddings(texts)
            processing_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return EmbeddingResponse(
                embeddings=embeddings,
                processing_time_ms=processing_time_ms,
                model_info=self.get_model_info()
            )
            
        except Exception as e:
            processing_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            raise RuntimeError(f"Embedding generation failed: {e}") from e
    
    async def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        if not text.strip():
            # Return zero vector for empty text
            return [0.0] * self.dimensions
        
        response = await self.embed_texts([text])
        if response.embeddings:
            return response.embeddings[0]
        else:
            raise RuntimeError("Failed to generate embedding for single text")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        return {
            "model_name": self.model_name,
            "dimensions": self.dimensions,
            "max_sequence_length": self.max_sequence_length,
            "is_loaded": self.is_loaded,
            "load_time": self._load_time.isoformat() if self._load_time else None,
            "config": self.config
        }
    
    def _validate_texts(self, texts: List[str]) -> List[str]:
        """Validate and preprocess input texts"""
        if not isinstance(texts, list):
            raise ValueError("texts must be a list")
        
        # Filter out empty strings and None values
        valid_texts = []
        for text in texts:
            if text is not None and isinstance(text, str) and text.strip():
                # Truncate if too long
                if len(text) > self.max_sequence_length:
                    text = text[:self.max_sequence_length]
                valid_texts.append(text.strip())
            else:
                # Add placeholder for empty texts to maintain indexing
                valid_texts.append("")
        
        return valid_texts


class StellaEmbedder(BaseEmbedder):
    """Stella embedding model implementation"""
    
    def __init__(self, config: Optional[StellaConfig] = None):
        if config is None:
            config = StellaConfig()
        
        super().__init__(config.model_dump() if hasattr(config, 'model_dump') else config.__dict__)
        self.stella_config = config
        self._device = None
        
    @property
    def model_name(self) -> str:
        return self.stella_config.model_name
    
    @property
    def dimensions(self) -> int:
        return self.stella_config.dimensions
    
    @property
    def max_sequence_length(self) -> int:
        return self.stella_config.max_length
    
    async def load_model(self) -> bool:
        """Load Stella model with platform optimizations"""
        if self.is_loaded:
            return True
        
        try:
            # Import here to avoid circular imports
            from sentence_transformers import SentenceTransformer
            import torch
            
            # Determine optimal device
            self._device = self._get_optimal_device()
            
            # Check if model is cached locally
            model_path = self.stella_config.model_path
            if model_path.exists() and any(model_path.iterdir()):
                # Load from local cache
                model_source = str(model_path)
            else:
                # Load from HuggingFace (will download if needed)
                model_source = self.stella_config.model_name
            
            # Load model
            self._model = SentenceTransformer(
                model_source,
                device=self._device,
                cache_folder=str(self.stella_config.cache_dir) if hasattr(self.stella_config, 'cache_dir') else None
            )
            
            # Configure model for inference
            self._model.eval()
            
            # Set precision if using GPU
            if self.stella_config.use_fp16 and self._device in ('cuda', 'mps'):
                if hasattr(self._model, 'half'):
                    self._model.half()
            
            self._is_loaded = True
            self._load_time = datetime.now()
            
            return True
            
        except Exception as e:
            self._model = None
            self._is_loaded = False
            raise RuntimeError(f"Failed to load Stella model: {e}") from e
    
    async def unload_model(self) -> None:
        """Unload Stella model to free memory"""
        if self._model is not None:
            try:
                # Move model to CPU to free GPU memory
                if hasattr(self._model, 'cpu'):
                    self._model.cpu()
                
                # Clear CUDA cache if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass
                
                del self._model
            except:
                pass
            
            self._model = None
            self._is_loaded = False
    
    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Stella model"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        # Validate and preprocess texts
        valid_texts = self._validate_texts(texts)
        
        if not valid_texts:
            return []
        
        try:
            # Generate embeddings
            embeddings = self._model.encode(
                valid_texts,
                batch_size=self.stella_config.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self.stella_config.normalize_embeddings
            )
            
            # Convert to list of lists
            return embeddings.tolist()
            
        except Exception as e:
            raise RuntimeError(f"Stella embedding generation failed: {e}") from e
    
    def _get_optimal_device(self) -> str:
        """Get optimal device for the current platform"""
        if self.stella_config.device:
            return self.stella_config.device
        
        try:
            import torch
            import platform
            
            # Auto-detect optimal device
            if torch.cuda.is_available():
                return "cuda"
            elif (platform.system() == "Darwin" and 
                  platform.machine() in ("arm64", "aarch64") and
                  hasattr(torch.backends, 'mps') and 
                  torch.backends.mps.is_available()):
                return "mps"  # Apple Silicon
            else:
                return "cpu"
                
        except ImportError:
            return "cpu"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed Stella model information"""
        base_info = super().get_model_info()
        
        stella_info = {
            "device": self._device,
            "cache_dir": str(self.stella_config.cache_dir),
            "is_cached": self.stella_config.is_model_cached,
            "use_fp16": self.stella_config.use_fp16,
            "normalize_embeddings": self.stella_config.normalize_embeddings,
            "batch_size": self.stella_config.batch_size
        }
        
        return {**base_info, **stella_info}


class EmbeddingManager:
    """Manager for multiple embedding providers"""
    
    def __init__(self):
        self._embedders: Dict[str, BaseEmbedder] = {}
        self._default_embedder: Optional[str] = None
    
    def register_embedder(self, name: str, embedder: BaseEmbedder, is_default: bool = False) -> None:
        """Register an embedding provider"""
        self._embedders[name] = embedder
        
        if is_default or self._default_embedder is None:
            self._default_embedder = name
    
    def get_embedder(self, name: Optional[str] = None) -> BaseEmbedder:
        """Get an embedding provider by name"""
        if name is None:
            name = self._default_embedder
        
        if name is None:
            raise ValueError("No embedder specified and no default set")
        
        if name not in self._embedders:
            raise ValueError(f"Embedder '{name}' not found")
        
        return self._embedders[name]
    
    def list_embedders(self) -> List[str]:
        """List all registered embedders"""
        return list(self._embedders.keys())
    
    async def embed_with_fallback(
        self, 
        texts: List[str], 
        preferred_embedder: Optional[str] = None,
        fallback_embedders: Optional[List[str]] = None
    ) -> EmbeddingResponse:
        """Generate embeddings with fallback to other providers"""
        embedders_to_try = []
        
        # Add preferred embedder first
        if preferred_embedder:
            embedders_to_try.append(preferred_embedder)
        
        # Add default if not already included
        if self._default_embedder and self._default_embedder not in embedders_to_try:
            embedders_to_try.append(self._default_embedder)
        
        # Add fallback embedders
        if fallback_embedders:
            for embedder_name in fallback_embedders:
                if embedder_name not in embedders_to_try:
                    embedders_to_try.append(embedder_name)
        
        # Try each embedder in order
        last_error = None
        for embedder_name in embedders_to_try:
            try:
                embedder = self.get_embedder(embedder_name)
                return await embedder.embed_texts(texts)
            except Exception as e:
                last_error = e
                continue
        
        # If all embedders failed
        if last_error:
            raise RuntimeError(f"All embedders failed. Last error: {last_error}") from last_error
        else:
            raise RuntimeError("No embedders available")
    
    async def cleanup_all(self) -> None:
        """Unload all embedders to free memory"""
        for embedder in self._embedders.values():
            try:
                await embedder.unload_model()
            except:
                pass