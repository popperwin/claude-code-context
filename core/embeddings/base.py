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
    
    async def embed_query(self, text: str, prompt: Optional[str] = "s2p_query") -> List[float]:
        """
        Generate embedding for a search query with optional prompt.
        
        This method is specifically designed for asymmetric search where queries
        need different encoding than documents. For Stella models, queries should
        use the "s2p_query" prompt while documents use no prompt.
        
        Args:
            text: Query text to embed
            prompt: Prompt name for the model (default: "s2p_query" for Stella)
            
        Returns:
            Query embedding vector
        """
        if not text.strip():
            # Return zero vector for empty text
            return [0.0] * self.dimensions
        
        # Default implementation falls back to embed_single for compatibility
        # Subclasses should override to implement prompt support
        return await self.embed_single(text)
    
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


# NOTE: Concrete StellaEmbedder implementation is in stella.py
# This module contains only abstract base classes and protocols


# NOTE: EmbeddingManager implementation removed - was unused dead code.
# Production code uses direct StellaEmbedder instantiation.
# Registry-based management is available in registry.py if needed.