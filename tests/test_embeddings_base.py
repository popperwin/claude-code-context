"""
Unit tests for embeddings base functionality.

Tests BaseEmbedder interface, error handling, model loading, batch processing, and EmbeddingManager.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from core.embeddings.base import (
    BaseEmbedder,
    EmbeddingRequest,
    EmbeddingResponse, 
    EmbeddingManager,
    EmbedderProtocol
)
from core.models.config import StellaConfig


class TestEmbeddingRequest:
    """Test EmbeddingRequest data class"""
    
    def test_basic_initialization(self):
        """Test basic request initialization"""
        request = EmbeddingRequest(texts=["hello", "world"])
        
        assert request.texts == ["hello", "world"]
        assert request.batch_id is None
        assert request.metadata is None
    
    def test_full_initialization(self):
        """Test request with all parameters"""
        metadata = {"source": "test"}
        request = EmbeddingRequest(
            texts=["test text"],
            batch_id="batch-123",
            metadata=metadata
        )
        
        assert request.texts == ["test text"]
        assert request.batch_id == "batch-123"
        assert request.metadata == metadata


class TestEmbeddingResponse:
    """Test EmbeddingResponse data class"""
    
    def test_basic_initialization(self):
        """Test basic response initialization"""
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        response = EmbeddingResponse(
            embeddings=embeddings,
            processing_time_ms=100.0
        )
        
        assert response.embeddings == embeddings
        assert response.processing_time_ms == 100.0
        assert response.batch_id is None
        assert response.model_info is None
    
    def test_full_initialization(self):
        """Test response with all parameters"""
        embeddings = [[0.1, 0.2]]
        model_info = {"model": "test"}
        
        response = EmbeddingResponse(
            embeddings=embeddings,
            processing_time_ms=50.0,
            batch_id="batch-456", 
            model_info=model_info
        )
        
        assert response.embeddings == embeddings
        assert response.processing_time_ms == 50.0
        assert response.batch_id == "batch-456"
        assert response.model_info == model_info
    
    def test_embedding_count_property(self):
        """Test embedding count calculation"""
        # Empty embeddings
        response = EmbeddingResponse(embeddings=[], processing_time_ms=0.0)
        assert response.embedding_count == 0
        
        # Multiple embeddings
        embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        response = EmbeddingResponse(embeddings=embeddings, processing_time_ms=100.0)
        assert response.embedding_count == 3
    
    def test_average_embedding_time_property(self):
        """Test average embedding time calculation"""
        # Empty embeddings (division by zero protection)
        response = EmbeddingResponse(embeddings=[], processing_time_ms=100.0)
        assert response.average_embedding_time_ms == 0.0
        
        # Normal case
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        response = EmbeddingResponse(embeddings=embeddings, processing_time_ms=100.0)
        assert response.average_embedding_time_ms == 50.0  # 100 / 2


class MockEmbedder(BaseEmbedder):
    """Mock embedder for testing BaseEmbedder functionality"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._model_name = "mock-model"
        self._dimensions = 384
        self._max_sequence_length = 512
        self._should_fail_load = False
        self._should_fail_generate = False
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @property
    def dimensions(self) -> int:
        return self._dimensions
    
    @property
    def max_sequence_length(self) -> int:
        return self._max_sequence_length
    
    async def load_model(self) -> bool:
        if self._should_fail_load:
            raise RuntimeError("Mock load failure")
        
        self._is_loaded = True
        self._model = "mock_model_object"
        self._load_time = datetime.now()
        return True
    
    async def unload_model(self) -> None:
        self._is_loaded = False
        self._model = None
        self._load_time = None
    
    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        if self._should_fail_generate:
            raise RuntimeError("Mock generation failure")
        
        return [[0.1] * self._dimensions for _ in texts]
    
    def set_should_fail_load(self, should_fail: bool):
        """Test helper to simulate load failures"""
        self._should_fail_load = should_fail
    
    def set_should_fail_generate(self, should_fail: bool):
        """Test helper to simulate generation failures"""
        self._should_fail_generate = should_fail


class TestBaseEmbedder:
    """Test BaseEmbedder abstract base class"""
    
    def test_initialization(self):
        """Test embedder initialization"""
        config = {"test_param": "test_value"}
        embedder = MockEmbedder(config)
        
        assert embedder.config == config
        assert not embedder.is_loaded
        assert embedder._model is None
        assert embedder._load_time is None
    
    def test_initialization_without_config(self):
        """Test embedder initialization without config"""
        embedder = MockEmbedder()
        
        assert embedder.config == {}
        assert not embedder.is_loaded
    
    def test_properties(self):
        """Test embedder properties"""
        embedder = MockEmbedder()
        
        assert embedder.model_name == "mock-model"
        assert embedder.dimensions == 384
        assert embedder.max_sequence_length == 512
    
    @pytest.mark.asyncio
    async def test_load_model_success(self):
        """Test successful model loading"""
        embedder = MockEmbedder()
        
        result = await embedder.load_model()
        
        assert result is True
        assert embedder.is_loaded
        assert embedder._model is not None
        assert embedder._load_time is not None
    
    @pytest.mark.asyncio
    async def test_load_model_failure(self):
        """Test model loading failure"""
        embedder = MockEmbedder()
        embedder.set_should_fail_load(True)
        
        with pytest.raises(RuntimeError, match="Mock load failure"):
            await embedder.load_model()
        
        assert not embedder.is_loaded
    
    @pytest.mark.asyncio
    async def test_unload_model(self):
        """Test model unloading"""
        embedder = MockEmbedder()
        
        # Load first
        await embedder.load_model()
        assert embedder.is_loaded
        
        # Unload
        await embedder.unload_model()
        assert not embedder.is_loaded
        assert embedder._model is None
    
    @pytest.mark.asyncio
    async def test_embed_texts_basic(self):
        """Test basic text embedding"""
        embedder = MockEmbedder()
        await embedder.load_model()
        
        texts = ["hello", "world"]
        response = await embedder.embed_texts(texts)
        
        assert isinstance(response, EmbeddingResponse)
        assert len(response.embeddings) == 2
        assert len(response.embeddings[0]) == 384
        assert response.processing_time_ms > 0
        assert response.model_info is not None
    
    @pytest.mark.asyncio
    async def test_embed_texts_empty_list(self):
        """Test embedding empty text list"""
        embedder = MockEmbedder()
        await embedder.load_model()
        
        response = await embedder.embed_texts([])
        
        assert isinstance(response, EmbeddingResponse)
        assert response.embeddings == []
        assert response.processing_time_ms == 0.0
    
    @pytest.mark.asyncio
    async def test_embed_texts_auto_load(self):
        """Test automatic model loading when not loaded"""
        embedder = MockEmbedder()
        assert not embedder.is_loaded
        
        texts = ["test"]
        response = await embedder.embed_texts(texts)
        
        # Should auto-load and succeed
        assert embedder.is_loaded
        assert len(response.embeddings) == 1
    
    @pytest.mark.asyncio
    async def test_embed_texts_generation_failure(self):
        """Test embedding generation failure"""
        embedder = MockEmbedder()
        await embedder.load_model()
        embedder.set_should_fail_generate(True)
        
        with pytest.raises(RuntimeError, match="Embedding generation failed"):
            await embedder.embed_texts(["test"])
    
    @pytest.mark.asyncio
    async def test_embed_single_basic(self):
        """Test single text embedding"""
        embedder = MockEmbedder()
        await embedder.load_model()
        
        embedding = await embedder.embed_single("hello world")
        
        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)
    
    @pytest.mark.asyncio
    async def test_embed_single_empty_text(self):
        """Test single embedding with empty text"""
        embedder = MockEmbedder()
        await embedder.load_model()
        
        # Empty string should return zero vector
        embedding = await embedder.embed_single("")
        
        assert len(embedding) == 384
        assert all(x == 0.0 for x in embedding)
        
        # Whitespace only should also return zero vector  
        embedding = await embedder.embed_single("   ")
        
        assert len(embedding) == 384
        assert all(x == 0.0 for x in embedding)
    
    @pytest.mark.asyncio
    async def test_embed_single_failure(self):
        """Test single embedding failure"""
        embedder = MockEmbedder()
        await embedder.load_model()
        embedder.set_should_fail_generate(True)
        
        with pytest.raises(RuntimeError):
            await embedder.embed_single("test")
    
    def test_get_model_info(self):
        """Test getting model information"""
        config = {"test_param": "value"}
        embedder = MockEmbedder(config)
        
        info = embedder.get_model_info()
        
        assert info["model_name"] == "mock-model"
        assert info["dimensions"] == 384
        assert info["max_sequence_length"] == 512
        assert info["is_loaded"] is False
        assert info["load_time"] is None
        assert info["config"] == config
    
    @pytest.mark.asyncio
    async def test_get_model_info_loaded(self):
        """Test getting model info when loaded"""
        embedder = MockEmbedder()
        await embedder.load_model()
        
        info = embedder.get_model_info()
        
        assert info["is_loaded"] is True
        assert info["load_time"] is not None
    
    def test_validate_texts_basic(self):
        """Test text validation and preprocessing"""
        embedder = MockEmbedder()
        
        texts = ["hello", "world", "test"]
        validated = embedder._validate_texts(texts)
        
        assert validated == ["hello", "world", "test"]
    
    def test_validate_texts_with_empty_and_none(self):
        """Test text validation with empty strings and None values"""
        embedder = MockEmbedder()
        
        texts = ["hello", "", None, "  ", "world"]
        validated = embedder._validate_texts(texts)
        
        # Empty strings and None should become empty strings
        assert validated == ["hello", "", "", "", "world"]
    
    def test_validate_texts_with_whitespace(self):
        """Test text validation with whitespace handling"""
        embedder = MockEmbedder()
        
        texts = ["  hello  ", "\tworld\n", "   test   "]
        validated = embedder._validate_texts(texts)
        
        # Should strip whitespace
        assert validated == ["hello", "world", "test"]
    
    def test_validate_texts_truncation(self):
        """Test text truncation for long texts"""
        embedder = MockEmbedder()
        
        # Create text longer than max_sequence_length (512)
        long_text = "x" * 600
        texts = [long_text]
        validated = embedder._validate_texts(texts)
        
        # Should be truncated to max length
        assert len(validated[0]) == 512
        assert validated[0] == "x" * 512
    
    def test_validate_texts_invalid_input(self):
        """Test text validation with invalid input"""
        embedder = MockEmbedder()
        
        with pytest.raises(ValueError, match="texts must be a list"):
            embedder._validate_texts("not a list")
    
    def test_validate_texts_non_string_elements(self):
        """Test text validation with non-string elements"""
        embedder = MockEmbedder()
        
        texts = ["hello", 123, {"key": "value"}, "world"]
        validated = embedder._validate_texts(texts)
        
        # Non-string elements should become empty strings
        assert validated == ["hello", "", "", "world"]


class TestEmbedderProtocol:
    """Test EmbedderProtocol interface"""
    
    def test_protocol_compliance(self):
        """Test that MockEmbedder implements the protocol"""
        embedder = MockEmbedder()
        
        # Should implement the protocol
        assert isinstance(embedder, EmbedderProtocol)


class TestEmbeddingManager:
    """Test EmbeddingManager functionality"""
    
    def test_manager_initialization(self):
        """Test manager initialization"""
        manager = EmbeddingManager()
        
        assert len(manager._embedders) == 0
        assert manager._default_embedder is None
    
    def test_register_embedder(self):
        """Test embedder registration"""
        manager = EmbeddingManager()
        embedder = MockEmbedder()
        
        manager.register_embedder("mock", embedder)
        
        assert "mock" in manager._embedders
        assert manager._embedders["mock"] is embedder
        assert manager._default_embedder == "mock"  # First registered becomes default
    
    def test_register_embedder_explicit_default(self):
        """Test registering embedder as explicit default"""
        manager = EmbeddingManager()
        embedder1 = MockEmbedder()
        embedder2 = MockEmbedder()
        
        manager.register_embedder("first", embedder1)
        manager.register_embedder("second", embedder2, is_default=True)
        
        assert manager._default_embedder == "second"
    
    def test_get_embedder_by_name(self):
        """Test getting embedder by name"""
        manager = EmbeddingManager()
        embedder = MockEmbedder()
        
        manager.register_embedder("test", embedder)
        
        retrieved = manager.get_embedder("test")
        assert retrieved is embedder
    
    def test_get_embedder_default(self):
        """Test getting default embedder"""
        manager = EmbeddingManager()
        embedder = MockEmbedder()
        
        manager.register_embedder("default", embedder, is_default=True)
        
        retrieved = manager.get_embedder()  # No name specified
        assert retrieved is embedder
    
    def test_get_embedder_no_default(self):
        """Test getting embedder when no default is set"""
        manager = EmbeddingManager()
        
        with pytest.raises(ValueError, match="No embedder specified and no default set"):
            manager.get_embedder()
    
    def test_get_embedder_not_found(self):
        """Test getting non-existent embedder"""
        manager = EmbeddingManager()
        
        with pytest.raises(ValueError, match="Embedder 'nonexistent' not found"):
            manager.get_embedder("nonexistent")
    
    def test_list_embedders(self):
        """Test listing registered embedders"""
        manager = EmbeddingManager()
        embedder1 = MockEmbedder()
        embedder2 = MockEmbedder()
        
        manager.register_embedder("first", embedder1)
        manager.register_embedder("second", embedder2)
        
        embedders = manager.list_embedders()
        assert set(embedders) == {"first", "second"}
    
    @pytest.mark.asyncio
    async def test_embed_with_fallback_success(self):
        """Test successful embedding with fallback"""
        manager = EmbeddingManager()
        embedder = MockEmbedder()
        
        manager.register_embedder("primary", embedder)
        await embedder.load_model()
        
        response = await manager.embed_with_fallback(
            ["test text"],
            preferred_embedder="primary"
        )
        
        assert isinstance(response, EmbeddingResponse)
        assert len(response.embeddings) == 1
    
    @pytest.mark.asyncio
    async def test_embed_with_fallback_to_default(self):
        """Test fallback to default embedder"""
        manager = EmbeddingManager()
        failing_embedder = MockEmbedder()
        working_embedder = MockEmbedder()
        
        failing_embedder.set_should_fail_generate(True)
        
        manager.register_embedder("failing", failing_embedder)
        manager.register_embedder("working", working_embedder, is_default=True)
        
        await working_embedder.load_model()
        
        response = await manager.embed_with_fallback(
            ["test text"],
            preferred_embedder="failing"
        )
        
        # Should fallback to working default embedder
        assert isinstance(response, EmbeddingResponse)
        assert len(response.embeddings) == 1
    
    @pytest.mark.asyncio
    async def test_embed_with_fallback_explicit_list(self):
        """Test fallback with explicit fallback list"""
        manager = EmbeddingManager()
        failing_embedder = MockEmbedder()
        working_embedder = MockEmbedder()
        
        failing_embedder.set_should_fail_generate(True)
        
        manager.register_embedder("failing", failing_embedder)
        manager.register_embedder("working", working_embedder)
        
        await working_embedder.load_model()
        
        response = await manager.embed_with_fallback(
            ["test text"],
            preferred_embedder="failing",
            fallback_embedders=["working"]
        )
        
        assert isinstance(response, EmbeddingResponse)
        assert len(response.embeddings) == 1
    
    @pytest.mark.asyncio
    async def test_embed_with_fallback_all_fail(self):
        """Test fallback when all embedders fail"""
        manager = EmbeddingManager()
        failing_embedder1 = MockEmbedder()
        failing_embedder2 = MockEmbedder()
        
        failing_embedder1.set_should_fail_generate(True)
        failing_embedder2.set_should_fail_generate(True)
        
        manager.register_embedder("failing1", failing_embedder1)
        manager.register_embedder("failing2", failing_embedder2, is_default=True)
        
        with pytest.raises(RuntimeError, match="All embedders failed"):
            await manager.embed_with_fallback(
                ["test text"],
                preferred_embedder="failing1"
            )
    
    @pytest.mark.asyncio
    async def test_embed_with_fallback_no_embedders(self):
        """Test fallback with no embedders available"""
        manager = EmbeddingManager()
        
        with pytest.raises(RuntimeError, match="No embedders available"):
            await manager.embed_with_fallback(["test text"])
    
    @pytest.mark.asyncio
    async def test_cleanup_all(self):
        """Test cleaning up all embedders"""
        manager = EmbeddingManager()
        embedder1 = MockEmbedder()
        embedder2 = MockEmbedder()
        
        manager.register_embedder("first", embedder1)
        manager.register_embedder("second", embedder2)
        
        # Load models
        await embedder1.load_model()
        await embedder2.load_model()
        
        assert embedder1.is_loaded
        assert embedder2.is_loaded
        
        # Cleanup
        await manager.cleanup_all()
        
        assert not embedder1.is_loaded
        assert not embedder2.is_loaded
    
    @pytest.mark.asyncio
    async def test_cleanup_all_with_errors(self):
        """Test cleanup with errors in unload"""
        manager = EmbeddingManager()
        
        class FailingUnloadEmbedder(MockEmbedder):
            async def unload_model(self):
                raise RuntimeError("Unload failed")
        
        failing_embedder = FailingUnloadEmbedder()
        normal_embedder = MockEmbedder()
        
        manager.register_embedder("failing", failing_embedder)
        manager.register_embedder("normal", normal_embedder)
        
        await normal_embedder.load_model()
        
        # Should handle errors gracefully
        await manager.cleanup_all()  # Should not raise
        
        assert not normal_embedder.is_loaded


class TestIntegrationScenarios:
    """Integration tests for embeddings base functionality"""
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self):
        """Test complete embedding workflow"""
        # Setup
        embedder = MockEmbedder({"test_config": "value"})
        
        # Load model
        success = await embedder.load_model()
        assert success
        assert embedder.is_loaded
        
        # Generate embeddings
        texts = ["hello world", "test text", "another example"]
        response = await embedder.embed_texts(texts)
        
        # Verify response
        assert len(response.embeddings) == 3
        assert all(len(emb) == 384 for emb in response.embeddings)
        assert response.processing_time_ms > 0
        
        # Test single embedding
        single_embedding = await embedder.embed_single("single text")
        assert len(single_embedding) == 384
        
        # Get model info
        info = embedder.get_model_info()
        assert info["is_loaded"] is True
        assert info["model_name"] == "mock-model"
        
        # Unload
        await embedder.unload_model()
        assert not embedder.is_loaded
    
    @pytest.mark.asyncio
    async def test_manager_with_multiple_embedders(self):
        """Test manager with multiple embedders and fallback"""
        manager = EmbeddingManager()
        
        # Register multiple embedders
        primary = MockEmbedder()
        backup = MockEmbedder()
        
        manager.register_embedder("primary", primary)
        manager.register_embedder("backup", backup, is_default=True)
        
        # Load backup only
        await backup.load_model()
        
        # Primary should fail, backup should work
        primary.set_should_fail_load(True)
        
        response = await manager.embed_with_fallback(
            ["test text"],
            preferred_embedder="primary",
            fallback_embedders=["backup"]
        )
        
        assert len(response.embeddings) == 1
        
        # Cleanup
        await manager.cleanup_all()
    
    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test error recovery scenarios"""
        embedder = MockEmbedder()
        
        # Test load failure recovery
        embedder.set_should_fail_load(True)
        
        with pytest.raises(RuntimeError):
            await embedder.load_model()
        
        # Should be able to recover
        embedder.set_should_fail_load(False)
        success = await embedder.load_model()
        assert success
        
        # Test generation failure
        embedder.set_should_fail_generate(True)
        
        with pytest.raises(RuntimeError):
            await embedder.embed_texts(["test"])
        
        # Should be able to recover
        embedder.set_should_fail_generate(False)
        response = await embedder.embed_texts(["test"])
        assert len(response.embeddings) == 1
        
        await embedder.unload_model()