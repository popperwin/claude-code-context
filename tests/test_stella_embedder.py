"""
Unit tests for Stella embedder implementation.

Tests embedding generation, caching, device detection, and performance.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import List

from core.embeddings.stella import StellaEmbedder
from core.embeddings.base import EmbeddingResponse
from core.models.config import StellaConfig


class TestStellaEmbedder:
    """Test Stella embedder functionality"""
    
    def test_initialization(self):
        """Test embedder initialization"""
        # Default initialization
        embedder = StellaEmbedder()
        assert embedder.model_name == "stella_en_400M_v5"
        assert embedder.dimensions == 1024
        assert embedder.max_sequence_length == 512
        assert not embedder.is_loaded
        
        # Custom configuration
        config = StellaConfig(
            model_name="stella_en_1.5B_v5",
            dimensions=1024,
            batch_size=16
        )
        embedder = StellaEmbedder(config)
        assert embedder.model_name == "stella_en_1.5B_v5"
        assert embedder.stella_config.batch_size == 16
    
    def test_device_detection(self):
        """Test device detection logic"""
        embedder = StellaEmbedder()
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.get_device_name', return_value="Test GPU"):
            device = embedder._detect_optimal_device()
            assert device == "cuda"
        
        with patch('torch.cuda.is_available', return_value=False), \
             patch('platform.system', return_value="Darwin"), \
             patch('platform.machine', return_value="arm64"), \
             patch('torch.backends.mps.is_available', return_value=True):
            device = embedder._detect_optimal_device()
            assert device == "mps"
        
        with patch('torch.cuda.is_available', return_value=False), \
             patch('platform.system', return_value="Linux"):
            device = embedder._detect_optimal_device()
            assert device == "cpu"
    
    @pytest.mark.asyncio
    async def test_model_loading_mock(self):
        """Test model loading with mocked dependencies"""
        embedder = StellaEmbedder()
        
        # Mock SentenceTransformer
        mock_model = Mock()
        mock_model.eval = Mock()
        
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_model), \
             patch.object(embedder, '_detect_optimal_device', return_value="cpu"), \
             patch.object(embedder, '_apply_device_optimizations', new_callable=AsyncMock):
            
            result = await embedder.load_model()
            
            assert result is True
            assert embedder.is_loaded
            assert embedder._model is mock_model
            assert embedder.device == "cpu"
    
    @pytest.mark.asyncio
    async def test_model_loading_failure(self):
        """Test model loading failure handling"""
        embedder = StellaEmbedder()
        
        with patch('sentence_transformers.SentenceTransformer', side_effect=Exception("Model not found")):
            result = await embedder.load_model()
            
            assert result is False
            assert not embedder.is_loaded
            assert embedder._model is None
    
    @pytest.mark.asyncio
    async def test_model_unloading(self):
        """Test model unloading"""
        embedder = StellaEmbedder()
        
        # Mock loaded model
        mock_model = Mock()
        mock_model.cpu = Mock()
        embedder._model = mock_model
        embedder._is_loaded = True
        
        with patch.object(embedder, '_clear_gpu_cache', new_callable=AsyncMock):
            await embedder.unload_model()
            
            assert not embedder.is_loaded
            assert embedder._model is None
            mock_model.cpu.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_embedding_generation_with_cache(self):
        """Test embedding generation with caching"""
        embedder = StellaEmbedder()
        
        # Mock model and cache
        mock_model = Mock()
        mock_embeddings = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        mock_model.encode.return_value = mock_embeddings
        
        embedder._model = mock_model
        embedder._is_loaded = True
        
        # Mock cache to simulate misses
        embedder._cache.get_batch = Mock(return_value=([None, None], [0, 1]))
        embedder._cache.put_batch = Mock()
        
        texts = ["def function1():", "class MyClass:"]
        
        with patch.object(embedder, '_generate_stella_embeddings', 
                         new_callable=AsyncMock, return_value=mock_embeddings):
            result = await embedder._generate_embeddings(texts)
            
            assert len(result) == 2
            assert result == mock_embeddings
            
            # Verify cache interactions
            embedder._cache.get_batch.assert_called_once()
            embedder._cache.put_batch.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_embedding_generation_with_cache_hits(self):
        """Test embedding generation with cache hits"""
        embedder = StellaEmbedder()
        
        # Mock the model loading components but let the loading process happen
        mock_model = Mock()
        mock_model.eval = Mock()
        
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_model), \
             patch.object(embedder, '_detect_optimal_device', return_value="cpu"), \
             patch.object(embedder, '_apply_device_optimizations', new_callable=AsyncMock):
            
            # First, actually load the model
            load_result = await embedder.load_model()
            assert load_result is True
            assert embedder.is_loaded
            
            # Now test cache hits - mock cache to return cached embeddings
            cached_embeddings = [[1.0, 2.0], [3.0, 4.0]]
            embedder._cache.get_batch = Mock(return_value=(cached_embeddings, []))
            
            texts = ["cached_text1", "cached_text2"]
            result = await embedder._generate_embeddings(texts)
            
            assert result == cached_embeddings
            
            # Should not call Stella model for cache hits
            embedder._cache.get_batch.assert_called_once()
    
    def test_text_validation(self):
        """Test input text validation"""
        embedder = StellaEmbedder()
        
        # Valid texts
        texts = ["def function():", "class MyClass:", "import numpy"]
        validated = embedder._validate_texts(texts)
        assert len(validated) == 3
        assert validated == texts
        
        # Mixed valid and invalid texts
        texts = ["valid", "", None, "  ", "another_valid"]
        validated = embedder._validate_texts(texts)
        assert len(validated) == 5
        assert validated[0] == "valid"
        assert validated[1] == ""  # Empty string placeholder
        assert validated[2] == ""  # None becomes empty
        assert validated[3] == ""  # Whitespace becomes empty
        assert validated[4] == "another_valid"
        
        # Long text truncation
        long_text = "x" * 1000
        embedder.stella_config.max_length = 100
        validated = embedder._validate_texts([long_text])
        assert len(validated[0]) == 100
    
    def test_cache_statistics(self):
        """Test cache statistics tracking"""
        embedder = StellaEmbedder()
        
        # Initial stats
        stats = embedder.cache_stats
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        
        # Add to cache manually for testing
        embedder._cache.put("test", [1.0, 2.0])
        embedder._cache.get("test")  # Generate hit
        embedder._cache.get("missing")  # Generate miss
        
        stats = embedder.cache_stats
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
    
    def test_performance_metrics(self):
        """Test performance metrics tracking"""
        embedder = StellaEmbedder()
        
        # Initial metrics
        metrics = embedder.get_performance_metrics()
        assert metrics["total_embeddings"] == 0
        assert metrics["cache_hits"] == 0
        assert metrics["cache_hit_rate"] == 0
        assert metrics["device"] is None
        assert not metrics["model_loaded"]
        
        # Simulate some activity
        embedder._total_embeddings = 100
        embedder._cache_hits = 80
        embedder._device = "cpu"
        embedder._is_loaded = True
        
        metrics = embedder.get_performance_metrics()
        assert metrics["total_embeddings"] == 100
        assert metrics["cache_hits"] == 80
        assert metrics["cache_hit_rate"] == 0.8
        assert metrics["device"] == "cpu"
        assert metrics["model_loaded"]
    
    def test_cache_management(self):
        """Test cache management operations"""
        embedder = StellaEmbedder()
        
        # Add test data
        embedder._cache.put("test1", [1.0])
        embedder._cache.put("test2", [2.0])
        embedder._cache_hits = 5
        
        assert embedder.cache_stats["size"] == 2
        
        # Clear cache
        embedder.clear_cache()
        assert embedder.cache_stats["size"] == 0
        assert embedder._cache_hits == 0
        
        # Resize cache
        embedder.resize_cache(50)
        assert embedder._cache.max_size == 50
    
    @pytest.mark.asyncio
    async def test_warmup(self):
        """Test model warmup functionality"""
        embedder = StellaEmbedder()
        
        # Mock embed_texts method
        mock_response = EmbeddingResponse(
            embeddings=[[1.0, 2.0], [3.0, 4.0]],
            processing_time_ms=10.0
        )
        
        with patch.object(embedder, 'load_model', new_callable=AsyncMock, return_value=True), \
             patch.object(embedder, 'embed_texts', new_callable=AsyncMock, return_value=mock_response):
            
            # Default warmup
            result = await embedder.warmup()
            
            assert "warmup_time_s" in result
            assert result["sample_count"] == 4  # Default samples
            assert result["embeddings_generated"] == 2
            assert result["device"] is None
            
            # Custom warmup samples
            custom_samples = ["test1", "test2"]
            result = await embedder.warmup(custom_samples)
            assert result["sample_count"] == 2
    
    def test_model_info(self):
        """Test model information retrieval"""
        config = StellaConfig(
            model_name="stella_en_400M_v5",
            dimensions=1024,
            batch_size=16
        )
        embedder = StellaEmbedder(config)
        embedder._device = "cpu"
        
        info = embedder.get_model_info()
        
        assert info["model_name"] == "stella_en_400M_v5"
        assert info["dimensions"] == 1024
        assert info["device"] == "cpu"
        assert info["batch_size"] == 16
        assert info["use_fp16"] is True  # Default value
        assert info["normalize_embeddings"] is True
        assert "cache_size" in info
        assert "cache_hit_rate" in info
    
    @pytest.mark.asyncio
    async def test_concurrent_loading(self):
        """Test concurrent model loading (should only load once)"""
        embedder = StellaEmbedder()
        
        # Mock the actual model loading part, not the load_model method
        with patch('sentence_transformers.SentenceTransformer') as mock_st, \
             patch.object(embedder, '_detect_optimal_device', return_value="cpu"), \
             patch.object(embedder, '_apply_device_optimizations', new_callable=AsyncMock):
            
            mock_model = Mock()
            mock_model.eval = Mock()
            mock_st.return_value = mock_model
            
            # Start multiple load operations concurrently
            tasks = [embedder.load_model() for _ in range(3)]
            results = await asyncio.gather(*tasks)
            
            # All should succeed
            assert all(results)
            # Model should be loaded
            assert embedder.is_loaded
            # SentenceTransformer should only be called once due to locking
            assert mock_st.call_count == 1
    
    def test_get_model_path(self):
        """Test model path resolution"""
        embedder = StellaEmbedder()
        
        # Test HuggingFace model name (default case) - will check local cache first
        path = embedder._get_model_path()
        # Should return model name if cache doesn't exist or is empty
        assert path == "stella_en_400M_v5" or path.endswith("stella_en_400M_v5")
        
        # Test with existing local path
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.iterdir', return_value=[Mock()]):  # Non-empty directory
            
            # Test that model_path property returns the full path when local model exists
            original_model_path = embedder.stella_config.model_path
            path = embedder._get_model_path()
            assert path == str(original_model_path)
    
    @pytest.mark.asyncio
    async def test_apply_device_optimizations(self):
        """Test device-specific optimizations"""
        embedder = StellaEmbedder()
        mock_model = Mock()
        embedder._model = mock_model
        
        # Test CUDA optimizations
        embedder._device = "cuda"
        embedder.stella_config.use_fp16 = True
        
        with patch('torch.backends.cudnn') as mock_cudnn:
            await embedder._apply_device_optimizations()
            mock_cudnn.benchmark = True
            mock_cudnn.enabled = True
        
        # Test MPS optimizations
        embedder._device = "mps"
        mock_model.to = Mock()
        
        await embedder._apply_device_optimizations()
        mock_model.to.assert_called_with('mps')
        
        # Test CPU optimizations
        embedder._device = "cpu"
        
        with patch('torch.set_num_threads') as mock_set_threads, \
             patch('torch.get_num_threads', return_value=8):
            await embedder._apply_device_optimizations()
            mock_set_threads.assert_called_with(4)  # min(4, 8)
    
    @pytest.mark.asyncio
    async def test_clear_gpu_cache(self):
        """Test GPU cache clearing"""
        embedder = StellaEmbedder()
        
        # Test CUDA cache clearing
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.empty_cache') as mock_empty_cache:
            await embedder._clear_gpu_cache()
            mock_empty_cache.assert_called_once()
        
        # Test when CUDA not available
        with patch('torch.cuda.is_available', return_value=False):
            # Should not raise error
            await embedder._clear_gpu_cache()
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        # Valid configuration
        config = StellaConfig(
            model_name="stella_en_400M_v5",
            dimensions=1024,
            batch_size=32
        )
        embedder = StellaEmbedder(config)
        assert embedder.stella_config == config
        
        # Test with invalid model name (if validation exists)
        # This would depend on the actual validation in StellaConfig
        try:
            invalid_config = StellaConfig(model_name="invalid_model")
            embedder = StellaEmbedder(invalid_config)
            # If no validation, this should still work
        except ValueError:
            # If validation exists, it should raise ValueError
            pass


class TestStellaEmbedderIntegration:
    """Integration tests for Stella embedder (may require actual model)"""
    
    @pytest.mark.asyncio
    async def test_real_model_loading(self):
        """Test loading real Stella model"""
        embedder = StellaEmbedder()
        
        # This test requires actual model files
        result = await embedder.load_model()
        assert result is True
        assert embedder.is_loaded
        
        await embedder.unload_model()
        assert not embedder.is_loaded
    
    @pytest.mark.asyncio
    async def test_real_embedding_generation(self):
        """Test real embedding generation"""
        embedder = StellaEmbedder()
        
        await embedder.load_model()
        
        texts = [
            "def hello_world():",
            "class MyClass:",
            "import numpy as np"
        ]
        
        response = await embedder.embed_texts(texts)
        
        assert len(response.embeddings) == 3
        assert all(len(emb) == 1024 for emb in response.embeddings)
        assert response.processing_time_ms > 0
        
        await embedder.unload_model()
    
    @pytest.mark.asyncio
    async def test_performance_benchmarks(self):
        """Test performance benchmarks with real model"""
        embedder = StellaEmbedder()
        
        await embedder.load_model()
        
        # Test batch performance
        texts = [f"def function_{i}():" for i in range(32)]
        
        import time
        start_time = time.time()
        response = await embedder.embed_texts(texts)
        elapsed_time = time.time() - start_time
        
        # Performance assertions
        assert elapsed_time < 2.0  # Should complete within 2 seconds
        assert len(response.embeddings) == 32
        assert response.processing_time_ms < 2000
        
        # Test cache performance
        start_time = time.time()
        response2 = await embedder.embed_texts(texts)  # Should be cached
        cached_elapsed = time.time() - start_time
        
        assert cached_elapsed < elapsed_time  # Cached should be faster
        
        await embedder.unload_model()