"""
Unit tests for embeddings registry functionality.

Tests registry management, provider loading, factory patterns, and lifecycle management.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, Optional
import hashlib
import json

from core.embeddings.registry import (
    EmbedderRegistry, 
    EmbedderManager,
    get_registry,
    create_embedder,
    list_available_embedders,
    get_default_embedder,
    _global_registry
)
from core.embeddings.base import BaseEmbedder
from core.models.config import StellaConfig


class MockEmbedder(BaseEmbedder):
    """Mock embedder for testing"""
    
    def __init__(self, config: Optional[Any] = None):
        super().__init__(config)
        self._model_name = "mock-model"
        self._dimensions = 384
        self._max_sequence_length = 512
    
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
        self._is_loaded = True
        self._model = "mock_model_object"
        return True
    
    async def unload_model(self) -> None:
        self._is_loaded = False
        self._model = None
    
    async def _generate_embeddings(self, texts):
        """Internal method to generate embeddings"""
        return [[0.1] * self._dimensions for _ in texts]
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": self._model_name,
            "dimensions": self._dimensions,
            "loaded": self._is_loaded,
            "max_sequence_length": self._max_sequence_length
        }


class MockConfig:
    """Mock config class for testing"""
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class TestEmbedderRegistry:
    """Test EmbedderRegistry functionality"""
    
    def test_registry_initialization(self):
        """Test registry initialization with built-in embedders"""
        registry = EmbedderRegistry()
        
        # Should have stella registered by default
        assert "stella" in registry.get_available_providers()
        assert registry.get_default_provider() == "stella"
        
        # Should have aliases registered
        aliases = registry.get_available_aliases()
        assert "local" in aliases
        assert "default" in aliases
        assert aliases["local"] == "stella"
        assert aliases["default"] == "stella"
    
    def test_register_embedder_basic(self):
        """Test basic embedder registration"""
        registry = EmbedderRegistry()
        
        # Register mock embedder
        registry.register_embedder(
            name="mock",
            embedder_class=MockEmbedder,
            description="Mock embedder for testing"
        )
        
        # Should be available
        assert "mock" in registry.get_available_providers()
        
        # Should be able to create instance
        embedder = registry.create_embedder("mock")
        assert isinstance(embedder, MockEmbedder)
    
    def test_register_embedder_with_config(self):
        """Test embedder registration with config class"""
        registry = EmbedderRegistry()
        
        registry.register_embedder(
            name="mock_with_config",
            embedder_class=MockEmbedder,
            config_class=MockConfig,
            description="Mock embedder with config"
        )
        
        # Should create with config
        embedder = registry.create_embedder(
            "mock_with_config",
            config={"test_param": "test_value"}
        )
        assert isinstance(embedder, MockEmbedder)
        assert hasattr(embedder.config, "test_param")
        assert embedder.config.test_param == "test_value"
    
    def test_register_embedder_with_factory(self):
        """Test embedder registration with custom factory"""
        registry = EmbedderRegistry()
        
        def custom_factory(config=None, **kwargs):
            embedder = MockEmbedder(config)
            embedder._model_name = kwargs.get("model_name", "factory-model")
            return embedder
        
        registry.register_embedder(
            name="mock_factory",
            embedder_class=MockEmbedder,
            factory=custom_factory,
            description="Mock embedder with factory"
        )
        
        # Should use factory
        embedder = registry.create_embedder(
            "mock_factory",
            model_name="custom-model"
        )
        assert isinstance(embedder, MockEmbedder)
        assert embedder.model_name == "custom-model"
    
    def test_register_embedder_validation(self):
        """Test embedder registration validation"""
        registry = EmbedderRegistry()
        
        # Invalid name
        with pytest.raises(ValueError, match="must be a non-empty string"):
            registry.register_embedder("", MockEmbedder)
        
        with pytest.raises(ValueError, match="must be a non-empty string"):
            registry.register_embedder(None, MockEmbedder)
        
        # Invalid class
        with pytest.raises(ValueError, match="must inherit from BaseEmbedder"):
            registry.register_embedder("invalid", str)
    
    def test_register_alias(self):
        """Test alias registration"""
        registry = EmbedderRegistry()
        
        # Register mock embedder first
        registry.register_embedder("mock", MockEmbedder)
        
        # Register alias
        registry.register_alias("test_alias", "mock")
        
        # Alias should work
        aliases = registry.get_available_aliases()
        assert "test_alias" in aliases
        assert aliases["test_alias"] == "mock"
        
        # Should be able to create via alias
        embedder = registry.create_embedder("test_alias")
        assert isinstance(embedder, MockEmbedder)
    
    def test_register_alias_validation(self):
        """Test alias registration validation"""
        registry = EmbedderRegistry()
        
        # Target doesn't exist
        with pytest.raises(ValueError, match="Target embedder 'nonexistent' not found"):
            registry.register_alias("alias", "nonexistent")
    
    def test_create_embedder_default(self):
        """Test creating embedder with default provider"""
        registry = EmbedderRegistry()
        
        # Should use default (stella)
        embedder = registry.create_embedder()
        assert embedder is not None
    
    def test_create_embedder_unknown_provider(self):
        """Test creating embedder with unknown provider"""
        registry = EmbedderRegistry()
        
        with pytest.raises(ValueError, match="Unknown embedder provider"):
            registry.create_embedder("nonexistent")
    
    def test_create_embedder_no_default(self):
        """Test creating embedder when no default is set"""
        registry = EmbedderRegistry()
        registry._default_provider = None
        
        with pytest.raises(ValueError, match="No embedder provider specified"):
            registry.create_embedder()
    
    def test_create_embedder_config_error(self):
        """Test creating embedder with invalid config"""
        registry = EmbedderRegistry()
        
        # Mock config class that raises error
        class FailingConfig:
            def __init__(self, **kwargs):
                raise ValueError("Invalid config")
        
        registry.register_embedder(
            "failing_config",
            MockEmbedder,
            config_class=FailingConfig
        )
        
        with pytest.raises(ValueError, match="Invalid configuration"):
            registry.create_embedder("failing_config", config={"param": "value"})
    
    def test_create_embedder_factory_error(self):
        """Test creating embedder with failing factory"""
        registry = EmbedderRegistry()
        
        def failing_factory(config=None, **kwargs):
            raise RuntimeError("Factory failed")
        
        registry.register_embedder(
            "failing_factory",
            MockEmbedder,
            factory=failing_factory
        )
        
        with pytest.raises(RuntimeError, match="Failed to create failing_factory embedder"):
            registry.create_embedder("failing_factory")
    
    def test_create_embedder_constructor_error(self):
        """Test creating embedder with failing constructor"""
        registry = EmbedderRegistry()
        
        class FailingEmbedder(BaseEmbedder):
            def __init__(self, config=None):
                raise RuntimeError("Constructor failed")
            
            @property
            def model_name(self) -> str:
                return "failing"
            
            @property
            def dimensions(self) -> int:
                return 384
            
            @property
            def max_sequence_length(self) -> int:
                return 512
            
            async def load_model(self) -> bool:
                return False
            
            async def unload_model(self) -> None:
                pass
            
            async def _generate_embeddings(self, texts):
                return []
        
        registry.register_embedder("failing_constructor", FailingEmbedder)
        
        with pytest.raises(RuntimeError, match="Failed to create failing_constructor embedder"):
            registry.create_embedder("failing_constructor")
    
    def test_set_default_provider(self):
        """Test setting default provider"""
        registry = EmbedderRegistry()
        registry.register_embedder("mock", MockEmbedder)
        
        # Set new default
        registry.set_default_provider("mock")
        assert registry.get_default_provider() == "mock"
        
        # Should work with alias
        registry.register_alias("mock_alias", "mock")
        registry.set_default_provider("mock_alias")
        assert registry.get_default_provider() == "mock"  # Should resolve alias
    
    def test_set_default_provider_unknown(self):
        """Test setting unknown default provider"""
        registry = EmbedderRegistry()
        
        with pytest.raises(ValueError, match="Unknown provider"):
            registry.set_default_provider("nonexistent")
    
    def test_get_provider_info(self):
        """Test getting provider information"""
        registry = EmbedderRegistry()
        registry.register_embedder(
            "mock",
            MockEmbedder,
            config_class=MockConfig,
            description="Test embedder"
        )
        
        info = registry.get_provider_info("mock")
        
        assert info["name"] == "mock"
        assert info["class"] == "MockEmbedder"
        assert info["has_config"] is True
        assert info["has_factory"] is False
        assert "config_class" in info
    
    def test_get_provider_info_unknown(self):
        """Test getting info for unknown provider"""
        registry = EmbedderRegistry()
        
        with pytest.raises(ValueError, match="Unknown provider"):
            registry.get_provider_info("nonexistent")
    
    def test_list_providers(self):
        """Test listing all providers"""
        registry = EmbedderRegistry()
        registry.register_embedder("mock", MockEmbedder)
        registry.register_alias("mock_alias", "mock")
        
        # With aliases
        providers = registry.list_providers(include_aliases=True)
        assert "stella" in providers
        assert "mock" in providers
        assert "_aliases" in providers
        assert "_default" in providers
        assert providers["_aliases"]["mock_alias"]["target"] == "mock"
        
        # Without aliases
        providers = registry.list_providers(include_aliases=False)
        assert "_aliases" not in providers
        assert "_default" in providers
    
    def test_validate_provider(self):
        """Test provider validation"""
        registry = EmbedderRegistry()
        registry.register_embedder("mock", MockEmbedder)
        
        # Valid provider
        assert registry.validate_provider("mock") is True
        assert registry.validate_provider("stella") is True
        
        # Invalid provider
        assert registry.validate_provider("nonexistent") is False
        
        # Provider with failing constructor
        class FailingEmbedder(BaseEmbedder):
            def __init__(self, config=None):
                raise RuntimeError("Failed")
            
            @property
            def model_name(self) -> str:
                return "failing"
            
            @property
            def dimensions(self) -> int:
                return 384
            
            @property
            def max_sequence_length(self) -> int:
                return 512
            
            async def load_model(self) -> bool:
                return False
            
            async def unload_model(self) -> None:
                pass
            
            async def _generate_embeddings(self, texts):
                return []
        
        registry.register_embedder("failing", FailingEmbedder)
        assert registry.validate_provider("failing") is False
    
    def test_cleanup_provider(self):
        """Test provider cleanup"""
        registry = EmbedderRegistry()
        registry.register_embedder("mock", MockEmbedder, config_class=MockConfig)
        registry.register_alias("mock_alias", "mock")
        
        # Set as default
        registry.set_default_provider("mock")
        
        # Cleanup
        registry.cleanup_provider("mock")
        
        # Should be removed
        assert "mock" not in registry.get_available_providers()
        assert "mock_alias" not in registry.get_available_aliases()
        
        # Default should change
        assert registry.get_default_provider() != "mock"
    
    def test_cleanup_provider_unknown(self):
        """Test cleanup of unknown provider"""
        registry = EmbedderRegistry()
        
        with pytest.raises(ValueError, match="Unknown provider"):
            registry.cleanup_provider("nonexistent")


class TestGlobalRegistryFunctions:
    """Test global registry functions"""
    
    def setup_method(self):
        """Reset global registry before each test"""
        import core.embeddings.registry
        core.embeddings.registry._global_registry = None
    
    def test_get_registry(self):
        """Test getting global registry"""
        registry1 = get_registry()
        registry2 = get_registry()
        
        # Should be singleton
        assert registry1 is registry2
        assert isinstance(registry1, EmbedderRegistry)
    
    def test_create_embedder_global(self):
        """Test creating embedder via global function"""
        embedder = create_embedder()  # Uses default
        assert embedder is not None
    
    def test_list_available_embedders(self):
        """Test listing available embedders via global function"""
        providers = list_available_embedders()
        assert isinstance(providers, list)
        assert "stella" in providers
    
    def test_get_default_embedder(self):
        """Test getting default embedder via global function"""
        embedder = get_default_embedder()
        assert embedder is not None


class TestEmbedderManager:
    """Test EmbedderManager functionality"""
    
    def test_manager_initialization(self):
        """Test manager initialization"""
        registry = EmbedderRegistry()
        manager = EmbedderManager(registry)
        
        assert manager.registry is registry
        assert len(manager._instances) == 0
        assert len(manager._instance_configs) == 0
    
    def test_manager_initialization_default_registry(self):
        """Test manager initialization with default registry"""
        manager = EmbedderManager()
        assert manager.registry is not None
    
    @pytest.mark.asyncio
    async def test_get_embedder_caching(self):
        """Test embedder instance caching"""
        registry = EmbedderRegistry()
        registry.register_embedder("mock", MockEmbedder)
        manager = EmbedderManager(registry)
        
        # First call should create instance
        embedder1 = await manager.get_embedder("mock")
        assert isinstance(embedder1, MockEmbedder)
        
        # Load the model to test caching
        await embedder1.load_model()
        
        # Second call should return cached instance
        embedder2 = await manager.get_embedder("mock")
        assert embedder1 is embedder2
    
    @pytest.mark.asyncio
    async def test_get_embedder_cache_miss(self):
        """Test cache miss when embedder is not loaded"""
        registry = EmbedderRegistry()
        registry.register_embedder("mock", MockEmbedder)
        manager = EmbedderManager(registry)
        
        # First call
        embedder1 = await manager.get_embedder("mock")
        # Don't load model
        
        # Second call should create new instance since first isn't loaded
        embedder2 = await manager.get_embedder("mock")
        # Should still cache the new instance
        assert len(manager._instances) >= 1
    
    def test_generate_cache_key(self):
        """Test cache key generation"""
        manager = EmbedderManager()
        
        key1 = manager._generate_cache_key("provider", {"param": "value"}, {"extra": "data"})
        key2 = manager._generate_cache_key("provider", {"param": "value"}, {"extra": "data"})
        key3 = manager._generate_cache_key("provider", {"param": "different"}, {"extra": "data"})
        
        # Same inputs should generate same key
        assert key1 == key2
        # Different inputs should generate different keys
        assert key1 != key3
        # Keys should be reasonable length
        assert len(key1) == 16
    
    @pytest.mark.asyncio
    async def test_cleanup_all(self):
        """Test cleaning up all managed instances"""
        registry = EmbedderRegistry()
        registry.register_embedder("mock", MockEmbedder)
        manager = EmbedderManager(registry)
        
        # Create some instances
        embedder1 = await manager.get_embedder("mock")
        embedder2 = await manager.get_embedder("mock", config={"param": "value"})
        
        # Load models
        await embedder1.load_model()
        await embedder2.load_model()
        
        assert len(manager._instances) >= 1
        assert embedder1.is_loaded
        
        # Cleanup
        await manager.cleanup_all()
        
        assert len(manager._instances) == 0
        assert len(manager._instance_configs) == 0
        assert not embedder1.is_loaded
    
    @pytest.mark.asyncio
    async def test_cleanup_all_with_errors(self):
        """Test cleanup with errors in unload_model"""
        registry = EmbedderRegistry()
        
        class FailingUnloadEmbedder(MockEmbedder):
            async def unload_model(self):
                self._is_loaded = False
                self._model = None
                raise RuntimeError("Unload failed")
        
        registry.register_embedder("failing", FailingUnloadEmbedder)
        manager = EmbedderManager(registry)
        
        # Create instance
        embedder = await manager.get_embedder("failing")
        await embedder.load_model()
        
        # Cleanup should handle errors gracefully
        await manager.cleanup_all()  # Should not raise
        
        assert len(manager._instances) == 0
    
    @pytest.mark.asyncio
    async def test_get_managed_instances(self):
        """Test getting information about managed instances"""
        registry = EmbedderRegistry()
        registry.register_embedder("mock", MockEmbedder)
        manager = EmbedderManager(registry)
        
        # Create instance
        embedder = await manager.get_embedder("mock", config={"param": "value"})
        await embedder.load_model()
        
        # Get instance info
        instances = manager.get_managed_instances()
        
        assert len(instances) >= 1
        
        # Check info structure
        for cache_key, info in instances.items():
            assert "provider" in info
            assert "is_loaded" in info
            assert "model_name" in info
            assert "dimensions" in info
            assert "model_info" in info
            
            if info["provider"] == "mock":
                assert info["is_loaded"] is True
                assert info["model_name"] == "mock-model"
                assert info["dimensions"] == 384


class TestRegistryIntegration:
    """Integration tests for registry components"""
    
    def setup_method(self):
        """Reset global registry before each test"""
        import core.embeddings.registry
        core.embeddings.registry._global_registry = None
    
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete workflow with registry and manager"""
        # Get registry and register custom embedder
        registry = get_registry()
        registry.register_embedder("test", MockEmbedder, config_class=MockConfig)
        
        # Create manager
        manager = EmbedderManager(registry)
        
        # Create embedder instances
        embedder1 = await manager.get_embedder("test", config={"param1": "value1"})
        embedder2 = await manager.get_embedder("test", config={"param2": "value2"})
        
        # Load models
        await embedder1.load_model()
        await embedder2.load_model()
        
        # Verify instances
        assert embedder1.is_loaded
        assert embedder2.is_loaded
        assert embedder1 is not embedder2  # Different configs = different instances
        
        # Get instance info
        instances = manager.get_managed_instances()
        assert len(instances) == 2
        
        # Cleanup
        await manager.cleanup_all()
        
        assert not embedder1.is_loaded
        assert not embedder2.is_loaded
        assert len(manager._instances) == 0
    
    def test_provider_lifecycle(self):
        """Test complete provider lifecycle"""
        registry = EmbedderRegistry()
        
        # Register
        registry.register_embedder("lifecycle_test", MockEmbedder, is_default=True)
        assert registry.get_default_provider() == "lifecycle_test"
        
        # Create instance
        embedder = registry.create_embedder()
        assert isinstance(embedder, MockEmbedder)
        
        # Get info
        info = registry.get_provider_info("lifecycle_test")
        assert info["is_default"] is True
        
        # Validate
        assert registry.validate_provider("lifecycle_test") is True
        
        # Cleanup
        registry.cleanup_provider("lifecycle_test")
        assert "lifecycle_test" not in registry.get_available_providers()
        assert registry.get_default_provider() != "lifecycle_test"