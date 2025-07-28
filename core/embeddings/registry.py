"""
Embedder registry and factory for managing multiple embedding providers.

Provides a centralized system for registering, creating, and managing embedders.
"""

import logging
from typing import Dict, Type, Optional, Any, List, Callable
from pathlib import Path

from .base import BaseEmbedder, EmbedderProtocol
from .stella import StellaEmbedder
from ..models.config import StellaConfig

logger = logging.getLogger(__name__)


class EmbedderRegistry:
    """Registry for embedder implementations with factory methods"""
    
    def __init__(self):
        """Initialize embedder registry"""
        self._embedders: Dict[str, Type[BaseEmbedder]] = {}
        self._configs: Dict[str, Type] = {}
        self._factories: Dict[str, Callable] = {}
        self._aliases: Dict[str, str] = {}
        self._default_provider: Optional[str] = None
        
        # Register built-in embedders
        self._register_builtin_embedders()
        
        logger.info("Initialized embedder registry")
    
    def _register_builtin_embedders(self) -> None:
        """Register built-in embedder implementations"""
        # Register Stella embedder
        self.register_embedder(
            name="stella",
            embedder_class=StellaEmbedder,
            config_class=StellaConfig,
            is_default=True,
            description="Stella embedding model (infgrad/stella_en_400M_v5)"
        )
        
        # Register aliases
        self.register_alias("local", "stella")
        self.register_alias("default", "stella")
        self.register_alias("infgrad", "stella")
    
    def register_embedder(
        self,
        name: str,
        embedder_class: Type[BaseEmbedder],
        config_class: Optional[Type] = None,
        factory: Optional[Callable] = None,
        is_default: bool = False,
        description: str = ""
    ) -> None:
        """
        Register a new embedder implementation.
        
        Args:
            name: Unique name for the embedder
            embedder_class: Embedder class implementing BaseEmbedder
            config_class: Configuration class for the embedder
            factory: Optional factory function for custom instantiation
            is_default: Whether this should be the default embedder
            description: Human-readable description
        """
        if not name or not isinstance(name, str):
            raise ValueError("Embedder name must be a non-empty string")
        
        if not issubclass(embedder_class, BaseEmbedder):
            raise ValueError("Embedder class must inherit from BaseEmbedder")
        
        # Register components
        self._embedders[name] = embedder_class
        
        if config_class:
            self._configs[name] = config_class
        
        if factory:
            self._factories[name] = factory
        
        # Set as default if specified or if no default exists
        if is_default or self._default_provider is None:
            self._default_provider = name
        
        logger.info(f"Registered embedder: {name} ({description})")
    
    def register_alias(self, alias: str, target: str) -> None:
        """
        Register an alias for an existing embedder.
        
        Args:
            alias: Alias name
            target: Target embedder name
        """
        if target not in self._embedders:
            raise ValueError(f"Target embedder '{target}' not found")
        
        self._aliases[alias] = target
        logger.debug(f"Registered alias: {alias} -> {target}")
    
    def create_embedder(
        self,
        provider: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> BaseEmbedder:
        """
        Create embedder instance.
        
        Args:
            provider: Embedder provider name (uses default if None)
            config: Configuration dictionary
            **kwargs: Additional configuration parameters
            
        Returns:
            Configured embedder instance
        """
        # Resolve provider name
        if provider is None:
            provider = self._default_provider
        
        if provider is None:
            raise ValueError("No embedder provider specified and no default set")
        
        # Resolve aliases
        resolved_provider = self._resolve_alias(provider)
        
        if resolved_provider not in self._embedders:
            available = list(self._embedders.keys())
            raise ValueError(
                f"Unknown embedder provider: {provider}. "
                f"Available providers: {available}"
            )
        
        embedder_class = self._embedders[resolved_provider]
        
        # Create configuration if available
        embedder_config = None
        if resolved_provider in self._configs:
            config_class = self._configs[resolved_provider]
            
            # Merge config dict and kwargs
            config_data = {}
            if config:
                config_data.update(config)
            config_data.update(kwargs)
            
            if config_data:
                try:
                    embedder_config = config_class(**config_data)
                except Exception as e:
                    logger.error(f"Failed to create config for {resolved_provider}: {e}")
                    raise ValueError(f"Invalid configuration for {resolved_provider}: {e}")
        
        # Use custom factory if available
        if resolved_provider in self._factories:
            factory = self._factories[resolved_provider]
            try:
                return factory(config=embedder_config, **kwargs)
            except Exception as e:
                logger.error(f"Factory failed for {resolved_provider}: {e}")
                raise RuntimeError(f"Failed to create {resolved_provider} embedder: {e}")
        
        # Use standard constructor
        try:
            if embedder_config:
                return embedder_class(embedder_config)
            else:
                return embedder_class()
        except Exception as e:
            logger.error(f"Failed to create {resolved_provider} embedder: {e}")
            raise RuntimeError(f"Failed to create {resolved_provider} embedder: {e}")
    
    def _resolve_alias(self, name: str) -> str:
        """Resolve alias to actual provider name"""
        return self._aliases.get(name, name)
    
    def get_available_providers(self) -> List[str]:
        """Get list of available embedder providers"""
        return list(self._embedders.keys())
    
    def get_available_aliases(self) -> Dict[str, str]:
        """Get dictionary of available aliases"""
        return self._aliases.copy()
    
    def get_default_provider(self) -> Optional[str]:
        """Get default embedder provider name"""
        return self._default_provider
    
    def set_default_provider(self, provider: str) -> None:
        """
        Set default embedder provider.
        
        Args:
            provider: Provider name to set as default
        """
        resolved_provider = self._resolve_alias(provider)
        
        if resolved_provider not in self._embedders:
            raise ValueError(f"Unknown provider: {provider}")
        
        self._default_provider = resolved_provider
        logger.info(f"Set default embedder provider: {resolved_provider}")
    
    def get_provider_info(self, provider: str) -> Dict[str, Any]:
        """
        Get information about a provider.
        
        Args:
            provider: Provider name
            
        Returns:
            Provider information dictionary
        """
        resolved_provider = self._resolve_alias(provider)
        
        if resolved_provider not in self._embedders:
            raise ValueError(f"Unknown provider: {provider}")
        
        embedder_class = self._embedders[resolved_provider]
        config_class = self._configs.get(resolved_provider)
        
        info = {
            "name": resolved_provider,
            "class": embedder_class.__name__,
            "module": embedder_class.__module__,
            "has_config": config_class is not None,
            "has_factory": resolved_provider in self._factories,
            "is_default": resolved_provider == self._default_provider
        }
        
        if config_class:
            info["config_class"] = config_class.__name__
        
        return info
    
    def list_providers(self, include_aliases: bool = True) -> Dict[str, Any]:
        """
        List all available providers with information.
        
        Args:
            include_aliases: Whether to include alias information
            
        Returns:
            Dictionary with provider information
        """
        providers = {}
        
        for provider in self._embedders:
            providers[provider] = self.get_provider_info(provider)
        
        if include_aliases:
            aliases_info = {}
            for alias, target in self._aliases.items():
                aliases_info[alias] = {
                    "target": target,
                    "resolved": self._resolve_alias(alias)
                }
            providers["_aliases"] = aliases_info
        
        providers["_default"] = self._default_provider
        
        return providers
    
    def validate_provider(self, provider: str) -> bool:
        """
        Validate that a provider exists and can be instantiated.
        
        Args:
            provider: Provider name to validate
            
        Returns:
            True if provider is valid
        """
        try:
            resolved_provider = self._resolve_alias(provider)
            
            if resolved_provider not in self._embedders:
                return False
            
            # Try to create a minimal instance
            embedder_class = self._embedders[resolved_provider]
            
            # For validation, we don't actually load the model
            if resolved_provider in self._configs:
                config_class = self._configs[resolved_provider]
                config = config_class()
                test_instance = embedder_class(config)
            else:
                test_instance = embedder_class()
            
            # Check basic interface compliance
            return isinstance(test_instance, BaseEmbedder)
            
        except Exception as e:
            logger.debug(f"Provider validation failed for {provider}: {e}")
            return False
    
    def cleanup_provider(self, provider: str) -> None:
        """
        Remove a provider from registry.
        
        Args:
            provider: Provider name to remove
        """
        resolved_provider = self._resolve_alias(provider)
        
        if resolved_provider not in self._embedders:
            raise ValueError(f"Unknown provider: {provider}")
        
        # Remove from all registries
        del self._embedders[resolved_provider]
        
        if resolved_provider in self._configs:
            del self._configs[resolved_provider]
        
        if resolved_provider in self._factories:
            del self._factories[resolved_provider]
        
        # Remove aliases pointing to this provider
        aliases_to_remove = [
            alias for alias, target in self._aliases.items() 
            if target == resolved_provider
        ]
        for alias in aliases_to_remove:
            del self._aliases[alias]
        
        # Reset default if necessary
        if self._default_provider == resolved_provider:
            self._default_provider = next(iter(self._embedders.keys()), None)
        
        logger.info(f"Removed provider: {resolved_provider}")


# Global registry instance
_global_registry: Optional[EmbedderRegistry] = None


def get_registry() -> EmbedderRegistry:
    """Get global embedder registry instance"""
    global _global_registry
    
    if _global_registry is None:
        _global_registry = EmbedderRegistry()
    
    return _global_registry


def create_embedder(
    provider: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> BaseEmbedder:
    """
    Convenience function to create embedder using global registry.
    
    Args:
        provider: Embedder provider name
        config: Configuration dictionary
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured embedder instance
    """
    registry = get_registry()
    return registry.create_embedder(provider, config, **kwargs)


def list_available_embedders() -> List[str]:
    """Get list of available embedder providers"""
    registry = get_registry()
    return registry.get_available_providers()


def get_default_embedder() -> BaseEmbedder:
    """Create default embedder instance"""
    registry = get_registry()
    return registry.create_embedder()


class EmbedderManager:
    """
    High-level manager for embedder instances with lifecycle management.
    
    Provides instance caching, automatic cleanup, and failover capabilities.
    """
    
    def __init__(self, registry: Optional[EmbedderRegistry] = None):
        """
        Initialize embedder manager.
        
        Args:
            registry: Embedder registry to use (uses global if None)
        """
        self.registry = registry or get_registry()
        self._instances: Dict[str, BaseEmbedder] = {}
        self._instance_configs: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Initialized embedder manager")
    
    async def get_embedder(
        self,
        provider: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> BaseEmbedder:
        """
        Get or create embedder instance with caching.
        
        Args:
            provider: Embedder provider name
            config: Configuration dictionary
            **kwargs: Additional configuration parameters
            
        Returns:
            Embedder instance (cached or newly created)
        """
        # Generate cache key
        cache_key = self._generate_cache_key(provider, config, kwargs)
        
        # Return cached instance if available
        if cache_key in self._instances:
            embedder = self._instances[cache_key]
            if embedder.is_loaded:
                return embedder
        
        # Create new instance
        embedder = self.registry.create_embedder(provider, config, **kwargs)
        
        # Cache instance and config
        self._instances[cache_key] = embedder
        self._instance_configs[cache_key] = {
            'provider': provider,
            'config': config or {},
            'kwargs': kwargs
        }
        
        return embedder
    
    def _generate_cache_key(
        self,
        provider: Optional[str],
        config: Optional[Dict[str, Any]],
        kwargs: Dict[str, Any]
    ) -> str:
        """Generate cache key for embedder instance"""
        import hashlib
        import json
        
        # Create deterministic key from parameters
        key_data = {
            'provider': provider,
            'config': config or {},
            'kwargs': kwargs
        }
        
        key_json = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_json.encode()).hexdigest()[:16]
    
    async def cleanup_all(self) -> None:
        """Cleanup all managed embedder instances"""
        for embedder in self._instances.values():
            try:
                await embedder.unload_model()
            except Exception as e:
                logger.warning(f"Error unloading embedder: {e}")
        
        self._instances.clear()
        self._instance_configs.clear()
        
        logger.info("Cleaned up all embedder instances")
    
    def get_managed_instances(self) -> Dict[str, Dict[str, Any]]:
        """Get information about managed instances"""
        instances = {}
        
        for cache_key, embedder in self._instances.items():
            config = self._instance_configs.get(cache_key, {})
            
            instances[cache_key] = {
                'provider': config.get('provider'),
                'is_loaded': embedder.is_loaded,
                'model_name': embedder.model_name,
                'dimensions': embedder.dimensions,
                'model_info': embedder.get_model_info()
            }
        
        return instances