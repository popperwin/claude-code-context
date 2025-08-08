"""
Qdrant connection management for MCP server.

Provides robust connection handling with retry logic, exponential backoff,
and connection pooling for optimal performance and reliability.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from contextlib import asynccontextmanager

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, CollectionInfo, CreateCollection,
    TextIndexParams, TokenizerType, PayloadSchemaType
)
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from claude_code_context.mcp_server.models import MCPServerConfig

logger = logging.getLogger(__name__)


class ConnectionError(Exception):
    """Qdrant connection error"""
    pass


class RetryConfig:
    """Retry configuration for connection attempts"""
    
    def __init__(
        self,
        max_attempts: int = 5,
        initial_delay: float = 1.0,
        max_delay: float = 30.0,
        backoff_factor: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number"""
        delay = self.initial_delay * (self.backoff_factor ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add up to 20% jitter to prevent thundering herd
            import random
            jitter_amount = delay * 0.2 * random.random()
            delay += jitter_amount
        
        return delay


class QdrantConnectionManager:
    """
    Manages Qdrant connections with retry logic and health monitoring.
    
    Features:
    - Exponential backoff for connection retries
    - Connection pooling and reuse
    - Health monitoring and auto-recovery
    - Collection management and validation
    - Performance metrics tracking
    """
    
    def __init__(self, config: MCPServerConfig):
        """Initialize connection manager with configuration"""
        self.config = config
        self.retry_config = RetryConfig()
        
        # Connection state
        self._client: Optional[QdrantClient] = None
        self._connected = False
        self._connection_lock = asyncio.Lock()
        self._last_health_check = 0.0
        self._health_check_interval = 30.0  # seconds
        
        # Performance metrics
        self._connection_attempts = 0
        self._successful_connections = 0
        self._failed_connections = 0
        self._total_request_time = 0.0
        self._request_count = 0
        
        logger.info(f"Initialized QdrantConnectionManager for {config.qdrant_url}")
    
    @property
    def is_connected(self) -> bool:
        """Check if currently connected to Qdrant"""
        return self._connected
    
    @property
    def client(self) -> Optional[QdrantClient]:
        """Get current client instance"""
        return self._client
    
    async def connect(self) -> bool:
        """
        Connect to Qdrant with retry logic.
        
        Returns:
            True if connection successful, False otherwise
        """
        async with self._connection_lock:
            if self._connected and self._client:
                # Verify existing connection is still valid
                if await self._verify_connection():
                    return True
                else:
                    await self._disconnect_internal()
            
            return await self._connect_with_retry()
    
    async def _connect_with_retry(self) -> bool:
        """Internal connection method with retry logic"""
        last_error = None
        
        for attempt in range(self.retry_config.max_attempts):
            self._connection_attempts += 1
            
            try:
                logger.info(f"Connecting to Qdrant (attempt {attempt + 1}/{self.retry_config.max_attempts})")
                
                # Create new client
                self._client = QdrantClient(
                    url=self.config.qdrant_url,
                    timeout=self.config.qdrant_timeout,
                    prefer_grpc=False  # Use HTTP for better compatibility
                )
                
                # Test connection
                start_time = time.time()
                collections = await asyncio.to_thread(self._client.get_collections)
                elapsed = time.time() - start_time
                
                self._connected = True
                self._successful_connections += 1
                self._last_health_check = time.time()
                
                logger.info(
                    f"✅ Connected to Qdrant in {elapsed:.3f}s "
                    f"({len(collections.collections)} collections available)"
                )
                return True
                
            except Exception as e:
                last_error = e
                self._failed_connections += 1
                
                logger.warning(
                    f"❌ Connection attempt {attempt + 1} failed: {e}"
                )
                
                # Clean up failed client
                if self._client:
                    try:
                        self._client.close()
                    except:
                        pass
                    self._client = None
                
                # Wait before retry (except on last attempt)
                if attempt < self.retry_config.max_attempts - 1:
                    delay = self.retry_config.get_delay(attempt)
                    logger.info(f"⏳ Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
        
        # All attempts failed
        self._connected = False
        logger.error(f"❌ Failed to connect to Qdrant after {self.retry_config.max_attempts} attempts")
        
        if last_error:
            raise ConnectionError(f"Failed to connect to Qdrant: {last_error}") from last_error
        
        return False
    
    async def disconnect(self) -> None:
        """Disconnect from Qdrant"""
        async with self._connection_lock:
            await self._disconnect_internal()
    
    async def _disconnect_internal(self) -> None:
        """Internal disconnect method"""
        if self._client:
            try:
                self._client.close()
                logger.info("Disconnected from Qdrant")
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")
            finally:
                self._client = None
        
        self._connected = False
    
    async def _verify_connection(self) -> bool:
        """Verify that the current connection is still valid"""
        if not self._client:
            return False
        
        try:
            # Quick health check
            await asyncio.to_thread(self._client.get_collections)
            self._last_health_check = time.time()
            return True
        except Exception as e:
            logger.warning(f"Connection verification failed: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check.
        
        Returns:
            Health status information
        """
        current_time = time.time()
        
        # Use cached result if recent
        if (self._connected and 
            current_time - self._last_health_check < self._health_check_interval):
            return await self._get_cached_health_status(current_time)
        
        try:
            if not self._connected:
                # Try to reconnect
                if not await self.connect():
                    return self._get_unhealthy_status("Not connected")
            
            # Perform detailed health check
            start_time = time.time()
            
            collections = await asyncio.to_thread(self._client.get_collections)
            
            # Check if our collection exists (with proper type suffix)
            # Import CollectionType to check for typed collection
            from core.storage.schemas import CollectionType
            collection_name = self.config.get_typed_collection_name(CollectionType.CODE)
            collection_exists = any(
                col.name == collection_name 
                for col in collections.collections
            )
            
            elapsed = time.time() - start_time
            self._last_health_check = current_time
            
            return {
                "status": "healthy",
                "connected": True,
                "response_time_ms": elapsed * 1000,
                "qdrant_url": self.config.qdrant_url,
                "collection_name": collection_name,
                "collection_exists": collection_exists,
                "total_collections": len(collections.collections),
                "connection_stats": self._get_connection_stats(),
                "last_check": current_time
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self._connected = False
            return self._get_unhealthy_status(str(e))
    
    async def _get_cached_health_status(self, current_time: float) -> Dict[str, Any]:
        """Get cached health status when recent check is available"""
        from core.storage.schemas import CollectionType
        return {
            "status": "healthy",
            "connected": True,
            "response_time_ms": None,  # Cached result
            "qdrant_url": self.config.qdrant_url,
            "collection_name": self.config.get_typed_collection_name(CollectionType.CODE),
            "collection_exists": None,  # Would need fresh check
            "total_collections": None,  # Would need fresh check
            "connection_stats": self._get_connection_stats(),
            "last_check": self._last_health_check,
            "cached": True
        }
    
    def _get_unhealthy_status(self, error_message: str) -> Dict[str, Any]:
        """Get unhealthy status response"""
        from core.storage.schemas import CollectionType
        return {
            "status": "unhealthy",
            "connected": False,
            "error": error_message,
            "qdrant_url": self.config.qdrant_url,
            "collection_name": self.config.get_typed_collection_name(CollectionType.CODE),
            "connection_stats": self._get_connection_stats(),
            "last_check": time.time()
        }
    
    def _get_connection_stats(self) -> Dict[str, Any]:
        """Get connection performance statistics"""
        success_rate = (
            self._successful_connections / max(self._connection_attempts, 1) * 100
        )
        
        avg_request_time = (
            self._total_request_time / max(self._request_count, 1) * 1000
            if self._request_count > 0 else None
        )
        
        return {
            "total_attempts": self._connection_attempts,
            "successful": self._successful_connections,
            "failed": self._failed_connections,
            "success_rate_percent": round(success_rate, 1),
            "total_requests": self._request_count,
            "avg_request_time_ms": round(avg_request_time, 2) if avg_request_time else None
        }
    
    async def ensure_collection_exists(self) -> bool:
        """
        Ensure the configured collection exists, creating it if necessary.
        
        Returns:
            True if collection exists or was created successfully
        """
        if not await self.connect():
            return False
        
        from core.storage.schemas import CollectionType
        collection_name = self.config.get_typed_collection_name(CollectionType.CODE)
        
        try:
            # Check if collection exists
            collections = await asyncio.to_thread(self._client.get_collections)
            existing_collections = {col.name for col in collections.collections}
            
            if collection_name in existing_collections:
                logger.info(f"✅ Collection '{collection_name}' already exists")
                return True
            
            # Create collection
            logger.info(f"📝 Creating collection '{collection_name}'...")
            
            await asyncio.to_thread(
                self._client.create_collection,
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=1024,  # Stella embedding dimensions
                    distance=Distance.COSINE
                )
            )
            
            # Create payload indexes for better search performance
            await self._create_payload_indexes(collection_name)
            
            logger.info(f"✅ Created collection '{collection_name}' with indexes")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create collection '{collection_name}': {e}")
            return False
    
    async def _create_payload_indexes(self, collection_name: str) -> None:
        """Create payload indexes for optimized search performance"""
        try:
            # Index for entity_type field
            await asyncio.to_thread(
                self._client.create_payload_index,
                collection_name=collection_name,
                field_name="entity_type",
                field_schema=PayloadSchemaType.KEYWORD
            )
            
            # Index for language field
            await asyncio.to_thread(
                self._client.create_payload_index,
                collection_name=collection_name,
                field_name="language",
                field_schema=PayloadSchemaType.KEYWORD
            )
            
            # Text index for content field
            await asyncio.to_thread(
                self._client.create_payload_index,
                collection_name=collection_name,
                field_name="content",
                field_schema=TextIndexParams(
                    type="text",
                    tokenizer=TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=20,
                    lowercase=True
                )
            )
            
            logger.info(f"Created payload indexes for collection '{collection_name}'")
            
        except Exception as e:
            logger.warning(f"Failed to create some payload indexes: {e}")
    
    @asynccontextmanager
    async def get_client(self):
        """
        Context manager for getting a connected client.
        
        Usage:
            async with connection_manager.get_client() as client:
                result = await asyncio.to_thread(client.search, ...)
        """
        if not await self.connect():
            raise ConnectionError("Unable to connect to Qdrant")
        
        start_time = time.time()
        try:
            yield self._client
            
            # Track successful request
            elapsed = time.time() - start_time
            self._total_request_time += elapsed
            self._request_count += 1
            
        except Exception as e:
            logger.error(f"Error during client operation: {e}")
            # Reset connection on certain errors
            if isinstance(e, (ResponseHandlingException, UnexpectedResponse)):
                self._connected = False
            raise
    
    async def collection_info(self) -> Optional[CollectionInfo]:
        """Get information about the configured collection"""
        try:
            async with self.get_client() as client:
                from core.storage.schemas import CollectionType
                collection_name = self.config.get_typed_collection_name(CollectionType.CODE)
                return await asyncio.to_thread(
                    client.get_collection, 
                    collection_name
                )
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive connection metrics"""
        from core.storage.schemas import CollectionType
        return {
            "connection_manager": {
                "connected": self._connected,
                "qdrant_url": self.config.qdrant_url,
                "collection_name": self.config.get_typed_collection_name(CollectionType.CODE),
                "last_health_check": self._last_health_check,
                "health_check_interval": self._health_check_interval
            },
            "performance": self._get_connection_stats(),
            "retry_config": {
                "max_attempts": self.retry_config.max_attempts,
                "initial_delay": self.retry_config.initial_delay,
                "max_delay": self.retry_config.max_delay,
                "backoff_factor": self.retry_config.backoff_factor
            }
        }