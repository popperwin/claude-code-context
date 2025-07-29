"""
Integration tests for Parser → Embeddings → Storage pipeline.

Tests the complete flow from Tree-sitter parsing through Stella embeddings
to Qdrant storage, validating Sprint 2 and Sprint 3 integration.
"""

import pytest
import tempfile
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Sprint 3 - Parser components
from core.parser.parallel_pipeline import ProcessParsingPipeline
from core.parser.base import ParseResult
from core.parser.registry import parser_registry

# Sprint 2 - Storage and embedding components
from core.storage import (
    HybridQdrantClient, CollectionManager, CollectionType, 
    BatchIndexer
)
from core.search import SearchMode, HybridSearcher
from core.embeddings import (
    StellaEmbedder, EmbeddingCache, get_default_cache,
    EmbeddingRequest, EmbeddingResponse
)
from core.models.entities import Entity, Relation, EntityType, RelationType, SourceLocation

logger = logging.getLogger(__name__)


@pytest.fixture
async def qdrant_client(stella_embedder):
    """Create a test Qdrant client with embedder"""
    client = HybridQdrantClient(
        url="http://localhost:6334",
        timeout=30.0,
        embedder=stella_embedder
    )
    
    # Ensure client is healthy
    try:
        health = await client.health_check()
        if not health.get("status") == "healthy":
            pytest.skip("Qdrant not available")
    except Exception:
        pytest.skip("Qdrant not available")
    
    yield client
    await client.disconnect()


@pytest.fixture
def stella_embedder():
    """Create Stella embedder with test configuration"""
    from core.models.config import StellaConfig
    
    config = StellaConfig(
        model_name="stella_en_400M_v5",
        device="cpu",  # Use CPU for test stability
        batch_size=8
    )
    
    embedder = StellaEmbedder(config=config)
    
    yield embedder
    
    # Cleanup
    if hasattr(embedder, 'cleanup'):
        embedder.cleanup()


@pytest.fixture
def collection_manager():
    """Create collection manager for test collections"""
    return CollectionManager(project_name="test-integration")


@pytest.fixture
def test_project():
    """Create a sample multi-language project"""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)
        
        # Python files
        (project_path / "main.py").write_text("""
'''Main application module'''

import os
import sys
from typing import List, Dict, Optional
from utils import process_data
from models import User, Product

def initialize_app() -> bool:
    '''Initialize the application'''
    logger.info("Starting application")
    return True

class ApplicationManager:
    '''Manages application lifecycle'''
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.users: List[User] = []
        self.products: List[Product] = []
    
    def add_user(self, user: User) -> None:
        '''Add a new user'''
        if user not in self.users:
            self.users.append(user)
    
    def get_user_products(self, user_id: str) -> List[Product]:
        '''Get products for a specific user'''
        return [p for p in self.products if p.owner_id == user_id]

async def async_data_processor(data: List[Dict]) -> List[Dict]:
    '''Process data asynchronously'''
    results = []
    for item in data:
        processed = await process_item_async(item)
        results.append(processed)
    return results
""")

        (project_path / "utils.py").write_text("""
'''Utility functions for data processing'''

import json
import hashlib
from typing import Any, Dict, List, Optional

def process_data(data: List[Dict]) -> List[Dict]:
    '''Process a list of data items'''
    return [process_item(item) for item in data]

def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
    '''Process a single data item'''
    return {
        'id': item.get('id', ''),
        'hash': generate_hash(item),
        'processed': True,
        'timestamp': get_current_timestamp()
    }

def generate_hash(data: Any) -> str:
    '''Generate hash for data'''
    json_str = json.dumps(data, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]

def get_current_timestamp() -> int:
    '''Get current Unix timestamp'''
    import time
    return int(time.time())

class DataValidator:
    '''Validates data structures'''
    
    @staticmethod
    def validate_item(item: Dict) -> bool:
        '''Validate a single item'''
        required_fields = ['id', 'type', 'data']
        return all(field in item for field in required_fields)
    
    @staticmethod
    def validate_batch(batch: List[Dict]) -> List[str]:
        '''Validate a batch of items, return errors'''
        errors = []
        for i, item in enumerate(batch):
            if not DataValidator.validate_item(item):
                errors.append(f"Item {i} is invalid")
        return errors
""")

        (project_path / "models.py").write_text("""
'''Data models for the application'''

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum

class UserRole(Enum):
    '''User role enumeration'''
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"

@dataclass
class User:
    '''User data model'''
    id: str
    name: str
    email: str
    role: UserRole = UserRole.USER
    metadata: Optional[Dict[str, Any]] = None
    
    def is_admin(self) -> bool:
        '''Check if user is admin'''
        return self.role == UserRole.ADMIN
    
    def get_display_name(self) -> str:
        '''Get user display name'''
        return f"{self.name} ({self.email})"

@dataclass  
class Product:
    '''Product data model'''
    id: str
    name: str
    description: str
    price: float
    owner_id: str
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def add_tag(self, tag: str) -> None:
        '''Add a tag to the product'''
        if tag not in self.tags:
            self.tags.append(tag)
    
    def get_formatted_price(self) -> str:
        '''Get formatted price string'''
        return f"${self.price:.2f}"

class ProductManager:
    '''Manages product operations'''
    
    def __init__(self):
        self.products: Dict[str, Product] = {}
    
    def add_product(self, product: Product) -> None:
        '''Add a product'''
        self.products[product.id] = product
    
    def get_product(self, product_id: str) -> Optional[Product]:
        '''Get a product by ID'''
        return self.products.get(product_id)
    
    def search_products(self, query: str) -> List[Product]:
        '''Search products by name or description'''
        query_lower = query.lower()
        return [
            product for product in self.products.values()
            if query_lower in product.name.lower() 
            or query_lower in product.description.lower()
        ]
""")

        # JavaScript file
        (project_path / "frontend.js").write_text("""
/**
 * Frontend application logic
 */

class ApiClient {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
        this.headers = {
            'Content-Type': 'application/json'
        };
    }
    
    async fetchUsers() {
        const response = await fetch(`${this.baseUrl}/users`, {
            headers: this.headers
        });
        return response.json();
    }
    
    async createUser(userData) {
        const response = await fetch(`${this.baseUrl}/users`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify(userData)
        });
        return response.json();
    }
    
    async updateUser(userId, userData) {
        const response = await fetch(`${this.baseUrl}/users/${userId}`, {
            method: 'PUT', 
            headers: this.headers,
            body: JSON.stringify(userData)
        });
        return response.json();
    }
}

class UserInterface {
    constructor(apiClient) {
        this.api = apiClient;
        this.users = [];
        this.selectedUser = null;
    }
    
    async loadUsers() {
        try {
            this.users = await this.api.fetchUsers();
            this.renderUserList();
        } catch (error) {
            console.error('Failed to load users:', error);
        }
    }
    
    renderUserList() {
        const container = document.getElementById('user-list');
        container.innerHTML = '';
        
        this.users.forEach(user => {
            const userElement = this.createUserElement(user);
            container.appendChild(userElement);
        });
    }
    
    createUserElement(user) {
        const div = document.createElement('div');
        div.className = 'user-item';
        div.innerHTML = `
            <h3>${user.name}</h3>
            <p>${user.email}</p>
            <span class="role">${user.role}</span>
        `;
        
        div.addEventListener('click', () => {
            this.selectUser(user);
        });
        
        return div;
    }
    
    selectUser(user) {
        this.selectedUser = user;
        this.renderUserDetails();
    }
    
    renderUserDetails() {
        if (!this.selectedUser) return;
        
        const container = document.getElementById('user-details');
        container.innerHTML = `
            <h2>User Details</h2>
            <p><strong>Name:</strong> ${this.selectedUser.name}</p>
            <p><strong>Email:</strong> ${this.selectedUser.email}</p>
            <p><strong>Role:</strong> ${this.selectedUser.role}</p>
        `;
    }
}

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
    const apiClient = new ApiClient('/api');
    const ui = new UserInterface(apiClient);
    ui.loadUsers();
});
""")

        # Configuration files
        (project_path / "config.json").write_text("""
{
    "app": {
        "name": "Test Application",
        "version": "1.0.0",
        "debug": true
    },
    "database": {
        "host": "localhost",
        "port": 5432,
        "name": "testdb",
        "pool_size": 10
    },
    "api": {
        "host": "0.0.0.0",
        "port": 8000,
        "cors_enabled": true,
        "rate_limit": 100
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "app.log"
    }
}
""")

        yield project_path


class TestParserStorageIntegration:
    """Test integration between parser and storage layers"""
    
    @pytest.mark.asyncio
    async def test_parse_to_storage_pipeline(
        self, 
        test_project: Path,
        qdrant_client: HybridQdrantClient,
        stella_embedder: StellaEmbedder,
        collection_manager: CollectionManager
    ):
        """Test complete pipeline from parsing to storage"""
        
        # Phase 1: Parse project files
        pipeline = ProcessParsingPipeline(max_workers=2, batch_size=3)
        parse_results, parse_stats = pipeline.parse_directory(test_project)
        
        # Verify parsing results
        assert len(parse_results) >= 4  # Python, JS, JSON files
        assert parse_stats.success_rate > 0.8
        assert parse_stats.total_entities > 0
        
        logger.info(f"Parsed {len(parse_results)} files with {parse_stats.total_entities} entities")
        
        # Phase 2: Create test collection
        collection_name = "test-integration-entities"
        
        try:
            # Clean up any existing collection
            await asyncio.to_thread(qdrant_client.client.delete_collection, collection_name)
        except:
            pass  # Collection might not exist
        
        # Create collection config
        config = collection_manager.create_collection_config(
            CollectionType.CODE,
            vector_size=stella_embedder.dimensions
        )
        config.name = collection_name
        
        # Create collection
        result = await qdrant_client.create_collection(config)
        success = result.success
        assert success, "Failed to create collection"
        
        # Phase 3: Generate embeddings and store entities
        batch_indexer = BatchIndexer(
            client=qdrant_client,
            embedder=stella_embedder,
            batch_size=10,
            max_retries=3
        )
        
        # Collect all entities from parse results
        all_entities = []
        for result in parse_results:
            all_entities.extend(result.entities)
        
        logger.info(f"Processing {len(all_entities)} entities for storage")
        
        # Index entities with embeddings
        indexing_result = await batch_indexer.index_entities(
            entities=all_entities,
            collection_name=collection_name
        )
        
        # Verify indexing results
        assert indexing_result.successful_entities > 0
        assert indexing_result.failed_entities == 0
        assert indexing_result.total_entities > 0
        
        logger.info(f"Indexed {indexing_result.successful_entities} entities successfully")
        
        # Phase 4: Verify storage and search
        logger.info("Starting Phase 4: Search verification")
        
        # Test semantic search to verify storage worked
        search_results = await qdrant_client.search_semantic(
            collection_name,
            "user management functions",
            limit=5
        )
        
        if len(search_results) == 0:
            logger.warning("No results for complex query, trying simpler search")
            # Try a simpler search to verify entities are stored
            search_results = await qdrant_client.search_semantic(
                collection_name,
                "function",
                limit=10
            )
        
        assert len(search_results) > 0, "Should return search results from stored entities"
        
        # Log results for debugging
        for i, result in enumerate(search_results[:3]):
            entity_name = result.point.payload.get("entity_name", "")
            entity_type = result.point.payload.get("entity_type", "")
            logger.info(f"Result {i+1}: {entity_name} ({entity_type})")
        
        # Verify search results contain relevant entities (maintain strict validation)
        found_user_related = any(
            "user" in result.point.payload.get("entity_name", "").lower() or
            "user" in result.point.payload.get("docstring", "").lower()
            for result in search_results
        )
        
        if not found_user_related:
            # If no user-related entities found, log what we got and still assert
            logger.warning("No user-related entities found. Available entities:")
            for result in search_results[:5]:
                entity_name = result.point.payload.get("entity_name", "")
                docstring = result.point.payload.get("docstring", "")[:50] if result.point.payload.get("docstring") else ""
                logger.warning(f"  - {entity_name}: {docstring}")
        
        assert found_user_related, "Should find user-related entities"
        
        logger.info(f"Search verification completed - {len(search_results)} relevant results")
        
        # Cleanup
        await asyncio.to_thread(qdrant_client.client.delete_collection, collection_name)
    
    @pytest.mark.asyncio
    async def test_relation_storage_retrieval(
        self,
        test_project: Path,
        qdrant_client: HybridQdrantClient,
        stella_embedder: StellaEmbedder,
        collection_manager: CollectionManager
    ):
        """Test storage and retrieval of entity relations"""
        
        # Parse project to get relations
        pipeline = ProcessParsingPipeline(max_workers=2, batch_size=2)
        parse_results, _ = pipeline.parse_directory(test_project)
        
        # Collect all relations
        all_relations = []
        for result in parse_results:
            all_relations.extend(result.relations)
        
        assert len(all_relations) > 0, "Should have extracted relations"
        logger.info(f"Found {len(all_relations)} relations")
        
        # Create relations collection
        relations_collection = "test-integration-relations"
        
        try:
            await asyncio.to_thread(qdrant_client.client.delete_collection, relations_collection)
        except:
            pass
        
        config = collection_manager.create_collection_config(
            CollectionType.RELATIONS,
            vector_size=stella_embedder.dimensions
        )
        config.name = relations_collection
        
        result = await qdrant_client.create_collection(config)
        success = result.success
        assert success
        
        # Index relations (convert to entities first since BatchIndexer only handles entities)
        batch_indexer = BatchIndexer(
            client=qdrant_client,
            embedder=stella_embedder,
            batch_size=5
        )
        
        # Convert relations to entities for indexing
        relation_entities = []
        for relation in all_relations:
            # Create entity representation of relation using proper constructor
            entity = Entity(
                id=f"rel_{relation.source_entity_id}_{relation.target_entity_id}",
                name=f"{relation.relation_type.value}_relation",
                qualified_name=f"{relation.relation_type.value}_relation",
                entity_type=EntityType.FUNCTION,  # Use generic type
                location=relation.location if relation.location else SourceLocation(
                    file_path=Path(""),
                    start_line=1,
                    end_line=1,
                    start_column=0,
                    end_column=0,
                    start_byte=0,
                    end_byte=0
                ),
                source_code=f"# Relation: {relation.relation_type.value}",
                signature=f"{relation.source_entity_id} -> {relation.target_entity_id}",
                docstring=f"Relation: {relation.relation_type.value}"
            )
            relation_entities.append(entity)
        
        indexing_result = await batch_indexer.index_entities(
            entities=relation_entities,
            collection_name=relations_collection
        )
        
        assert indexing_result.successful_entities > 0
        assert indexing_result.failed_entities == 0
        
        # Test relation queries using direct client methods
        # Search for import relations using payload search
        import_results = await qdrant_client.search_payload(
            relations_collection,
            "imports_relation",
            limit=10
        )
        
        # Also try broader search if specific didn't work
        if len(import_results) == 0:
            import_results = await qdrant_client.search_payload(
                relations_collection,
                "relation",
                limit=10
            )
        
        assert len(import_results) > 0, "Should find relation entities in storage"
        
        # Verify relation structure (adapted for entity representation)
        for result in import_results:
            payload = result.point.payload
            assert "entity_name" in payload, f"Missing entity_name in payload: {payload.keys()}"
            assert "signature" in payload, f"Missing signature in payload: {payload.keys()}"
            # Relations are stored as entities with signature containing "source -> target" relationship
            
        logger.info(f"Successfully verified {len(import_results)} relation entities with proper structure")
        
        logger.info(f"Found {len(import_results)} import relations")
        
        # Cleanup
        await asyncio.to_thread(qdrant_client.client.delete_collection, relations_collection)
    
    @pytest.mark.asyncio 
    async def test_multi_project_isolation(
        self,
        qdrant_client: HybridQdrantClient,
        stella_embedder: StellaEmbedder,
        collection_manager: CollectionManager
    ):
        """Test that different projects are properly isolated in collections"""
        
        # Create two test projects with different content
        projects = {}
        
        for project_id in ["project-a", "project-b"]:
            with tempfile.TemporaryDirectory() as temp_dir:
                project_path = Path(temp_dir)
                
                # Create different content for each project
                (project_path / f"{project_id}.py").write_text(f"""
def {project_id.replace('-', '_')}_function():
    '''Function specific to {project_id}'''
    return "{project_id}"

class {project_id.replace('-', '_').title()}Class:
    '''Class specific to {project_id}'''
    
    def method(self):
        return "{project_id}-method"
""")
                
                # Parse project
                pipeline = ProcessParsingPipeline(max_workers=1, batch_size=5)
                results, stats = pipeline.parse_directory(project_path)
                
                projects[project_id] = {
                    "results": results,
                    "stats": stats,
                    "entities": [entity for result in results for entity in result.entities]
                }
        
        # Create separate collections for each project
        collections = {}
        
        for project_id in ["project-a", "project-b"]:
            collection_name = f"test-isolation-{project_id}"
            collections[project_id] = collection_name
            
            # Clean up existing
            try:
                await asyncio.to_thread(qdrant_client.client.delete_collection, collection_name)
            except:
                pass
            
            # Create collection
            config = collection_manager.create_collection_config(
                CollectionType.CODE,
                vector_size=stella_embedder.dimensions
            )
            config.name = collection_name
            
            result = await qdrant_client.create_collection(config)
            success = result.success
            assert success
            
            # Index project entities
            batch_indexer = BatchIndexer(
                client=qdrant_client,
                embedder=stella_embedder
            )
            
            result = await batch_indexer.index_entities(
                entities=projects[project_id]["entities"],
                collection_name=collection_name
            )
            
            assert result.successful_entities > 0
            assert result.failed_entities == 0
        
        # Test isolation: search in project-a should not return project-b results
        searcher = HybridSearcher(qdrant_client)
        
        # Search for project-a specific content in project-a collection
        project_a_results = await qdrant_client.search_payload(
            collections["project-a"],
            "project_a_function",
            limit=10
        )
        
        # Verify only project-a entities are returned
        assert len(project_a_results) > 0
        for result in project_a_results:
            name = result.point.payload.get("entity_name", "")
            assert "project_a" in name or "project-a" in name.replace("_", "-")
            assert "project_b" not in name and "project-b" not in name.replace("_", "-")
        
        # Search for project-a content in project-b collection (should return nothing)
        cross_results = await qdrant_client.search_payload(
            collections["project-b"],
            "project_a_function",
            limit=10
        )
        
        # Should not find project-a content in project-b collection
        project_a_in_b = any(
            "project_a" in result.point.payload.get("entity_name", "")
            for result in cross_results
        )
        assert not project_a_in_b, "Project isolation failed"
        
        logger.info("Multi-project isolation verified successfully")
        
        # Cleanup
        for collection_name in collections.values():
            await asyncio.to_thread(qdrant_client.client.delete_collection, collection_name)
    
    @pytest.mark.asyncio
    async def test_embedding_cache_integration(
        self,
        test_project: Path,
        stella_embedder: StellaEmbedder
    ):
        """Test that embedding cache works with parser output"""
        
        # Parse project
        pipeline = ProcessParsingPipeline(max_workers=1, batch_size=5)
        parse_results, _ = pipeline.parse_directory(test_project)
        
        # Collect entities
        entities = [entity for result in parse_results for entity in result.entities]
        assert len(entities) > 0
        
        # Clear cache and time first embedding
        get_default_cache().clear()
        
        import time
        start_time = time.perf_counter()
        
        # Generate embeddings first time (should be slow)
        first_response = await stella_embedder.embed_texts([
            f"{entity.name}: {entity.docstring or entity.signature}"
            for entity in entities[:5]  # Test with subset
        ])
        
        first_time = time.perf_counter() - start_time
        first_embeddings = first_response.embeddings
        
        # Generate same embeddings again (should be faster due to cache)
        start_time = time.perf_counter()
        
        second_response = await stella_embedder.embed_texts([
            f"{entity.name}: {entity.docstring or entity.signature}"
            for entity in entities[:5]
        ])
        
        second_embeddings = second_response.embeddings
        
        second_time = time.perf_counter() - start_time
        
        # Verify results are identical
        assert len(first_embeddings) == len(second_embeddings)
        for i in range(len(first_embeddings)):
            # Compare the actual embedding vectors
            assert first_embeddings[i] == second_embeddings[i]
        
        # Cache should make second run faster (allow some variance for test stability)
        cache_speedup = first_time / second_time if second_time > 0 else float('inf')
        logger.info(f"Cache speedup: {cache_speedup:.2f}x")
        
        # Should see some improvement (even if small due to test overhead)
        assert cache_speedup >= 1.0, "Cache should provide some speedup"
    
    @pytest.mark.asyncio
    async def test_performance_benchmarks(
        self,
        test_project: Path,
        qdrant_client: HybridQdrantClient,
        stella_embedder: StellaEmbedder,
        collection_manager: CollectionManager
    ):
        """Benchmark end-to-end parsing to storage performance"""
        
        import time
        
        # Benchmark parsing phase
        parse_start = time.perf_counter()
        
        pipeline = ProcessParsingPipeline(max_workers=4, batch_size=5)
        parse_results, parse_stats = pipeline.parse_directory(test_project)
        
        parse_time = time.perf_counter() - parse_start
        
        # Collect entities
        all_entities = [entity for result in parse_results for entity in result.entities]
        
        # Benchmark embedding + storage phase
        collection_name = "test-performance-benchmark"
        
        try:
            await asyncio.to_thread(qdrant_client.client.delete_collection, collection_name)
        except:
            pass
        
        config = collection_manager.create_collection_config(
            CollectionType.CODE,
            vector_size=stella_embedder.dimensions
        )
        config.name = collection_name
        
        await qdrant_client.create_collection(config)
        
        storage_start = time.perf_counter()
        
        batch_indexer = BatchIndexer(
            client=qdrant_client,
            embedder=stella_embedder,
            batch_size=20  # Larger batch for performance
        )
        
        indexing_result = await batch_indexer.index_entities(
            entities=all_entities,
            collection_name=collection_name
        )
        
        storage_time = time.perf_counter() - storage_start
        total_time = parse_time + storage_time
        
        # Performance metrics
        entities_per_second = len(all_entities) / total_time
        parse_rate = parse_stats.files_per_second
        
        logger.info(f"Performance Benchmarks:")
        logger.info(f"  Parse time: {parse_time:.3f}s ({parse_rate:.1f} files/sec)")
        logger.info(f"  Storage time: {storage_time:.3f}s")
        logger.info(f"  Total time: {total_time:.3f}s")
        logger.info(f"  Entities/sec: {entities_per_second:.1f}")
        logger.info(f"  Total entities: {len(all_entities)}")
        
        # Assertions for reasonable performance
        assert entities_per_second > 0.5, "Should process at least 0.5 entities/second"
        assert parse_rate > 1.0, "Should parse at least 1 file/second"
        assert indexing_result.successful_entities > 0
        assert indexing_result.successful_entities == len(all_entities)
        
        # Test search performance
        search_start = time.perf_counter()
        
        search_results = await qdrant_client.search_hybrid(
            collection_name,
            "application manager user data",
            limit=10
        )
        
        search_time = time.perf_counter() - search_start
        
        logger.info(f"  Search time: {search_time*1000:.1f}ms")
        logger.info(f"  Search results: {len(search_results)}")
        
        # Search should be fast and return results
        assert search_time < 1.0, "Search should complete in under 1 second"
        assert len(search_results) > 0, "Should return search results"
        
        # Cleanup
        await asyncio.to_thread(qdrant_client.client.delete_collection, collection_name)


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-s"])