"""
Comprehensive real-life end-to-end tests for Entity Lifecycle Integration.

Tests the complete entity lifecycle with real repositories: cloning, initial indexing,
entity scanning, incremental sync, file modifications, entity deletions, and 
real search results across multiple languages using the pure entity-level architecture.
"""

import pytest
import os
import shutil
import subprocess
import uuid
import time
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path

# Fix tokenizer fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from core.indexer.entity_lifecycle_integration import EntityLifecycleIntegrator, EntityOperationResult
from core.indexer.entity_scanner import EntityScanner, EntityScanRequest
from core.indexer.entity_detector import EntityChangeDetector
from core.sync.lifecycle import EntityLifecycleManager
from core.sync.engine import ProjectCollectionSyncEngine
from core.indexer.hybrid_indexer import HybridIndexer, IndexingJobConfig
from core.storage.client import HybridQdrantClient
from core.parser.parallel_pipeline import ProcessParsingPipeline
from core.embeddings.stella import StellaEmbedder
from core.indexer.cache import CacheManager
from core.models.entities import Entity, EntityType
from core.sync.events import FileSystemEvent, EventType
from core.storage.schemas import CollectionManager, CollectionType


class TestEntityLifecycleIntegrationRealLife:
    """Real-life integration tests for entity lifecycle operations"""
    
    @classmethod
    def setup_class(cls):
        """Setup shared test resources with GitHub repositories and shared collections"""
        cls.test_dir = Path("test-harness/temp-entity-lifecycle").resolve()
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Track shared collections for cleanup
        cls.shared_collections = set()
        cls.created_files = {}  # Track files we create for cleanup
        
        # Clone smaller, focused repositories for entity testing
        cls.repos = {
            "python-small": {
                "url": "https://github.com/pallets/click.git",
                "path": cls.test_dir / "click",
                "branch": "main",
                "language": "python"
            },
            "typescript-small": {
                "url": "https://github.com/sindresorhus/is.git", 
                "path": cls.test_dir / "is",
                "branch": "main",
                "language": "typescript"
            },
            "javascript-small": {
                "url": "https://github.com/jonschlinkert/is-number.git",
                "path": cls.test_dir / "is-number",
                "branch": "master", 
                "language": "javascript"
            },
            "go-small": {
                "url": "https://github.com/gin-gonic/gin.git",
                "path": cls.test_dir / "gin",
                "branch": "master",
                "language": "go"
            }
        }
        
        # Clone repositories (shallow clone for speed)
        for repo_name, repo_info in cls.repos.items():
            if not repo_info["path"].exists():
                try:
                    subprocess.run([
                        "git", "clone", "--depth", "1", "--single-branch",
                        repo_info["url"], str(repo_info["path"])
                    ], check=True, capture_output=True, timeout=60)
                    print(f"Cloned {repo_name} successfully")
                except subprocess.TimeoutExpired:
                    print(f"Timeout cloning {repo_name}, skipping")
                except subprocess.CalledProcessError as e:
                    print(f"Failed to clone {repo_name}: {e}")
    
    @classmethod 
    def teardown_class(cls):
        """Cleanup shared test resources"""
        # Clean up shared collections
        if hasattr(cls, 'shared_collections'):
            try:
                import requests
                for collection_name in cls.shared_collections:
                    try:
                        requests.delete(f"http://localhost:6334/collections/{collection_name}", timeout=5)
                        print(f"Cleaned up collection: {collection_name}")
                    except Exception:
                        pass  # Ignore cleanup errors
            except Exception:
                pass
        
        # Clean up created test files
        if hasattr(cls, 'created_files'):
            for repo_name, files in cls.created_files.items():
                for file_path in files:
                    try:
                        if file_path.exists():
                            file_path.unlink()
                    except Exception:
                        pass
        
        # Clean up test repositories 
        if hasattr(cls, 'test_dir') and cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
            print("Cleaned up test repositories")
    
    def setup_method(self):
        """Setup test instance"""
        self.client = HybridQdrantClient("http://localhost:6334")
        
        # Initialize components (will be done in async setup)
        self.parser_pipeline = None
        self.embedder = None
        self.cache_manager = None
        self.indexer = None
        self.integrator = None
        
        # Clean up any old test collections (but not shared ones)
        self._cleanup_old_test_collections()
    
    async def _async_setup(self):
        """Async setup for components that need event loop"""
        if self.indexer is None:
            self.parser_pipeline = ProcessParsingPipeline(max_workers=2, batch_size=10)
            self.embedder = StellaEmbedder()
            
            # Pre-load the Stella model to exclude loading time from performance measurements
            await self.embedder.load_model()
            
            # Create new client with embedder - the old client was missing embedder
            self.client = HybridQdrantClient("http://localhost:6334", embedder=self.embedder)
            
            self.cache_manager = CacheManager()
            self.indexer = HybridIndexer(
                parser_pipeline=self.parser_pipeline,
                embedder=self.embedder,
                storage_client=self.client,
                cache_manager=self.cache_manager
            )
    
    def teardown_method(self):
        """Cleanup after each test"""
        # Don't clean up shared collections in teardown_method
        pass
    
    def _cleanup_old_test_collections(self):
        """Clean up old test collections (but not shared entity collections)"""
        try:
            import requests
            response = requests.get("http://localhost:6334/collections", timeout=5)
            if response.status_code == 200:
                collections = response.json().get("result", {}).get("collections", [])
                for collection in collections:
                    collection_name = collection.get("name", "")
                    # Only delete old UUID-based collections, not our shared ones
                    old_patterns = ["temp-", "quality-test-", "integration-test-", "entity-test-temp-"]
                    # Don't delete our shared "test-entity-{repo}" collections
                    if (any(pattern in collection_name for pattern in old_patterns) and 
                        not collection_name.startswith("test-entity-")):
                        requests.delete(f"http://localhost:6334/collections/{collection_name}", timeout=5)
        except Exception:
            pass  # Ignore cleanup errors
    
    async def get_or_create_entity_collection(self, repo_name: str) -> str:
        """Get existing entity collection for repo or create it if needed"""
        # Setup async components first
        await self._async_setup()
        
        # Use CollectionManager to get the proper collection name format
        base_project_name = f"test-entity-{repo_name}"
        collection_manager = CollectionManager(project_name=base_project_name)
        actual_collection_name = collection_manager.get_collection_name(CollectionType.CODE)
        
        print(f"Looking for collection: {actual_collection_name}")
        
        # Check if collection already exists with sufficient entities
        try:
            collection_info = await self.client.get_collection_info(actual_collection_name)
            if collection_info and collection_info.get("points_count", 0) > 50:
                # Collection exists and has entities, reuse it
                self.__class__.shared_collections.add(actual_collection_name)
                print(f"✅ REUSING existing collection: {actual_collection_name} ({collection_info.get('points_count', 0)} entities)")
                return actual_collection_name
        except Exception as e:
            print(f"Collection {actual_collection_name} doesn't exist: {e}")
        
        repo_info = self.repos[repo_name]
        repo_path = repo_info["path"]
        
        if not repo_path.exists():
            raise pytest.skip(f"Repository {repo_name} not available")
        
        # Create and index new collection using pure entity approach
        print(f"🔄 CREATING new entity collection: {actual_collection_name}")
        
        config = IndexingJobConfig(
            project_path=repo_path,
            project_name=base_project_name,  # Use base name, CollectionManager handles suffix
            include_patterns=["*.py", "*.js", "*.ts", "*.go", "*.rs", "*.java"],
            exclude_patterns=["**/node_modules/**", "**/target/**", "**/.git/**", "**/build/**"],
            batch_size=25,
            # Use entity-level configuration
            entity_scan_mode="full_rescan",
            enable_entity_monitoring=True,
            entity_batch_size=50
        )
        
        await self.indexer.index_project(config)
        
        # Verify entities were created
        collection_info = await self.client.get_collection_info(actual_collection_name)
        entity_count = collection_info.get("points_count", 0) if collection_info else 0
        print(f"✅ CREATED collection {actual_collection_name} with {entity_count} entities")
        
        # Track for cleanup
        self.__class__.shared_collections.add(actual_collection_name)
        return actual_collection_name
    
    async def create_integrator_for_repo(self, repo_name: str) -> EntityLifecycleIntegrator:
        """Create EntityLifecycleIntegrator for a repository"""
        repo_info = self.repos[repo_name]
        repo_path = repo_info["path"]
        
        if not repo_path.exists():
            raise pytest.skip(f"Repository {repo_name} not available")
        
        # Get or create collection
        collection_name = await self.get_or_create_entity_collection(repo_name)
        
        # Create CollectionManager for the project
        base_project_name = f"test-entity-{repo_name}"
        collection_manager = CollectionManager(project_name=base_project_name)
        
        # Create integrator with CollectionManager
        integrator = EntityLifecycleIntegrator(
            storage_client=self.client,
            project_path=repo_path,
            collection_name=collection_name,
            enable_real_time_sync=True,
            batch_size=25,
            collection_manager=collection_manager
        )
        
        return integrator
    
    def create_test_file(self, repo_name: str, filename: str, content: str) -> Path:
        """Create a test file in the repository and track it for cleanup"""
        repo_path = self.repos[repo_name]["path"]
        test_file = repo_path / filename
        
        # Write content
        test_file.write_text(content, encoding='utf-8')
        
        # Track for cleanup
        if repo_name not in self.__class__.created_files:
            self.__class__.created_files[repo_name] = []
        self.__class__.created_files[repo_name].append(test_file)
        
        return test_file

    @pytest.mark.asyncio
    async def test_bulk_entity_creation_python_repo(self):
        """Test bulk entity creation with real Python repository"""
        integrator = await self.create_integrator_for_repo("python-small")
        
        # Create some test Python files
        test_files = []
        
        # Create a simple Python module  
        python_content = '''
def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two integers."""
    return a + b

class DataProcessor:
    """Process and analyze data."""
    
    def __init__(self, data: list):
        self.data = data
    
    def process(self) -> dict:
        """Process the data and return results."""
        return {
            "count": len(self.data),
            "sum": sum(self.data) if self.data else 0
        }

GLOBAL_CONFIG = {
    "version": "1.0.0",
    "debug": True
}
'''
        test_file = self.create_test_file("python-small", "test_entity_creation.py", python_content)
        test_files.append(test_file)
        
        # Create another Python file with classes
        class_content = '''
from typing import Optional, List

class UserManager:
    """Manage user operations."""
    
    def create_user(self, name: str, email: str) -> dict:
        """Create a new user."""
        return {"name": name, "email": email, "id": hash(email)}
    
    def find_user(self, user_id: int) -> Optional[dict]:
        """Find a user by ID."""
        # Implementation would go here
        return None
    
    async def async_operation(self) -> List[str]:
        """Perform an async operation."""
        return ["result1", "result2"]
'''
        test_file2 = self.create_test_file("python-small", "test_user_manager.py", class_content)
        test_files.append(test_file2)
        
        # Test bulk entity creation
        result = await integrator.bulk_entity_create(
            file_paths=test_files,
            progress_callback=lambda processed, total, data: print(f"Progress: {processed}/{total} - {data.get('entities_found', 0)} entities")
        )
        
        # Verify result
        assert result.success, f"Bulk entity creation failed: {result.error_message}"
        assert result.entities_created > 0, "Should have created entities"
        assert result.entities_affected > 0, "Should have affected entities"
        assert result.operation_time_ms > 0, "Should have measurable operation time"
        
        print(f"Created {result.entities_created} entities in {result.operation_time_ms:.1f}ms")
        print(f"Metadata: {result.metadata}")
        
        # Verify entities exist in collection
        search_results = await self.client.search_hybrid(
            collection_name=integrator.collection_name,
            query="calculate_sum",
            limit=5
        )
        
        assert len(search_results) > 0, "Should find the created function"
        
        # Check if we can find our specific entities
        found_entities = []
        for result in search_results:
            entity_name = result.point.payload.get("entity_name", "")
            if entity_name in ["calculate_sum", "DataProcessor", "UserManager"]:
                found_entities.append(entity_name)
        
        assert len(found_entities) > 0, f"Should find our created entities, found: {found_entities}"

    @pytest.mark.asyncio
    async def test_entity_modification_and_sync_javascript_repo(self):
        """Test entity modification and synchronization with real JavaScript repository"""
        integrator = await self.create_integrator_for_repo("javascript-small")
        
        # Create initial JavaScript file
        initial_content = '''
/**
 * Calculate the total price with tax
 * @param {number} price - Base price
 * @param {number} taxRate - Tax rate (0.1 for 10%)
 * @returns {number} Total price with tax
 */
function calculateTotal(price, taxRate = 0.1) {
    return price * (1 + taxRate);
}

class ShoppingCart {
    constructor() {
        this.items = [];
        this.discountCode = null;
    }
    
    /**
     * Add item to cart
     * @param {Object} item - Item to add
     */
    addItem(item) {
        this.items.push(item);
    }
    
    getTotal() {
        return this.items.reduce((sum, item) => sum + item.price, 0);
    }
}

const DEFAULT_CONFIG = {
    currency: 'USD',
    locale: 'en-US'
};
'''
        test_file = self.create_test_file("javascript-small", "shopping_cart.js", initial_content)
        
        # Initial entity creation
        create_result = await integrator.bulk_entity_create([test_file])
        assert create_result.success, "Initial creation should succeed"
        initial_entities = create_result.entities_created
        
        print(f"Initial creation: {initial_entities} entities")
        
        # Modify the file - add new method, modify existing, remove something
        modified_content = '''
/**
 * Calculate the total price with tax and discount
 * @param {number} price - Base price  
 * @param {number} taxRate - Tax rate (0.1 for 10%)
 * @param {number} discount - Discount amount
 * @returns {number} Total price with tax and discount
 */
function calculateTotal(price, taxRate = 0.1, discount = 0) {
    return price * (1 + taxRate) - discount;
}

/**
 * New utility function for formatting currency
 * @param {number} amount - Amount to format
 * @returns {string} Formatted currency string
 */
function formatCurrency(amount) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(amount);
}

class ShoppingCart {
    constructor() {
        this.items = [];
        this.discountCode = null;
        this.taxRate = 0.1; // New property
    }
    
    /**
     * Add item to cart with validation
     * @param {Object} item - Item to add
     */
    addItem(item) {
        if (!item || !item.price) {
            throw new Error('Invalid item');
        }
        this.items.push(item);
    }
    
    getTotal() {
        return this.items.reduce((sum, item) => sum + item.price, 0);
    }
    
    /**
     * New method: Get total with tax
     * @returns {number} Total with tax
     */
    getTotalWithTax() {
        return this.getTotal() * (1 + this.taxRate);
    }
    
    /**
     * New method: Apply discount code
     * @param {string} code - Discount code
     */
    applyDiscount(code) {
        this.discountCode = code;
    }
}

// Modified constant
const DEFAULT_CONFIG = {
    currency: 'USD',
    locale: 'en-US',
    taxRate: 0.1 // New field
};

// New constant
const DISCOUNT_CODES = {
    'SAVE10': 0.1,
    'SAVE20': 0.2
};
'''
        
        # Write modified content
        test_file.write_text(modified_content, encoding='utf-8')
        
        # Test bulk entity update with change detection
        update_result = await integrator.bulk_entity_update(
            file_paths=[test_file],
            detect_changes=True,
            progress_callback=lambda processed, total, data: print(f"Update progress: {processed}/{total} - {data}")
        )
        
        assert update_result.success, f"Bulk entity update failed: {update_result.error_message}"
        
        print(f"Update results: created={update_result.entities_created}, "
              f"updated={update_result.entities_updated}, deleted={update_result.entities_deleted}")
        
        # Verify we can find the new entities
        search_results = await self.client.search_hybrid(
            collection_name=integrator.collection_name,
            query="formatCurrency",
            limit=5
        )
        
        assert len(search_results) > 0, "Should find the new formatCurrency function"
        
        # Verify we can find modified entities
        search_results = await self.client.search_hybrid(
            collection_name=integrator.collection_name,
            query="getTotalWithTax",
            limit=5
        )
        
        assert len(search_results) > 0, "Should find the new getTotalWithTax method"

    @pytest.mark.asyncio 
    async def test_file_deletion_and_cascade_operations_go_repo(self):
        """Test file deletion with cascade operations using real Go repository"""
        integrator = await self.create_integrator_for_repo("go-small")
        
        # Create Go files with dependencies
        main_content = '''
package main

import (
    "fmt"
    "net/http"
)

// Server represents our HTTP server
type Server struct {
    port string
    handlers map[string]http.HandlerFunc
}

// NewServer creates a new server instance  
func NewServer(port string) *Server {
    return &Server{
        port: port,
        handlers: make(map[string]http.HandlerFunc),
    }
}

// RegisterHandler registers a new HTTP handler
func (s *Server) RegisterHandler(path string, handler http.HandlerFunc) {
    s.handlers[path] = handler
}

// Start starts the HTTP server
func (s *Server) Start() error {
    mux := http.NewServeMux()
    for path, handler := range s.handlers {
        mux.HandleFunc(path, handler)
    }
    
    fmt.Printf("Server starting on port %s\\n", s.port)
    return http.ListenAndServe(":"+s.port, mux)
}

const DefaultPort = "8080"
'''
        
        handlers_content = '''
package main

import (
    "encoding/json"
    "net/http"
)

// Response represents an API response  
type Response struct {
    Message string `json:"message"`
    Status  int    `json:"status"`
}

// HealthHandler handles health check requests
func HealthHandler(w http.ResponseWriter, r *http.Request) {
    response := Response{
        Message: "OK",
        Status:  200,
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(response)
}

// NotFoundHandler handles 404 requests
func NotFoundHandler(w http.ResponseWriter, r *http.Request) {
    response := Response{
        Message: "Not Found", 
        Status:  404,
    }
    
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(http.StatusNotFound)
    json.NewEncoder(w).Encode(response)
}

var GlobalConfig = map[string]interface{}{
    "debug": true,
    "version": "1.0.0",
}
'''
        
        main_file = self.create_test_file("go-small", "test_server.go", main_content)
        handlers_file = self.create_test_file("go-small", "test_handlers.go", handlers_content)
        
        # Create entities initially
        create_result = await integrator.bulk_entity_create([main_file, handlers_file])
        assert create_result.success, "Initial creation should succeed"
        
        initial_entities = create_result.entities_created
        print(f"Created {initial_entities} entities from {len([main_file, handlers_file])} files")
        
        # Verify entities exist before deletion
        search_results = await self.client.search_hybrid(
            collection_name=integrator.collection_name,
            query="HealthHandler",
            limit=5
        )
        assert len(search_results) > 0, "Should find HealthHandler before deletion"
        

        # Test file deletion with cascade
        delete_result = await integrator.bulk_entity_delete(
            file_paths=[handlers_file],  # Delete the handlers file
            cascade_relationships=True
        )
        
        assert delete_result.success, f"Bulk entity delete failed: {delete_result.error_message}"
        assert delete_result.entities_deleted > 0, "Should have deleted entities"
        
        print(f"Deleted {delete_result.entities_deleted} entities in {delete_result.operation_time_ms:.1f}ms")
        
        # Verify entities are gone
        search_results = await self.client.search_hybrid(
            collection_name=integrator.collection_name, 
            query="HealthHandler",
            limit=5
        )
        
        # Should not find HealthHandler anymore (or find much fewer results)
        handler_results = [r for r in search_results if "HealthHandler" in r.point.payload.get("entity_name", "")]
        assert len(handler_results) == 0, "HealthHandler should be deleted"
        
        # But should still find entities from the main file
        search_results = await self.client.search_hybrid(
            collection_name=integrator.collection_name,
            query="NewServer", 
            limit=5
        )
        assert len(search_results) > 0, "Should still find NewServer from main file"

    @pytest.mark.asyncio
    async def test_atomic_entity_replacement_typescript_repo(self):
        """Test atomic entity replacement with real TypeScript repository"""
        integrator = await self.create_integrator_for_repo("typescript-small")
        
        # Create initial TypeScript file
        initial_content = '''
interface User {
    id: number;
    name: string;
    email: string;
}

interface Product {
    id: number;
    title: string;
    price: number;
}

class UserService {
    private users: User[] = [];
    
    constructor() {
        this.users = [];
    }
    
    addUser(user: Omit<User, 'id'>): User {
        const newUser: User = {
            ...user,
            id: this.users.length + 1
        };
        this.users.push(newUser);
        return newUser;
    }
    
    getUser(id: number): User | undefined {
        return this.users.find(user => user.id === id);
    }
    
    getAllUsers(): User[] {
        return [...this.users];
    }
}

export const DEFAULT_USER: User = {
    id: 0,
    name: 'Default',
    email: 'default@example.com'
};

export { UserService, User, Product };
'''
        
        test_file = self.create_test_file("typescript-small", "user_service.ts", initial_content)
        
        # Create initial entities
        create_result = await integrator.bulk_entity_create([test_file])
        assert create_result.success, "Initial creation should succeed"
        
        # Search for initial entities to verify they exist
        search_results = await self.client.search_hybrid(
            collection_name=integrator.collection_name,
            query="UserService",
            limit=10
        )
        
        initial_user_service_results = len([r for r in search_results if "UserService" in r.point.payload.get("entity_name", "")])
        assert initial_user_service_results > 0, "Should find UserService initially"
        
        # Create completely new content (atomic replacement)
        replacement_content = '''
interface Customer {
    customerId: string;
    fullName: string;
    emailAddress: string;
    phone?: string;
    address: Address;
}

interface Address {
    street: string;
    city: string;
    country: string;
    postalCode: string;
}

interface Order {
    orderId: string;
    customerId: string;
    items: OrderItem[];
    totalAmount: number;
    orderDate: Date;
}

interface OrderItem {
    productId: string;
    quantity: number;
    unitPrice: number;
}

class CustomerService {
    private customers: Map<string, Customer> = new Map();
    
    constructor() {
        this.customers = new Map();
    }
    
    async createCustomer(customerData: Omit<Customer, 'customerId'>): Promise<Customer> {
        const customerId = this.generateCustomerId();
        const customer: Customer = {
            ...customerData,
            customerId
        };
        
        this.customers.set(customerId, customer);
        return customer;
    }
    
    async findCustomer(customerId: string): Promise<Customer | null> {
        return this.customers.get(customerId) || null;
    }
    
    async updateCustomerAddress(customerId: string, address: Address): Promise<boolean> {
        const customer = this.customers.get(customerId);
        if (!customer) return false;
        
        customer.address = address;
        this.customers.set(customerId, customer);
        return true;
    }
    
    private generateCustomerId(): string {
        return `CUST_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }
    
    async getCustomerOrders(customerId: string): Promise<Order[]> {
        // Implementation would fetch from order service
        return [];
    }
}

class OrderService {
    private orders: Map<string, Order> = new Map();
    
    async createOrder(orderData: Omit<Order, 'orderId' | 'orderDate'>): Promise<Order> {
        const order: Order = {
            ...orderData,
            orderId: this.generateOrderId(),
            orderDate: new Date()
        };
        
        this.orders.set(order.orderId, order);
        return order;
    }
    
    private generateOrderId(): string {
        return `ORDER_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }
}

export const BUSINESS_CONFIG = {
    maxOrderItems: 50,
    defaultCountry: 'US',
    supportedCurrencies: ['USD', 'EUR', 'GBP']
};

export { CustomerService, OrderService, Customer, Order, Address };
'''
        
        # Write replacement content
        test_file.write_text(replacement_content, encoding='utf-8')
        
        # Perform atomic entity replacement
        replacement_result = await integrator.atomic_entity_replacement(
            file_path=test_file
        )
        
        assert replacement_result.success, f"Atomic replacement failed: {replacement_result.error_message}"
        
        print(f"Atomic replacement: created={replacement_result.entities_created}, "
              f"deleted={replacement_result.entities_deleted} in {replacement_result.operation_time_ms:.1f}ms")
        
        # Verify old entities are gone
        search_results = await self.client.search_hybrid(
            collection_name=integrator.collection_name,
            query="UserService",
            limit=10
        )
        
        user_service_results = [r for r in search_results if "UserService" in r.point.payload.get("entity_name", "")]
        assert len(user_service_results) == 0, "Old UserService should be gone after atomic replacement"
        
        # Verify new entities exist
        search_results = await self.client.search_hybrid(
            collection_name=integrator.collection_name,
            query="CustomerService",
            limit=10
        )
        
        customer_service_results = [r for r in search_results if "CustomerService" in r.point.payload.get("entity_name", "")]
        assert len(customer_service_results) > 0, "New CustomerService should exist after atomic replacement"
        
        # Verify new interfaces exist
        search_results = await self.client.search_hybrid(
            collection_name=integrator.collection_name,
            query="Customer",
            limit=10
        )
        
        customer_interface_results = [r for r in search_results if r.point.payload.get("entity_type") == "interface"]
        assert len(customer_interface_results) > 0, "New Customer interface should exist"

    @pytest.mark.asyncio
    async def test_real_time_sync_integration(self):
        """Test real-time synchronization integration"""
        integrator = await self.create_integrator_for_repo("python-small")
        
        # Enable real-time sync
        sync_enabled = await integrator.enable_sync()
        if not sync_enabled:
            pytest.skip("Real-time sync not available or failed to enable")
        
        # Get initial status
        initial_status = integrator.get_integration_status()
        print(f"Initial integration status: {initial_status['integrator_info']}")
        
        # Create a file that should trigger real-time sync
        realtime_content = '''
def realtime_function():
    """This function was created to test real-time sync."""
    return "real-time sync working"

class RealtimeClass:
    """Test class for real-time synchronization."""
    
    def method_one(self):
        return "method one"
    
    def method_two(self):
        return "method two"
'''
        
        test_file = self.create_test_file("python-small", "realtime_test.py", realtime_content)
        
        # Give a moment for any real-time processing
        import asyncio
        await asyncio.sleep(1)
        
        # Manually trigger entity creation (since we may not have full real-time monitoring in tests)
        create_result = await integrator.bulk_entity_create([test_file])
        assert create_result.success, "Real-time entity creation should succeed"
        
        # Test that entities can be found
        search_results = await self.client.search_hybrid(
            collection_name=integrator.collection_name,
            query="realtime_function",
            limit=5
        )
        
        assert len(search_results) > 0, "Should find real-time created entities"
        
        # Disable real-time sync
        sync_disabled = await integrator.disable_real_time_sync()
        assert sync_disabled, "Should be able to disable real-time sync"
        
        # Get final status
        final_status = integrator.get_integration_status()
        assert final_status["performance"]["total_operations"] > 0, "Should have recorded operations"

    @pytest.mark.asyncio
    async def test_entity_mapping_rebuild_and_health(self):
        """Test entity mapping rebuild and health monitoring"""
        integrator = await self.create_integrator_for_repo("javascript-small")
        
        # Create some entities first
        test_content = '''
function testFunction() {
    return "test";
}

class TestClass {
    constructor() {
        this.value = 42;
    }
    
    getValue() {
        return this.value;
    }
}

const TEST_CONSTANT = "test value";
'''
        
        test_file = self.create_test_file("javascript-small", "mapping_test.js", test_content)
        
        # Create entities  
        create_result = await integrator.bulk_entity_create([test_file])
        assert create_result.success, "Entity creation should succeed"
        
        # Test mapping rebuild
        rebuild_result = await integrator.rebuild_entity_mappings()
        assert rebuild_result.success, f"Mapping rebuild failed: {rebuild_result.error_message}"
        assert rebuild_result.entities_affected > 0, "Should have rebuilt mappings for entities"
        
        print(f"Rebuilt mappings for {rebuild_result.entities_affected} entities in {rebuild_result.operation_time_ms:.1f}ms")
        
        # Check integration status and health
        status = integrator.get_integration_status()
        
        # Verify status structure
        assert "integrator_info" in status
        assert "performance" in status  
        assert "mapping_state" in status
        assert "component_status" in status
        
        # Verify mapping state looks healthy
        mapping_state = status["mapping_state"]
        assert mapping_state["total_entities"] > 0, "Should have entities in mapping"
        assert mapping_state["health_score"] > 0.5, "Health score should be reasonable"
        assert mapping_state["last_sync_time"] is not None, "Should have sync time"
        
        # Verify component status
        component_status = status["component_status"]
        assert "lifecycle_manager" in component_status
        assert "entity_scanner" in component_status
        
        print(f"Integration health: {mapping_state['health_score']:.2f}")
        print(f"Total entities tracked: {mapping_state['total_entities']}")
        print(f"Performance: {status['performance']['total_operations']} operations, "
              f"avg {status['performance']['average_operation_time_ms']:.1f}ms")

    @pytest.mark.asyncio
    async def test_cross_language_entity_operations(self):
        """Test entity operations across multiple programming languages"""
        
        # Test with multiple languages
        languages = ["python-small", "javascript-small"]
        integrators = {}
        
        # Create integrators for each language
        for lang in languages:
            try:
                integrators[lang] = await self.create_integrator_for_repo(lang)
            except Exception as e:
                print(f"Skipping {lang}: {e}")
                continue
        
        if len(integrators) < 2:
            pytest.skip("Need at least 2 language repositories for cross-language test")
        
        # Create test files in each language
        test_files = {}
        
        # Python file
        if "python-small" in integrators:
            python_content = '''
def cross_lang_python_function(data):
    """Process data in Python."""
    return {"language": "python", "data": data}

class PythonProcessor:
    def process(self, items):
        return [item.upper() for item in items]
'''
            test_files["python-small"] = self.create_test_file("python-small", "cross_lang_test.py", python_content)
        
        # JavaScript file
        if "javascript-small" in integrators:
            js_content = '''
/**
 * Process data in JavaScript
 * @param {any} data - Data to process
 * @returns {Object} Processed result
 */
function crossLangJsFunction(data) {
    return { language: "javascript", data: data };
}

class JsProcessor {
    process(items) {
        return items.map(item => item.toLowerCase());
    }
}

const JS_CONFIG = {
    version: "1.0",
    enabled: true
};
'''
            test_files["javascript-small"] = self.create_test_file("javascript-small", "cross_lang_test.js", js_content)
        
        # Create multiple test files for each language to get 100+ entities for realistic performance testing
        for lang in integrators.keys():
            if lang in test_files:
                # Create 20 additional files for this language to get more entities
                additional_files = []
                for i in range(20):
                    if lang == "python-small":
                        content = f'''
def function_{i}():
    """Function {i} for performance testing."""
    def inner_function_{i}(data):
        return data * {i}
    
    class TestClass{i}:
        def __init__(self):
            self.value_{i} = {i}
        
        def method_{i}(self, x):
            return x + {i}
        
        def calculate_{i}(self, a, b):
            return a * b + {i}

class UtilityClass{i}:
    def process_{i}(self, items):
        return [item for item in items if len(str(item)) > {i % 5}]

CONSTANT_{i} = {i * 100}
CONFIG_{i} = {{"enabled": True, "value": {i}}}
'''
                    else:  # javascript-small
                        content = f'''
function testFunction{i}() {{
    // Function {i} for performance testing
    function innerFunction{i}(data) {{
        return data * {i};
    }}
    
    function calculateValue{i}(a, b) {{
        return a + b + {i};
    }}
    
    return {{
        inner: innerFunction{i},
        calc: calculateValue{i},
        value: {i}
    }};
}}

class TestClass{i} {{
    constructor() {{
        this.id = {i};
        this.data = [];
    }}
    
    process{i}(items) {{
        return items.filter(x => x > {i});
    }}
    
    transform{i}(data) {{
        return data.map(x => x + {i});
    }}
}}

const CONFIG_{i} = {{
    id: {i},
    enabled: true,
    threshold: {i * 10}
}};

const UTILS_{i} = {{
    helper{i}: (x) => x * {i},
    validator{i}: (x) => x > {i}
}};
'''
                    
                    additional_files.append(self.create_test_file(lang, f"perf_test_{i}.{'py' if lang == 'python-small' else 'js'}", content))
                
                # Add the additional files to test_files
                if isinstance(test_files[lang], list):
                    test_files[lang].extend(additional_files)
                else:
                    test_files[lang] = [test_files[lang]] + additional_files

        # Create entities in each language using multiple files for realistic performance testing
        creation_results = {}
        for lang, integrator in integrators.items():
            if lang in test_files:
                files_to_process = test_files[lang] if isinstance(test_files[lang], list) else [test_files[lang]]
                
                result = await integrator.bulk_entity_create(files_to_process)
                creation_results[lang] = result
                assert result.success, f"Creation failed for {lang}: {result.error_message}"
                
                entities_per_second = result.entities_created / (result.operation_time_ms / 1000) if result.operation_time_ms > 0 else 0
                print(f"{lang}: Created {result.entities_created} entities from {len(files_to_process)} files ({entities_per_second:.1f} entities/sec)")
        
        # Search across languages
        all_collections = list(integrators.values())
        search_terms = ["cross", "process", "function"]
        
        for term in search_terms[:1]:  # Test one term to avoid too many searches
            found_languages = []
            
            for lang, integrator in integrators.items():
                search_results = await self.client.search_hybrid(
                    collection_name=integrator.collection_name,
                    query=term,
                    limit=5
                )
                
                if search_results:
                    found_languages.append(lang)
                    print(f"Found '{term}' in {lang}: {len(search_results)} results")
            
            # Should find results in multiple languages
            if len(integrators) > 1:
                assert len(found_languages) > 0, f"Should find '{term}' in at least one language"
        
        # Test performance across languages
        total_operations = 0
        total_time = 0.0
        
        for lang, integrator in integrators.items():
            status = integrator.get_integration_status()
            ops = status["performance"]["total_operations"]
            time_ms = status["performance"]["total_operations_time_ms"]
            avg_time = status["performance"]["average_operation_time_ms"]
            
            total_operations += ops
            total_time += time_ms
            
            print(f"{lang}: {ops} operations, {time_ms:.1f}ms total (avg: {avg_time:.1f}ms per operation)")
        
        # Calculate total entities created across all languages
        total_entities = sum(result.entities_created for result in creation_results.values())
        total_processing_time = sum(result.operation_time_ms for result in creation_results.values())
        
        avg_time_per_op = total_time / total_operations if total_operations > 0 else 0
        avg_entities_per_sec = (total_entities / (total_processing_time / 1000)) if total_processing_time > 0 else 0
        
        print(f"Cross-language performance: {total_operations} operations, {avg_time_per_op:.1f}ms avg per operation")
        print(f"Entity processing performance: {total_entities} entities, {avg_entities_per_sec:.1f} entities/sec")
        
        # Performance assertion based on entities/sec (accounting for model warmup overhead)
        # Target: ≥30 entities/sec for bulk operations (includes ~3200ms warmup cost)
        assert avg_entities_per_sec >= 30, f"Entity processing rate too low: {avg_entities_per_sec:.1f} entities/sec (target: ≥30)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])