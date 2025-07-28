"""
Tests for Rust parser with comprehensive entity and relation extraction.

Tests the RustParser implementation to ensure correct extraction of:
- Modules and uses (imports)
- Functions with parameters and return types
- Structs with fields (regular, tuple, unit)
- Enums with variants
- Traits with method signatures
- Impl blocks (trait implementations and inherent impls)
- Constants and static variables
- Type aliases
- Macros
- Relations between entities
"""

import pytest
from pathlib import Path

from core.parser.rust_parser import RustParser
from core.parser.registry import parser_registry
from core.models.entities import (
    Entity, EntityType, Relation, RelationType, 
    SourceLocation, Visibility
)


@pytest.fixture
def rust_parser():
    """Create a Rust parser instance for testing"""
    return RustParser()


@pytest.fixture
def sample_rust_code():
    """Sample Rust code for testing entity extraction"""
    return '''// Sample Rust code for testing
use std::collections::HashMap;
use std::fs::File;
use crate::utils::helpers;

// Constants
const MAX_SIZE: usize = 1024;
const PI: f64 = 3.14159;

// Static variables
static mut COUNTER: u32 = 0;
static CONFIG: &str = "default";

// Type alias
type UserId = u64;
type ResultType<T> = Result<T, String>;

// Struct definitions
#[derive(Debug, Clone)]
pub struct User {
    pub id: UserId,
    pub name: String,
    email: String,
    age: Option<u8>,
}

// Tuple struct
pub struct Point(pub f64, pub f64);

// Unit struct
pub struct EmptyStruct;

// Enum definition
#[derive(Debug, PartialEq)]
pub enum Status {
    Active,
    Inactive,
    Pending { reason: String },
    Suspended(String),
}

// Trait definition
pub trait Drawable {
    fn draw(&self);
    fn area(&self) -> f64;
    
    // Default implementation
    fn description(&self) -> String {
        "A drawable object".to_string()
    }
}

// Functions
pub fn create_user(name: String, email: String) -> User {
    User {
        id: generate_id(),
        name,
        email,
        age: None,
    }
}

pub async fn fetch_data(url: &str) -> Result<String, String> {
    // Async function implementation
    Ok("data".to_string())
}

pub unsafe fn raw_memory_access(ptr: *mut u8) {
    // Unsafe function
}

// Impl blocks
impl User {
    pub fn new(name: String, email: String) -> Self {
        Self {
            id: generate_id(),
            name,
            email,
            age: None,
        }
    }
    
    pub fn set_age(&mut self, age: u8) {
        self.age = Some(age);
    }
    
    fn validate_email(&self) -> bool {
        self.email.contains('@')
    }
}

impl Drawable for User {
    fn draw(&self) {
        println!("Drawing user: {}", self.name);
    }
    
    fn area(&self) -> f64 {
        0.0 // Users don't have area
    }
}

// Helper function
fn generate_id() -> UserId {
    42
}

// Module declaration
pub mod utils {
    pub fn helper_function() -> i32 {
        100
    }
}

// Macro definition
macro_rules! debug_print {
    ($($arg:tt)*) => {
        println!("DEBUG: {}", format!($($arg)*));
    };
}'''


@pytest.fixture
def complex_rust_code():
    """More complex Rust code with advanced features"""
    return '''use std::sync::{Arc, Mutex};
use std::thread;
use tokio::time::{sleep, Duration};

// Generic struct
pub struct Container<T> {
    items: Vec<T>,
    capacity: usize,
}

// Generic enum
pub enum Either<L, R> {
    Left(L),
    Right(R),
}

// Complex trait with associated types
pub trait Iterator {
    type Item;
    
    fn next(&mut self) -> Option<Self::Item>;
    
    fn collect<B: FromIterator<Self::Item>>(self) -> B
    where
        Self: Sized,
    {
        FromIterator::from_iter(self)
    }
}

// Trait with lifetime parameters
pub trait Validator<'a> {
    fn validate(&self, input: &'a str) -> bool;
}

// Complex impl with generics
impl<T: Clone> Container<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            items: Vec::new(),
            capacity,
        }
    }
    
    pub fn add(&mut self, item: T) -> Result<(), &'static str> {
        if self.items.len() >= self.capacity {
            Err("Container is full")
        } else {
            self.items.push(item);
            Ok(())
        }
    }
}

// Async trait implementation
impl Container<String> {
    pub async fn process_all(&self) -> Vec<String> {
        let mut results = Vec::new();
        for item in &self.items {
            sleep(Duration::from_millis(10)).await;
            results.push(item.clone());
        }
        results
    }
}

// Function with complex types
pub fn process_data<T, E>(
    data: Vec<T>,
    processor: impl Fn(T) -> Result<T, E>,
) -> Result<Vec<T>, E>
where
    T: Clone,
    E: std::fmt::Debug,
{
    data.into_iter()
        .map(processor)
        .collect()
}

// Const generic function
pub fn create_array<const N: usize>() -> [i32; N] {
    [0; N]
}'''


class TestRustParser:
    """Test suite for Rust parser"""
    
    def test_parser_registration(self, rust_parser):
        """Test that Rust parser is properly registered"""
        assert rust_parser.language == "rust"
        assert rust_parser.can_parse(Path("test.rs"))
        assert not rust_parser.can_parse(Path("test.py"))
        
        # Test registry integration
        registered_parser = parser_registry.get_parser("rust")
        assert registered_parser is not None
        assert isinstance(registered_parser, RustParser)
        
        # Test file extension mapping
        file_parser = parser_registry.get_parser_for_file(Path("example.rs"))
        assert file_parser is not None
        assert isinstance(file_parser, RustParser)
    
    def test_extract_modules(self, rust_parser, tmp_path, sample_rust_code):
        """Test module declaration extraction"""
        test_file = tmp_path / "test.rs"
        test_file.write_text(sample_rust_code)
        
        result = rust_parser.parse_file(test_file)
        entities = result.entities
        
        # Filter modules
        modules = [e for e in entities if e.entity_type == EntityType.MODULE]
        assert len(modules) >= 1
        
        # Check utils module
        utils_module = next((m for m in modules if m.name == "utils"), None)
        assert utils_module is not None
        assert utils_module.visibility == Visibility.PUBLIC
        assert utils_module.signature == "mod utils"
        assert utils_module.metadata["module_name"] == "utils"
        assert utils_module.metadata["is_inline_module"] == True
    
    def test_extract_uses(self, rust_parser, tmp_path, sample_rust_code):
        """Test use statement extraction"""
        test_file = tmp_path / "test.rs"
        test_file.write_text(sample_rust_code)
        
        result = rust_parser.parse_file(test_file)
        entities = result.entities
        
        # Filter uses
        uses = [e for e in entities if e.entity_type == EntityType.IMPORT]
        assert len(uses) >= 3
        
        # Check specific imports
        import_names = [use.name for use in uses]
        import_paths = [use.qualified_name for use in uses]
        
        assert "HashMap" in import_names
        assert "File" in import_names
        assert "helpers" in import_names
        
        assert "std::collections::HashMap" in import_paths
        assert "std::fs::File" in import_paths
        assert "crate::utils::helpers" in import_paths
        
        # Check standard library detection
        hashmap_import = next((u for u in uses if u.name == "HashMap"), None)
        assert hashmap_import is not None
        assert hashmap_import.metadata["is_standard_library"] == True
        
        helpers_import = next((u for u in uses if u.name == "helpers"), None)
        assert helpers_import is not None
        assert helpers_import.metadata["is_standard_library"] == False
    
    def test_extract_structs(self, rust_parser, tmp_path, sample_rust_code):
        """Test struct definition extraction"""
        test_file = tmp_path / "test.rs"
        test_file.write_text(sample_rust_code)
        
        result = rust_parser.parse_file(test_file)
        entities = result.entities
        
        # Filter structs
        structs = [e for e in entities if e.entity_type == EntityType.STRUCT]
        assert len(structs) >= 3
        
        struct_names = [s.name for s in structs]
        assert "User" in struct_names
        assert "Point" in struct_names
        assert "EmptyStruct" in struct_names
        
        # Check User struct
        user_struct = next((s for s in structs if s.name == "User"), None)
        assert user_struct is not None
        assert user_struct.visibility == Visibility.PUBLIC
        assert user_struct.signature == "struct User"
        assert user_struct.metadata["struct_name"] == "User"
        assert user_struct.metadata["is_tuple_struct"] == False
        assert user_struct.metadata["is_unit_struct"] == False
        
        # Check fields
        fields = user_struct.metadata["fields"]
        assert len(fields) >= 4
        field_names = [f["name"] for f in fields]
        assert "id" in field_names
        assert "name" in field_names
        assert "email" in field_names
        assert "age" in field_names
        
        # Check Point tuple struct
        point_struct = next((s for s in structs if s.name == "Point"), None)
        assert point_struct is not None
        assert point_struct.metadata["is_tuple_struct"] == True
        
        # Check EmptyStruct unit struct
        empty_struct = next((s for s in structs if s.name == "EmptyStruct"), None)
        assert empty_struct is not None
        assert empty_struct.metadata["is_unit_struct"] == True
    
    def test_extract_enums(self, rust_parser, tmp_path, sample_rust_code):
        """Test enum definition extraction"""
        test_file = tmp_path / "test.rs"
        test_file.write_text(sample_rust_code)
        
        result = rust_parser.parse_file(test_file)
        entities = result.entities
        
        # Filter enums
        enums = [e for e in entities if e.entity_type == EntityType.ENUM]
        assert len(enums) >= 1
        
        # Check Status enum
        status_enum = next((e for e in enums if e.name == "Status"), None)
        assert status_enum is not None
        assert status_enum.visibility == Visibility.PUBLIC
        assert status_enum.signature == "enum Status"
        assert status_enum.metadata["enum_name"] == "Status"
        
        # Check variants
        variants = status_enum.metadata["variants"]
        assert len(variants) >= 4
        variant_names = [v["name"] for v in variants]
        assert "Active" in variant_names
        assert "Inactive" in variant_names
        assert "Pending" in variant_names
        assert "Suspended" in variant_names
    
    def test_extract_traits(self, rust_parser, tmp_path, sample_rust_code):
        """Test trait definition extraction"""
        test_file = tmp_path / "test.rs"
        test_file.write_text(sample_rust_code)
        
        result = rust_parser.parse_file(test_file)
        entities = result.entities
        
        # Filter traits
        traits = [e for e in entities if e.entity_type == EntityType.TRAIT]
        assert len(traits) >= 1
        
        # Check Drawable trait
        drawable_trait = next((t for t in traits if t.name == "Drawable"), None)
        assert drawable_trait is not None
        assert drawable_trait.visibility == Visibility.PUBLIC
        assert drawable_trait.signature == "trait Drawable"
        assert drawable_trait.metadata["trait_name"] == "Drawable"
        
        # Check methods
        methods = drawable_trait.metadata["methods"]
        assert len(methods) >= 3
        method_names = [m["name"] for m in methods]
        assert "draw" in method_names
        assert "area" in method_names
        assert "description" in method_names
    
    def test_extract_functions(self, rust_parser, tmp_path, sample_rust_code):
        """Test function extraction"""
        test_file = tmp_path / "test.rs"
        test_file.write_text(sample_rust_code)
        
        result = rust_parser.parse_file(test_file)
        entities = result.entities
        
        # Filter functions
        functions = [e for e in entities if e.entity_type == EntityType.FUNCTION]
        assert len(functions) >= 4
        
        function_names = [f.name for f in functions]
        assert "create_user" in function_names
        assert "fetch_data" in function_names
        assert "raw_memory_access" in function_names
        assert "generate_id" in function_names
        
        # Check create_user function
        create_user_func = next((f for f in functions if f.name == "create_user"), None)
        assert create_user_func is not None
        assert create_user_func.visibility == Visibility.PUBLIC
        assert "pub fn create_user" in create_user_func.signature
        
        # Check parameters
        parameters = create_user_func.metadata["parameters"]
        assert len(parameters) == 2
        assert parameters[0]["name"] == "name"
        assert parameters[0]["type"] == "String"
        assert parameters[1]["name"] == "email"
        assert parameters[1]["type"] == "String"
        
        # Check fetch_data async function
        fetch_data_func = next((f for f in functions if f.name == "fetch_data"), None)
        assert fetch_data_func is not None
        assert fetch_data_func.is_async == True
        assert fetch_data_func.metadata["is_async"] == True
        assert "async" in fetch_data_func.signature
        
        # Check raw_memory_access unsafe function
        unsafe_func = next((f for f in functions if f.name == "raw_memory_access"), None)
        assert unsafe_func is not None
        assert unsafe_func.metadata["is_unsafe"] == True
        assert "unsafe" in unsafe_func.signature
    
    def test_extract_impls(self, rust_parser, tmp_path, sample_rust_code):
        """Test impl block extraction"""
        test_file = tmp_path / "test.rs"
        test_file.write_text(sample_rust_code)
        
        result = rust_parser.parse_file(test_file)
        entities = result.entities
        
        # Filter impls
        impls = [e for e in entities if e.entity_type == EntityType.IMPLEMENTATION]
        assert len(impls) >= 2
        
        # Check inherent impl
        user_impl = next((i for i in impls if i.name == "impl User"), None)
        assert user_impl is not None
        assert user_impl.metadata["impl_type"] == "User"
        assert user_impl.metadata["trait_name"] is None
        assert user_impl.metadata["is_trait_impl"] == False
        
        # Check methods
        methods = user_impl.metadata["methods"]
        assert len(methods) >= 3
        method_names = [m["name"] for m in methods]
        assert "new" in method_names
        assert "set_age" in method_names
        assert "validate_email" in method_names
        
        # Check trait impl
        drawable_impl = next((i for i in impls if "Drawable for User" in i.name), None)
        assert drawable_impl is not None
        assert drawable_impl.metadata["impl_type"] == "User"
        assert drawable_impl.metadata["trait_name"] == "Drawable"
        assert drawable_impl.metadata["is_trait_impl"] == True
    
    def test_extract_constants(self, rust_parser, tmp_path, sample_rust_code):
        """Test constant declaration extraction"""
        test_file = tmp_path / "test.rs"
        test_file.write_text(sample_rust_code)
        
        result = rust_parser.parse_file(test_file)
        entities = result.entities
        
        # Filter constants
        constants = [e for e in entities if e.entity_type == EntityType.CONSTANT]
        assert len(constants) >= 2
        
        constant_names = [c.name for c in constants]
        assert "MAX_SIZE" in constant_names
        assert "PI" in constant_names
        
        # Check MAX_SIZE constant
        max_size = next((c for c in constants if c.name == "MAX_SIZE"), None)
        assert max_size is not None
        assert max_size.visibility == Visibility.PRIVATE  # No pub keyword
        assert max_size.metadata["const_name"] == "MAX_SIZE"
        assert max_size.metadata["is_constant"] == True
        assert "const MAX_SIZE:" in max_size.signature
    
    def test_extract_static_vars(self, rust_parser, tmp_path, sample_rust_code):
        """Test static variable extraction"""
        test_file = tmp_path / "test.rs"
        test_file.write_text(sample_rust_code)
        
        result = rust_parser.parse_file(test_file)
        entities = result.entities
        
        # Filter static variables
        statics = [e for e in entities if e.entity_type == EntityType.VARIABLE and e.metadata.get("is_static")]
        assert len(statics) >= 2
        
        static_names = [s.name for s in statics]
        assert "COUNTER" in static_names
        assert "CONFIG" in static_names
        
        # Check COUNTER static
        counter = next((s for s in statics if s.name == "COUNTER"), None)
        assert counter is not None
        assert counter.metadata["is_mutable"] == True
        assert counter.metadata["is_static"] == True
        
        # Check CONFIG static
        config = next((s for s in statics if s.name == "CONFIG"), None)
        assert config is not None
        assert config.metadata["is_mutable"] == False
        assert config.metadata["is_static"] == True
    
    def test_extract_type_aliases(self, rust_parser, tmp_path, sample_rust_code):
        """Test type alias extraction"""
        test_file = tmp_path / "test.rs"
        test_file.write_text(sample_rust_code)
        
        result = rust_parser.parse_file(test_file)
        entities = result.entities
        
        # Filter type aliases
        types = [e for e in entities if e.entity_type == EntityType.TYPE]
        assert len(types) >= 2
        
        type_names = [t.name for t in types]
        assert "UserId" in type_names
        assert "ResultType" in type_names
        
        # Check UserId type
        user_id_type = next((t for t in types if t.name == "UserId"), None)
        assert user_id_type is not None
        assert user_id_type.visibility == Visibility.PRIVATE
        assert user_id_type.metadata["type_name"] == "UserId"
        assert user_id_type.metadata["is_alias"] == True
        assert "type UserId =" in user_id_type.signature
    
    def test_extract_macros(self, rust_parser, tmp_path, sample_rust_code):
        """Test macro definition extraction"""
        test_file = tmp_path / "test.rs"
        test_file.write_text(sample_rust_code)
        
        result = rust_parser.parse_file(test_file)
        entities = result.entities
        
        # Filter macros
        macros = [e for e in entities if e.entity_type == EntityType.MACRO]
        assert len(macros) >= 1
        
        # Check debug_print macro
        debug_macro = next((m for m in macros if m.name == "debug_print"), None)
        assert debug_macro is not None
        assert debug_macro.metadata["macro_name"] == "debug_print"
        assert debug_macro.signature == "macro_rules! debug_print"
    
    def test_visibility_detection(self, rust_parser, tmp_path):
        """Test Rust visibility detection"""
        content = '''// Public items
pub struct PublicStruct {
    pub public_field: String,
    private_field: i32,
}

pub fn public_function() {}

pub const PUBLIC_CONST: i32 = 42;

// Private items (default)
struct PrivateStruct {
    field: String,
}

fn private_function() {}

const PRIVATE_CONST: i32 = 42;

// Pub(crate) items
pub(crate) struct CrateStruct {}

pub(crate) fn crate_function() {}'''
        
        test_file = tmp_path / "test.rs"
        test_file.write_text(content)
        
        result = rust_parser.parse_file(test_file)
        entities = result.entities
        
        # Check public entities
        public_entities = [e for e in entities if e.visibility == Visibility.PUBLIC]
        public_names = [e.name for e in public_entities]
        
        assert "PublicStruct" in public_names
        assert "public_function" in public_names
        assert "PUBLIC_CONST" in public_names
        
        # Check private entities
        private_entities = [e for e in entities if e.visibility == Visibility.PRIVATE]
        private_names = [e.name for e in private_entities]
        
        assert "PrivateStruct" in private_names
        assert "private_function" in private_names
        assert "PRIVATE_CONST" in private_names
    
    def test_complex_generics(self, rust_parser, tmp_path, complex_rust_code):
        """Test parsing of complex generic structures"""
        test_file = tmp_path / "test.rs"
        test_file.write_text(complex_rust_code)
        
        result = rust_parser.parse_file(test_file)
        entities = result.entities
        
        # Should extract various entity types
        entity_types = [e.entity_type for e in entities]
        assert EntityType.STRUCT in entity_types
        assert EntityType.ENUM in entity_types
        assert EntityType.TRAIT in entity_types
        assert EntityType.FUNCTION in entity_types
        assert EntityType.IMPLEMENTATION in entity_types
        
        # Check specific entities
        entity_names = [e.name for e in entities]
        assert "Container" in entity_names
        assert "Either" in entity_names
        assert "Iterator" in entity_names
        assert "Validator" in entity_names
        assert "process_data" in entity_names
        assert "create_array" in entity_names
    
    def test_empty_file_parsing(self, rust_parser, tmp_path):
        """Test parsing empty Rust file"""
        empty_file = tmp_path / "empty.rs"
        empty_file.write_text("")
        
        result = rust_parser.parse_file(empty_file)
        
        assert result.success is True
        assert result.entity_count == 0
        assert result.relation_count == 0
    
    def test_metadata_completeness(self, rust_parser, tmp_path, sample_rust_code):
        """Test that extracted entities have complete metadata"""
        test_file = tmp_path / "test.rs"
        test_file.write_text(sample_rust_code)
        
        result = rust_parser.parse_file(test_file)
        entities = result.entities
        
        for entity in entities:
            # All entities should have basic metadata
            assert entity.id is not None
            assert entity.name is not None
            assert entity.entity_type is not None
            assert entity.location is not None
            assert entity.signature is not None
            assert entity.source_code is not None
            assert entity.source_hash is not None
            assert entity.visibility is not None
            assert entity.metadata is not None
            
            # All entities should have language metadata
            assert entity.metadata["language"] == "rust"
            assert "ast_node_type" in entity.metadata
            
            # Type-specific metadata checks
            if entity.entity_type == EntityType.STRUCT:
                assert "struct_name" in entity.metadata
                assert "fields" in entity.metadata
                assert "field_count" in entity.metadata
                assert "is_tuple_struct" in entity.metadata
                assert "is_unit_struct" in entity.metadata
            
            elif entity.entity_type == EntityType.FUNCTION:
                assert "function_name" in entity.metadata
                assert "parameters" in entity.metadata
                assert "is_async" in entity.metadata
                assert "is_unsafe" in entity.metadata
                assert "is_const" in entity.metadata
                assert "parameter_count" in entity.metadata
            
            elif entity.entity_type == EntityType.ENUM:
                assert "enum_name" in entity.metadata
                assert "variants" in entity.metadata
                assert "variant_count" in entity.metadata
            
            elif entity.entity_type == EntityType.TRAIT:
                assert "trait_name" in entity.metadata
                assert "methods" in entity.metadata
                assert "method_count" in entity.metadata
            
            elif entity.entity_type == EntityType.IMPLEMENTATION:
                assert "impl_type" in entity.metadata
                assert "is_trait_impl" in entity.metadata
                assert "methods" in entity.metadata
                assert "method_count" in entity.metadata
            
            elif entity.entity_type == EntityType.IMPORT:
                assert "import_path" in entity.metadata
                assert "imported_name" in entity.metadata
                assert "is_standard_library" in entity.metadata
                assert "is_external_crate" in entity.metadata