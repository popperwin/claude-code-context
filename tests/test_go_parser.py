"""
Tests for Go parser with comprehensive entity and relation extraction.

Tests the GoParser implementation to ensure correct extraction of:
- Packages and imports
- Functions and methods with receivers
- Structs with fields
- Interfaces with method signatures
- Variables and constants with types
- Type definitions and aliases
- Relations between entities
"""

import pytest
from pathlib import Path

from core.parser.go_parser import GoParser
from core.parser.registry import parser_registry
from core.models.entities import (
    Entity, EntityType, Relation, RelationType, 
    SourceLocation, Visibility
)


@pytest.fixture
def go_parser():
    """Create a Go parser instance for testing"""
    return GoParser()


@pytest.fixture
def sample_go_code():
    """Sample Go code for testing entity extraction"""
    return '''// Package declaration
package main

import (
    "fmt"
    "os"
    "github.com/example/pkg"
    utils "github.com/example/utils"
)

// Constants
const (
    MaxConnections = 100
    DefaultTimeout = 30
    _internalFlag  = true
)

// Variables
var (
    GlobalCounter int = 0
    logger        *Logger
    _privateVar   string
)

// Type definitions
type UserID int64
type Handler func(string) error

// Struct definition
type User struct {
    ID       UserID `json:"id" db:"user_id"`
    Name     string `json:"name"`
    Email    string `json:"email"`
    Age      int    `json:"age,omitempty"`
    _private string
}

// Interface definition
type Writer interface {
    Write([]byte) (int, error)
    Close() error
}

// Function
func NewUser(name, email string) *User {
    return &User{
        ID:    UserID(generateID()),
        Name:  name,
        Email: email,
    }
}

// Methods with receivers
func (u *User) GetDisplayName() string {
    return fmt.Sprintf("%s <%s>", u.Name, u.Email)
}

func (u *User) IsValid() bool {
    return u.Name != "" && u.Email != ""
}

func (u User) GetAge() int {
    return u.Age
}

// Function with goroutines
func ProcessUsers(users []User) {
    for _, user := range users {
        go func(u User) {
            fmt.Printf("Processing user: %s\\n", u.Name)
        }(user)
    }
}

// Main function
func main() {
    user := NewUser("John Doe", "john@example.com")
    fmt.Println(user.GetDisplayName())
    
    if user.IsValid() {
        ProcessUsers([]User{*user})
    }
}'''


@pytest.fixture
def complex_go_code():
    """More complex Go code with advanced features"""
    return '''package server

import (
    "context"
    "net/http"
    "time"
    
    "github.com/gorilla/mux"
    "github.com/sirupsen/logrus"
)

// Generic type definition (Go 1.18+)
type Result[T any] struct {
    Value T
    Error error
}

// Interface with embedded interface
type ReadWriteCloser interface {
    Reader
    Writer
    Closer
}

type Reader interface {
    Read([]byte) (int, error)
}

type Writer interface {
    Write([]byte) (int, error)
}

type Closer interface {
    Close() error
}

// Struct with embedded types
type Server struct {
    *http.Server
    Router *mux.Router
    logger *logrus.Logger
    config Config
}

type Config struct {
    Port         int           `yaml:"port" env:"PORT"`
    ReadTimeout  time.Duration `yaml:"read_timeout"`
    WriteTimeout time.Duration `yaml:"write_timeout"`
}

// Constructor function
func NewServer(cfg Config) *Server {
    return &Server{
        Server: &http.Server{
            Addr:         fmt.Sprintf(":%d", cfg.Port),
            ReadTimeout:  cfg.ReadTimeout,
            WriteTimeout: cfg.WriteTimeout,
        },
        Router: mux.NewRouter(),
        logger: logrus.New(),
        config: cfg,
    }
}

// Method with context and error handling
func (s *Server) Start(ctx context.Context) error {
    s.Server.Handler = s.Router
    
    go func() {
        <-ctx.Done()
        s.logger.Info("Shutting down server...")
        s.Server.Shutdown(context.Background())
    }()
    
    s.logger.Infof("Starting server on %s", s.Server.Addr)
    return s.Server.ListenAndServe()
}

// Method implementing interface
func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
    s.Router.ServeHTTP(w, r)
}

// Generic function
func MapSlice[T, R any](slice []T, fn func(T) R) []R {
    result := make([]R, len(slice))
    for i, v := range slice {
        result[i] = fn(v)
    }
    return result
}'''


class TestGoParser:
    """Test suite for Go parser"""
    
    def test_parser_registration(self, go_parser):
        """Test that Go parser is properly registered"""
        assert go_parser.language == "go"
        assert go_parser.can_parse(Path("test.go"))
        assert not go_parser.can_parse(Path("test.py"))
        
        # Test registry integration
        registered_parser = parser_registry.get_parser("go")
        assert registered_parser is not None
        assert isinstance(registered_parser, GoParser)
        
        # Test file extension mapping
        file_parser = parser_registry.get_parser_for_file(Path("example.go"))
        assert file_parser is not None
        assert isinstance(file_parser, GoParser)
    
    def test_extract_packages(self, go_parser, tmp_path):
        """Test package declaration extraction"""
        content = '''package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}'''
        
        test_file = tmp_path / "test.go"
        test_file.write_text(content)
        
        result = go_parser.parse_file(test_file)
        entities = result.entities
        
        # Filter packages (using MODULE entity type)
        packages = [e for e in entities if e.entity_type == EntityType.MODULE]
        assert len(packages) == 1
        
        package = packages[0]
        assert package.name == "main"
        assert package.signature == "package main"
        assert package.visibility == Visibility.PUBLIC
        assert package.metadata["package_name"] == "main"
    
    def test_extract_imports(self, go_parser, tmp_path):
        """Test import declaration extraction"""
        content = '''package main

import (
    "fmt"
    "os"
    "github.com/example/pkg"
    utils "github.com/example/utils"
)'''
        
        test_file = tmp_path / "test.go"
        test_file.write_text(content)
        
        result = go_parser.parse_file(test_file)
        entities = result.entities
        
        # Filter imports
        imports = [e for e in entities if e.entity_type == EntityType.IMPORT]
        assert len(imports) >= 4
        
        # Check specific imports
        import_names = [imp.name for imp in imports]
        import_paths = [imp.qualified_name for imp in imports]
        
        assert "fmt" in import_names
        assert "os" in import_names
        assert "pkg" in import_names
        assert "utils" in import_names
        
        assert "fmt" in import_paths
        assert "os" in import_paths
        assert "github.com/example/pkg" in import_paths
        assert "github.com/example/utils" in import_paths
        
        # Check alias
        utils_import = next((imp for imp in imports if imp.name == "utils"), None)
        assert utils_import is not None
        assert utils_import.metadata["alias"] == "utils"
        assert utils_import.qualified_name == "github.com/example/utils"
        
        # Check standard library detection
        fmt_import = next((imp for imp in imports if imp.name == "fmt"), None)
        assert fmt_import is not None
        assert fmt_import.metadata["is_standard_library"] == True
        
        pkg_import = next((imp for imp in imports if imp.name == "pkg"), None)
        assert pkg_import is not None
        assert pkg_import.metadata["is_standard_library"] == False
    
    def test_extract_structs(self, go_parser, tmp_path, sample_go_code):
        """Test struct definition extraction"""
        test_file = tmp_path / "test.go"
        test_file.write_text(sample_go_code)
        
        result = go_parser.parse_file(test_file)
        entities = result.entities
        
        # Filter structs
        structs = [e for e in entities if e.entity_type == EntityType.STRUCT]
        assert len(structs) >= 1
        
        # Check User struct
        user_struct = next((s for s in structs if s.name == "User"), None)
        assert user_struct is not None
        assert user_struct.visibility == Visibility.PUBLIC  # Capitalized name
        assert user_struct.signature == "type User struct"
        assert user_struct.metadata["struct_name"] == "User"
        
        # Check fields
        fields = user_struct.metadata["fields"]
        assert len(fields) >= 4
        
        # Check specific fields
        field_names = []
        for field in fields:
            field_names.extend(field["names"])
        
        assert "ID" in field_names
        assert "Name" in field_names
        assert "Email" in field_names
        assert "Age" in field_names
        
        # Check field with tags
        id_field = next((f for f in fields if "ID" in f["names"]), None)
        assert id_field is not None
        assert id_field["type"] == "UserID"
        assert id_field["tag"] is not None
        assert "json:" in id_field["tag"]
    
    def test_extract_interfaces(self, go_parser, tmp_path, sample_go_code):
        """Test interface definition extraction"""
        test_file = tmp_path / "test.go"
        test_file.write_text(sample_go_code)
        
        result = go_parser.parse_file(test_file)
        entities = result.entities
        
        # Filter interfaces
        interfaces = [e for e in entities if e.entity_type == EntityType.INTERFACE]
        assert len(interfaces) >= 1
        
        # Check Writer interface
        writer_interface = next((i for i in interfaces if i.name == "Writer"), None)
        assert writer_interface is not None
        assert writer_interface.visibility == Visibility.PUBLIC  # Capitalized name
        assert writer_interface.signature == "type Writer interface"
        assert writer_interface.metadata["interface_name"] == "Writer"
        
        # Check methods
        methods = writer_interface.metadata["methods"]
        assert len(methods) >= 2
        
        method_names = [m["name"] for m in methods]
        assert "Write" in method_names
        assert "Close" in method_names
    
    def test_extract_functions(self, go_parser, tmp_path, sample_go_code):
        """Test function extraction"""
        test_file = tmp_path / "test.go"
        test_file.write_text(sample_go_code)
        
        result = go_parser.parse_file(test_file)
        entities = result.entities
        
        # Filter functions
        functions = [e for e in entities if e.entity_type == EntityType.FUNCTION]
        assert len(functions) >= 3
        
        function_names = [f.name for f in functions]
        assert "NewUser" in function_names
        assert "ProcessUsers" in function_names
        assert "main" in function_names
        
        # Check NewUser function
        new_user_func = next((f for f in functions if f.name == "NewUser"), None)
        assert new_user_func is not None
        assert new_user_func.visibility == Visibility.PUBLIC  # Capitalized name
        assert "func NewUser" in new_user_func.signature
        
        # Check parameters
        parameters = new_user_func.metadata["parameters"]
        assert len(parameters) == 2
        assert parameters[0]["name"] == "name"
        assert parameters[0]["type"] == "string"
        assert parameters[1]["name"] == "email"
        assert parameters[1]["type"] == "string"
        
        # Check ProcessUsers function (should detect goroutines)
        process_func = next((f for f in functions if f.name == "ProcessUsers"), None)
        assert process_func is not None
        assert process_func.is_async == True  # Contains goroutines
        assert process_func.metadata["has_goroutines"] == True
    
    def test_extract_methods(self, go_parser, tmp_path, sample_go_code):
        """Test method extraction with receivers"""
        test_file = tmp_path / "test.go"
        test_file.write_text(sample_go_code)
        
        result = go_parser.parse_file(test_file)
        entities = result.entities
        
        # Filter methods
        methods = [e for e in entities if e.entity_type == EntityType.METHOD]
        assert len(methods) >= 3
        
        method_names = [m.name for m in methods]
        assert "GetDisplayName" in method_names
        assert "IsValid" in method_names
        assert "GetAge" in method_names
        
        # Check GetDisplayName method
        get_display_method = next((m for m in methods if m.name == "GetDisplayName"), None)
        assert get_display_method is not None
        assert get_display_method.visibility == Visibility.PUBLIC  # Capitalized name
        
        # Check receiver
        receiver = get_display_method.metadata["receiver"]
        assert receiver is not None
        assert receiver["type"] == "*User"
        assert receiver["name"] == "u"
        
        # Check signature includes receiver
        assert "func (u *User)" in get_display_method.signature
        
        # Check GetAge method (value receiver)
        get_age_method = next((m for m in methods if m.name == "GetAge"), None)
        assert get_age_method is not None
        
        # Check value receiver
        receiver = get_age_method.metadata["receiver"]
        assert receiver is not None
        assert receiver["type"] == "User"  # No pointer
    
    def test_extract_variables(self, go_parser, tmp_path, sample_go_code):
        """Test variable declaration extraction"""
        test_file = tmp_path / "test.go"
        test_file.write_text(sample_go_code)
        
        result = go_parser.parse_file(test_file)
        entities = result.entities
        
        # Filter variables
        variables = [e for e in entities if e.entity_type == EntityType.VARIABLE]
        assert len(variables) >= 3
        
        variable_names = [v.name for v in variables]
        assert "GlobalCounter" in variable_names
        assert "logger" in variable_names
        assert "_privateVar" in variable_names
        
        # Check GlobalCounter variable
        global_counter = next((v for v in variables if v.name == "GlobalCounter"), None)
        assert global_counter is not None
        assert global_counter.visibility == Visibility.PUBLIC  # Capitalized name
        assert global_counter.metadata["type"] == "int"
        assert global_counter.metadata["value"] == "0"
        assert "var GlobalCounter int = 0" in global_counter.signature
        
        # Check private variable
        private_var = next((v for v in variables if v.name == "_privateVar"), None)
        assert private_var is not None
        assert private_var.visibility == Visibility.PRIVATE  # Starts with underscore
        assert private_var.metadata["type"] == "string"
    
    def test_extract_constants(self, go_parser, tmp_path, sample_go_code):
        """Test constant declaration extraction"""
        test_file = tmp_path / "test.go"
        test_file.write_text(sample_go_code)
        
        result = go_parser.parse_file(test_file)
        entities = result.entities
        
        # Filter constants
        constants = [e for e in entities if e.entity_type == EntityType.CONSTANT]
        assert len(constants) >= 3
        
        constant_names = [c.name for c in constants]
        assert "MaxConnections" in constant_names
        assert "DefaultTimeout" in constant_names
        assert "_internalFlag" in constant_names
        
        # Check MaxConnections constant
        max_conn = next((c for c in constants if c.name == "MaxConnections"), None)
        assert max_conn is not None
        assert max_conn.visibility == Visibility.PUBLIC  # Capitalized name
        assert max_conn.metadata["value"] == "100"
        assert max_conn.metadata["is_constant"] == True
        assert "const MaxConnections" in max_conn.signature
        
        # Check private constant
        internal_flag = next((c for c in constants if c.name == "_internalFlag"), None)
        assert internal_flag is not None
        assert internal_flag.visibility == Visibility.PRIVATE  # Starts with underscore
        assert internal_flag.metadata["value"] == "true"
    
    def test_extract_types(self, go_parser, tmp_path, sample_go_code):
        """Test type alias extraction"""
        test_file = tmp_path / "test.go"
        test_file.write_text(sample_go_code)
        
        result = go_parser.parse_file(test_file)
        entities = result.entities
        
        # Filter type aliases
        types = [e for e in entities if e.entity_type == EntityType.TYPE]
        assert len(types) >= 2
        
        type_names = [t.name for t in types]
        assert "UserID" in type_names
        assert "Handler" in type_names
        
        # Check UserID type
        user_id_type = next((t for t in types if t.name == "UserID"), None)
        assert user_id_type is not None
        assert user_id_type.visibility == Visibility.PUBLIC  # Capitalized name
        assert user_id_type.metadata["underlying_type"] == "int64"
        assert user_id_type.metadata["is_alias"] == True
        assert user_id_type.signature == "type UserID int64"
        
        # Check Handler type (function type)
        handler_type = next((t for t in types if t.name == "Handler"), None)
        assert handler_type is not None
        assert handler_type.metadata["underlying_type"] == "func(string) error"
    
    def test_visibility_detection(self, go_parser, tmp_path):
        """Test Go visibility detection based on naming conventions"""
        content = '''package test

// Public entities (capitalized)
type PublicStruct struct {
    PublicField string
}

func PublicFunction() {}

var PublicVariable int

const PublicConstant = 42

// Private entities (lowercase or underscore)
type privateStruct struct {
    privateField string
}

func privateFunction() {}

var privateVariable int

const privateConstant = 42

var _internalVariable int

func _internalFunction() {}'''
        
        test_file = tmp_path / "test.go"
        test_file.write_text(content)
        
        result = go_parser.parse_file(test_file)
        entities = result.entities
        
        # Check public entities
        public_entities = [e for e in entities if e.visibility == Visibility.PUBLIC]
        public_names = [e.name for e in public_entities]
        
        assert "PublicStruct" in public_names
        assert "PublicFunction" in public_names
        assert "PublicVariable" in public_names
        assert "PublicConstant" in public_names
        
        # Check private entities
        private_entities = [e for e in entities if e.visibility == Visibility.PRIVATE]
        private_names = [e.name for e in private_entities]
        
        assert "privateStruct" in private_names
        assert "privateFunction" in private_names
        assert "privateVariable" in private_names
        assert "privateConstant" in private_names
        assert "_internalVariable" in private_names
        assert "_internalFunction" in private_names
    
    def test_goroutine_detection(self, go_parser, tmp_path):
        """Test detection of goroutines in functions"""
        content = '''package main

// Function without goroutines
func syncFunction() {
    println("This is synchronous")
}

// Function with goroutines
func asyncFunction() {
    go func() {
        println("This runs in a goroutine")
    }()
    
    go processData()
}

func processData() {
    println("Processing data")
}'''
        
        test_file = tmp_path / "test.go"
        test_file.write_text(content)
        
        result = go_parser.parse_file(test_file)
        entities = result.entities
        
        # Filter functions
        functions = [e for e in entities if e.entity_type == EntityType.FUNCTION]
        
        # Check sync function
        sync_func = next((f for f in functions if f.name == "syncFunction"), None)
        assert sync_func is not None
        assert sync_func.is_async == False
        assert sync_func.metadata["has_goroutines"] == False
        
        # Check async function
        async_func = next((f for f in functions if f.name == "asyncFunction"), None)
        assert async_func is not None
        assert async_func.is_async == True
        assert async_func.metadata["has_goroutines"] == True
        
        # Check regular function (not called with go)
        process_func = next((f for f in functions if f.name == "processData"), None)
        assert process_func is not None
        assert process_func.is_async == False
        assert process_func.metadata["has_goroutines"] == False
    
    def test_complex_structures(self, go_parser, tmp_path, complex_go_code):
        """Test parsing of complex Go structures"""
        test_file = tmp_path / "test.go"
        test_file.write_text(complex_go_code)
        
        result = go_parser.parse_file(test_file)
        entities = result.entities
        
        # Should extract various entity types
        entity_types = [e.entity_type for e in entities]
        assert EntityType.MODULE in entity_types  # package
        assert EntityType.IMPORT in entity_types  # imports
        assert EntityType.STRUCT in entity_types  # structs
        assert EntityType.INTERFACE in entity_types  # interfaces
        assert EntityType.FUNCTION in entity_types  # functions
        assert EntityType.METHOD in entity_types  # methods
        
        # Check specific entities
        entity_names = [e.name for e in entities]
        assert "server" in entity_names  # package
        assert "Server" in entity_names  # struct
        assert "Config" in entity_names  # struct
        assert "ReadWriteCloser" in entity_names  # interface
        assert "NewServer" in entity_names  # function
        assert "Start" in entity_names  # method
    
    def test_error_handling(self, go_parser, tmp_path):
        """Test parser behavior with invalid or malformed Go code"""
        # Test with empty file
        empty_file = tmp_path / "empty.go"
        empty_file.write_text("")
        
        result = go_parser.parse_file(empty_file)
        assert result.success is True
        assert result.entity_count == 0
        assert result.relation_count == 0
        
        # Test with malformed Go code
        malformed_file = tmp_path / "malformed.go"
        malformed_file.write_text("this is not valid go code { } [ invalid")
        
        result = go_parser.parse_file(malformed_file)
        # Should not crash, but may have empty or limited results
        assert isinstance(result.entities, list)
        assert isinstance(result.relations, list)
    
    def test_metadata_completeness(self, go_parser, tmp_path, sample_go_code):
        """Test that extracted entities have complete metadata"""
        test_file = tmp_path / "test.go"
        test_file.write_text(sample_go_code)
        
        result = go_parser.parse_file(test_file)
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
            assert entity.metadata["language"] == "go"
            assert "ast_node_type" in entity.metadata
            
            # Type-specific metadata checks
            if entity.entity_type == EntityType.STRUCT:
                assert "struct_name" in entity.metadata
                assert "fields" in entity.metadata
                assert "field_count" in entity.metadata
            
            elif entity.entity_type == EntityType.METHOD:
                assert "function_name" in entity.metadata
                assert "parameters" in entity.metadata
                assert "return_types" in entity.metadata
                if "receiver" in entity.metadata:
                    assert "receiver_type" in entity.metadata
                    assert "receiver_name" in entity.metadata
            
            elif entity.entity_type == EntityType.IMPORT:
                assert "import_path" in entity.metadata
                assert "is_standard_library" in entity.metadata