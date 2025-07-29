"""
Tests for C parser with comprehensive entity and relation extraction.

Tests the CParser implementation to ensure correct extraction of:
- Function declarations and definitions
- Struct, union, and enum declarations
- Typedef declarations
- Global variable declarations
- Macro definitions
- Include statements
- Relations between entities
"""

import pytest
from pathlib import Path

from core.parser.c_parser import CParser
from core.parser.registry import parser_registry
from core.models.entities import (
    Entity, EntityType, Relation, RelationType, 
    SourceLocation, Visibility
)


@pytest.fixture
def c_parser():
    """Create a C parser instance for testing"""
    return CParser()


@pytest.fixture
def sample_c_code():
    """Sample C code for testing entity extraction"""
    return '''#include <stdio.h>
#include <stdlib.h>
#include "myheader.h"

#define MAX_SIZE 100
#define VERSION "1.0.0"
#define MIN(a, b) ((a) < (b) ? (a) : (b))

typedef struct {
    int id;
    char name[50];
} User;

typedef union {
    int as_int;
    float as_float;
    char as_char;
} Value;

typedef enum {
    STATUS_PENDING,
    STATUS_ACTIVE,
    STATUS_INACTIVE
} Status;

typedef int UserID;

int global_counter = 0;
const char* app_name = "MyApp";

// Function declarations
int add(int a, int b);
void print_user(const User* user);
Status get_status(UserID id);

// Function definitions
int add(int a, int b) {
    return a + b;
}

void print_user(const User* user) {
    if (user != NULL) {
        printf("User: %s (ID: %d)\n", user->name, user->id);
    }
}

Status get_status(UserID id) {
    if (id > 0) {
        return STATUS_ACTIVE;
    }
    return STATUS_INACTIVE;
}

int main(int argc, char* argv[]) {
    User user = {1, "John Doe"};
    int result = add(5, 3);
    
    print_user(&user);
    printf("Result: %d\n", result);
    
    Status status = get_status(user.id);
    return 0;
}'''


@pytest.fixture
def header_file_code():
    """Sample C header file for testing"""
    return '''#ifndef MYHEADER_H
#define MYHEADER_H

#include <stdint.h>

#define API_VERSION 2
#define MAX_USERS 1000

typedef struct Point {
    double x;
    double y;
    double z;
} Point;

typedef enum Color {
    RED,
    GREEN,
    BLUE,
    ALPHA
} Color;

// Function prototypes
Point create_point(double x, double y, double z);
double distance(const Point* p1, const Point* p2);
void set_color(Color color);

// Global variables
extern int debug_mode;
extern const char* version_string;

#endif // MYHEADER_H'''


@pytest.fixture
def complex_c_code():
    """Complex C code with advanced features"""
    return '''#include <stdio.h>
#include <string.h>

#define BUFFER_SIZE 1024
#define STRINGIFY(x) #x
#define CONCAT(a, b) a ## b

typedef struct Node {
    int data;
    struct Node* next;
    struct Node* prev;
} Node;

typedef union FloatBits {
    float value;
    struct {
        unsigned int mantissa : 23;
        unsigned int exponent : 8;
        unsigned int sign : 1;
    } bits;
} FloatBits;

typedef enum LogLevel {
    LOG_DEBUG = 0,
    LOG_INFO = 1,
    LOG_WARN = 2,
    LOG_ERROR = 3
} LogLevel;

typedef int (*CompareFunc)(const void* a, const void* b);
typedef void (*DestroyFunc)(void* data);

static int instance_count = 0;
static const double PI = 3.14159265359;

// Function pointer array
static CompareFunc comparers[] = {
    NULL,
    NULL,
    NULL
};

Node* create_node(int data) {
    Node* node = malloc(sizeof(Node));
    if (node) {
        node->data = data;
        node->next = NULL;
        node->prev = NULL;
        instance_count++;
    }
    return node;
}

void destroy_node(Node* node) {
    if (node) {
        free(node);
        instance_count--;
    }
}

int compare_nodes(const void* a, const void* b) {
    const Node* na = (const Node*)a;
    const Node* nb = (const Node*)b;
    return na->data - nb->data;
}

void log_message(LogLevel level, const char* format, ...) {
    va_list args;
    va_start(args, format);
    
    switch (level) {
        case LOG_DEBUG:
            printf("[DEBUG] ");
            break;
        case LOG_INFO:
            printf("[INFO] ");
            break;
        case LOG_WARN:
            printf("[WARN] ");
            break;
        case LOG_ERROR:
            printf("[ERROR] ");
            break;
    }
    
    vprintf(format, args);
    va_end(args);
}'''


class TestCParserBasics:
    """Test basic C parser functionality"""
    
    def test_parser_initialization(self, c_parser):
        """Test parser initialization and properties"""
        assert c_parser.get_language_name() == "c"
        assert ".c" in c_parser.get_supported_extensions()
        assert ".h" in c_parser.get_supported_extensions()
    
    def test_can_parse_c_files(self, c_parser):
        """Test file extension detection"""
        assert c_parser.can_parse(Path("main.c"))
        assert c_parser.can_parse(Path("header.h"))
        assert c_parser.can_parse(Path("library.c"))
        assert not c_parser.can_parse(Path("script.js"))
        assert not c_parser.can_parse(Path("code.py"))
    
    def test_parser_registration(self):
        """Test that C parser is registered correctly"""
        # Check C parser registration
        c_parser = parser_registry.get_parser("c")
        assert c_parser is not None
        assert isinstance(c_parser, CParser)
        
        # Check file mapping
        c_file = Path("main.c")
        file_parser = parser_registry.get_parser_for_file(c_file)
        assert file_parser is not None
        assert isinstance(file_parser, CParser)


class TestCEntityExtraction:
    """Test entity extraction from C code"""
    
    def test_function_extraction(self, c_parser, tmp_path, sample_c_code):
        """Test extraction of function declarations and definitions"""
        test_file = tmp_path / "main.c"
        test_file.write_text(sample_c_code)
        
        result = c_parser.parse_file(test_file)
        
        # Find function entities
        functions = [e for e in result.entities if e.entity_type in [EntityType.FUNCTION, EntityType.INTERFACE]]
        function_names = [f.name for f in functions]
        
        # Check function declarations and definitions
        assert "add" in function_names
        assert "print_user" in function_names
        assert "get_status" in function_names
        assert "main" in function_names
        
        # Check function details
        add_func = next((f for f in functions if f.name == "add"), None)
        assert add_func is not None
        assert "int add(" in add_func.signature
        assert add_func.metadata["return_type"] == "int"
        assert add_func.metadata["parameter_count"] == 2
        assert "int a" in add_func.metadata["parameters"]
        assert "int b" in add_func.metadata["parameters"]
    
    def test_struct_extraction(self, c_parser, tmp_path, sample_c_code):
        """Test extraction of struct declarations"""
        test_file = tmp_path / "main.c"
        test_file.write_text(sample_c_code)
        
        result = c_parser.parse_file(test_file)
        
        # Find struct entities (mapped to CLASS)
        structs = [e for e in result.entities if e.entity_type == EntityType.CLASS and e.metadata.get("is_struct")]
        struct_names = [s.name for s in structs]
        
        # Check struct extraction
        assert "User" in struct_names
        
        # Check struct details
        user_struct = next((s for s in structs if s.name == "User"), None)
        assert user_struct is not None
        assert "struct User" in user_struct.signature
        assert user_struct.metadata["field_count"] == 2
        assert "int id" in user_struct.metadata["fields"]
        assert "char name[50]" in user_struct.metadata["fields"]
    
    def test_union_extraction(self, c_parser, tmp_path, sample_c_code):
        """Test extraction of union declarations"""
        test_file = tmp_path / "main.c"
        test_file.write_text(sample_c_code)
        
        result = c_parser.parse_file(test_file)
        
        # Find union entities (mapped to CLASS)
        unions = [e for e in result.entities if e.entity_type == EntityType.CLASS and e.metadata.get("is_union")]
        union_names = [u.name for u in unions]
        
        # Check union extraction
        assert "Value" in union_names
        
        # Check union details
        value_union = next((u for u in unions if u.name == "Value"), None)
        assert value_union is not None
        assert "union Value" in value_union.signature
        assert value_union.metadata["field_count"] == 3
        assert "int as_int" in value_union.metadata["fields"]
        assert "float as_float" in value_union.metadata["fields"]
        assert "char as_char" in value_union.metadata["fields"]
    
    def test_enum_extraction(self, c_parser, tmp_path, sample_c_code):
        """Test extraction of enum declarations"""
        test_file = tmp_path / "main.c"
        test_file.write_text(sample_c_code)
        
        result = c_parser.parse_file(test_file)
        
        # Find enum entities
        enums = [e for e in result.entities if e.entity_type == EntityType.ENUM]
        enum_names = [e.name for e in enums]
        
        # Check enum extraction
        assert "Status" in enum_names
        
        # Check enum details
        status_enum = next((e for e in enums if e.name == "Status"), None)
        assert status_enum is not None
        assert "enum Status" in status_enum.signature
        assert status_enum.metadata["value_count"] == 3
        assert "STATUS_PENDING" in status_enum.metadata["values"]
        assert "STATUS_ACTIVE" in status_enum.metadata["values"]
        assert "STATUS_INACTIVE" in status_enum.metadata["values"]
    
    def test_typedef_extraction(self, c_parser, tmp_path, sample_c_code):
        """Test extraction of typedef declarations"""
        test_file = tmp_path / "main.c"
        test_file.write_text(sample_c_code)
        
        result = c_parser.parse_file(test_file)
        
        # Find typedef entities
        typedefs = [e for e in result.entities if e.entity_type == EntityType.TYPE_ALIAS]
        typedef_names = [t.name for t in typedefs]
        
        # Check typedef extraction
        assert "UserID" in typedef_names
        
        # Check typedef details
        userid_typedef = next((t for t in typedefs if t.name == "UserID"), None)
        assert userid_typedef is not None
        assert "typedef" in userid_typedef.signature
        assert "int" in userid_typedef.signature
        assert userid_typedef.metadata["base_type"] == "int"
    
    def test_macro_extraction(self, c_parser, tmp_path, sample_c_code):
        """Test extraction of macro definitions"""
        test_file = tmp_path / "main.c"
        test_file.write_text(sample_c_code)
        
        result = c_parser.parse_file(test_file)
        
        # Find macro entities (mapped to CONSTANT)
        macros = [e for e in result.entities if e.entity_type == EntityType.CONSTANT and e.metadata.get("is_macro")]
        macro_names = [m.name for m in macros]
        
        # Check macro extraction
        assert "MAX_SIZE" in macro_names
        assert "VERSION" in macro_names
        assert "MIN" in macro_names
        
        # Check macro details
        max_size_macro = next((m for m in macros if m.name == "MAX_SIZE"), None)
        assert max_size_macro is not None
        assert "#define MAX_SIZE" in max_size_macro.signature
        assert "100" in max_size_macro.metadata["macro_value"]
    
    def test_global_variable_extraction(self, c_parser, tmp_path, sample_c_code):
        """Test extraction of global variables"""
        test_file = tmp_path / "main.c"
        test_file.write_text(sample_c_code)
        
        result = c_parser.parse_file(test_file)
        
        # Find global variable entities
        variables = [e for e in result.entities if e.entity_type == EntityType.VARIABLE and e.metadata.get("is_global")]
        variable_names = [v.name for v in variables]
        
        # Check global variable extraction
        assert "global_counter" in variable_names
        assert "app_name" in variable_names
        
        # Check variable details
        counter_var = next((v for v in variables if v.name == "global_counter"), None)
        assert counter_var is not None
        assert "int global_counter" in counter_var.signature
        assert counter_var.metadata["variable_type"] == "int"
    
    def test_include_extraction(self, c_parser, tmp_path, sample_c_code):
        """Test extraction of include statements"""
        test_file = tmp_path / "main.c"
        test_file.write_text(sample_c_code)
        
        result = c_parser.parse_file(test_file)
        
        # Find include entities
        includes = [e for e in result.entities if e.entity_type == EntityType.IMPORT]
        include_names = [i.name for i in includes]
        
        # Check include extraction
        assert "stdio.h" in include_names
        assert "stdlib.h" in include_names
        assert "myheader.h" in include_names
        
        # Check include details
        stdio_include = next((i for i in includes if i.name == "stdio.h"), None)
        assert stdio_include is not None
        assert "#include <stdio.h>" in stdio_include.signature
        assert stdio_include.metadata["is_system_include"] is True
        
        myheader_include = next((i for i in includes if i.name == "myheader.h"), None)
        assert myheader_include is not None
        assert "#include \"myheader.h\"" in myheader_include.signature
        assert myheader_include.metadata["is_local_include"] is True


class TestCRelationExtraction:
    """Test relation extraction from C code"""
    
    def test_function_call_relations(self, c_parser, tmp_path, sample_c_code):
        """Test extraction of function call relations"""
        test_file = tmp_path / "main.c"
        test_file.write_text(sample_c_code)
        
        result = c_parser.parse_file(test_file)
        
        # Find function call relations
        call_relations = [r for r in result.relations if r.relation_type == RelationType.CALLS]
        
        assert len(call_relations) > 0
        
        # Check that main calls other functions
        main_calls = [r for r in call_relations if "main" in r.source_entity_id]
        assert len(main_calls) > 0
        
        # Check specific function calls
        call_targets = [r.target_entity_id for r in call_relations]
        assert any("add" in target for target in call_targets)
        assert any("print_user" in target for target in call_targets)
    
    def test_type_usage_relations(self, c_parser, tmp_path, sample_c_code):
        """Test extraction of type usage relations"""
        test_file = tmp_path / "main.c"
        test_file.write_text(sample_c_code)
        
        result = c_parser.parse_file(test_file)
        
        # Find type usage relations
        usage_relations = [r for r in result.relations if r.relation_type == RelationType.USES_TYPE]
        
        assert len(usage_relations) > 0
        
        # Check that variables use custom types
        usage_targets = [r.target_entity_id for r in usage_relations]
        assert any("User" in target for target in usage_targets)
        assert any("Status" in target for target in usage_targets)
    
    def test_include_relations(self, c_parser, tmp_path, sample_c_code):
        """Test extraction of include relations"""
        test_file = tmp_path / "main.c"
        test_file.write_text(sample_c_code)
        
        result = c_parser.parse_file(test_file)
        
        # Find import relations
        import_relations = [r for r in result.relations if r.relation_type == RelationType.IMPORTS]
        
        assert len(import_relations) > 0
        
        # Check that file imports headers
        import_targets = [r.target_entity_id for r in import_relations]
        assert any("stdio.h" in target for target in import_targets)
        assert any("stdlib.h" in target for target in import_targets)
        assert any("myheader.h" in target for target in import_targets)


class TestCParserEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_c_file_parsing(self, c_parser, tmp_path):
        """Test parsing empty C files"""
        test_cases = [
            "",
            "// Just a comment",
            "/* Block comment */",
            "#include <stdio.h>\n// Only include"
        ]
        
        for i, c_content in enumerate(test_cases):
            test_file = tmp_path / f"empty_{i}.c"
            test_file.write_text(c_content)
            
            result = c_parser.parse_file(test_file)
            
            assert result.success is True
            # May have minimal entities for include-only files
            if "#include" in c_content:
                assert result.entity_count >= 1
    
    def test_syntax_error_handling(self, c_parser, tmp_path):
        """Test handling of C syntax errors"""
        invalid_c = '''int main() {
    printf("Hello World"  // Missing closing parenthesis and semicolon
    return 0;
}'''
        
        test_file = tmp_path / "invalid.c"
        test_file.write_text(invalid_c)
        
        result = c_parser.parse_file(test_file)
        
        # Should handle gracefully
        assert result is not None
        # Tree-sitter is usually robust with partial parsing
    
    def test_complex_c_structures(self, c_parser, tmp_path, complex_c_code):
        """Test parsing complex C code with advanced features"""
        test_file = tmp_path / "complex.c"
        test_file.write_text(complex_c_code)
        
        result = c_parser.parse_file(test_file)
        
        assert result.success is True
        assert result.entity_count > 10
        
        # Check that complex structures are handled
        entity_names = [e.name for e in result.entities]
        assert "Node" in entity_names
        assert "FloatBits" in entity_names
        assert "LogLevel" in entity_names
        assert "CompareFunc" in entity_names
        assert "create_node" in entity_names
        assert "log_message" in entity_names
    
    def test_header_file_parsing(self, c_parser, tmp_path, header_file_code):
        """Test parsing C header files"""
        test_file = tmp_path / "myheader.h"
        test_file.write_text(header_file_code)
        
        result = c_parser.parse_file(test_file)
        
        assert result.success is True
        
        # Check header-specific entities
        entity_names = [e.name for e in result.entities]
        assert "Point" in entity_names
        assert "Color" in entity_names
        assert "create_point" in entity_names
        assert "API_VERSION" in entity_names
        
        # Check extern variables
        variables = [e for e in result.entities if e.entity_type == EntityType.VARIABLE]
        variable_names = [v.name for v in variables]
        assert "debug_mode" in variable_names
        assert "version_string" in variable_names
    
    def test_function_declarations_vs_definitions(self, c_parser, tmp_path, sample_c_code):
        """Test distinction between function declarations and definitions"""
        test_file = tmp_path / "main.c"
        test_file.write_text(sample_c_code)
        
        result = c_parser.parse_file(test_file)
        
        # Find all function entities
        all_functions = [e for e in result.entities if e.entity_type in [EntityType.FUNCTION, EntityType.INTERFACE]]
        
        # Check that we have both declarations and definitions
        declarations = [f for f in all_functions if f.metadata.get("is_declaration")]
        definitions = [f for f in all_functions if f.metadata.get("is_definition")]
        
        # Should have function declarations (prototypes) and definitions
        assert len(declarations) > 0 or len(definitions) > 0
        
        # Check specific function types
        add_functions = [f for f in all_functions if f.name == "add"]
        if len(add_functions) > 1:
            # Should have both declaration and definition
            has_decl = any(f.metadata.get("is_declaration") for f in add_functions)
            has_def = any(f.metadata.get("is_definition") for f in add_functions)
            assert has_decl or has_def


class TestCParserIntegration:
    """Integration tests with registry and file discovery"""
    
    def test_registry_integration(self, tmp_path):
        """Test integration with parser registry"""
        # Create C files
        files = []
        for i in range(3):
            c_file = tmp_path / f"module_{i}.c"
            c_content = f'''#include <stdio.h>

int function_{i}(int x) {{
    return x * {i + 1};
}}

int main() {{
    return function_{i}(5);
}}'''
            c_file.write_text(c_content)
            files.append(c_file)
        
        # Use registry to discover and parse files
        parseable_files = parser_registry.discover_files(tmp_path)
        c_files = [f for f in parseable_files if f.suffix in [".c", ".h"]]
        
        assert len(c_files) == 3
        
        # Parse files in parallel
        results = parser_registry.parse_files_parallel(c_files)
        
        assert len(results) == 3
        assert all(r.success for r in results)
        
        # Check that entities were extracted
        total_entities = sum(r.entity_count for r in results)
        assert total_entities >= 6  # At least 2 functions per file


# Performance benchmarks
@pytest.mark.benchmark
class TestCParserPerformance:
    """Performance benchmarks for C parser"""
    
    def test_parsing_speed(self, c_parser, tmp_path, benchmark):
        """Benchmark C parsing speed"""
        # Create large C file
        large_c = '''#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ITEMS 1000

typedef struct Item {
    int id;
    char name[100];
    double value;
} Item;

typedef enum ItemType {
    TYPE_UNKNOWN,
    TYPE_PRODUCT,
    TYPE_SERVICE
} ItemType;

// Global variables
static Item items[MAX_ITEMS];
static int item_count = 0;
static const double PI = 3.14159;

// Function prototypes
int add_item(const char* name, double value, ItemType type);
Item* find_item(int id);
void remove_item(int id);
void print_all_items(void);
double calculate_total(void);

'''
        
        # Add many functions
        for i in range(20):
            large_c += f'''
int process_item_{i}(Item* item) {{
    if (!item) return -1;
    
    item->value *= {i + 1}.0;
    printf("Processing item %d: %s\\n", item->id, item->name);
    
    return item->id;
}}

double calculate_value_{i}(double base) {{
    return base * {i + 1} * PI;
}}
'''
        
        large_c += '''
int main() {
    printf("C Performance Test\\n");
    return 0;
}'''
        
        test_file = tmp_path / "benchmark.c"
        test_file.write_text(large_c)
        
        # Benchmark parsing
        def parse_file():
            return c_parser.parse_file(test_file)
        
        result = benchmark(parse_file)
        
        # Verify parsing worked
        assert result.success
        assert result.entity_count > 30
        
        # Performance targets (guidelines)
        assert result.parse_time < 2.0  # Should parse in under 2 seconds