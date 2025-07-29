"""
Tests for C++ parser with comprehensive entity and relation extraction.

Tests the CppParser implementation to ensure correct extraction of:
- Class declarations with inheritance and templates
- Function declarations and definitions
- Struct and union declarations with templates
- Enum declarations (including enum class)
- Namespace declarations
- Template declarations
- Method declarations within classes
- Constructor and destructor methods
- Typedef declarations
- Global variable declarations
- Include statements
- Relations between entities (inheritance, calls, type usage)
"""

import pytest
from pathlib import Path

from core.parser.cpp_parser import CppParser
from core.parser.registry import parser_registry
from core.models.entities import (
    Entity, EntityType, Relation, RelationType, 
    SourceLocation, Visibility
)


@pytest.fixture
def cpp_parser():
    """Create a C++ parser instance for testing"""
    return CppParser()


@pytest.fixture
def sample_cpp_code():
    """Sample C++ code for testing entity extraction"""
    return '''#include <iostream>
#include <vector>
#include <memory>
#include "custom_header.hpp"

namespace MyNamespace {
    
    // Forward declarations
    class ForwardClass;
    
    // Template class with inheritance
    template<typename T, class U = int>
    class TemplateBase {
    public:
        virtual ~TemplateBase() = default;
        virtual void pure_virtual() = 0;
        
    protected:
        T data;
        U extra;
    };
    
    // Enum class
    enum class Status : int {
        PENDING = 0,
        ACTIVE = 1,
        INACTIVE = 2,
        DELETED = 3
    };
    
    // Regular enum
    enum Color {
        RED, GREEN, BLUE, ALPHA
    };
    
    // Struct with inheritance
    struct Point3D : public Point2D {
        double z;
        
        Point3D(double x, double y, double z) : Point2D(x, y), z(z) {}
        
        double distance() const {
            return sqrt(x*x + y*y + z*z);
        }
    };
    
    // Union
    union Value {
        int as_int;
        float as_float;
        double as_double;
        char as_char;
    };
    
    // Class with multiple inheritance and templates
    class DataProcessor : public TemplateBase<std::string>, private Logger {
    private:
        std::vector<std::string> data_;
        std::unique_ptr<Value> value_;
        Status current_status_;
        
    public:
        // Constructors
        DataProcessor();
        DataProcessor(const std::vector<std::string>& data);
        DataProcessor(const DataProcessor& other) = delete;
        DataProcessor(DataProcessor&& other) noexcept;
        
        // Destructor
        virtual ~DataProcessor() override;
        
        // Assignment operators
        DataProcessor& operator=(const DataProcessor& other) = delete;
        DataProcessor& operator=(DataProcessor&& other) noexcept;
        
        // Public methods
        void process_data();
        std::string get_result() const;
        void set_status(Status status);
        Status get_status() const { return current_status_; }
        
        // Template method
        template<typename Func>
        void apply_function(Func&& func);
        
        // Static method
        static DataProcessor create_default();
        
        // Pure virtual implementation
        void pure_virtual() override {
            process_data();
        }
        
    protected:
        void log_operation(const std::string& op);
        
    private:
        void internal_process();
        bool validate_data() const;
    };
    
    // Template function
    template<typename T>
    T max_value(const T& a, const T& b) {
        return (a > b) ? a : b;
    }
    
    // Global functions
    void global_init();
    int calculate_sum(const std::vector<int>& values);
    std::unique_ptr<DataProcessor> create_processor();
    
    // Global variables
    const int MAX_PROCESSORS = 10;
    static std::vector<DataProcessor*> active_processors;
    extern bool debug_enabled;
    
    // Function pointer typedef
    typedef void (*ProcessorCallback)(const DataProcessor&);
    
    // Template typedef
    template<typename T>
    using ProcessorMap = std::unordered_map<std::string, std::unique_ptr<T>>;
    
} // namespace MyNamespace

// Global namespace functions
int main(int argc, char* argv[]) {
    MyNamespace::global_init();
    
    auto processor = MyNamespace::create_processor();
    processor->process_data();
    
    std::string result = processor->get_result();
    std::cout << "Result: " << result << std::endl;
    
    return 0;
}

// Template specialization
template<>
class MyNamespace::TemplateBase<int> {
public:
    void specialized_method() {
        // Specialized implementation
    }
};'''


@pytest.fixture
def header_cpp_code():
    """Sample C++ header file for testing"""
    return '''#ifndef CUSTOM_HEADER_HPP
#define CUSTOM_HEADER_HPP

#include <string>
#include <vector>

namespace Graphics {
    
    // Abstract base class
    class Drawable {
    public:
        virtual ~Drawable() = default;
        virtual void draw() const = 0;
        virtual void update(double delta_time) = 0;
        
        // Non-virtual methods
        void set_visible(bool visible) { visible_ = visible; }
        bool is_visible() const { return visible_; }
        
    protected:
        bool visible_ = true;
    };
    
    // Template class
    template<int N>
    class Matrix {
    private:
        double data_[N][N];
        
    public:
        Matrix();
        Matrix(const Matrix& other);
        Matrix& operator=(const Matrix& other);
        
        double& operator()(int row, int col);
        const double& operator()(int row, int col) const;
        
        Matrix operator+(const Matrix& other) const;
        Matrix operator*(const Matrix& other) const;
        
        void identity();
        double determinant() const;
        
        static Matrix zero();
        static Matrix identity_matrix();
    };
    
    // Specialized classes
    using Matrix2D = Matrix<2>;
    using Matrix3D = Matrix<3>;
    using Matrix4D = Matrix<4>;
    
    // Struct for simple data
    struct Vertex {
        double x, y, z;
        double nx, ny, nz;  // Normal
        double u, v;        // Texture coordinates
        
        Vertex() : x(0), y(0), z(0), nx(0), ny(1), nz(0), u(0), v(0) {}
        Vertex(double x, double y, double z) : x(x), y(y), z(z), nx(0), ny(1), nz(0), u(0), v(0) {}
    };
    
    // Enum for render modes
    enum class RenderMode {
        WIREFRAME,
        SOLID,
        TEXTURED,
        SHADED
    };
    
    // Global constants
    extern const double PI;
    extern const double EPSILON;
    
    // Global functions
    double degrees_to_radians(double degrees);
    double radians_to_degrees(double radians);
    
    // Template functions
    template<typename T>
    T clamp(const T& value, const T& min_val, const T& max_val);
    
    template<typename Container>
    void clear_container(Container& container);
    
} // namespace Graphics

#endif // CUSTOM_HEADER_HPP'''


@pytest.fixture
def complex_cpp_code():
    """Complex C++ code with advanced features"""
    return '''#include <memory>
#include <functional>
#include <type_traits>

namespace Advanced {
    
    // SFINAE and template metaprogramming
    template<typename T>
    struct is_smart_pointer : std::false_type {};
    
    template<typename T>
    struct is_smart_pointer<std::unique_ptr<T>> : std::true_type {};
    
    template<typename T>
    struct is_smart_pointer<std::shared_ptr<T>> : std::true_type {};
    
    // Variadic template class
    template<typename... Args>
    class VariadicProcessor {
    private:
        std::tuple<Args...> data_;
        
    public:
        template<std::size_t I>
        auto get() -> decltype(std::get<I>(data_)) {
            return std::get<I>(data_);
        }
        
        template<typename Func>
        void for_each(Func&& func) {
            for_each_impl(std::forward<Func>(func), std::index_sequence_for<Args...>{});
        }
        
    private:
        template<typename Func, std::size_t... Is>
        void for_each_impl(Func&& func, std::index_sequence<Is...>) {
            (func(std::get<Is>(data_)), ...);  // C++17 fold expression
        }
    };
    
    // Concept-like template constraints (C++20 style but compatible)
    template<typename T>
    class Container {
        static_assert(std::is_default_constructible_v<T>, "T must be default constructible");
        static_assert(std::is_copy_constructible_v<T>, "T must be copy constructible");
        
    private:
        std::vector<T> items_;
        mutable std::mutex mutex_;
        
    public:
        void add(const T& item) {
            std::lock_guard<std::mutex> lock(mutex_);
            items_.push_back(item);
        }
        
        void add(T&& item) {
            std::lock_guard<std::mutex> lock(mutex_);
            items_.push_back(std::move(item));
        }
        
        template<typename... Args>
        void emplace(Args&&... args) {
            std::lock_guard<std::mutex> lock(mutex_);
            items_.emplace_back(std::forward<Args>(args)...);
        }
        
        auto begin() const { return items_.begin(); }
        auto end() const { return items_.end(); }
        
        std::size_t size() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return items_.size();
        }
    };
    
    // CRTP (Curiously Recurring Template Pattern)
    template<typename Derived>
    class Singleton {
    protected:
        Singleton() = default;
        ~Singleton() = default;
        
    public:
        static Derived& instance() {
            static Derived instance_;
            return instance_;
        }
        
        Singleton(const Singleton&) = delete;
        Singleton& operator=(const Singleton&) = delete;
        Singleton(Singleton&&) = delete;
        Singleton& operator=(Singleton&&) = delete;
    };
    
    // Policy-based design
    template<typename T, 
             template<typename> class OwnershipPolicy,
             template<typename> class CheckingPolicy>
    class SmartPtr : private OwnershipPolicy<T>, private CheckingPolicy<T> {
    private:
        T* ptr_;
        
    public:
        explicit SmartPtr(T* ptr = nullptr) : ptr_(ptr) {}
        
        ~SmartPtr() {
            OwnershipPolicy<T>::destroy(ptr_);
        }
        
        T& operator*() {
            CheckingPolicy<T>::check(ptr_);
            return *ptr_;
        }
        
        T* operator->() {
            CheckingPolicy<T>::check(ptr_);
            return ptr_;
        }
        
        T* get() const { return ptr_; }
    };
    
    // Policy implementations
    template<typename T>
    class RefCountedOwnership {
    public:
        void destroy(T* ptr) {
            if (ptr && --ref_count_ == 0) {
                delete ptr;
            }
        }
        
    private:
        static int ref_count_;
    };
    
    template<typename T>
    class NoChecking {
    public:
        static void check(T* ptr) { /* No checking */ }
    };
    
    template<typename T>
    class AssertChecking {
    public:
        static void check(T* ptr) {
            assert(ptr != nullptr);
        }
    };
    
    // Type aliases for common smart pointer configurations
    template<typename T>
    using SafePtr = SmartPtr<T, RefCountedOwnership, AssertChecking>;
    
    template<typename T>
    using UnsafePtr = SmartPtr<T, RefCountedOwnership, NoChecking>;
    
    // Function objects and lambdas
    class EventHandler {
    private:
        std::vector<std::function<void()>> callbacks_;
        
    public:
        template<typename Func>
        void add_callback(Func&& func) {
            callbacks_.emplace_back(std::forward<Func>(func));
        }
        
        void trigger_all() {
            for (auto& callback : callbacks_) {
                callback();
            }
        }
        
        void clear() {
            callbacks_.clear();
        }
    };
    
    // Operator overloading example
    class Vector3D {
    private:
        double x_, y_, z_;
        
    public:
        Vector3D(double x = 0, double y = 0, double z = 0) : x_(x), y_(y), z_(z) {}
        
        // Arithmetic operators
        Vector3D operator+(const Vector3D& other) const {
            return Vector3D(x_ + other.x_, y_ + other.y_, z_ + other.z_);
        }
        
        Vector3D operator-(const Vector3D& other) const {
            return Vector3D(x_ - other.x_, y_ - other.y_, z_ - other.z_);
        }
        
        Vector3D operator*(double scalar) const {
            return Vector3D(x_ * scalar, y_ * scalar, z_ * scalar);
        }
        
        // Compound assignment operators
        Vector3D& operator+=(const Vector3D& other) {
            x_ += other.x_;
            y_ += other.y_;
            z_ += other.z_;
            return *this;
        }
        
        // Comparison operators
        bool operator==(const Vector3D& other) const {
            return x_ == other.x_ && y_ == other.y_ && z_ == other.z_;
        }
        
        bool operator!=(const Vector3D& other) const {
            return !(*this == other);
        }
        
        // Stream operators
        friend std::ostream& operator<<(std::ostream& os, const Vector3D& vec) {
            return os << "Vector3D(" << vec.x_ << ", " << vec.y_ << ", " << vec.z_ << ")";
        }
        
        // Getters
        double x() const { return x_; }
        double y() const { return y_; }
        double z() const { return z_; }
    };
    
} // namespace Advanced'''


class TestCppParserBasics:
    """Test basic C++ parser functionality"""
    
    def test_parser_initialization(self, cpp_parser):
        """Test parser initialization and properties"""
        assert cpp_parser.get_language_name() == "cpp"
        extensions = cpp_parser.get_supported_extensions()
        assert ".cpp" in extensions
        assert ".hpp" in extensions
        assert ".cxx" in extensions
        assert ".cc" in extensions
        assert ".hxx" in extensions
        assert ".h" in extensions
    
    def test_can_parse_cpp_files(self, cpp_parser):
        """Test file extension detection"""
        assert cpp_parser.can_parse(Path("main.cpp"))
        assert cpp_parser.can_parse(Path("header.hpp"))
        assert cpp_parser.can_parse(Path("source.cxx"))
        assert cpp_parser.can_parse(Path("header.hxx"))
        assert cpp_parser.can_parse(Path("old_style.cc"))
        assert cpp_parser.can_parse(Path("c_header.h"))
        assert not cpp_parser.can_parse(Path("script.js"))
        assert not cpp_parser.can_parse(Path("code.py"))
    
    def test_parser_registration(self):
        """Test that C++ parser is registered correctly"""
        # Check C++ parser registration
        cpp_parser = parser_registry.get_parser("cpp")
        assert cpp_parser is not None
        assert isinstance(cpp_parser, CppParser)
        
        # Check file mapping
        cpp_file = Path("main.cpp")
        file_parser = parser_registry.get_parser_for_file(cpp_file)
        assert file_parser is not None
        assert isinstance(file_parser, CppParser)


class TestCppEntityExtraction:
    """Test entity extraction from C++ code"""
    
    def test_class_extraction(self, cpp_parser, tmp_path, sample_cpp_code):
        """Test extraction of class declarations"""
        test_file = tmp_path / "main.cpp"
        test_file.write_text(sample_cpp_code)
        
        result = cpp_parser.parse_file(test_file)
        
        # Find class entities
        classes = [e for e in result.entities if e.entity_type == EntityType.CLASS and e.metadata.get("is_class")]
        class_names = [c.name for c in classes]
        
        # Check class extraction
        assert "TemplateBase" in class_names
        assert "DataProcessor" in class_names
        
        # Check class details
        data_processor = next((c for c in classes if c.name == "DataProcessor"), None)
        assert data_processor is not None
        assert "class DataProcessor" in data_processor.signature
        assert data_processor.metadata.get("has_inheritance") is True
        assert len(data_processor.metadata.get("base_classes", [])) >= 2  # TemplateBase and Logger
        assert data_processor.metadata.get("method_count", 0) > 5
    
    def test_struct_extraction(self, cpp_parser, tmp_path, sample_cpp_code):
        """Test extraction of struct declarations"""
        test_file = tmp_path / "main.cpp"
        test_file.write_text(sample_cpp_code)
        
        result = cpp_parser.parse_file(test_file)
        
        # Find struct entities
        structs = [e for e in result.entities if e.entity_type == EntityType.CLASS and e.metadata.get("is_struct")]
        struct_names = [s.name for s in structs]
        
        # Check struct extraction
        assert "Point3D" in struct_names
        
        # Check struct details
        point3d = next((s for s in structs if s.name == "Point3D"), None)
        assert point3d is not None
        assert "struct Point3D" in point3d.signature
        assert point3d.metadata.get("has_inheritance") is True
        assert "Point2D" in point3d.metadata.get("base_classes", [])
    
    def test_union_extraction(self, cpp_parser, tmp_path, sample_cpp_code):
        """Test extraction of union declarations"""
        test_file = tmp_path / "main.cpp"
        test_file.write_text(sample_cpp_code)
        
        result = cpp_parser.parse_file(test_file)
        
        # Find union entities
        unions = [e for e in result.entities if e.entity_type == EntityType.CLASS and e.metadata.get("is_union")]
        union_names = [u.name for u in unions]
        
        # Check union extraction
        assert "Value" in union_names
        
        # Check union details
        value_union = next((u for u in unions if u.name == "Value"), None)
        assert value_union is not None
        assert "union Value" in value_union.signature
        assert value_union.metadata.get("field_count", 0) >= 4
    
    def test_enum_extraction(self, cpp_parser, tmp_path, sample_cpp_code):
        """Test extraction of enum declarations"""
        test_file = tmp_path / "main.cpp"
        test_file.write_text(sample_cpp_code)
        
        result = cpp_parser.parse_file(test_file)
        
        # Find enum entities
        enums = [e for e in result.entities if e.entity_type == EntityType.ENUM]
        enum_names = [e.name for e in enums]
        
        # Check enum extraction
        assert "Status" in enum_names
        assert "Color" in enum_names
        
        # Check enum class details
        status_enum = next((e for e in enums if e.name == "Status"), None)
        assert status_enum is not None
        assert "enum class Status" in status_enum.signature or "enum Status" in status_enum.signature
        assert status_enum.metadata.get("value_count", 0) >= 4
        
        # Check regular enum details
        color_enum = next((e for e in enums if e.name == "Color"), None)
        assert color_enum is not None
        assert color_enum.metadata.get("value_count", 0) >= 4
    
    def test_function_extraction(self, cpp_parser, tmp_path, sample_cpp_code):
        """Test extraction of function declarations and definitions"""
        test_file = tmp_path / "main.cpp"
        test_file.write_text(sample_cpp_code)
        
        result = cpp_parser.parse_file(test_file)
        
        # Find function entities (not methods)
        functions = [e for e in result.entities if e.entity_type in [EntityType.FUNCTION, EntityType.INTERFACE] and not e.metadata.get("is_method")]
        function_names = [f.name for f in functions]
        
        # Check function extraction
        assert "main" in function_names
        assert "global_init" in function_names
        assert "calculate_sum" in function_names
        assert "max_value" in function_names  # Template function
        
        # Check function details
        main_func = next((f for f in functions if f.name == "main"), None)
        assert main_func is not None
        assert "int main(" in main_func.signature
        assert main_func.metadata.get("parameter_count", 0) == 2
    
    def test_method_extraction(self, cpp_parser, tmp_path, sample_cpp_code):
        """Test extraction of method declarations within classes"""
        test_file = tmp_path / "main.cpp"
        test_file.write_text(sample_cpp_code)
        
        result = cpp_parser.parse_file(test_file)
        
        # Find method entities
        methods = [e for e in result.entities if e.entity_type == EntityType.METHOD]
        method_names = [m.name for m in methods]
        
        # Check method extraction
        assert "process_data" in method_names
        assert "get_result" in method_names
        assert "set_status" in method_names
        assert "distance" in method_names  # From Point3D struct
        
        # Check method details
        process_data = next((m for m in methods if m.name == "process_data"), None)
        assert process_data is not None
        assert process_data.metadata.get("is_method") is True
    
    def test_namespace_extraction(self, cpp_parser, tmp_path, sample_cpp_code):
        """Test extraction of namespace declarations"""
        test_file = tmp_path / "main.cpp"
        test_file.write_text(sample_cpp_code)
        
        result = cpp_parser.parse_file(test_file)
        
        # Find namespace entities
        namespaces = [e for e in result.entities if e.entity_type == EntityType.MODULE and e.metadata.get("is_namespace")]
        namespace_names = [n.name for n in namespaces]
        
        # Check namespace extraction
        assert "MyNamespace" in namespace_names
        
        # Check namespace details
        my_namespace = next((n for n in namespaces if n.name == "MyNamespace"), None)
        assert my_namespace is not None
        assert "namespace MyNamespace" in my_namespace.signature
        assert my_namespace.metadata.get("member_count", 0) > 0
    
    def test_template_extraction(self, cpp_parser, tmp_path, sample_cpp_code):
        """Test extraction of template declarations"""
        test_file = tmp_path / "main.cpp"
        test_file.write_text(sample_cpp_code)
        
        result = cpp_parser.parse_file(test_file)
        
        # Find template entities
        templates = [e for e in result.entities if e.metadata.get("is_template")]
        
        # Should find template classes and functions
        assert len(templates) > 0
        
        # Check for specific templates
        template_names = [t.name for t in templates]
        template_signatures = [t.signature for t in templates]
        
        # Look for template indicators in signatures
        template_sigs = [sig for sig in template_signatures if "template<" in sig]
        assert len(template_sigs) > 0
    
    def test_typedef_extraction(self, cpp_parser, tmp_path, sample_cpp_code):
        """Test extraction of typedef declarations"""
        test_file = tmp_path / "main.cpp"
        test_file.write_text(sample_cpp_code)
        
        result = cpp_parser.parse_file(test_file)
        
        # Find typedef entities
        typedefs = [e for e in result.entities if e.entity_type == EntityType.TYPE_ALIAS]
        typedef_names = [t.name for t in typedefs]
        
        # Check typedef extraction
        assert "ProcessorCallback" in typedef_names
        
        # Check typedef details
        callback_typedef = next((t for t in typedefs if t.name == "ProcessorCallback"), None)
        assert callback_typedef is not None
        assert "typedef" in callback_typedef.signature
    
    def test_global_variable_extraction(self, cpp_parser, tmp_path, sample_cpp_code):
        """Test extraction of global variables"""
        test_file = tmp_path / "main.cpp"
        test_file.write_text(sample_cpp_code)
        
        result = cpp_parser.parse_file(test_file)
        
        # Find global variable entities
        variables = [e for e in result.entities if e.entity_type == EntityType.VARIABLE and e.metadata.get("is_global")]
        variable_names = [v.name for v in variables]
        
        # Check global variable extraction
        assert "MAX_PROCESSORS" in variable_names
        assert "active_processors" in variable_names
        
        # Check variable details
        max_processors = next((v for v in variables if v.name == "MAX_PROCESSORS"), None)
        assert max_processors is not None
        assert "const int MAX_PROCESSORS" in max_processors.signature
    
    def test_include_extraction(self, cpp_parser, tmp_path, sample_cpp_code):
        """Test extraction of include statements"""
        test_file = tmp_path / "main.cpp"
        test_file.write_text(sample_cpp_code)
        
        result = cpp_parser.parse_file(test_file)
        
        # Find include entities
        includes = [e for e in result.entities if e.entity_type == EntityType.IMPORT]
        include_names = [i.name for i in includes]
        
        # Check include extraction
        assert "iostream" in include_names
        assert "vector" in include_names
        assert "memory" in include_names
        assert "custom_header.hpp" in include_names
        
        # Check include details
        iostream_include = next((i for i in includes if i.name == "iostream"), None)
        assert iostream_include is not None
        assert iostream_include.metadata.get("is_system_include") is True
        
        custom_include = next((i for i in includes if i.name == "custom_header.hpp"), None)
        assert custom_include is not None
        assert custom_include.metadata.get("is_local_include") is True


class TestCppRelationExtraction:
    """Test relation extraction from C++ code"""
    
    def test_function_call_relations(self, cpp_parser, tmp_path, sample_cpp_code):
        """Test extraction of function call relations"""
        test_file = tmp_path / "main.cpp"
        test_file.write_text(sample_cpp_code)
        
        result = cpp_parser.parse_file(test_file)
        
        # Find function call relations
        call_relations = [r for r in result.relations if r.relation_type == RelationType.CALLS]
        
        assert len(call_relations) > 0
        
        # Check that main calls other functions
        main_calls = [r for r in call_relations if "main" in r.source_entity_id]
        assert len(main_calls) > 0
    
    def test_inheritance_relations(self, cpp_parser, tmp_path, sample_cpp_code):
        """Test extraction of inheritance relations"""
        test_file = tmp_path / "main.cpp"
        test_file.write_text(sample_cpp_code)
        
        result = cpp_parser.parse_file(test_file)
        
        # Find inheritance relations
        inheritance_relations = [r for r in result.relations if r.relation_type == RelationType.INHERITS]
        
        assert len(inheritance_relations) > 0
        
        # Check specific inheritance relationships
        relation_contexts = [r.context for r in inheritance_relations if r.context]
        inheritance_contexts = [ctx for ctx in relation_contexts if "inherits from" in ctx]
        assert len(inheritance_contexts) > 0
    
    def test_type_usage_relations(self, cpp_parser, tmp_path, sample_cpp_code):
        """Test extraction of type usage relations"""
        test_file = tmp_path / "main.cpp"
        test_file.write_text(sample_cpp_code)
        
        result = cpp_parser.parse_file(test_file)
        
        # Find type usage relations
        usage_relations = [r for r in result.relations if r.relation_type == RelationType.USES_TYPE]
        
        assert len(usage_relations) > 0
        
        # Check that variables use custom types
        usage_contexts = [r.context for r in usage_relations if r.context]
        type_usage_contexts = [ctx for ctx in usage_contexts if "uses type" in ctx]
        assert len(type_usage_contexts) > 0
    
    def test_include_relations(self, cpp_parser, tmp_path, sample_cpp_code):
        """Test extraction of include relations"""
        test_file = tmp_path / "main.cpp"
        test_file.write_text(sample_cpp_code)
        
        result = cpp_parser.parse_file(test_file)
        
        # Find import relations
        import_relations = [r for r in result.relations if r.relation_type == RelationType.IMPORTS]
        
        assert len(import_relations) > 0
        
        # Check that file imports headers
        import_contexts = [r.context for r in import_relations if r.context]
        include_contexts = [ctx for ctx in import_contexts if "includes" in ctx]
        assert len(include_contexts) > 0


class TestCppParserEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_cpp_file_parsing(self, cpp_parser, tmp_path):
        """Test parsing empty C++ files"""
        test_cases = [
            "",
            "// Just a comment",
            "/* Block comment */",
            "#include <iostream>\n// Only include"
        ]
        
        for i, cpp_content in enumerate(test_cases):
            test_file = tmp_path / f"empty_{i}.cpp"
            test_file.write_text(cpp_content)
            
            result = cpp_parser.parse_file(test_file)
            
            assert result.success is True
            # May have minimal entities for include-only files
            if "#include" in cpp_content:
                assert result.entity_count >= 1
    
    def test_syntax_error_handling(self, cpp_parser, tmp_path):
        """Test handling of C++ syntax errors"""
        invalid_cpp = '''class TestClass {
    public:
        void method() {
            std::cout << "Hello World"  // Missing semicolon and closing brace
        // Missing closing brace for method
    // Missing closing brace for class
'''
        
        test_file = tmp_path / "invalid.cpp"
        test_file.write_text(invalid_cpp)
        
        result = cpp_parser.parse_file(test_file)
        
        # Should handle gracefully
        assert result is not None
        # Tree-sitter is usually robust with partial parsing
    
    def test_complex_cpp_structures(self, cpp_parser, tmp_path, complex_cpp_code):
        """Test parsing complex C++ code with advanced features"""
        test_file = tmp_path / "complex.cpp"
        test_file.write_text(complex_cpp_code)
        
        result = cpp_parser.parse_file(test_file)
        
        assert result.success is True
        assert result.entity_count > 15
        
        # Check that complex structures are handled
        entity_names = [e.name for e in result.entities]
        assert "VariadicProcessor" in entity_names
        assert "Container" in entity_names
        assert "Singleton" in entity_names
        assert "SmartPtr" in entity_names
        assert "EventHandler" in entity_names
        assert "Vector3D" in entity_names
    
    def test_header_file_parsing(self, cpp_parser, tmp_path, header_cpp_code):
        """Test parsing C++ header files"""
        test_file = tmp_path / "custom_header.hpp"
        test_file.write_text(header_cpp_code)
        
        result = cpp_parser.parse_file(test_file)
        
        assert result.success is True
        
        # Check header-specific entities
        entity_names = [e.name for e in result.entities]
        assert "Drawable" in entity_names
        assert "Matrix" in entity_names
        assert "Vertex" in entity_names
        assert "RenderMode" in entity_names
        
        # Check template classes
        template_entities = [e for e in result.entities if e.metadata.get("is_template")]
        assert len(template_entities) > 0
    
    def test_visibility_detection(self, cpp_parser, tmp_path):
        """Test detection of public/private/protected visibility"""
        visibility_cpp = '''class VisibilityTest {
private:
    int private_field;
    void private_method();
    
protected:
    int protected_field;
    void protected_method();
    
public:
    int public_field;
    void public_method();
    
    VisibilityTest();  // Constructor
    ~VisibilityTest(); // Destructor
};'''
        
        test_file = tmp_path / "visibility.cpp"
        test_file.write_text(visibility_cpp)
        
        result = cpp_parser.parse_file(test_file)
        
        assert result.success is True
        
        # Find method entities and check visibility
        methods = [e for e in result.entities if e.entity_type == EntityType.METHOD]
        
        # Should have methods with different visibility levels
        visibilities = [m.visibility for m in methods]
        
        # At least some methods should have different visibility
        unique_visibilities = set(visibilities)
        assert len(unique_visibilities) > 1
    
    def test_template_specialization(self, cpp_parser, tmp_path):
        """Test template specialization handling"""
        template_cpp = '''template<typename T>
class TemplateClass {
public:
    void generic_method(T value) {}
};

// Full specialization
template<>
class TemplateClass<int> {
public:
    void specialized_method(int value) {}
};

// Partial specialization
template<typename T>
class TemplateClass<T*> {
public:
    void pointer_method(T* value) {}
};'''
        
        test_file = tmp_path / "template.cpp"
        test_file.write_text(template_cpp)
        
        result = cpp_parser.parse_file(test_file)
        
        assert result.success is True
        
        # Should extract template classes
        classes = [e for e in result.entities if e.entity_type == EntityType.CLASS]
        class_names = [c.name for c in classes]
        
        assert "TemplateClass" in class_names


class TestCppParserIntegration:
    """Integration tests with registry and file discovery"""
    
    def test_registry_integration(self, tmp_path):
        """Test integration with parser registry"""
        # Create C++ files
        files = []
        for i in range(3):
            cpp_file = tmp_path / f"module_{i}.cpp"
            cpp_content = f'''#include <iostream>

namespace Module{i} {{
    class TestClass{i} {{
    public:
        void method_{i}() {{
            std::cout << "Method {i}" << std::endl;
        }}
    }};
    
    void function_{i}() {{
        TestClass{i} obj;
        obj.method_{i}();
    }}
}}

int main() {{
    Module{i}::function_{i}();
    return 0;
}}'''
            cpp_file.write_text(cpp_content)
            files.append(cpp_file)
        
        # Use registry to discover and parse files
        parseable_files = parser_registry.discover_files(tmp_path)
        cpp_files = [f for f in parseable_files if f.suffix in [".cpp", ".hpp", ".cxx", ".cc", ".hxx", ".h"]]
        
        assert len(cpp_files) == 3
        
        # Parse files in parallel
        results = parser_registry.parse_files_parallel(cpp_files)
        
        assert len(results) == 3
        assert all(r.success for r in results)
        
        # Check that entities were extracted
        total_entities = sum(r.entity_count for r in results)
        assert total_entities >= 9  # At least 3 classes, 3 functions, 3 main functions
    
    def test_multiple_file_types(self, tmp_path):
        """Test parsing different C++ file extensions"""
        file_contents = {
            "main.cpp": '''#include "header.hpp"
int main() { return 0; }''',
            "header.hpp": '''#ifndef HEADER_HPP
#define HEADER_HPP
class MyClass {};
#endif''',
            "source.cxx": '''#include <vector>
void function() {}''',
            "legacy.cc": '''struct LegacyStruct { int x; };''',
            "old_header.h": '''extern int global_var;'''
        }
        
        files = []
        for filename, content in file_contents.items():
            file_path = tmp_path / filename
            file_path.write_text(content)
            files.append(file_path)
        
        # Parse all files
        results = []
        for file_path in files:
            parser = parser_registry.get_parser_for_file(file_path)
            if parser:
                result = parser.parse_file(file_path)
                results.append(result)
        
        assert len(results) == 5
        assert all(r.success for r in results)
        
        # Check that different file types were parsed correctly
        total_entities = sum(r.entity_count for r in results)
        assert total_entities >= 5  # At least one entity per file


# Performance benchmarks
@pytest.mark.benchmark
class TestCppParserPerformance:
    """Performance benchmarks for C++ parser"""
    
    def test_parsing_speed(self, cpp_parser, tmp_path, benchmark):
        """Benchmark C++ parsing speed"""
        # Create large C++ file
        large_cpp = '''#include <iostream>
#include <vector>
#include <memory>
#include <string>

namespace Performance {
    
    template<typename T>
    class Container {
    private:
        std::vector<T> items_;
        
    public:
        void add(const T& item) { items_.push_back(item); }
        T get(size_t index) const { return items_[index]; }
        size_t size() const { return items_.size(); }
    };
    
    enum class ProcessingMode {
        FAST, NORMAL, THOROUGH, EXHAUSTIVE
    };
    
    struct ProcessingConfig {
        ProcessingMode mode;
        int iterations;
        double threshold;
        bool enable_logging;
        
        ProcessingConfig() : mode(ProcessingMode::NORMAL), iterations(100), threshold(0.5), enable_logging(false) {}
    };
    
    class DataProcessor {
    private:
        Container<std::string> data_;
        ProcessingConfig config_;
        
    public:
        DataProcessor(const ProcessingConfig& config) : config_(config) {}
        
        void process() {
            for (int i = 0; i < config_.iterations; ++i) {
                process_iteration(i);
            }
        }
        
    private:
        void process_iteration(int iteration) {
            if (config_.enable_logging) {
                log_iteration(iteration);
            }
            perform_calculation(iteration);
        }
        
        void log_iteration(int iteration) {
            std::cout << "Processing iteration: " << iteration << std::endl;
        }
        
        double perform_calculation(int iteration) {
            return iteration * config_.threshold;
        }
    };
    
'''
        
        # Add many classes and functions
        for i in range(20):
            large_cpp += f'''
    class ProcessorModule{i} {{
    private:
        Container<int> values_{i}_;
        double factor_{i}_;
        
    public:
        ProcessorModule{i}() : factor_{i}_({i + 1}.0) {{}}
        
        void initialize() {{
            for (int j = 0; j < 50; ++j) {{
                values_{i}_.add(j * factor_{i}_);
            }}
        }}
        
        double calculate_sum() {{
            double sum = 0.0;
            for (size_t k = 0; k < values_{i}_.size(); ++k) {{
                sum += values_{i}_.get(k);
            }}
            return sum;
        }}
        
        double calculate_average() {{
            if (values_{i}_.size() == 0) return 0.0;
            return calculate_sum() / values_{i}_.size();
        }}
        
        void reset() {{
            values_{i}_ = Container<int>();
        }}
    }};
    
    void process_module_{i}() {{
        ProcessorModule{i} module;
        module.initialize();
        double avg = module.calculate_average();
        std::cout << "Module {i} average: " << avg << std::endl;
        module.reset();
    }}
'''
        
        large_cpp += '''
} // namespace Performance

int main() {
    Performance::ProcessingConfig config;
    config.mode = Performance::ProcessingMode::FAST;
    config.iterations = 10;
    config.enable_logging = true;
    
    Performance::DataProcessor processor(config);
    processor.process();
    
    // Process all modules
'''
        
        for i in range(20):
            large_cpp += f'    Performance::process_module_{i}();\n'
        
        large_cpp += '''
    return 0;
}'''
        
        test_file = tmp_path / "benchmark.cpp"
        test_file.write_text(large_cpp)
        
        # Benchmark parsing
        def parse_file():
            return cpp_parser.parse_file(test_file)
        
        result = benchmark(parse_file)
        
        # Verify parsing worked
        assert result.success
        assert result.entity_count > 50
        
        # Performance targets (guidelines)
        assert result.parse_time < 3.0  # Should parse in under 3 seconds