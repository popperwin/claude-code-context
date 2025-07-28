"""
Tests for Java parser implementation.
"""

import pytest
from pathlib import Path
from core.parser.java_parser import JavaParser
from core.models.entities import EntityType, Visibility


class TestJavaParser:
    """Test suite for Java parser"""
    
    @pytest.fixture
    def java_parser(self):
        """Create a Java parser instance"""
        return JavaParser()
    
    @pytest.fixture
    def sample_java_code(self):
        """Sample Java code for testing"""
        return '''// Sample Java code for testing
package com.example.app;

import java.util.List;
import java.util.ArrayList;
import static java.util.Collections.sort;
import com.example.utils.*;

/**
 * Main application class
 */
@Component
@Service("userService")
public class UserService implements UserRepository {
    
    // Constants
    public static final String DEFAULT_NAME = "Unknown";
    private static final int MAX_USERS = 1000;
    
    // Fields
    private String name;
    private int age;
    private List<String> roles;
    public volatile boolean active;
    
    // Constructors
    public UserService() {
        this.name = DEFAULT_NAME;
        this.roles = new ArrayList<>();
    }
    
    public UserService(String name, int age) {
        this.name = name;
        this.age = age;
        this.roles = new ArrayList<>();
    }
    
    // Methods
    public String getName() {
        return name;
    }
    
    public void setName(String name) {
        this.name = name;
    }
    
    @Override
    public List<User> findAll() throws DatabaseException {
        // Implementation here
        return new ArrayList<>();
    }
    
    public static void main(String[] args) {
        UserService service = new UserService();
        System.out.println("Service created: " + service.getName());
    }
    
    // Private helper method
    private void validateUser(User user) {
        if (user == null) {
            throw new IllegalArgumentException("User cannot be null");
        }
    }
}

/**
 * User entity interface
 */
public interface UserRepository {
    List<User> findAll() throws DatabaseException;
    void save(User user);
    User findById(Long id);
}

/**
 * User data class
 */
public class User {
    private Long id;
    private String username;
    private String email;
    
    public User(Long id, String username, String email) {
        this.id = id;
        this.username = username;
        this.email = email;
    }
    
    // Getters and setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    
    public String getUsername() { return username; }
    public void setUsername(String username) { this.username = username; }
    
    public String getEmail() { return email; }
    public void setEmail(String email) { this.email = email; }
}

/**
 * Status enumeration
 */
public enum UserStatus {
    ACTIVE("Active user"),
    INACTIVE("Inactive user"),
    PENDING("Pending verification"),
    SUSPENDED("Account suspended");
    
    private final String description;
    
    UserStatus(String description) {
        this.description = description;
    }
    
    public String getDescription() {
        return description;
    }
}

/**
 * Generic utility class
 */
public abstract class BaseRepository<T extends Entity> {
    
    protected List<T> items;
    
    public BaseRepository() {
        this.items = new ArrayList<>();
    }
    
    public abstract void save(T item);
    public abstract T findById(Long id);
    
    public int count() {
        return items.size();
    }
}

/**
 * Custom annotation
 */
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
public @interface Service {
    String value() default "";
    int priority() default 0;
}
'''
    
    def test_parser_registration(self, java_parser):
        """Test that Java parser is properly registered"""
        from core.parser.registry import parser_registry
        
        # Check parser is registered
        registered_parser = parser_registry.get_parser("java")
        assert registered_parser is not None
        assert isinstance(registered_parser, JavaParser)
        
        # Check file extensions
        assert java_parser.can_parse(Path("test.java"))
        assert not java_parser.can_parse(Path("test.py"))
        assert not java_parser.can_parse(Path("test.js"))
    
    def test_extract_packages(self, java_parser, tmp_path, sample_java_code):
        """Test package extraction"""
        test_file = tmp_path / "test.java"
        test_file.write_text(sample_java_code)
        
        result = java_parser.parse_file(test_file)
        entities = result.entities
        
        # Filter packages
        packages = [e for e in entities if e.entity_type == EntityType.MODULE]
        assert len(packages) >= 1
        
        # Check main package
        main_package = next((p for p in packages if p.name == "com.example.app"), None)
        assert main_package is not None
        assert main_package.visibility == Visibility.PUBLIC
        assert main_package.signature == "package com.example.app"
        assert main_package.metadata["package_name"] == "com.example.app"
    
    def test_extract_imports(self, java_parser, tmp_path, sample_java_code):
        """Test import extraction"""
        test_file = tmp_path / "test.java"
        test_file.write_text(sample_java_code)
        
        result = java_parser.parse_file(test_file)
        entities = result.entities
        
        # Filter imports
        imports = [e for e in entities if e.entity_type == EntityType.IMPORT]
        assert len(imports) >= 4
        
        import_names = [i.name for i in imports]
        assert "java.util.List" in import_names
        assert "java.util.ArrayList" in import_names
        assert "java.util.Collections.sort" in import_names  # static import
        assert "com.example.utils.*" in import_names  # wildcard import
        
        # Check static import
        static_import = next((i for i in imports if "Collections.sort" in i.name), None)
        assert static_import is not None
        assert static_import.metadata["is_static"] == True
        
        # Check wildcard import
        wildcard_import = next((i for i in imports if "com.example.utils.*" in i.name), None)
        assert wildcard_import is not None
        assert wildcard_import.metadata["is_wildcard"] == True
    
    def test_extract_classes(self, java_parser, tmp_path, sample_java_code):
        """Test class extraction"""
        test_file = tmp_path / "test.java"
        test_file.write_text(sample_java_code)
        
        result = java_parser.parse_file(test_file)
        entities = result.entities
        
        # Filter classes
        classes = [e for e in entities if e.entity_type == EntityType.CLASS]
        assert len(classes) >= 3
        
        class_names = [c.name for c in classes]
        assert "UserService" in class_names
        assert "User" in class_names
        assert "BaseRepository" in class_names
        
        # Check UserService class
        user_service = next((c for c in classes if c.name == "UserService"), None)
        assert user_service is not None
        assert user_service.visibility == Visibility.PUBLIC
        assert "public class UserService" in user_service.signature
        assert user_service.metadata["interfaces"] == ["UserRepository"]
        assert user_service.metadata["method_count"] >= 5
        assert user_service.metadata["field_count"] >= 4
        
        # Check BaseRepository class
        base_repo = next((c for c in classes if c.name == "BaseRepository"), None)
        assert base_repo is not None
        assert base_repo.metadata["is_abstract"] == True
        assert "T" in base_repo.metadata["type_parameters"]
    
    def test_extract_interfaces(self, java_parser, tmp_path, sample_java_code):
        """Test interface extraction"""
        test_file = tmp_path / "test.java"
        test_file.write_text(sample_java_code)
        
        result = java_parser.parse_file(test_file)
        entities = result.entities
        
        # Filter interfaces
        interfaces = [e for e in entities if e.entity_type == EntityType.INTERFACE]
        assert len(interfaces) >= 1
        
        # Check UserRepository interface
        user_repo = next((i for i in interfaces if i.name == "UserRepository"), None)
        assert user_repo is not None
        assert user_repo.visibility == Visibility.PUBLIC
        assert user_repo.signature == "public interface UserRepository"
        assert user_repo.metadata["method_count"] >= 3
        
        # Check methods
        methods = user_repo.metadata["methods"]
        method_names = [m["name"] for m in methods]
        assert "findAll" in method_names
        assert "save" in method_names
        assert "findById" in method_names
    
    def test_extract_enums(self, java_parser, tmp_path, sample_java_code):
        """Test enum extraction"""
        test_file = tmp_path / "test.java"
        test_file.write_text(sample_java_code)
        
        result = java_parser.parse_file(test_file)
        entities = result.entities
        
        # Filter enums
        enums = [e for e in entities if e.entity_type == EntityType.ENUM]
        assert len(enums) >= 1
        
        # Check UserStatus enum
        user_status = next((e for e in enums if e.name == "UserStatus"), None)
        assert user_status is not None
        assert user_status.visibility == Visibility.PUBLIC
        assert user_status.signature == "public enum UserStatus"
        assert user_status.metadata["constant_count"] >= 4
        
        # Check constants
        constants = user_status.metadata["constants"]
        assert "ACTIVE" in constants
        assert "INACTIVE" in constants
        assert "PENDING" in constants
        assert "SUSPENDED" in constants
    
    def test_extract_methods(self, java_parser, tmp_path, sample_java_code):
        """Test method extraction"""
        test_file = tmp_path / "test.java"
        test_file.write_text(sample_java_code)
        
        result = java_parser.parse_file(test_file)
        entities = result.entities
        
        # Filter methods
        methods = [e for e in entities if e.entity_type == EntityType.METHOD]
        assert len(methods) >= 10
        
        method_names = [m.name for m in methods]
        assert "getName" in method_names
        assert "setName" in method_names
        assert "findAll" in method_names
        assert "main" in method_names
        assert "validateUser" in method_names
        
        # Check getName method
        get_name = next((m for m in methods if m.name == "getName"), None)
        assert get_name is not None
        assert get_name.visibility == Visibility.PUBLIC
        assert get_name.metadata["return_type"] == "String"
        assert get_name.metadata["parameter_count"] == 0
        
        # Check setName method
        set_name = next((m for m in methods if m.name == "setName"), None)
        assert set_name is not None
        assert set_name.visibility == Visibility.PUBLIC
        assert set_name.metadata["return_type"] == "void"
        assert set_name.metadata["parameter_count"] == 1
        assert set_name.metadata["parameters"][0]["name"] == "name"
        assert set_name.metadata["parameters"][0]["type"] == "String"
        
        # Check main method
        main_method = next((m for m in methods if m.name == "main"), None)
        assert main_method is not None
        assert main_method.metadata["is_static"] == True
        assert main_method.metadata["parameter_count"] == 1
        
        # Check findAll method
        find_all = next((m for m in methods if m.name == "findAll"), None)
        assert find_all is not None
        assert "DatabaseException" in find_all.metadata["throws"]
        
        # Check private method
        validate_user = next((m for m in methods if m.name == "validateUser"), None)
        assert validate_user is not None
        assert validate_user.visibility == Visibility.PRIVATE
    
    def test_extract_fields(self, java_parser, tmp_path, sample_java_code):
        """Test field extraction"""
        test_file = tmp_path / "test.java"
        test_file.write_text(sample_java_code)
        
        result = java_parser.parse_file(test_file)
        entities = result.entities
        
        # Filter fields
        fields = [e for e in entities if e.entity_type == EntityType.VARIABLE]
        assert len(fields) >= 8
        
        field_names = [f.name for f in fields]
        assert "DEFAULT_NAME" in field_names
        assert "MAX_USERS" in field_names
        assert "name" in field_names
        assert "age" in field_names
        assert "roles" in field_names
        assert "active" in field_names
        
        # Check constant field
        default_name = next((f for f in fields if f.name == "DEFAULT_NAME"), None)
        assert default_name is not None
        assert default_name.visibility == Visibility.PUBLIC
        assert default_name.metadata["is_static"] == True
        assert default_name.metadata["is_final"] == True
        assert default_name.metadata["field_type"] == "String"
        
        # Check private field
        name_field = next((f for f in fields if f.name == "name" and f.metadata["field_type"] == "String"), None)
        assert name_field is not None
        assert name_field.visibility == Visibility.PRIVATE
        assert name_field.metadata["is_static"] == False
        
        # Check volatile field
        active_field = next((f for f in fields if f.name == "active"), None)
        assert active_field is not None
        assert active_field.metadata["is_volatile"] == True
    
    def test_extract_annotations(self, java_parser, tmp_path, sample_java_code):
        """Test annotation extraction"""
        test_file = tmp_path / "test.java"
        test_file.write_text(sample_java_code)
        
        result = java_parser.parse_file(test_file)
        entities = result.entities
        
        # Filter annotations (they appear as special classes)
        annotations = [e for e in entities if e.entity_type == EntityType.CLASS and e.metadata.get("is_annotation", False)]
        assert len(annotations) >= 1
        
        # Check Service annotation
        service_annotation = next((a for a in annotations if a.name == "Service"), None)
        assert service_annotation is not None
        assert service_annotation.visibility == Visibility.PUBLIC
        assert service_annotation.signature == "@interface Service"
        assert service_annotation.metadata["is_annotation"] == True
    
    def test_visibility_detection(self, java_parser, tmp_path):
        """Test visibility modifier detection"""
        test_code = '''
        public class PublicClass {}
        class PackageClass {}
        
        public class TestClass {
            public String publicField;
            private String privateField;
            protected String protectedField;
            String packageField;
            
            public void publicMethod() {}
            private void privateMethod() {}
            protected void protectedMethod() {}
            void packageMethod() {}
        }
        '''
        
        test_file = tmp_path / "test.java"
        test_file.write_text(test_code)
        
        result = java_parser.parse_file(test_file)
        entities = result.entities
        
        # Check class visibility
        public_class = next((e for e in entities if e.name == "PublicClass"), None)
        assert public_class is not None
        assert public_class.visibility == Visibility.PUBLIC
        
        package_class = next((e for e in entities if e.name == "PackageClass"), None)
        assert package_class is not None
        assert package_class.visibility == Visibility.PACKAGE_PRIVATE
        
        # Check field visibility
        fields = [e for e in entities if e.entity_type == EntityType.VARIABLE]
        
        public_field = next((f for f in fields if f.name == "publicField"), None)
        assert public_field is not None
        assert public_field.visibility == Visibility.PUBLIC
        
        private_field = next((f for f in fields if f.name == "privateField"), None)
        assert private_field is not None
        assert private_field.visibility == Visibility.PRIVATE
        
        protected_field = next((f for f in fields if f.name == "protectedField"), None)
        assert protected_field is not None
        assert protected_field.visibility == Visibility.PROTECTED
        
        package_field = next((f for f in fields if f.name == "packageField"), None)
        assert package_field is not None
        assert package_field.visibility == Visibility.PACKAGE_PRIVATE
        
        # Check method visibility
        methods = [e for e in entities if e.entity_type == EntityType.METHOD]
        
        public_method = next((m for m in methods if m.name == "publicMethod"), None)
        assert public_method is not None
        assert public_method.visibility == Visibility.PUBLIC
        
        private_method = next((m for m in methods if m.name == "privateMethod"), None)
        assert private_method is not None
        assert private_method.visibility == Visibility.PRIVATE
        
        protected_method = next((m for m in methods if m.name == "protectedMethod"), None)
        assert protected_method is not None
        assert protected_method.visibility == Visibility.PROTECTED
        
        package_method = next((m for m in methods if m.name == "packageMethod"), None)
        assert package_method is not None
        assert package_method.visibility == Visibility.PACKAGE_PRIVATE
    
    def test_generics_support(self, java_parser, tmp_path):
        """Test generic type parameter support"""
        test_code = '''
        public class GenericClass<T, U extends Comparable<U>> {
            private T value;
            private List<U> items;
            
            public <V> V process(V input, Function<T, V> mapper) {
                return mapper.apply(value);
            }
        }
        
        public interface GenericInterface<K, V> {
            V get(K key);
            void put(K key, V value);
        }
        '''
        
        test_file = tmp_path / "test.java"
        test_file.write_text(test_code)
        
        result = java_parser.parse_file(test_file)
        entities = result.entities
        
        # Check generic class
        generic_class = next((e for e in entities if e.name == "GenericClass"), None)
        assert generic_class is not None
        assert "T" in generic_class.metadata["type_parameters"]
        assert "U" in generic_class.metadata["type_parameters"]
        assert "<T, U extends Comparable<U>>" in generic_class.signature
        
        # Check generic interface
        generic_interface = next((e for e in entities if e.name == "GenericInterface"), None)
        assert generic_interface is not None
        assert "K" in generic_interface.metadata["type_parameters"]
        assert "V" in generic_interface.metadata["type_parameters"]
    
    def test_empty_file_parsing(self, java_parser, tmp_path):
        """Test parsing of empty Java file"""
        test_file = tmp_path / "empty.java"
        test_file.write_text("")
        
        result = java_parser.parse_file(test_file)
        assert result.entities == []
        assert result.relations == []
        assert result.parse_time > 0
    
    def test_invalid_java_parsing(self, java_parser, tmp_path):
        """Test parsing of invalid Java code"""
        test_code = '''
        public class InvalidClass {
            // Missing closing brace and invalid syntax
            public void method( {
                String s = "unclosed string
        '''
        
        test_file = tmp_path / "invalid.java"
        test_file.write_text(test_code)
        
        result = java_parser.parse_file(test_file)
        
        # Should still parse partially
        assert len(result.entities) >= 0
        assert result.partial_parse == True
        assert len(result.syntax_errors) > 0
    
    def test_metadata_completeness(self, java_parser, tmp_path, sample_java_code):
        """Test that extracted entities have complete metadata"""
        test_file = tmp_path / "test.java"
        test_file.write_text(sample_java_code)
        
        result = java_parser.parse_file(test_file)
        entities = result.entities
        
        # Should extract various entity types
        entity_types = [e.entity_type for e in entities]
        assert EntityType.MODULE in entity_types      # package
        assert EntityType.IMPORT in entity_types      # imports
        assert EntityType.CLASS in entity_types       # classes
        assert EntityType.INTERFACE in entity_types   # interfaces
        assert EntityType.ENUM in entity_types        # enums
        assert EntityType.METHOD in entity_types      # methods
        assert EntityType.VARIABLE in entity_types    # fields
        
        # Check that each entity has required metadata
        for entity in entities:
            assert entity.id is not None
            assert entity.name is not None
            assert entity.entity_type is not None
            assert entity.location is not None
            assert entity.signature is not None
            assert entity.source_code is not None
            assert entity.source_hash is not None
            assert entity.visibility is not None
            assert entity.metadata is not None
            assert entity.metadata["language"] == "java"
            assert "ast_node_type" in entity.metadata