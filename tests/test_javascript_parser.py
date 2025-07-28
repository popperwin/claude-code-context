"""
Tests for JavaScript/TypeScript parser with comprehensive entity and relation extraction.

Tests the JavaScriptParser and TypeScriptParser implementations to ensure correct extraction of:
- Functions (regular, arrow, async)
- Classes with inheritance
- Variables and constants (var, let, const)
- Import/export statements (ES6 modules)
- JSX components (React)
- TypeScript interfaces, types, enums
- Relations between entities
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from core.parser.javascript_parser import JavaScriptParser, TypeScriptParser
from core.parser.registry import parser_registry
from core.models.entities import (
    Entity, EntityType, Relation, RelationType, 
    SourceLocation, Visibility
)


@pytest.fixture
def javascript_parser():
    """Create a JavaScript parser instance for testing"""
    return JavaScriptParser()

@pytest.fixture
def typescript_parser():
    """Create a TypeScript parser instance for testing"""
    return TypeScriptParser()


@pytest.fixture
def sample_javascript_code():
    """Sample JavaScript code for testing entity extraction"""
    return '''/**
 * Module for user management functionality.
 * @fileoverview User management utilities
 */

import React from 'react';
import { Component } from 'react';
import { fetchData, processUser } from './api';
import * as Utils from './utils';

// Module-level constants
const MAX_USERS = 100;
const DEFAULT_TIMEOUT = 5000;
var globalCounter = 0;
let currentUser = null;

/**
 * Base user class with essential properties.
 */
class User {
    constructor(name, email) {
        this.name = name;
        this.email = email;
        this.id = User.generateId();
    }
    
    /**
     * Get user display name
     * @returns {string} Formatted display name
     */
    getDisplayName() {
        return `${this.name} <${this.email}>`;
    }
    
    static generateId() {
        return Math.random().toString(36);
    }
    
    async validateEmail() {
        const result = await fetchData(`/validate/${this.email}`);
        return result.valid;
    }
}

/**
 * Admin user extending base User class.
 */
class AdminUser extends User {
    constructor(name, email, permissions = []) {
        super(name, email);
        this.permissions = permissions;
    }
    
    hasPermission(permission) {
        return this.permissions.includes(permission);
    }
    
    get isAdmin() {
        return true;
    }
}

/**
 * Process array of users with async operations
 * @param {Array<User>} users - Array of user objects
 * @returns {Promise<Array>} Processed user data
 */
async function processUsers(users) {
    const results = [];
    for (const user of users) {
        const processed = await processUser(user);
        results.push(processed);
    }
    return results;
}

/**
 * Create user from form data
 * @param {Object} formData - Form data object
 * @returns {User} New user instance
 */
const createUserFromForm = (formData) => {
    const { name, email } = formData;
    return new User(name, email);
};

// Arrow function with destructuring
const getUserInfo = ({ name, email, id }) => ({
    displayName: `${name} (${id})`,
    contact: email
});

/**
 * React component for user display
 */
function UserProfile({ user, onEdit }) {
    const handleClick = () => {
        onEdit(user.id);
    };
    
    return (
        <div className="user-profile">
            <h2>{user.getDisplayName()}</h2>
            <button onClick={handleClick}>Edit</button>
        </div>
    );
}

// Export statements
export { User, AdminUser };
export default processUsers;
export const utilities = Utils;
'''


@pytest.fixture
def sample_typescript_code():
    """Sample TypeScript code for testing entity extraction"""
    return '''/**
 * TypeScript interfaces and types for user management
 */

import { EventEmitter } from 'events';
import type { ReactNode } from 'react';

// Type aliases
export type UserId = string;
export type UserRole = 'admin' | 'user' | 'guest';
export type UserStatus = 'active' | 'inactive' | 'pending';

// Interface definitions
export interface IUser {
    id: UserId;
    name: string;
    email: string;
    role: UserRole;
    status: UserStatus;
    createdAt?: Date;
}

export interface IUserRepository {
    findById(id: UserId): Promise<IUser | null>;
    create(userData: Omit<IUser, 'id'>): Promise<IUser>;
    update(id: UserId, updates: Partial<IUser>): Promise<IUser>;
    delete(id: UserId): Promise<boolean>;
}

// Generic interface
export interface ApiResponse<T> {
    data: T;
    status: number;
    message?: string;
}

// Enum definitions
export enum UserPermission {
    READ = 'read',
    WRITE = 'write',
    DELETE = 'delete',
    ADMIN = 'admin'
}

export enum NotificationTypes {
    EMAIL,
    SMS,
    PUSH
}

/**
 * Generic user service class with TypeScript features
 */
export class UserService<T extends IUser> extends EventEmitter {
    private users: Map<UserId, T> = new Map();
    private readonly repository: IUserRepository;
    
    constructor(repository: IUserRepository) {
        super();
        this.repository = repository;
    }
    
    /**
     * Add user with generic constraints
     */
    async addUser<U extends T>(userData: Omit<U, 'id'>): Promise<U> {
        const user = await this.repository.create(userData) as U;
        this.users.set(user.id, user);
        this.emit('userAdded', user);
        return user;
    }
    
    /**
     * Get user by ID with optional type narrowing
     */
    getUser(id: UserId): T | undefined {
        return this.users.get(id);
    }
    
    /**
     * Filter users by predicate function
     */
    filterUsers(predicate: (user: T) => boolean): T[] {
        return Array.from(this.users.values()).filter(predicate);
    }
    
    /**
     * Optional parameter method
     */
    updateUserStatus(id: UserId, status: UserStatus, reason?: string): void {
        const user = this.users.get(id);
        if (user) {
            user.status = status;
            this.emit('statusChanged', { user, reason });
        }
    }
}

/**
 * React component with TypeScript props
 */
interface UserListProps {
    users: IUser[];
    onUserSelect?: (user: IUser) => void;
    className?: string;
}

export const UserList: React.FC<UserListProps> = ({ 
    users, 
    onUserSelect,
    className = 'user-list' 
}) => {
    return (
        <div className={className}>
            {users.map(user => (
                <div key={user.id} onClick={() => onUserSelect?.(user)}>
                    {user.name} - {user.role}
                </div>
            ))}
        </div>
    );
};

// Utility type functions
export const isAdminUser = (user: IUser): user is IUser & { role: 'admin' } => {
    return user.role === 'admin';
};

// Namespace declaration
export namespace UserUtils {
    export const formatUserName = (user: IUser): string => {
        return `${user.name} (${user.role})`;
    };
    
    export const validateEmail = (email: string): boolean => {
        return /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/.test(email);
    };
}
'''


@pytest.fixture
def complex_javascript_code():
    """Complex JavaScript with modern features"""
    return '''
import { debounce, throttle } from 'lodash';
import EventEmitter from 'events';

/**
 * Advanced JavaScript patterns and modern syntax
 */

// Destructuring with defaults
const { 
    API_URL = 'http://localhost:3000',
    MAX_RETRIES = 3,
    TIMEOUT = 5000 
} = process.env;

// Class with private fields and decorators
class DataManager extends EventEmitter {
    #cache = new Map();
    #isInitialized = false;
    
    constructor(options = {}) {
        super();
        this.options = { ...this.defaultOptions, ...options };
        this.debouncedSave = debounce(this.save.bind(this), 300);
    }
    
    get defaultOptions() {
        return {
            autoSave: true,
            compression: false,
            maxCacheSize: 100
        };
    }
    
    // Async generator method
    async* fetchDataStream(query) {
        let page = 1;
        let hasMore = true;
        
        while (hasMore) {
            const response = await this.fetchPage(query, page);
            yield* response.items;
            
            hasMore = response.hasMore;
            page++;
        }
    }
    
    // Method with complex parameters
    async processWithRetry(
        operation,
        { maxRetries = MAX_RETRIES, delay = 1000 } = {}
    ) {
        for (let attempt = 1; attempt <= maxRetries; attempt++) {
            try {
                return await operation();
            } catch (error) {
                if (attempt === maxRetries) throw error;
                await this.delay(delay * attempt);
            }
        }
    }
    
    // Template literal method
    generateQuery(table, conditions = {}) {
        const whereClause = Object.entries(conditions)
            .map(([key, value]) => `${key} = '${value}'`)
            .join(' AND ');
            
        return `
            SELECT * FROM ${table}
            ${whereClause ? `WHERE ${whereClause}` : ''}
            ORDER BY created_at DESC
        `;
    }
}

// Higher-order function
const withErrorBoundary = (WrappedComponent) => {
    return class ErrorBoundary extends React.Component {
        constructor(props) {
            super(props);
            this.state = { hasError: false };
        }
        
        static getDerivedStateFromError(error) {
            return { hasError: true };
        }
        
        componentDidCatch(error, errorInfo) {
            console.error('Error caught by boundary:', error, errorInfo);
        }
        
        render() {
            if (this.state.hasError) {
                return <div>Something went wrong.</div>;
            }
            
            return <WrappedComponent {...this.props} />;
        }
    };
};

// Advanced React hooks pattern
const useDataFetcher = (url, dependencies = []) => {
    const [data, setData] = React.useState(null);
    const [loading, setLoading] = React.useState(false);
    const [error, setError] = React.useState(null);
    
    React.useEffect(() => {
        let cancelled = false;
        
        const fetchData = async () => {
            setLoading(true);
            setError(null);
            
            try {
                const response = await fetch(url);
                const result = await response.json();
                
                if (!cancelled) {
                    setData(result);
                }
            } catch (err) {
                if (!cancelled) {
                    setError(err.message);
                }
            } finally {
                if (!cancelled) {
                    setLoading(false);
                }
            }
        };
        
        fetchData();
        
        return () => {
            cancelled = true;
        };
    }, dependencies);
    
    return { data, loading, error };
};

export { DataManager, withErrorBoundary, useDataFetcher };
export default DataManager;
'''


class TestJavaScriptParserBasics:
    """Test basic JavaScript parser functionality"""
    
    def test_parser_initialization(self, javascript_parser):
        """Test parser initialization and properties"""
        assert javascript_parser.get_language_name() == "javascript"
        assert ".js" in javascript_parser.get_supported_extensions()
        assert ".jsx" in javascript_parser.get_supported_extensions()
        assert ".mjs" in javascript_parser.get_supported_extensions()
        assert ".cjs" in javascript_parser.get_supported_extensions()
    
    def test_can_parse_javascript_files(self, javascript_parser):
        """Test file extension detection"""
        assert javascript_parser.can_parse(Path("test.js"))
        assert javascript_parser.can_parse(Path("test.jsx"))
        assert javascript_parser.can_parse(Path("test.mjs"))
        assert javascript_parser.can_parse(Path("test.cjs"))
        assert not javascript_parser.can_parse(Path("test.py"))
        assert not javascript_parser.can_parse(Path("test.txt"))
    
    def test_typescript_parser_initialization(self, typescript_parser):
        """Test TypeScript parser initialization"""
        assert typescript_parser.get_language_name() == "typescript"
        assert ".ts" in typescript_parser.get_supported_extensions()
        assert ".tsx" in typescript_parser.get_supported_extensions()
        assert ".d.ts" in typescript_parser.get_supported_extensions()
    
    def test_parser_registration(self):
        """Test that JavaScript parsers are registered correctly"""
        # Check JavaScript parser registration
        js_parser = parser_registry.get_parser("javascript")
        assert js_parser is not None
        assert isinstance(js_parser, JavaScriptParser)
        
        # Check TypeScript parser registration
        ts_parser = parser_registry.get_parser("typescript")
        assert ts_parser is not None
        assert isinstance(ts_parser, TypeScriptParser)
        
        # Check file mapping
        js_file = Path("test.js")
        js_file_parser = parser_registry.get_parser_for_file(js_file)
        assert js_file_parser is not None
        assert isinstance(js_file_parser, JavaScriptParser)
        
        ts_file = Path("test.ts")
        ts_file_parser = parser_registry.get_parser_for_file(ts_file)
        assert ts_file_parser is not None
        assert isinstance(ts_file_parser, TypeScriptParser)


class TestJavaScriptEntityExtraction:
    """Test entity extraction from JavaScript code"""
    
    def test_function_extraction(self, javascript_parser, tmp_path, sample_javascript_code):
        """Test extraction of function entities"""
        test_file = tmp_path / "test_functions.js"
        test_file.write_text(sample_javascript_code)
        
        result = javascript_parser.parse_file(test_file)
        
        # Find function entities
        functions = [e for e in result.entities if e.entity_type == EntityType.FUNCTION]
        function_names = [f.name for f in functions]
        
        # Check regular functions
        assert "processUsers" in function_names
        assert "UserProfile" in function_names  # React component
        
        # Check arrow functions
        assert "createUserFromForm" in function_names
        assert "getUserInfo" in function_names
        
        # Check async function
        process_users = next((f for f in functions if f.name == "processUsers"), None)
        assert process_users is not None
        assert process_users.is_async is True
        assert "async function" in process_users.signature
        assert "Process array of users" in process_users.docstring
    
    def test_class_extraction(self, javascript_parser, tmp_path, sample_javascript_code):
        """Test extraction of class entities"""
        test_file = tmp_path / "test_classes.js"
        test_file.write_text(sample_javascript_code)
        
        result = javascript_parser.parse_file(test_file)
        
        # Find class entities
        classes = [e for e in result.entities if e.entity_type == EntityType.CLASS]
        assert len(classes) == 2
        
        # Check User class
        user_class = next((c for c in classes if c.name == "User"), None)
        assert user_class is not None
        assert user_class.qualified_name == "User"
        assert user_class.visibility == Visibility.PUBLIC
        assert "Base user class" in user_class.docstring
        assert user_class.metadata["superclasses"] == []
        
        # Check AdminUser with inheritance
        admin_class = next((c for c in classes if c.name == "AdminUser"), None)
        assert admin_class is not None
        assert admin_class.qualified_name == "AdminUser"
        assert "User" in admin_class.metadata["superclasses"]
        assert "Admin user extending" in admin_class.docstring
    
    def test_method_extraction(self, javascript_parser, tmp_path, sample_javascript_code):
        """Test extraction of method entities"""
        test_file = tmp_path / "test_methods.js"
        test_file.write_text(sample_javascript_code)
        
        result = javascript_parser.parse_file(test_file)
        
        # Find method entities
        methods = [e for e in result.entities if e.entity_type == EntityType.METHOD]
        method_names = [m.name for m in methods]
        
        expected_methods = ["constructor", "getDisplayName", "generateId", "validateEmail", "hasPermission"]
        for expected in expected_methods:
            assert expected in method_names
        
        # Check static method
        generate_id = next((m for m in methods if m.name == "generateId"), None)
        assert generate_id is not None
        assert generate_id.metadata["is_static"] is True
        
        # Check async method
        validate_email = next((m for m in methods if m.name == "validateEmail"), None)
        assert validate_email is not None
        assert validate_email.is_async is True
        
        # Check getter method
        is_admin = next((m for m in methods if m.name == "isAdmin"), None)
        assert is_admin is not None
        assert is_admin.metadata.get("is_getter", False)
    
    def test_variable_extraction(self, javascript_parser, tmp_path, sample_javascript_code):
        """Test extraction of variable and constant entities"""
        test_file = tmp_path / "test_variables.js"
        test_file.write_text(sample_javascript_code)
        
        result = javascript_parser.parse_file(test_file)
        
        # Find variable and constant entities
        variables = [e for e in result.entities if e.entity_type == EntityType.VARIABLE]
        constants = [e for e in result.entities if e.entity_type == EntityType.CONSTANT]
        
        variable_names = [v.name for v in variables]
        constant_names = [c.name for c in constants]
        
        # Check constants (const declarations)
        assert "MAX_USERS" in constant_names
        assert "DEFAULT_TIMEOUT" in constant_names
        
        # Check variables (var, let)
        assert "globalCounter" in variable_names
        assert "currentUser" in variable_names
        
        # Check variable metadata
        global_counter = next((v for v in variables if v.name == "globalCounter"), None)
        assert global_counter is not None
        assert global_counter.metadata["declaration_type"] == "var"
    
    def test_import_export_extraction(self, javascript_parser, tmp_path, sample_javascript_code):
        """Test extraction of import and export entities"""
        test_file = tmp_path / "test_imports.js"
        test_file.write_text(sample_javascript_code)
        
        result = javascript_parser.parse_file(test_file)
        
        # Find import entities
        imports = [e for e in result.entities if e.entity_type == EntityType.IMPORT]
        exports = [e for e in result.entities if e.entity_type == EntityType.EXPORT]
        
        import_names = [i.name for i in imports]
        export_names = [e.name for e in exports]
        
        # Check imports
        assert "React" in import_names  # default import
        assert "Component" in import_names  # named import
        assert "fetchData" in import_names
        assert "processUser" in import_names
        assert "Utils" in import_names  # namespace import
        
        # Check exports
        assert "User" in export_names
        assert "AdminUser" in export_names
        assert "processUsers" in export_names  # default export
        assert "utilities" in export_names
        
        # Check import metadata
        react_import = next((i for i in imports if i.name == "React"), None)
        assert react_import is not None
        assert react_import.metadata["module_name"] == "react"
        assert react_import.metadata["is_default_import"] is True
    
    def test_jsx_component_extraction(self, javascript_parser, tmp_path, sample_javascript_code):
        """Test extraction of JSX components"""
        test_file = tmp_path / "test_jsx.js"
        test_file.write_text(sample_javascript_code)
        
        result = javascript_parser.parse_file(test_file)
        
        # UserProfile should be detected as a function (React component)
        functions = [e for e in result.entities if e.entity_type == EntityType.FUNCTION]
        user_profile = next((f for f in functions if f.name == "UserProfile"), None)
        
        assert user_profile is not None
        # Check for framework-agnostic JSX detection instead of React-specific
        assert user_profile.metadata.get("returns_jsx", False) or user_profile.metadata.get("contains_jsx", False)
        assert "React component" in user_profile.docstring


class TestTypeScriptEntityExtraction:
    """Test entity extraction from TypeScript code"""
    
    def test_interface_extraction(self, typescript_parser, tmp_path, sample_typescript_code):
        """Test extraction of TypeScript interface entities"""
        test_file = tmp_path / "test_interfaces.ts"
        test_file.write_text(sample_typescript_code)
        
        result = typescript_parser.parse_file(test_file)
        
        # Find interface entities
        interfaces = [e for e in result.entities if e.entity_type == EntityType.INTERFACE]
        interface_names = [i.name for i in interfaces]
        
        expected_interfaces = ["IUser", "IUserRepository", "ApiResponse", "UserListProps"]
        for expected in expected_interfaces:
            assert expected in interface_names
        
        # Check generic interface
        api_response = next((i for i in interfaces if i.name == "ApiResponse"), None)
        assert api_response is not None
        assert "T" in api_response.metadata.get("generic_parameters", [])
    
    def test_type_alias_extraction(self, typescript_parser, tmp_path, sample_typescript_code):
        """Test extraction of TypeScript type aliases"""
        test_file = tmp_path / "test_types.ts"
        test_file.write_text(sample_typescript_code)
        
        result = typescript_parser.parse_file(test_file)
        
        # Find type entities
        types = [e for e in result.entities if e.entity_type == EntityType.TYPE]
        type_names = [t.name for t in types]
        
        expected_types = ["UserId", "UserRole", "UserStatus"]
        for expected in expected_types:
            assert expected in type_names
        
        # Check union type
        user_role = next((t for t in types if t.name == "UserRole"), None)
        assert user_role is not None
        assert "admin" in str(user_role.metadata.get("type_definition", ""))
    
    def test_enum_extraction(self, typescript_parser, tmp_path, sample_typescript_code):
        """Test extraction of TypeScript enum entities"""
        test_file = tmp_path / "test_enums.ts"
        test_file.write_text(sample_typescript_code)
        
        result = typescript_parser.parse_file(test_file)
        
        # Find enum entities
        enums = [e for e in result.entities if e.entity_type == EntityType.ENUM]
        enum_names = [e.name for e in enums]
        
        expected_enums = ["UserPermission", "NotificationTypes"]
        for expected in expected_enums:
            assert expected in enum_names
        
        # Check string enum
        user_permission = next((e for e in enums if e.name == "UserPermission"), None)
        assert user_permission is not None
        assert "READ" in user_permission.metadata.get("members", [])
    
    def test_generic_class_extraction(self, typescript_parser, tmp_path, sample_typescript_code):
        """Test extraction of generic TypeScript classes"""
        test_file = tmp_path / "test_generics.ts"
        test_file.write_text(sample_typescript_code)
        
        result = typescript_parser.parse_file(test_file)
        
        # Find class entities
        classes = [e for e in result.entities if e.entity_type == EntityType.CLASS]
        user_service = next((c for c in classes if c.name == "UserService"), None)
        
        assert user_service is not None
        assert "T" in user_service.metadata.get("generic_parameters", [])
        assert "EventEmitter" in user_service.metadata["superclasses"]
    
    def test_namespace_extraction(self, typescript_parser, tmp_path, sample_typescript_code):
        """Test extraction of TypeScript namespace entities"""
        test_file = tmp_path / "test_namespace.ts"
        test_file.write_text(sample_typescript_code)
        
        result = typescript_parser.parse_file(test_file)
        
        # Find namespace entities
        namespaces = [e for e in result.entities if e.entity_type == EntityType.NAMESPACE]
        namespace_names = [n.name for n in namespaces]
        
        assert "UserUtils" in namespace_names
        
        user_utils = next((n for n in namespaces if n.name == "UserUtils"), None)
        assert user_utils is not None
        assert user_utils.visibility == Visibility.PUBLIC


class TestJavaScriptRelationExtraction:
    """Test relation extraction from JavaScript code"""
    
    def test_inheritance_relations(self, javascript_parser, tmp_path, sample_javascript_code):
        """Test extraction of inheritance relations"""
        test_file = tmp_path / "test_inheritance.js"
        test_file.write_text(sample_javascript_code)
        
        result = javascript_parser.parse_file(test_file)
        
        # Find inheritance relations
        inheritance_relations = [r for r in result.relations if r.relation_type == RelationType.INHERITS]
        
        assert len(inheritance_relations) >= 1
        
        # Check AdminUser inherits from User
        admin_inherits = next(
            (r for r in inheritance_relations if "AdminUser" in r.source_entity_id), 
            None
        )
        assert admin_inherits is not None
        assert "User" in admin_inherits.target_entity_id
    
    def test_import_relations(self, javascript_parser, tmp_path, sample_javascript_code):
        """Test extraction of import relations"""
        test_file = tmp_path / "test_import_relations.js"
        test_file.write_text(sample_javascript_code)
        
        result = javascript_parser.parse_file(test_file)
        
        # Find import relations
        import_relations = [r for r in result.relations if r.relation_type == RelationType.IMPORTS]
        
        assert len(import_relations) > 0
        
        # Check React import
        react_import = next(
            (r for r in import_relations if "React" in r.target_entity_id),
            None
        )
        assert react_import is not None
    
    def test_call_relations(self, javascript_parser, tmp_path, complex_javascript_code):
        """Test extraction of function call relations"""
        test_file = tmp_path / "test_calls.js"
        test_file.write_text(complex_javascript_code)
        
        result = javascript_parser.parse_file(test_file)
        
        # Find call relations
        call_relations = [r for r in result.relations if r.relation_type == RelationType.CALLS]
        
        assert len(call_relations) > 0


class TestJavaScriptJSXSupport:
    """Test JSX and TSX support in JavaScript/TypeScript parsers"""
    
    @pytest.fixture
    def jsx_code(self):
        """Sample JSX code for testing"""
        return '''
import React from 'react';

function UserProfile({ user, onEdit }) {
    const handleClick = () => {
        onEdit(user.id);
    };
    
    return (
        <div className="user-profile">
            <h2>{user.name}</h2>
            <button onClick={handleClick}>Edit</button>
        </div>
    );
}

const UserCard = ({ user }) => (
    <div className="card">
        <img src={user.avatar} alt={user.name} />
        <span>{user.email}</span>
    </div>
);

export { UserProfile, UserCard };
'''
    
    @pytest.fixture
    def tsx_code(self):
        """Sample TSX code for testing"""
        return '''
import React from 'react';

interface User {
    id: string;
    name: string;
    email: string;
}

interface UserListProps {
    users: User[];
    onUserSelect?: (user: User) => void;
}

export const UserList: React.FC<UserListProps> = ({ users, onUserSelect }) => {
    return (
        <div className="user-list">
            {users.map(user => (
                <div key={user.id} onClick={() => onUserSelect?.(user)}>
                    <span>{user.name}</span> - <span>{user.email}</span>
                </div>
            ))}
        </div>
    );
};
'''
    
    def test_jsx_parsing(self, javascript_parser, tmp_path, jsx_code):
        """Test JSX file parsing with React components"""
        jsx_file = tmp_path / "component.jsx"
        jsx_file.write_text(jsx_code)
        
        result = javascript_parser.parse_file(jsx_file)
        
        assert result.success is True
        assert result.entity_count > 0
        
        # Check for React functions
        functions = [e for e in result.entities if e.entity_type == EntityType.FUNCTION]
        function_names = [f.name for f in functions]
        
        assert "UserProfile" in function_names
        assert "UserCard" in function_names
        assert "handleClick" in function_names
        
        # Check for React import
        imports = [e for e in result.entities if e.entity_type == EntityType.IMPORT]
        import_names = [i.name for i in imports]
        assert "React" in import_names
    
    def test_tsx_parsing(self, typescript_parser, tmp_path, tsx_code):
        """Test TSX file parsing with TypeScript interfaces and React components"""
        tsx_file = tmp_path / "component.tsx"
        tsx_file.write_text(tsx_code)
        
        result = typescript_parser.parse_file(tsx_file)
        
        assert result.success is True
        assert result.entity_count > 0
        
        # Check for TypeScript interfaces
        interfaces = [e for e in result.entities if e.entity_type == EntityType.INTERFACE]
        interface_names = [i.name for i in interfaces]
        
        assert "User" in interface_names
        assert "UserListProps" in interface_names
        
        # Check for React function
        functions = [e for e in result.entities if e.entity_type == EntityType.FUNCTION]
        function_names = [f.name for f in functions]
        assert "UserList" in function_names
        
        # Check for React import
        imports = [e for e in result.entities if e.entity_type == EntityType.IMPORT]
        import_names = [i.name for i in imports]
        assert "React" in import_names
    
    def test_jsx_extensions_supported(self, javascript_parser):
        """Test that JSX extensions are supported"""
        assert javascript_parser.can_parse(Path("component.jsx"))
        assert ".jsx" in javascript_parser.get_supported_extensions()
    
    def test_tsx_extensions_supported(self, typescript_parser):
        """Test that TSX extensions are supported"""
        assert typescript_parser.can_parse(Path("component.tsx"))
        assert ".tsx" in typescript_parser.get_supported_extensions()
    
    def test_react_component_detection(self, javascript_parser, tmp_path):
        """Test detection of React components"""
        react_code = '''
import React from 'react';

function MyComponent(props) {
    return <div>{props.children}</div>;
}

const AnotherComponent = () => {
    return <span>Hello</span>;
};

// Regular function (not React component)
function utilityFunction() {
    return "not jsx";
}

export { MyComponent, AnotherComponent };
'''
        
        jsx_file = tmp_path / "components.jsx"
        jsx_file.write_text(react_code)
        
        result = javascript_parser.parse_file(jsx_file)
        
        assert result.success is True
        
        functions = [e for e in result.entities if e.entity_type == EntityType.FUNCTION]
        
        # All functions should be detected
        function_names = [f.name for f in functions]
        assert "MyComponent" in function_names
        assert "AnotherComponent" in function_names
        assert "utilityFunction" in function_names


class TestJavaScriptParserEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_file_parsing(self, javascript_parser, tmp_path):
        """Test parsing empty JavaScript file"""
        empty_file = tmp_path / "empty.js"
        empty_file.write_text("")
        
        result = javascript_parser.parse_file(empty_file)
        
        assert result.success is True
        assert result.entity_count == 0
        assert result.relation_count == 0
    
    def test_syntax_error_handling(self, javascript_parser, tmp_path):
        """Test handling of syntax errors"""
        bad_file = tmp_path / "syntax_error.js"
        bad_file.write_text("function test( { invalid syntax here")
        
        result = javascript_parser.parse_file(bad_file)
        
        # Parser should handle gracefully
        assert result is not None
        # May have partial results or be marked as failed
    
    def test_modern_syntax_parsing(self, javascript_parser, tmp_path, complex_javascript_code):
        """Test parsing modern JavaScript syntax"""
        test_file = tmp_path / "modern.js"
        test_file.write_text(complex_javascript_code)
        
        result = javascript_parser.parse_file(test_file)
        
        assert result.success is True
        assert result.entity_count > 0
        
        # Should extract modern features
        classes = [e for e in result.entities if e.entity_type == EntityType.CLASS]
        assert len(classes) > 0
        
        functions = [e for e in result.entities if e.entity_type == EntityType.FUNCTION]
        assert len(functions) > 0
    
    def test_unicode_handling(self, javascript_parser, tmp_path):
        """Test handling of Unicode characters"""
        unicode_content = '''
/**
 * 测试 Unicode 支持
 */

function 测试函数(参数) {
    return `结果: ${参数}`;
}

class 测试类 {
    constructor(名称) {
        this.名称 = 名称;
    }
}

export { 测试函数, 测试类 };
'''
        unicode_file = tmp_path / "unicode.js"
        unicode_file.write_text(unicode_content, encoding='utf-8')
        
        result = javascript_parser.parse_file(unicode_file)
        
        assert result.success is True
        
        # Check Unicode entities were extracted
        functions = [e for e in result.entities if e.entity_type == EntityType.FUNCTION]
        classes = [e for e in result.entities if e.entity_type == EntityType.CLASS]
        
        assert any("测试函数" in f.name for f in functions)
        assert any("测试类" in c.name for c in classes)


class TestJavaScriptParserIntegration:
    """Integration tests with registry and file discovery"""
    
    def test_registry_integration(self, tmp_path):
        """Test integration with parser registry"""
        # Create JavaScript files
        files = []
        for i in range(3):
            js_file = tmp_path / f"module_{i}.js"
            js_file.write_text(f"function function_{i}() {{ return {i}; }}")
            files.append(js_file)
        
        # Use registry to discover and parse files
        parseable_files = parser_registry.discover_files(tmp_path)
        js_files = [f for f in parseable_files if f.suffix == ".js"]
        
        assert len(js_files) == 3
        
        # Parse files in parallel
        results = parser_registry.parse_files_parallel(js_files)
        
        assert len(results) == 3
        assert all(r.success for r in results)
        
        # Check that functions were extracted
        total_functions = sum(r.entity_count for r in results)
        assert total_functions >= 3
    
    def test_mixed_file_types(self, tmp_path):
        """Test parsing mixed JavaScript and TypeScript files"""
        # Create mixed files
        js_file = tmp_path / "script.js"
        js_file.write_text("function jsFunction() { return 'js'; }")
        
        ts_file = tmp_path / "script.ts"
        ts_file.write_text("interface ITest { name: string; }\nfunction tsFunction(): string { return 'ts'; }")
        
        # Parse both files
        parseable_files = parser_registry.discover_files(tmp_path)
        results = parser_registry.parse_files_parallel(parseable_files)
        
        assert len(results) == 2
        assert all(r.success for r in results)
        
        # Check different entity types were extracted
        all_entities = []
        for result in results:
            all_entities.extend(result.entities)
        
        entity_types = {e.entity_type for e in all_entities}
        assert EntityType.FUNCTION in entity_types
        assert EntityType.INTERFACE in entity_types  # From TypeScript file


# Performance benchmarks
@pytest.mark.benchmark
class TestJavaScriptParserPerformance:
    """Performance benchmarks for JavaScript parser"""
    
    def test_parsing_speed(self, javascript_parser, tmp_path, benchmark):
        """Benchmark JavaScript parsing speed"""
        # Create moderately complex JavaScript file
        complex_code = '''
import React from 'react';
import { useState, useEffect } from 'react';

class DataProcessor {
    constructor(config) {
        this.config = config;
        this.cache = new Map();
    }
    
    async processItems(items) {
        const results = [];
        for (const item of items) {
            const processed = await this.processItem(item);
            results.push(processed);
        }
        return results;
    }
    
    processItem(item) {
        return new Promise(resolve => {
            setTimeout(() => resolve(item.toUpperCase()), 10);
        });
    }
}

const useDataProcessor = (config) => {
    const [processor] = useState(() => new DataProcessor(config));
    const [data, setData] = useState([]);
    
    useEffect(() => {
        processor.processItems(['a', 'b', 'c'])
            .then(setData);
    }, [processor]);
    
    return { data, processor };
};

export { DataProcessor, useDataProcessor };
''' * 5  # Repeat to make it larger
        
        test_file = tmp_path / "benchmark.js"
        test_file.write_text(complex_code)
        
        # Benchmark parsing
        def parse_file():
            return javascript_parser.parse_file(test_file)
        
        result = benchmark(parse_file)
        
        # Verify parsing worked
        assert result.success
        assert result.entity_count > 0
        
        # Performance targets (guidelines)
        assert result.parse_time < 1.0  # Should parse in under 1 second