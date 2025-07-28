"""
Tests for Python parser with comprehensive entity and relation extraction.

Tests the PythonParser implementation to ensure correct extraction of:
- Classes with inheritance
- Functions and methods with signatures
- Variables and constants
- Import statements
- Relations between entities
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from core.parser.python_parser import PythonParser
from core.parser.registry import parser_registry
from core.models.entities import (
    Entity, EntityType, Relation, RelationType, 
    SourceLocation, Visibility
)


@pytest.fixture
def python_parser():
    """Create a Python parser instance for testing"""
    return PythonParser()


@pytest.fixture
def sample_python_code():
    """Sample Python code for testing entity extraction"""
    return '''"""Module docstring for testing."""

import os
import sys
from typing import List, Dict, Optional
from datetime import datetime as dt

# Module-level constant
MAX_CONNECTIONS = 100
_private_var = "secret"

class BaseClass:
    """Base class with methods."""
    
    def __init__(self, name: str):
        self.name = name
    
    def get_name(self) -> str:
        """Get the name."""
        return self.name

class DerivedClass(BaseClass):
    """Derived class inheriting from BaseClass."""
    
    def __init__(self, name: str, value: int):
        super().__init__(name)
        self.value = value
    
    @property 
    def display_name(self) -> str:
        return f"{self.name}_{self.value}"
    
    @staticmethod
    def create_default():
        return DerivedClass("default", 0)
    
    async def async_method(self) -> None:
        """Async method example."""
        await some_async_call()

def module_function(items: List[str]) -> Dict[str, int]:
    """Process items and return counts."""
    result = {}
    for item in items:
        result[item] = len(item)
    return result

async def async_function(data: Optional[str] = None) -> str:
    """Async function example."""
    if data:
        return await process_data(data)
    return "empty"

# Test constant detection
DEBUG_MODE = True
'''


@pytest.fixture  
def complex_python_code():
    """More complex Python code with decorators and nested structures"""
    return '''
from functools import wraps
from typing import Callable, Any
import logging

logger = logging.getLogger(__name__)

def retry_decorator(max_attempts: int = 3):
    """Decorator for retrying functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
            return None
        return wrapper
    return decorator

class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, connection_string: str):
        self._connection_string = connection_string
        self._connections = []
    
    @retry_decorator(max_attempts=5)
    def connect(self) -> bool:
        """Connect to database with retry logic."""
        # Implementation here
        return True
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
'''


class TestPythonParserBasics:
    """Test basic Python parser functionality"""
    
    def test_parser_initialization(self, python_parser):
        """Test parser initialization and properties"""
        assert python_parser.get_language_name() == "python"
        assert ".py" in python_parser.get_supported_extensions()
        assert ".pyi" in python_parser.get_supported_extensions()
        assert ".pyw" in python_parser.get_supported_extensions()
    
    def test_can_parse_python_files(self, python_parser):
        """Test file extension detection"""
        assert python_parser.can_parse(Path("test.py"))
        assert python_parser.can_parse(Path("test.pyi"))
        assert python_parser.can_parse(Path("test.pyw"))
        assert not python_parser.can_parse(Path("test.js"))
        assert not python_parser.can_parse(Path("test.txt"))
    
    def test_parser_registration(self):
        """Test that Python parser is registered correctly"""
        # Check if parser is registered in global registry
        parser = parser_registry.get_parser("python")
        assert parser is not None
        assert isinstance(parser, PythonParser)
        
        # Check file mapping
        test_file = Path("test.py")
        file_parser = parser_registry.get_parser_for_file(test_file)
        assert file_parser is not None
        assert isinstance(file_parser, PythonParser)


class TestPythonEntityExtraction:
    """Test entity extraction from Python code"""
    
    def test_class_extraction(self, python_parser, tmp_path, sample_python_code):
        """Test extraction of class entities"""
        test_file = tmp_path / "test_classes.py"
        test_file.write_text(sample_python_code)
        
        result = python_parser.parse_file(test_file)
        
        # Find class entities
        classes = [e for e in result.entities if e.entity_type == EntityType.CLASS]
        assert len(classes) == 2
        
        # Check BaseClass
        base_class = next((c for c in classes if c.name == "BaseClass"), None)
        assert base_class is not None
        assert base_class.qualified_name == "BaseClass"
        assert base_class.visibility == Visibility.PUBLIC
        assert "Base class with methods" in base_class.docstring
        assert base_class.metadata["superclasses"] == []
        
        # Check DerivedClass with inheritance
        derived_class = next((c for c in classes if c.name == "DerivedClass"), None)
        assert derived_class is not None
        assert derived_class.qualified_name == "DerivedClass"
        assert "BaseClass" in derived_class.metadata["superclasses"]
        assert "Derived class inheriting" in derived_class.docstring
    
    def test_function_extraction(self, python_parser, tmp_path, sample_python_code):
        """Test extraction of function entities"""
        test_file = tmp_path / "test_functions.py"
        test_file.write_text(sample_python_code)
        
        result = python_parser.parse_file(test_file)
        
        # Find function entities (not methods)
        functions = [e for e in result.entities if e.entity_type == EntityType.FUNCTION]
        function_names = [f.name for f in functions]
        
        assert "module_function" in function_names
        assert "async_function" in function_names
        
        # Check module_function details
        module_func = next((f for f in functions if f.name == "module_function"), None)
        assert module_func is not None
        assert "List[str]" in module_func.signature
        assert "Dict[str, int]" in module_func.signature
        assert "Process items and return counts" in module_func.docstring
    
    def test_method_extraction(self, python_parser, tmp_path, sample_python_code):
        """Test extraction of method entities"""
        test_file = tmp_path / "test_methods.py" 
        test_file.write_text(sample_python_code)
        
        result = python_parser.parse_file(test_file)
        
        # Find method entities
        methods = [e for e in result.entities if e.entity_type == EntityType.METHOD]
        method_names = [m.name for m in methods]
        
        expected_methods = ["__init__", "get_name", "display_name", "create_default", "async_method"]
        for expected in expected_methods:
            assert expected in method_names
        
        # Check async method
        async_method = next((m for m in methods if m.name == "async_method"), None)
        assert async_method is not None
        assert async_method.is_async is True
        assert "async" in async_method.signature
        
        # Check property method
        property_method = next((m for m in methods if m.name == "display_name"), None)
        assert property_method is not None
        assert property_method.metadata["is_property"] is True
        assert "@property" in property_method.metadata["decorators"]
        
        # Check static method
        static_method = next((m for m in methods if m.name == "create_default"), None)
        assert static_method is not None
        assert static_method.metadata["is_static"] is True
        assert "@staticmethod" in static_method.metadata["decorators"]
    
    def test_variable_extraction(self, python_parser, tmp_path, sample_python_code):
        """Test extraction of variable and constant entities"""
        test_file = tmp_path / "test_variables.py"
        test_file.write_text(sample_python_code)
        
        result = python_parser.parse_file(test_file)
        
        # Find variable and constant entities
        variables = [e for e in result.entities if e.entity_type == EntityType.VARIABLE]
        constants = [e for e in result.entities if e.entity_type == EntityType.CONSTANT]
        
        variable_names = [v.name for v in variables]
        constant_names = [c.name for c in constants]
        
        # Check constants (uppercase with underscores)
        assert "MAX_CONNECTIONS" in constant_names
        assert "DEBUG_MODE" in constant_names
        
        # Check private variable
        assert "_private_var" in variable_names
        private_var = next((v for v in variables if v.name == "_private_var"), None)
        assert private_var is not None
        assert private_var.visibility == Visibility.PROTECTED
    
    def test_import_extraction(self, python_parser, tmp_path, sample_python_code):
        """Test extraction of import entities"""
        test_file = tmp_path / "test_imports.py"
        test_file.write_text(sample_python_code)
        
        result = python_parser.parse_file(test_file)
        
        # Find import entities
        imports = [e for e in result.entities if e.entity_type == EntityType.IMPORT]
        import_names = [i.name for i in imports]
        
        # Check regular imports
        assert "os" in import_names
        assert "sys" in import_names
        
        # Check from imports
        assert "List" in import_names
        assert "Dict" in import_names  
        assert "Optional" in import_names
        assert "dt" in import_names  # aliased import
        
        # Check import metadata
        dt_import = next((i for i in imports if i.name == "dt"), None)
        assert dt_import is not None
        assert dt_import.metadata["module_name"] == "datetime"
        assert dt_import.metadata["is_from_import"] is True


class TestPythonRelationExtraction:
    """Test relation extraction from Python code"""
    
    def test_inheritance_relations(self, python_parser, tmp_path, sample_python_code):
        """Test extraction of inheritance relations"""
        test_file = tmp_path / "test_inheritance.py"
        test_file.write_text(sample_python_code)
        
        result = python_parser.parse_file(test_file)
        
        # Find inheritance relations
        inheritance_relations = [r for r in result.relations if r.relation_type == RelationType.INHERITS]
        
        assert len(inheritance_relations) >= 1
        
        # Check DerivedClass inherits from BaseClass
        derived_inherits = next(
            (r for r in inheritance_relations if "DerivedClass" in r.source_entity_id), 
            None
        )
        assert derived_inherits is not None
        assert "BaseClass" in derived_inherits.target_entity_id
        assert "DerivedClass(BaseClass)" in derived_inherits.context
    
    def test_containment_relations(self, python_parser, tmp_path, sample_python_code):
        """Test extraction of containment relations"""
        test_file = tmp_path / "test_containment.py"
        test_file.write_text(sample_python_code)
        
        result = python_parser.parse_file(test_file)
        
        # Find containment relations
        containment_relations = [r for r in result.relations if r.relation_type == RelationType.CONTAINS]
        
        # Should have classes containing their methods
        assert len(containment_relations) > 0
        
        # Check that BaseClass contains its methods
        base_contains = [r for r in containment_relations if "BaseClass" in r.source_entity_id]
        assert len(base_contains) >= 2  # __init__ and get_name
    
    def test_decorator_relations(self, python_parser, tmp_path, complex_python_code):
        """Test extraction of decorator relations"""
        test_file = tmp_path / "test_decorators.py"
        test_file.write_text(complex_python_code)
        
        result = python_parser.parse_file(test_file)
        
        # Find decorator relations
        decorator_relations = [r for r in result.relations if r.relation_type == RelationType.DECORATES]
        
        assert len(decorator_relations) > 0
        
        # Check that retry_decorator decorates connect method
        retry_decorates = next(
            (r for r in decorator_relations if "retry_decorator" in r.source_entity_id),
            None
        )
        assert retry_decorates is not None
    
    def test_import_relations(self, python_parser, tmp_path, sample_python_code):
        """Test extraction of import relations"""
        test_file = tmp_path / "test_import_relations.py"
        test_file.write_text(sample_python_code)
        
        result = python_parser.parse_file(test_file)
        
        # Find import relations
        import_relations = [r for r in result.relations if r.relation_type == RelationType.IMPORTS]
        
        assert len(import_relations) > 0
        
        # Check that file imports modules
        os_import = next(
            (r for r in import_relations if "os" in r.target_entity_id),
            None
        )
        assert os_import is not None


class TestPythonParserEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_file_parsing(self, python_parser, tmp_path):
        """Test parsing empty Python file"""
        empty_file = tmp_path / "empty.py"
        empty_file.write_text("")
        
        result = python_parser.parse_file(empty_file)
        
        assert result.success is True
        assert result.entity_count == 0
        assert result.relation_count == 0
    
    def test_syntax_error_handling(self, python_parser, tmp_path):
        """Test handling of syntax errors"""
        bad_file = tmp_path / "syntax_error.py"
        bad_file.write_text("def func(\n  invalid syntax here\n  pass")
        
        result = python_parser.parse_file(bad_file)
        
        # Parser should handle gracefully
        assert result is not None
        # May have partial results or be marked as failed
    
    def test_large_file_handling(self, python_parser, tmp_path):
        """Test parsing large Python file"""
        large_content = "# Large file\n" + "x = 1\n" * 1000
        large_file = tmp_path / "large.py"
        large_file.write_text(large_content)
        
        result = python_parser.parse_file(large_file)
        
        assert result is not None
        assert result.file_size > 0
        # Should extract many variable entities
        variables = [e for e in result.entities if e.entity_type == EntityType.VARIABLE]
        assert len(variables) > 0
    
    def test_unicode_handling(self, python_parser, tmp_path):
        """Test handling of Unicode characters"""
        unicode_content = '''
"""测试 Unicode 支持."""

def 测试函数(参数: str) -> str:
    """Unicode function name and parameter."""
    return f"结果: {参数}"

class 测试类:
    """Unicode class name."""
    pass
'''
        unicode_file = tmp_path / "unicode.py"
        unicode_file.write_text(unicode_content, encoding='utf-8')
        
        result = python_parser.parse_file(unicode_file)
        
        assert result.success is True
        
        # Check Unicode entities were extracted
        functions = [e for e in result.entities if e.entity_type == EntityType.FUNCTION]
        classes = [e for e in result.entities if e.entity_type == EntityType.CLASS]
        
        assert any("测试函数" in f.name for f in functions)
        assert any("测试类" in c.name for c in classes)


class TestPythonParserIntegration:
    """Integration tests with registry and file discovery"""
    
    def test_registry_integration(self, tmp_path):
        """Test integration with parser registry"""
        # Create Python files
        files = []
        for i in range(3):
            py_file = tmp_path / f"module_{i}.py"
            py_file.write_text(f"def function_{i}(): pass")
            files.append(py_file)
        
        # Use registry to discover and parse files
        parseable_files = parser_registry.discover_files(tmp_path)
        python_files = [f for f in parseable_files if f.suffix == ".py"]
        
        assert len(python_files) == 3
        
        # Parse files in parallel
        results = parser_registry.parse_files_parallel(python_files)
        
        assert len(results) == 3
        assert all(r.success for r in results)
        
        # Check that functions were extracted
        total_functions = sum(r.entity_count for r in results)
        assert total_functions >= 3
    
    def test_performance_monitoring(self, python_parser, tmp_path, sample_python_code):
        """Test parse timing and performance monitoring"""
        test_file = tmp_path / "performance_test.py"
        test_file.write_text(sample_python_code)
        
        result = python_parser.parse_file(test_file)
        
        # Check timing information
        assert result.parse_time is not None
        assert result.parse_time >= 0
        assert result.file_size > 0
        assert result.file_hash is not None
        assert len(result.file_hash) > 0


# Performance benchmarks
@pytest.mark.benchmark
class TestPythonParserPerformance:
    """Performance benchmarks for Python parser"""
    
    def test_parsing_speed(self, python_parser, tmp_path, benchmark):
        """Benchmark parsing speed"""
        # Create moderately complex Python file
        complex_code = '''
import sys
from typing import List, Dict, Optional

class DataProcessor:
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.results = []
    
    def process_items(self, items: List[str]) -> Dict[str, int]:
        result = {}
        for item in items:
            result[item] = self._process_single(item)
        return result
    
    def _process_single(self, item: str) -> int:
        return len(item) * 2

    @property
    def item_count(self) -> int:
        return len(self.results)

def main():
    processor = DataProcessor({"mode": "fast"})
    items = ["a", "b", "c"] * 100
    result = processor.process_items(items)
    return result
''' * 10  # Repeat to make it larger
        
        test_file = tmp_path / "benchmark.py"
        test_file.write_text(complex_code)
        
        # Benchmark parsing
        def parse_file():
            return python_parser.parse_file(test_file)
        
        result = benchmark(parse_file)
        
        # Verify parsing worked
        assert result.success
        assert result.entity_count > 0
        
        # Performance targets (these are guidelines)
        assert result.parse_time < 1.0  # Should parse in under 1 second