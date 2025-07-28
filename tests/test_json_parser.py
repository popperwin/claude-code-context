"""
Tests for JSON parser with comprehensive entity and relation extraction.

Tests the JSONParser implementation to ensure correct extraction of:
- JSON objects and nested structures
- Arrays and data collections
- Key-value pairs with type detection
- Configuration sections
- Relations between entities
"""

import pytest
from pathlib import Path

from core.parser.json_parser import JSONParser
from core.parser.registry import parser_registry
from core.models.entities import (
    Entity, EntityType, Relation, RelationType, 
    SourceLocation, Visibility
)


@pytest.fixture
def json_parser():
    """Create a JSON parser instance for testing"""
    return JSONParser()


@pytest.fixture
def sample_json_code():
    """Sample JSON code for testing entity extraction"""
    return '''{
  "name": "test-project",
  "version": "1.0.0",
  "description": "A test project for JSON parsing",
  "main": "index.js",
  "scripts": {
    "start": "node index.js",
    "test": "jest",
    "build": "webpack --mode production",
    "dev": "webpack-dev-server --mode development"
  },
  "dependencies": {
    "express": "^4.18.0",
    "lodash": "^4.17.21",
    "axios": "^1.3.0"
  },
  "devDependencies": {
    "jest": "^29.0.0",
    "webpack": "^5.70.0",
    "webpack-dev-server": "^4.7.0"
  },
  "config": {
    "port": 3000,
    "host": "localhost",
    "database": {
      "host": "localhost",
      "port": 5432,
      "name": "testdb",
      "credentials": {
        "username": "admin",
        "password": "secret"
      }
    },
    "features": [
      "authentication",
      "logging",
      "monitoring"
    ]
  },
  "author": "Test Author",
  "license": "MIT",
  "keywords": ["test", "json", "parser"]
}'''


@pytest.fixture
def complex_json_code():
    """Complex JSON with various structures"""
    return '''{
  "api": {
    "version": "v1",
    "endpoints": {
      "users": {
        "GET": "/api/v1/users",
        "POST": "/api/v1/users",
        "PUT": "/api/v1/users/:id",
        "DELETE": "/api/v1/users/:id"
      },
      "auth": {
        "login": "/api/v1/auth/login",
        "logout": "/api/v1/auth/logout",
        "refresh": "/api/v1/auth/refresh"
      }
    },
    "middleware": [
      "cors",
      "helmet",
      "compression"
    ]
  },
  "database": {
    "connections": [
      {
        "name": "primary",
        "type": "postgresql",
        "host": "localhost",
        "port": 5432,
        "ssl": true
      },
      {
        "name": "cache",
        "type": "redis",
        "host": "localhost",
        "port": 6379,
        "ssl": false
      }
    ],
    "migrations": {
      "directory": "./migrations",
      "tableName": "knex_migrations"
    }
  },
  "logging": {
    "level": "info",
    "transports": [
      {
        "type": "console",
        "format": "json"
      },
      {
        "type": "file",
        "filename": "app.log",
        "maxsize": "10MB",
        "maxFiles": 5
      }
    ]
  },
  "features": {
    "authentication": true,
    "rateLimit": {
      "enabled": true,
      "max": 100,
      "windowMs": 900000
    },
    "cors": {
      "origin": ["http://localhost:3000", "https://app.example.com"],
      "credentials": true
    }
  }
}'''


@pytest.fixture
def jsonc_code():
    """JSONC code with comments for testing"""
    return '''{
  // Main configuration
  "name": "test-app",
  "version": "1.0.0",
  
  /* 
   * Development configuration
   * Used for local development
   */
  "development": {
    "port": 3000,
    "debug": true // Enable debug mode
  },
  
  // Production settings
  "production": {
    "port": 8080,
    "debug": false,
    "compression": true
  }
}'''


class TestJSONParserBasics:
    """Test basic JSON parser functionality"""
    
    def test_parser_initialization(self, json_parser):
        """Test parser initialization and properties"""
        assert json_parser.get_language_name() == "json"
        assert ".json" in json_parser.get_supported_extensions()
        assert ".jsonc" in json_parser.get_supported_extensions()
        assert ".json5" in json_parser.get_supported_extensions()
    
    def test_can_parse_json_files(self, json_parser):
        """Test file extension detection"""
        assert json_parser.can_parse(Path("config.json"))
        assert json_parser.can_parse(Path("package.json"))
        assert json_parser.can_parse(Path("tsconfig.jsonc"))
        assert json_parser.can_parse(Path("config.json5"))
        assert not json_parser.can_parse(Path("script.js"))
        assert not json_parser.can_parse(Path("data.yaml"))
    
    def test_parser_registration(self):
        """Test that JSON parser is registered correctly"""
        # Check JSON parser registration
        json_parser = parser_registry.get_parser("json")
        assert json_parser is not None
        assert isinstance(json_parser, JSONParser)
        
        # Check file mapping
        json_file = Path("config.json")
        file_parser = parser_registry.get_parser_for_file(json_file)
        assert file_parser is not None
        assert isinstance(file_parser, JSONParser)


class TestJSONEntityExtraction:
    """Test entity extraction from JSON code"""
    
    def test_root_entity_extraction(self, json_parser, tmp_path, sample_json_code):
        """Test extraction of root JSON entity"""
        test_file = tmp_path / "package.json"
        test_file.write_text(sample_json_code)
        
        result = json_parser.parse_file(test_file)
        
        # Find root entity
        root_entities = [e for e in result.entities if e.entity_type == EntityType.MODULE]
        assert len(root_entities) == 1
        
        root_entity = root_entities[0]
        assert root_entity.name == "root"
        assert root_entity.qualified_name == "$"
        assert root_entity.metadata["json_path"] == "$"
        assert root_entity.metadata["value_type"] == "dict"
    
    def test_property_extraction(self, json_parser, tmp_path, sample_json_code):
        """Test extraction of JSON properties"""
        test_file = tmp_path / "package.json"
        test_file.write_text(sample_json_code)
        
        result = json_parser.parse_file(test_file)
        
        # Find constant entities (simple properties)
        constants = [e for e in result.entities if e.entity_type == EntityType.CONSTANT]
        constant_names = [c.name for c in constants]
        
        # Check simple string/number properties
        assert "name" in constant_names
        assert "version" in constant_names
        assert "description" in constant_names
        assert "main" in constant_names
        assert "author" in constant_names
        assert "license" in constant_names
        
        # Check name property details
        name_entity = next((c for c in constants if c.name == "name"), None)
        assert name_entity is not None
        assert name_entity.qualified_name == "$.name"
        assert '"name": "test-project"' in name_entity.signature
    
    def test_object_extraction(self, json_parser, tmp_path, sample_json_code):
        """Test extraction of JSON objects"""
        test_file = tmp_path / "package.json"
        test_file.write_text(sample_json_code)
        
        result = json_parser.parse_file(test_file)
        
        # Find class entities (objects)
        classes = [e for e in result.entities if e.entity_type == EntityType.CLASS]
        class_names = [c.name for c in classes]
        
        # Check nested objects
        assert "scripts" in class_names
        assert "dependencies" in class_names
        assert "devDependencies" in class_names
        assert "config" in class_names
        assert "database" in class_names
        assert "credentials" in class_names
        
        # Check scripts object details
        scripts_entity = next((c for c in classes if c.name == "scripts"), None)
        assert scripts_entity is not None
        assert scripts_entity.qualified_name == "$.scripts"
        assert scripts_entity.metadata["json_path"] == "$.scripts"
        assert scripts_entity.metadata["property_count"] == 4
        assert "start" in scripts_entity.metadata["properties"]
        assert "test" in scripts_entity.metadata["properties"]
    
    def test_array_extraction(self, json_parser, tmp_path, sample_json_code):
        """Test extraction of JSON arrays"""
        test_file = tmp_path / "package.json"
        test_file.write_text(sample_json_code)
        
        result = json_parser.parse_file(test_file)
        
        # Find variable entities (arrays)
        variables = [e for e in result.entities if e.entity_type == EntityType.VARIABLE]
        variable_names = [v.name for v in variables]
        
        # Check arrays
        assert "features" in variable_names
        assert "keywords" in variable_names
        
        # Check features array details
        features_entity = next((v for v in variables if v.name == "features"), None)
        assert features_entity is not None
        assert features_entity.qualified_name == "$.config.features"
        assert features_entity.metadata["json_path"] == "$.config.features"
        assert features_entity.metadata["array_length"] == 3
        assert "str" in features_entity.metadata["element_types"]
    
    def test_configuration_sections(self, json_parser, tmp_path, sample_json_code):
        """Test extraction of configuration sections"""
        test_file = tmp_path / "package.json"
        test_file.write_text(sample_json_code)
        
        result = json_parser.parse_file(test_file)
        
        # Find namespace entities (config sections)
        namespaces = [e for e in result.entities if e.entity_type == EntityType.NAMESPACE]
        namespace_names = [n.name for n in namespaces]
        
        # Check configuration sections
        assert "scripts" in namespace_names
        assert "dependencies" in namespace_names
        assert "devDependencies" in namespace_names
        assert "config" in namespace_names
        
        # Check config section details
        config_entity = next((n for n in namespaces if n.name == "config"), None)
        assert config_entity is not None
        assert config_entity.metadata["is_config_section"] is True
        assert config_entity.qualified_name == "$.config"
    
    def test_nested_structures(self, json_parser, tmp_path, complex_json_code):
        """Test extraction of deeply nested JSON structures"""
        test_file = tmp_path / "complex.json"
        test_file.write_text(complex_json_code)
        
        result = json_parser.parse_file(test_file)
        
        assert result.success is True
        assert result.entity_count > 10
        
        # Check nested entity paths
        entity_paths = [e.metadata.get("json_path", "") for e in result.entities]
        
        # Check deep nesting
        assert "$.api.endpoints.users" in entity_paths
        assert "$.database.connections" in entity_paths
        assert "$.features.rateLimit" in entity_paths
        
        # Check array with objects
        connections_entity = next(
            (e for e in result.entities if e.metadata.get("json_path") == "$.database.connections"),
            None
        )
        assert connections_entity is not None
        assert connections_entity.entity_type == EntityType.VARIABLE
        assert connections_entity.metadata["has_objects"] is True
    
    def test_jsonc_parsing(self, json_parser, tmp_path, jsonc_code):
        """Test parsing JSONC files with comments"""
        test_file = tmp_path / "config.jsonc"
        test_file.write_text(jsonc_code)
        
        result = json_parser.parse_file(test_file)
        
        assert result.success is True
        assert result.entity_count > 0
        
        # Check that properties were extracted despite comments
        constants = [e for e in result.entities if e.entity_type == EntityType.CONSTANT]
        constant_names = [c.name for c in constants]
        
        assert "name" in constant_names
        assert "version" in constant_names
        
        # Check nested objects were parsed
        classes = [e for e in result.entities if e.entity_type == EntityType.CLASS]
        class_names = [c.name for c in classes]
        
        assert "development" in class_names
        assert "production" in class_names


class TestJSONRelationExtraction:
    """Test relation extraction from JSON code"""
    
    def test_containment_relations(self, json_parser, tmp_path, sample_json_code):
        """Test extraction of containment relations"""
        test_file = tmp_path / "package.json"
        test_file.write_text(sample_json_code)
        
        result = json_parser.parse_file(test_file)
        
        # Find containment relations
        containment_relations = [r for r in result.relations if r.relation_type == RelationType.CONTAINS]
        
        assert len(containment_relations) > 0
        
        # Check that config contains database
        config_contains_db = next(
            (r for r in containment_relations 
             if "config" in r.source_entity_id and "database" in r.target_entity_id),
            None
        )
        assert config_contains_db is not None
        
        # Check that database contains credentials
        db_contains_creds = next(
            (r for r in containment_relations 
             if "database" in r.source_entity_id and "credentials" in r.target_entity_id),
            None
        )
        assert db_contains_creds is not None
    
    def test_dependency_relations(self, json_parser, tmp_path, sample_json_code):
        """Test extraction of dependency relations"""
        test_file = tmp_path / "package.json"
        test_file.write_text(sample_json_code)
        
        result = json_parser.parse_file(test_file)
        
        # Find dependency relations
        dependency_relations = [r for r in result.relations if r.relation_type == RelationType.DEPENDS_ON]
        
        assert len(dependency_relations) > 0
        
        # Check dependencies
        dependency_names = [r.target_entity_id for r in dependency_relations]
        assert any("express" in dep for dep in dependency_names)
        assert any("lodash" in dep for dep in dependency_names)
        assert any("jest" in dep for dep in dependency_names)


class TestJSONParserEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_json_parsing(self, json_parser, tmp_path):
        """Test parsing empty JSON files"""
        test_cases = [
            "{}",
            "[]",
            '{"empty": {}}'
        ]
        
        for i, json_content in enumerate(test_cases):
            test_file = tmp_path / f"empty_{i}.json"
            test_file.write_text(json_content)
            
            result = json_parser.parse_file(test_file)
            
            assert result.success is True
            # Should have at least root entity
            assert result.entity_count >= 1
    
    def test_invalid_json_handling(self, json_parser, tmp_path):
        """Test handling of invalid JSON"""
        invalid_json = '''{"name": "test", "invalid": syntax here}'''
        
        test_file = tmp_path / "invalid.json"
        test_file.write_text(invalid_json)
        
        result = json_parser.parse_file(test_file)
        
        # Should handle gracefully
        assert result is not None
        # May have empty results due to parse error
        assert result.entity_count == 0
    
    def test_large_json_parsing(self, json_parser, tmp_path):
        """Test parsing large JSON structures"""
        # Create large JSON with many properties
        large_data = {
            "items": [{"id": i, "name": f"item_{i}", "active": i % 2 == 0} for i in range(100)],
            "metadata": {f"key_{i}": f"value_{i}" for i in range(50)},
            "config": {
                "nested": {
                    "deep": {
                        "very_deep": {
                            "properties": {f"prop_{i}": i for i in range(20)}
                        }
                    }
                }
            }
        }
        
        import json
        test_file = tmp_path / "large.json"
        test_file.write_text(json.dumps(large_data, indent=2))
        
        result = json_parser.parse_file(test_file)
        
        assert result.success is True
        assert result.entity_count > 10
        
        # Should handle arrays with many items
        items_entity = next(
            (e for e in result.entities if e.name == "items"),
            None
        )
        assert items_entity is not None
        assert items_entity.metadata["array_length"] == 100
    
    def test_unicode_handling(self, json_parser, tmp_path):
        """Test handling of Unicode in JSON"""
        unicode_json = '''{
  "name": "测试项目",
  "description": "这是一个测试项目",
  "emoji": "🚀✨",
  "multilingual": {
    "english": "Hello World",
    "chinese": "你好世界",
    "japanese": "こんにちは世界",
    "arabic": "مرحبا بالعالم"
  }
}'''
        
        test_file = tmp_path / "unicode.json"
        test_file.write_text(unicode_json, encoding='utf-8')
        
        result = json_parser.parse_file(test_file)
        
        assert result.success is True
        
        # Check Unicode entities were extracted
        constants = [e for e in result.entities if e.entity_type == EntityType.CONSTANT]
        
        # Find the Chinese name
        name_entity = next((c for c in constants if c.name == "name"), None)
        assert name_entity is not None
        assert "测试项目" in name_entity.source_code


class TestJSONParserIntegration:
    """Integration tests with registry and file discovery"""
    
    def test_registry_integration(self, tmp_path):
        """Test integration with parser registry"""
        # Create JSON files
        files = []
        for i in range(3):
            json_file = tmp_path / f"config_{i}.json"
            json_content = f'{{"name": "config_{i}", "version": "{i}.0.0"}}'
            json_file.write_text(json_content)
            files.append(json_file)
        
        # Use registry to discover and parse files
        parseable_files = parser_registry.discover_files(tmp_path)
        json_files = [f for f in parseable_files if f.suffix == ".json"]
        
        assert len(json_files) == 3
        
        # Parse files in parallel
        results = parser_registry.parse_files_parallel(json_files)
        
        assert len(results) == 3
        assert all(r.success for r in results)
        
        # Check that entities were extracted
        total_entities = sum(r.entity_count for r in results)
        assert total_entities >= 6  # At least 2 entities per file
    
    def test_package_json_parsing(self, json_parser, tmp_path):
        """Test parsing real-world package.json"""
        package_json = '''{
  "name": "my-awesome-project",
  "version": "2.1.0",
  "description": "An awesome Node.js project",
  "main": "dist/index.js",
  "types": "dist/index.d.ts",
  "scripts": {
    "build": "tsc",
    "start": "node dist/index.js",
    "dev": "ts-node src/index.ts",
    "test": "jest",
    "lint": "eslint src/**/*.ts",
    "format": "prettier --write src/**/*.ts"
  },
  "dependencies": {
    "express": "^4.18.2",
    "typescript": "^4.9.5",
    "@types/node": "^18.15.0"
  },
  "devDependencies": {
    "jest": "^29.5.0",
    "eslint": "^8.36.0",
    "prettier": "^2.8.7",
    "ts-node": "^10.9.1"
  },
  "engines": {
    "node": ">=16.0.0",
    "npm": ">=8.0.0"
  },
  "repository": {
    "type": "git",
    "url": "https://github.com/user/my-awesome-project.git"
  },
  "keywords": ["nodejs", "typescript", "api"],
  "author": "John Doe <john@example.com>",
  "license": "MIT"
}'''
        
        test_file = tmp_path / "package.json"
        test_file.write_text(package_json)
        
        result = json_parser.parse_file(test_file)
        
        assert result.success is True
        assert result.entity_count > 15
        
        # Check package.json specific entities
        entity_names = [e.name for e in result.entities]
        assert "scripts" in entity_names
        assert "dependencies" in entity_names
        assert "devDependencies" in entity_names
        assert "engines" in entity_names
        assert "repository" in entity_names
        
        # Check that configuration sections are identified
        config_sections = [e for e in result.entities if e.metadata.get("is_config_section")]
        config_names = [c.name for c in config_sections]
        assert "scripts" in config_names
        assert "dependencies" in config_names
        assert "devDependencies" in config_names


# Performance benchmarks
@pytest.mark.benchmark
class TestJSONParserPerformance:
    """Performance benchmarks for JSON parser"""
    
    def test_parsing_speed(self, json_parser, tmp_path, benchmark):
        """Benchmark JSON parsing speed"""
        # Create large JSON file
        import json
        large_data = {
            "metadata": {"version": "1.0.0", "created": "2024-01-01"},
            "items": [
                {
                    "id": i,
                    "name": f"item_{i}",
                    "tags": [f"tag_{j}" for j in range(i % 5)],
                    "config": {
                        "enabled": i % 2 == 0,
                        "priority": i % 10,
                        "settings": {f"key_{k}": f"value_{k}" for k in range(i % 3)}
                    }
                }
                for i in range(100)
            ],
            "configuration": {
                "database": {"host": "localhost", "port": 5432},
                "cache": {"host": "localhost", "port": 6379},
                "logging": {"level": "info", "format": "json"}
            }
        }
        
        test_file = tmp_path / "benchmark.json"
        test_file.write_text(json.dumps(large_data, indent=2))
        
        # Benchmark parsing
        def parse_file():
            return json_parser.parse_file(test_file)
        
        result = benchmark(parse_file)
        
        # Verify parsing worked
        assert result.success
        assert result.entity_count > 50
        
        # Performance targets (guidelines)
        assert result.parse_time < 2.0  # Should parse in under 2 seconds