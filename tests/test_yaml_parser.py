"""
Tests for YAML parser with comprehensive entity and relation extraction.

Tests the YAMLParser implementation to ensure correct extraction of:
- YAML objects and nested structures
- Arrays and data collections
- Key-value pairs with type detection
- Configuration sections
- Relations between entities
- Multi-document YAML support
"""

import pytest
from pathlib import Path

from core.parser.yaml_parser import YAMLParser
from core.parser.registry import parser_registry
from core.models.entities import (
    Entity, EntityType, Relation, RelationType, 
    SourceLocation, Visibility
)


@pytest.fixture
def yaml_parser():
    """Create a YAML parser instance for testing"""
    return YAMLParser()


@pytest.fixture
def sample_yaml_code():
    """Sample YAML code for testing entity extraction"""
    return '''name: test-project
version: 1.0.0
description: A test project for YAML parsing
main: index.js

scripts:
  start: node index.js
  test: jest
  build: webpack --mode production
  dev: webpack-dev-server --mode development

dependencies:
  express: "^4.18.0"
  lodash: "^4.17.21"
  axios: "^1.3.0"

devDependencies:
  jest: "^29.0.0"
  webpack: "^5.70.0"
  webpack-dev-server: "^4.7.0"

config:
  port: 3000
  host: localhost
  database:
    host: localhost
    port: 5432
    name: testdb
    credentials:
      username: admin
      password: secret
  features:
    - authentication
    - logging
    - monitoring

author: Test Author
license: MIT
keywords:
  - test
  - yaml
  - parser'''


@pytest.fixture
def docker_compose_yaml():
    """Docker Compose YAML for testing configuration sections"""
    return '''version: '3.8'

services:
  web:
    build: .
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
    depends_on:
      - database
      - redis
    volumes:
      - ./src:/app/src
    
  database:
    image: postgres:13
    environment:
      POSTGRES_DB: myapp
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: secret
    volumes:
      - db_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
  
  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"

volumes:
  db_data:

networks:
  default:
    driver: bridge'''


@pytest.fixture
def multi_document_yaml():
    """Multi-document YAML for testing"""
    return '''---
name: document1
type: config
settings:
  debug: true
  port: 3000

---
name: document2
type: data
items:
  - id: 1
    name: item1
  - id: 2
    name: item2

---
name: document3
type: service
config:
  host: localhost
  timeout: 30'''


@pytest.fixture
def yaml_with_anchors():
    """YAML with anchors and aliases"""
    return '''defaults: &defaults
  timeout: 30
  retries: 3
  debug: false

development:
  <<: *defaults
  debug: true
  host: localhost

production:
  <<: *defaults
  host: prod.example.com
  ssl: true

staging:
  <<: *defaults
  host: staging.example.com'''


class TestYAMLParserBasics:
    """Test basic YAML parser functionality"""
    
    def test_parser_initialization(self, yaml_parser):
        """Test parser initialization and properties"""
        assert yaml_parser.get_language_name() == "yaml"
        assert ".yaml" in yaml_parser.get_supported_extensions()
        assert ".yml" in yaml_parser.get_supported_extensions()
    
    def test_can_parse_yaml_files(self, yaml_parser):
        """Test file extension detection"""
        assert yaml_parser.can_parse(Path("config.yaml"))
        assert yaml_parser.can_parse(Path("docker-compose.yml"))
        assert yaml_parser.can_parse(Path("kubernetes.yaml"))
        assert not yaml_parser.can_parse(Path("script.js"))
        assert not yaml_parser.can_parse(Path("data.json"))
    
    def test_parser_registration(self):
        """Test that YAML parser is registered correctly"""
        # Check YAML parser registration
        yaml_parser = parser_registry.get_parser("yaml")
        assert yaml_parser is not None
        assert isinstance(yaml_parser, YAMLParser)
        
        # Check file mapping
        yaml_file = Path("config.yaml")
        file_parser = parser_registry.get_parser_for_file(yaml_file)
        assert file_parser is not None
        assert isinstance(file_parser, YAMLParser)


class TestYAMLEntityExtraction:
    """Test entity extraction from YAML code"""
    
    def test_root_entity_extraction(self, yaml_parser, tmp_path, sample_yaml_code):
        """Test extraction of root YAML entity"""
        test_file = tmp_path / "config.yaml"
        test_file.write_text(sample_yaml_code)
        
        result = yaml_parser.parse_file(test_file)
        
        # Find root entity
        root_entities = [e for e in result.entities if e.entity_type == EntityType.MODULE]
        assert len(root_entities) == 1
        
        root_entity = root_entities[0]
        assert root_entity.name == "root"
        assert root_entity.qualified_name == "$"
        assert root_entity.metadata["yaml_path"] == "$"
        assert root_entity.metadata["value_type"] == "dict"
    
    def test_property_extraction(self, yaml_parser, tmp_path, sample_yaml_code):
        """Test extraction of YAML properties"""
        test_file = tmp_path / "config.yaml"
        test_file.write_text(sample_yaml_code)
        
        result = yaml_parser.parse_file(test_file)
        
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
        assert "test-project" in name_entity.signature
    
    def test_object_extraction(self, yaml_parser, tmp_path, sample_yaml_code):
        """Test extraction of YAML objects"""
        test_file = tmp_path / "config.yaml"
        test_file.write_text(sample_yaml_code)
        
        result = yaml_parser.parse_file(test_file)
        
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
        assert scripts_entity.metadata["yaml_path"] == "$.scripts"
        assert scripts_entity.metadata["property_count"] == 4
        assert "start" in scripts_entity.metadata["properties"]
        assert "test" in scripts_entity.metadata["properties"]
    
    def test_array_extraction(self, yaml_parser, tmp_path, sample_yaml_code):
        """Test extraction of YAML arrays"""
        test_file = tmp_path / "config.yaml"
        test_file.write_text(sample_yaml_code)
        
        result = yaml_parser.parse_file(test_file)
        
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
        assert features_entity.metadata["yaml_path"] == "$.config.features"
        assert features_entity.metadata["array_length"] == 3
        assert "str" in features_entity.metadata["element_types"]
    
    def test_configuration_sections(self, yaml_parser, tmp_path, sample_yaml_code):
        """Test extraction of configuration sections"""
        test_file = tmp_path / "config.yaml"
        test_file.write_text(sample_yaml_code)
        
        result = yaml_parser.parse_file(test_file)
        
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
    
    def test_docker_compose_sections(self, yaml_parser, tmp_path, docker_compose_yaml):
        """Test extraction of Docker Compose configuration sections"""
        test_file = tmp_path / "docker-compose.yml"
        test_file.write_text(docker_compose_yaml)
        
        result = yaml_parser.parse_file(test_file)
        
        # Find namespace entities (config sections)
        namespaces = [e for e in result.entities if e.entity_type == EntityType.NAMESPACE]
        namespace_names = [n.name for n in namespaces]
        
        # Check Docker Compose specific sections
        assert "services" in namespace_names
        assert "volumes" in namespace_names
        assert "networks" in namespace_names
    
    def test_multi_document_yaml(self, yaml_parser, tmp_path, multi_document_yaml):
        """Test parsing multi-document YAML files"""
        test_file = tmp_path / "multi-doc.yaml"
        test_file.write_text(multi_document_yaml)
        
        result = yaml_parser.parse_file(test_file)
        
        assert result.success is True
        assert result.entity_count > 5
        
        # Check that multiple document roots were created
        root_entities = [e for e in result.entities if e.entity_type == EntityType.MODULE]
        assert len(root_entities) >= 3  # Should have document roots
        
        # Check entity paths include document prefixes
        entity_paths = [e.metadata.get("yaml_path", "") for e in result.entities]
        
        # Should have entities from different documents
        doc_paths = [path for path in entity_paths if path.startswith("$doc")]
        assert len(doc_paths) > 0


class TestYAMLRelationExtraction:
    """Test relation extraction from YAML code"""
    
    def test_containment_relations(self, yaml_parser, tmp_path, sample_yaml_code):
        """Test extraction of containment relations"""
        test_file = tmp_path / "config.yaml"
        test_file.write_text(sample_yaml_code)
        
        result = yaml_parser.parse_file(test_file)
        
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
    
    def test_dependency_relations(self, yaml_parser, tmp_path, sample_yaml_code):
        """Test extraction of dependency relations"""
        test_file = tmp_path / "config.yaml"
        test_file.write_text(sample_yaml_code)
        
        result = yaml_parser.parse_file(test_file)
        
        # Find dependency relations
        dependency_relations = [r for r in result.relations if r.relation_type == RelationType.DEPENDS_ON]
        
        assert len(dependency_relations) > 0
        
        # Check dependencies
        dependency_names = [r.target_entity_id for r in dependency_relations]
        assert any("express" in dep for dep in dependency_names)
        assert any("lodash" in dep for dep in dependency_names)
        assert any("jest" in dep for dep in dependency_names)
    
    def test_docker_compose_dependencies(self, yaml_parser, tmp_path, docker_compose_yaml):
        """Test extraction of Docker Compose service dependencies"""
        test_file = tmp_path / "docker-compose.yml"
        test_file.write_text(docker_compose_yaml)
        
        result = yaml_parser.parse_file(test_file)
        
        # Check that web service dependencies are extracted
        dependency_relations = [r for r in result.relations if r.relation_type == RelationType.DEPENDS_ON]
        
        # Should have dependencies for database and redis
        dependency_names = [r.target_entity_id for r in dependency_relations]
        assert any("database" in dep for dep in dependency_names)
        assert any("redis" in dep for dep in dependency_names)
    
    def test_yaml_anchor_references(self, yaml_parser, tmp_path, yaml_with_anchors):
        """Test extraction of YAML anchor and alias references"""
        test_file = tmp_path / "anchors.yaml"
        test_file.write_text(yaml_with_anchors)
        
        result = yaml_parser.parse_file(test_file)
        
        # Find reference relations
        reference_relations = [r for r in result.relations if r.relation_type == RelationType.REFERENCES]
        
        # Should find references to defaults anchor
        assert len(reference_relations) > 0


class TestYAMLParserEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_yaml_parsing(self, yaml_parser, tmp_path):
        """Test parsing empty YAML files"""
        test_cases = [
            "{}",
            "[]",
            "key: value",
            "# Just a comment"
        ]
        
        for i, yaml_content in enumerate(test_cases):
            test_file = tmp_path / f"empty_{i}.yaml"
            test_file.write_text(yaml_content)
            
            result = yaml_parser.parse_file(test_file)
            
            assert result.success is True
            # Should have at least root entity for non-comment cases
            if yaml_content != "# Just a comment":
                assert result.entity_count >= 1
    
    def test_invalid_yaml_handling(self, yaml_parser, tmp_path):
        """Test handling of invalid YAML"""
        invalid_yaml = '''name: test
invalid: [unclosed bracket
another: value'''
        
        test_file = tmp_path / "invalid.yaml"
        test_file.write_text(invalid_yaml)
        
        result = yaml_parser.parse_file(test_file)
        
        # Should handle gracefully
        assert result is not None
        # May have empty results due to parse error
        assert result.entity_count == 0
    
    def test_complex_yaml_parsing(self, yaml_parser, tmp_path):
        """Test parsing complex YAML structures"""
        complex_yaml = '''
metadata:
  name: complex-app
  namespace: production
  labels:
    app: myapp
    version: v1.2.3
    
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
      - name: web
        image: nginx:1.21
        ports:
        - containerPort: 80
        env:
        - name: ENV
          value: production
        - name: DEBUG
          value: "false"
        resources:
          limits:
            cpu: 500m
            memory: 512Mi
          requests:
            cpu: 100m
            memory: 128Mi
      - name: sidecar
        image: sidecar:latest
        command: ["/bin/sh"]
        args: ["-c", "while true; do sleep 30; done"]
'''
        
        test_file = tmp_path / "complex.yaml"
        test_file.write_text(complex_yaml)
        
        result = yaml_parser.parse_file(test_file)
        
        assert result.success is True
        assert result.entity_count > 10
        
        # Should handle deeply nested structures
        entity_paths = [e.metadata.get("yaml_path", "") for e in result.entities]
        
        # Check deep nesting
        assert any("spec.template.spec.containers" in path for path in entity_paths)
        assert any("spec.selector.matchLabels" in path for path in entity_paths)
    
    def test_unicode_handling(self, yaml_parser, tmp_path):
        """Test handling of Unicode in YAML"""
        unicode_yaml = '''name: 测试项目
description: 这是一个测试项目
emoji: 🚀✨
multilingual:
  english: Hello World
  chinese: 你好世界
  japanese: こんにちは世界
  arabic: مرحبا بالعالم
  emoji_list:
    - 😀
    - 🎉
    - 🔥'''
        
        test_file = tmp_path / "unicode.yaml"
        test_file.write_text(unicode_yaml, encoding='utf-8')
        
        result = yaml_parser.parse_file(test_file)
        
        assert result.success is True
        
        # Check Unicode entities were extracted
        constants = [e for e in result.entities if e.entity_type == EntityType.CONSTANT]
        
        # Find the Chinese name
        name_entity = next((c for c in constants if c.name == "name"), None)
        assert name_entity is not None
        assert "测试项目" in name_entity.source_code


class TestYAMLParserIntegration:
    """Integration tests with registry and file discovery"""
    
    def test_registry_integration(self, tmp_path):
        """Test integration with parser registry"""
        # Create YAML files
        files = []
        for i in range(3):
            yaml_file = tmp_path / f"config_{i}.yaml"
            yaml_content = f'''name: config_{i}
version: {i}.0.0
description: Configuration file {i}'''
            yaml_file.write_text(yaml_content)
            files.append(yaml_file)
        
        # Use registry to discover and parse files
        parseable_files = parser_registry.discover_files(tmp_path)
        yaml_files = [f for f in parseable_files if f.suffix in [".yaml", ".yml"]]
        
        assert len(yaml_files) == 3
        
        # Parse files in parallel
        results = parser_registry.parse_files_parallel(yaml_files)
        
        assert len(results) == 3
        assert all(r.success for r in results)
        
        # Check that entities were extracted
        total_entities = sum(r.entity_count for r in results)
        assert total_entities >= 9  # At least 3 entities per file
    
    def test_kubernetes_yaml_parsing(self, yaml_parser, tmp_path):
        """Test parsing real-world Kubernetes YAML"""
        kubernetes_yaml = '''apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
  namespace: default
  labels:
    app: nginx
    version: "1.0"
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.21
        ports:
        - containerPort: 80
          protocol: TCP
        env:
        - name: ENV
          value: production
        resources:
          limits:
            cpu: 500m
            memory: 512Mi
          requests:
            cpu: 100m
            memory: 128Mi
        livenessProbe:
          httpGet:
            path: /health
            port: 80
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 80
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: nginx-service
  namespace: default
spec:
  selector:
    app: nginx
  ports:
  - port: 80
    targetPort: 80
    protocol: TCP
  type: ClusterIP'''
        
        test_file = tmp_path / "kubernetes.yaml"
        test_file.write_text(kubernetes_yaml)
        
        result = yaml_parser.parse_file(test_file)
        
        assert result.success is True
        assert result.entity_count > 15
        
        # Check Kubernetes specific entities
        entity_names = [e.name for e in result.entities]
        assert "metadata" in entity_names
        assert "spec" in entity_names
        assert "containers" in entity_names
        
        # Check that configuration sections are identified
        config_sections = [e for e in result.entities if e.metadata.get("is_config_section")]
        # Should identify some Kubernetes sections as config
        assert len(config_sections) > 0


# Performance benchmarks
@pytest.mark.benchmark
class TestYAMLParserPerformance:
    """Performance benchmarks for YAML parser"""
    
    def test_parsing_speed(self, yaml_parser, tmp_path, benchmark):
        """Benchmark YAML parsing speed"""
        # Create large YAML file
        large_yaml = '''metadata:
  name: large-config
  version: 1.0.0

services:'''
        
        # Add many services
        for i in range(50):
            large_yaml += f'''
  service_{i}:
    image: app:latest
    ports:
      - "{3000 + i}:{3000 + i}"
    environment:
      SERVICE_ID: "{i}"
      PORT: "{3000 + i}"
    volumes:
      - ./data_{i}:/app/data
    depends_on:
      - database_{i % 5}
    labels:
      service.id: "{i}"
      service.type: "web"'''
        
        # Add databases
        large_yaml += '''

databases:'''
        for i in range(5):
            large_yaml += f'''
  database_{i}:
    image: postgres:13
    environment:
      POSTGRES_DB: "db_{i}"
      POSTGRES_USER: "user_{i}"
      POSTGRES_PASSWORD: "pass_{i}"
    volumes:
      - db_data_{i}:/var/lib/postgresql/data
    ports:
      - "{5432 + i}:{5432 + i}"'''
        
        test_file = tmp_path / "benchmark.yaml"
        test_file.write_text(large_yaml)
        
        # Benchmark parsing
        def parse_file():
            return yaml_parser.parse_file(test_file)
        
        result = benchmark(parse_file)
        
        # Verify parsing worked
        assert result.success
        assert result.entity_count > 100
        
        # Performance targets (guidelines)
        assert result.parse_time < 3.0  # Should parse in under 3 seconds