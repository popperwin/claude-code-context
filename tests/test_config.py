"""
Unit tests for configuration models.

Tests configuration validation, loading, and template processing.
"""

import pytest
import tempfile
import json
from pathlib import Path
from core.models.config import (
    QdrantConfig, StellaConfig, IndexingConfig, ProjectConfig, GlobalSettings
)


class TestQdrantConfig:
    """Test QdrantConfig model"""
    
    def test_default_qdrant_config(self):
        """Test default Qdrant configuration"""
        config = QdrantConfig()
        
        assert config.url == "http://localhost:6333"
        assert config.timeout == 60.0
        assert config.batch_size == 100
        assert config.vector_size == 1024
        assert config.distance_metric == "cosine"
        assert isinstance(config.collections, dict)
    
    def test_url_validation(self):
        """Test URL validation"""
        # Valid URLs
        valid_urls = [
            "http://localhost:6333",
            "https://qdrant.example.com",
            "http://192.168.1.100:6333"
        ]
        
        for url in valid_urls:
            config = QdrantConfig(url=url)
            assert config.url == url.rstrip('/')
        
        # Invalid URLs
        invalid_urls = [
            "localhost:6333",  # Missing protocol
            "ftp://localhost:6333",  # Wrong protocol
            ""  # Empty
        ]
        
        for url in invalid_urls:
            with pytest.raises(ValueError, match="Qdrant URL must start with"):
                QdrantConfig(url=url)
    
    def test_distance_metric_validation(self):
        """Test distance metric validation"""
        valid_metrics = ["cosine", "euclidean", "dot"]
        
        for metric in valid_metrics:
            config = QdrantConfig(distance_metric=metric)
            assert config.distance_metric == metric.lower()
        
        # Invalid metric
        with pytest.raises(ValueError, match="Distance metric must be one of"):
            QdrantConfig(distance_metric="invalid")
    
    def test_collection_name_generation(self):
        """Test collection name generation"""
        config = QdrantConfig()
        
        # Test single collection
        collection_name = config.get_collection_name("my-project", "code")
        assert collection_name == "my-project-code"
        
        # Test all collections
        collections = config.get_all_collections("my-project")
        expected = {
            "code": "my-project-code",
            "relations": "my-project-relations",
            "embeddings": "my-project-embeddings"
        }
        assert collections == expected
        
        # Test name normalization
        collection_name = config.get_collection_name("My_Project Name", "code")
        assert collection_name == "my-project-name-code"


class TestStellaConfig:
    """Test StellaConfig model"""
    
    def test_default_stella_config(self):
        """Test default Stella configuration"""
        config = StellaConfig()
        
        assert config.model_name == "stella_en_400M_v5"
        assert config.dimensions == 1024
        assert config.batch_size == 32
        assert config.max_length == 512
        assert config.normalize_embeddings is True
        assert config.use_fp16 is True
        assert config.cache_dir.name == "stella"
    
    def test_model_name_validation(self):
        """Test model name validation"""
        valid_models = [
            "stella_en_400M_v5",
            "stella_en_1.5B_v5",
            "stella_base_en_v2"
        ]
        
        for model in valid_models:
            config = StellaConfig(model_name=model)
            assert config.model_name == model
        
        # Invalid model
        with pytest.raises(ValueError, match="Model must be one of"):
            StellaConfig(model_name="invalid_model")
    
    def test_cache_dir_creation(self):
        """Test cache directory creation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "test_cache"
            
            # Cache directory should be created
            config = StellaConfig(cache_dir=cache_path)
            assert config.cache_dir.exists()
            assert config.cache_dir.is_dir()
    
    def test_model_path_property(self):
        """Test model path property"""
        config = StellaConfig()
        expected_path = config.cache_dir / config.model_name
        assert config.model_path == expected_path
    
    def test_is_model_cached_property(self):
        """Test is_model_cached property"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "test_cache"
            config = StellaConfig(cache_dir=cache_path)
            
            # Model not cached (empty directory)
            assert config.is_model_cached is False
            
            # Create some files in model directory
            model_dir = config.model_path
            model_dir.mkdir(parents=True)
            (model_dir / "config.json").touch()
            
            assert config.is_model_cached is True
    
    def test_device_detection(self):
        """Test device detection logic"""
        config = StellaConfig()
        
        # Should return a valid device
        device = config.get_device()
        assert device in ["cpu", "cuda", "mps"]
        
        # Test with explicit device
        config = StellaConfig(device="cpu")
        assert config.get_device() == "cpu"


class TestIndexingConfig:
    """Test IndexingConfig model"""
    
    def test_default_indexing_config(self):
        """Test default indexing configuration"""
        config = IndexingConfig()
        
        assert len(config.include_patterns) > 0
        assert len(config.exclude_patterns) > 0
        assert config.max_file_size_mb == 10
        assert config.extract_docstrings is True
        assert config.include_test_files is True
    
    def test_pattern_validation(self):
        """Test pattern validation"""
        # Valid patterns
        valid_patterns = ["*.py", "*.js", "test/*"]
        config = IndexingConfig(include_patterns=valid_patterns)
        assert config.include_patterns == valid_patterns
        
        # Empty patterns should fail
        with pytest.raises(ValueError, match="Pattern list cannot be empty"):
            IndexingConfig(include_patterns=[])
    
    def test_should_index_file(self):
        """Test file indexing decision"""
        config = IndexingConfig(
            include_patterns=["*.py", "*.js"],
            exclude_patterns=["*test*", "node_modules/*"]
        )
        
        # Should index
        assert config.should_index_file(Path("main.py")) is True
        assert config.should_index_file(Path("app.js")) is True
        
        # Should not index (excluded)
        assert config.should_index_file(Path("test_main.py")) is False
        assert config.should_index_file(Path("node_modules/lib.js")) is False
        
        # Should not index (not included)
        assert config.should_index_file(Path("readme.txt")) is False
    
    def test_max_file_size_bytes(self):
        """Test max file size conversion"""
        config = IndexingConfig(max_file_size_mb=5)
        assert config.max_file_size_bytes == 5 * 1024 * 1024


class TestProjectConfig:
    """Test ProjectConfig model"""
    
    def test_valid_project_config(self):
        """Test creating a valid project configuration"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            
            config = ProjectConfig(
                name="test-project",
                path=project_path,
                collection_prefix="test-project"
            )
            
            assert config.name == "test-project"
            assert config.path == project_path.resolve()
            assert config.collection_prefix == "test-project"
            assert isinstance(config.qdrant, QdrantConfig)
            assert isinstance(config.stella, StellaConfig)
            assert isinstance(config.indexing, IndexingConfig)
    
    def test_name_validation(self):
        """Test project name validation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            
            # Valid names
            valid_names = ["test-project", "test_project", "test project", "123test"]
            for name in valid_names:
                config = ProjectConfig(
                    name=name,
                    path=project_path,
                    collection_prefix="test"
                )
                assert config.name == name.strip()
            
            # Invalid names
            invalid_names = ["", "   ", "test@project", "test/project"]
            for name in invalid_names:
                with pytest.raises(ValueError, match="Project name must be alphanumeric"):
                    ProjectConfig(
                        name=name,
                        path=project_path,
                        collection_prefix="test"
                    )
    
    def test_collection_prefix_validation(self):
        """Test collection prefix validation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            
            # Valid prefixes
            valid_prefixes = ["test-project", "test_project", "123test"]
            for prefix in valid_prefixes:
                config = ProjectConfig(
                    name="test",
                    path=project_path,
                    collection_prefix=prefix
                )
                assert config.collection_prefix == prefix.lower()
            
            # Invalid prefixes
            invalid_prefixes = ["", "test project", "test@project"]
            for prefix in invalid_prefixes:
                with pytest.raises(ValueError, match="Collection prefix must be alphanumeric"):
                    ProjectConfig(
                        name="test",
                        path=project_path,
                        collection_prefix=prefix
                    )
    
    def test_path_validation(self):
        """Test project path validation"""
        # Non-existent path
        with pytest.raises(ValueError, match="Project path does not exist"):
            ProjectConfig(
                name="test",
                path=Path("/non/existent/path"),
                collection_prefix="test"
            )
        
        # File instead of directory
        with tempfile.NamedTemporaryFile() as temp_file:
            with pytest.raises(ValueError, match="Project path is not a directory"):
                ProjectConfig(
                    name="test",
                    path=Path(temp_file.name),
                    collection_prefix="test"
                )
    
    def test_collection_prefix_user_control(self):
        """Test that collection prefix respects user input"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            
            config = ProjectConfig(
                name="My Project_Name",
                path=project_path,
                collection_prefix="custom-prefix"
            )
            
            # Should preserve user input (lowercased)
            assert config.collection_prefix == "custom-prefix"
    
    def test_collection_names_generation(self):
        """Test collection names generation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            
            config = ProjectConfig(
                name="test-project",
                path=project_path,
                collection_prefix="test-project"
            )
            
            collections = config.get_collection_names()
            expected = {
                "code": "test-project-code",
                "relations": "test-project-relations", 
                "embeddings": "test-project-embeddings"
            }
            assert collections == expected
    
    def test_config_directory_methods(self):
        """Test configuration directory methods"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            
            config = ProjectConfig(
                name="test-project",
                path=project_path,
                collection_prefix="test-project"
            )
            
            config_dir = config.get_config_dir()
            assert config_dir == (project_path / ".claude-indexer").resolve()
            assert config_dir.exists()  # Should be created
            
            config_file = config.get_config_file()
            assert config_file == config_dir / "config.json"
    
    def test_is_initialized_property(self):
        """Test is_initialized property"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            
            config = ProjectConfig(
                name="test-project",
                path=project_path,
                collection_prefix="test-project"
            )
            
            # Not initialized (no config file)
            assert config.is_initialized is False
            
            # Create config file
            config_dir = config.get_config_dir()
            config_file = config.get_config_file()
            config_file.write_text("{}")
            
            assert config.is_initialized is True
    
    def test_serialization(self):
        """Test configuration serialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            
            config = ProjectConfig(
                name="test-project",
                path=project_path,
                collection_prefix="test-project"
            )
            
            # Test to_dict
            config_dict = config.to_dict()
            assert isinstance(config_dict, dict)
            assert config_dict["name"] == "test-project"
            assert isinstance(config_dict["path"], str)  # Should be serialized as string
            
            # Test from_dict
            restored_config = ProjectConfig.from_dict(config_dict)
            assert restored_config.name == config.name
            assert restored_config.path == config.path
            assert restored_config.collection_prefix == config.collection_prefix


class TestGlobalSettings:
    """Test GlobalSettings model"""
    
    def test_default_global_settings(self):
        """Test default global settings"""
        import os
        # Temporarily unset test environment LOG_LEVEL to test defaults
        old_log_level = os.environ.pop("CLAUDE_INDEXER_LOG_LEVEL", None)
        try:
            settings = GlobalSettings()
            
            assert settings.default_qdrant_url == "http://localhost:6333"
            assert settings.default_stella_model == "stella_en_400M_v5"
            assert settings.max_concurrent_projects == 10
            assert settings.log_level == "INFO"
            assert settings.enable_telemetry is False
        finally:
            # Restore test environment
            if old_log_level:
                os.environ["CLAUDE_INDEXER_LOG_LEVEL"] = old_log_level
        
        # Directories should be created
        assert settings.global_cache_dir.exists()
        assert settings.global_config_dir.exists()
    
    def test_stella_cache_dir(self):
        """Test Stella cache directory property"""
        settings = GlobalSettings()
        
        stella_dir = settings.stella_cache_dir
        assert stella_dir.name == "stella"
        assert stella_dir.parent == settings.global_cache_dir
        assert stella_dir.exists()  # Should be created
    
    def test_projects_registry_file(self):
        """Test projects registry file property"""
        settings = GlobalSettings()
        
        registry_file = settings.projects_registry_file
        assert registry_file.name == "projects.json"
        assert registry_file.parent == settings.global_config_dir
    
    def test_log_file(self):
        """Test log file property"""
        # Logging disabled
        settings = GlobalSettings(log_to_file=False)
        assert settings.get_log_file() is None
        
        # Logging enabled
        settings = GlobalSettings(log_to_file=True)
        log_file = settings.get_log_file()
        assert log_file is not None
        assert log_file.name == "claude-indexer.log"
        assert log_file.parent.name == "logs"