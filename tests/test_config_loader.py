"""
Unit tests for configuration loader functionality.

Tests configuration loading, template substitution, environment overrides, project setup, and validation.
"""

import json
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, MagicMock
from typing import Dict, Any

from config.loader import ConfigurationLoader
from core.models.config import ProjectConfig, GlobalSettings
from config.defaults import DEFAULT_SETTINGS, ENV_VAR_MAPPING


class TestConfigurationLoader:
    """Test ConfigurationLoader functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.loader = ConfigurationLoader()
    
    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_loader_initialization(self):
        """Test loader initialization"""
        loader = ConfigurationLoader()
        
        assert isinstance(loader.global_settings, GlobalSettings)
        assert loader.template_dir.exists()
        assert isinstance(loader.config_cache, dict)
        assert len(loader.config_cache) == 0
    
    def test_load_project_config_new_project(self):
        """Test loading configuration for new project"""
        project_path = self.temp_path / "test_project"
        project_path.mkdir()
        
        config = self.loader.load_project_config(project_path, "test-project")
        
        assert isinstance(config, ProjectConfig)
        assert config.name == "test-project"
        assert config.path.resolve() == project_path.resolve()
        assert config.collection_prefix == "test-project"
    
    def test_load_project_config_existing(self):
        """Test loading existing configuration"""
        project_path = self.temp_path / "existing_project"
        project_path.mkdir()
        
        # Create config directory and file
        config_dir = project_path / ".claude-indexer"
        config_dir.mkdir()
        config_file = config_dir / "config.json"
        
        config_data = {
            "name": "existing-project",
            "path": str(project_path),
            "collection_prefix": "existing",
            "qdrant": DEFAULT_SETTINGS["qdrant"],
            "stella": DEFAULT_SETTINGS["stella"],
            "indexing": DEFAULT_SETTINGS["indexing"],
            "version": "1.0.0"
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        config = self.loader.load_project_config(project_path)
        
        assert config.name == "existing-project"
        assert config.collection_prefix == "existing"
    
    def test_load_project_config_caching(self):
        """Test configuration caching"""
        project_path = self.temp_path / "cached_project"
        project_path.mkdir()
        
        # First load
        config1 = self.loader.load_project_config(project_path, "cached")
        
        # Second load should return cached version
        config2 = self.loader.load_project_config(project_path, "cached")
        
        assert config1 is config2
        assert len(self.loader.config_cache) == 1
    
    def test_load_existing_config_invalid_json(self):
        """Test loading config with invalid JSON"""
        project_path = self.temp_path / "invalid_project"
        project_path.mkdir()
        
        config_dir = project_path / ".claude-indexer"
        config_dir.mkdir()
        config_file = config_dir / "config.json"
        
        # Write invalid JSON
        with open(config_file, 'w') as f:
            f.write("{ invalid json content")
        
        # Should fall back to creating new config - uses directory name as fallback
        config = self.loader.load_project_config(project_path)  # No explicit name, uses dir
        
        assert isinstance(config, ProjectConfig)
        assert config.name == "invalid-project"  # Uses sanitized directory name
    
    def test_load_existing_config_path_conversion(self):
        """Test path conversion in existing config"""
        project_path = self.temp_path / "path_project"
        project_path.mkdir()
        
        config_dir = project_path / ".claude-indexer"
        config_dir.mkdir()
        config_file = config_dir / "config.json"
        
        # Use a custom cache directory within the temp directory
        custom_cache_dir = str(self.temp_path / "custom_cache_dir")
        
        config_data = {
            "name": "path-project",
            "path": str(project_path),
            "collection_prefix": "path",
            "qdrant": DEFAULT_SETTINGS["qdrant"],
            "stella": {
                **DEFAULT_SETTINGS["stella"],
                "cache_dir": custom_cache_dir
            },
            "indexing": DEFAULT_SETTINGS["indexing"],
            "version": "1.0.0"
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        config = self.loader.load_project_config(project_path)
        
        assert isinstance(config.path, Path)
        assert isinstance(config.stella.cache_dir, Path)
        assert str(config.stella.cache_dir) == custom_cache_dir
    
    def test_create_project_config_name_sanitization(self):
        """Test project name sanitization"""
        project_path = self.temp_path / "Test Project_Name"
        project_path.mkdir()
        
        config = self.loader._create_project_config(project_path, "Test Project_Name")
        
        assert config.name == "test-project-name"
        assert config.collection_prefix == "test-project-name"
    
    def test_substitute_template_vars_dict(self):
        """Test template variable substitution in dictionary"""
        data = {
            "name": "${project_name}",
            "path": "${project_path}",
            "nested": {
                "value": "${project_name}-suffix"
            }
        }
        
        substitutions = {
            "project_name": "test-project",
            "project_path": "/path/to/project"
        }
        
        result = self.loader._substitute_template_vars(data, substitutions)
        
        assert result["name"] == "test-project"
        assert result["path"] == "/path/to/project"
        assert result["nested"]["value"] == "test-project-suffix"
    
    def test_substitute_template_vars_list(self):
        """Test template variable substitution in list"""
        data = ["${project_name}", "static-value", "${project_name}-suffix"]
        substitutions = {"project_name": "test"}
        
        result = self.loader._substitute_template_vars(data, substitutions)
        
        assert result == ["test", "static-value", "test-suffix"]
    
    def test_substitute_template_vars_invalid_template(self):
        """Test template substitution with invalid template"""
        data = {"invalid": "${invalid_var}"}
        substitutions = {"project_name": "test"}
        
        # Should return original string for invalid template
        result = self.loader._substitute_template_vars(data, substitutions)
        
        assert result["invalid"] == "${invalid_var}"
    
    def test_apply_env_overrides(self):
        """Test environment variable overrides"""
        config_data = {
            "qdrant": {"url": "http://localhost:6333"},
            "stella": {"model_name": "default_model"}
        }
        
        with patch.dict(os.environ, {
            "CLAUDE_INDEXER_QDRANT_URL": "http://custom:6333",
            "CLAUDE_INDEXER_STELLA_MODEL": "custom_model"
        }):
            result = self.loader._apply_env_overrides(config_data)
        
        assert result["qdrant"]["url"] == "http://custom:6333"
        assert result["stella"]["model_name"] == "custom_model"
    
    def test_set_nested_value(self):
        """Test setting nested dictionary values"""
        data = {"level1": {"level2": {}}}
        
        self.loader._set_nested_value(data, "level1.level2.new_key", "test_value")
        
        assert data["level1"]["level2"]["new_key"] == "test_value"
    
    def test_set_nested_value_creates_missing_keys(self):
        """Test setting nested value creates missing intermediate keys"""
        data = {}
        
        self.loader._set_nested_value(data, "new.nested.key", "value")
        
        assert data["new"]["nested"]["key"] == "value"
    
    def test_convert_env_value_boolean_true(self):
        """Test boolean conversion for true values"""
        true_values = ["true", "True", "TRUE", "yes", "1", "on"]
        
        for value in true_values:
            result = self.loader._convert_env_value(value)
            assert result is True
    
    def test_convert_env_value_boolean_false(self):
        """Test boolean conversion for false values"""
        false_values = ["false", "False", "FALSE", "no", "0", "off"]
        
        for value in false_values:
            result = self.loader._convert_env_value(value)
            assert result is False
    
    def test_convert_env_value_numeric(self):
        """Test numeric conversion"""
        assert self.loader._convert_env_value("42") == 42
        assert self.loader._convert_env_value("3.14") == 3.14
        assert self.loader._convert_env_value("invalid") == "invalid"
    
    def test_save_project_config_success(self):
        """Test successful project configuration saving"""
        project_path = self.temp_path / "save_project"
        project_path.mkdir()
        
        config = self.loader.load_project_config(project_path, "save-test")
        
        # Save should create directory and file
        result = self.loader.save_project_config(config)
        
        assert result is True
        assert config.get_config_file().exists()
        
        # Check cache is updated
        cache_key = str(config.path)
        assert cache_key in self.loader.config_cache
    
    def test_save_project_config_failure(self):
        """Test project configuration saving failure"""
        # Create config with invalid path
        config = Mock(spec=ProjectConfig)
        config.get_config_dir.side_effect = OSError("Permission denied")
        config.path = Path("/invalid/path")
        
        result = self.loader.save_project_config(config)
        
        assert result is False
    
    def test_setup_project_new(self):
        """Test complete project setup for new project"""
        project_path = self.temp_path / "setup_project"
        project_path.mkdir()
        
        config = self.loader.setup_project(project_path, "setup-test")
        
        assert isinstance(config, ProjectConfig)
        assert config.name == "setup-test"
        
        # Check directory structure
        assert (project_path / ".claude-indexer").exists()
        assert (project_path / ".claude").exists()
        assert (project_path / ".claude" / "hooks").exists()
        assert (project_path / ".claude" / "settings.json").exists()
    
    def test_setup_project_existing_no_overwrite(self):
        """Test setup of already initialized project without overwrite"""
        project_path = self.temp_path / "existing_setup"
        project_path.mkdir()
        
        # First setup
        config1 = self.loader.setup_project(project_path, "existing")
        
        # Second setup without overwrite
        config2 = self.loader.setup_project(project_path, "existing", overwrite=False)
        
        assert config1.name == config2.name
    
    def test_setup_project_nonexistent_path(self):
        """Test setup with nonexistent project path"""
        nonexistent_path = self.temp_path / "nonexistent"
        
        with pytest.raises(ValueError, match="Project path does not exist"):
            self.loader.setup_project(nonexistent_path)
    
    def test_setup_project_auto_name(self):
        """Test setup with automatic project name"""
        project_path = self.temp_path / "Auto Project Name"
        project_path.mkdir()
        
        config = self.loader.setup_project(project_path)
        
        assert config.name == "auto-project-name"
    
    def test_create_project_structure(self):
        """Test project directory structure creation"""
        project_path = self.temp_path / "structure_test"
        project_path.mkdir()
        
        config = Mock()
        self.loader._create_project_structure(project_path, config)
        
        assert (project_path / ".claude-indexer").exists()
        assert (project_path / ".claude").exists()
        assert (project_path / ".claude" / "hooks").exists()
    
    def test_setup_claude_hooks(self):
        """Test Claude Code hooks setup"""
        project_path = self.temp_path / "hooks_test"
        project_path.mkdir()
        
        config = Mock()
        config.name = "hooks-test"
        config.collection_prefix = "hooks-test"
        config.qdrant.url = "http://localhost:6333"
        
        self.loader._setup_claude_hooks(project_path, config)
        
        settings_file = project_path / ".claude" / "settings.json"
        assert settings_file.exists()
        
        with open(settings_file) as f:
            settings = json.load(f)
        
        assert "hooks" in settings
        assert "user_prompt_submit" in settings["hooks"]
        assert settings["hooks"]["user_prompt_submit"][0]["environment"]["PROJECT_NAME"] == "hooks-test"
    
    @patch('requests.get')
    def test_validate_qdrant_connection_success(self, mock_get):
        """Test successful Qdrant connection validation"""
        # Mock successful responses
        mock_health = Mock()
        mock_health.status_code = 200
        mock_collections = Mock()
        mock_collections.status_code = 200
        
        mock_get.side_effect = [mock_health, mock_collections]
        
        config = Mock()
        config.qdrant.url = "http://localhost:6333"
        config.qdrant.timeout = 60.0
        
        result = self.loader.validate_qdrant_connection(config)
        
        assert result is True
        assert mock_get.call_count == 2
    
    @patch('requests.get')
    def test_validate_qdrant_connection_health_fail(self, mock_get):
        """Test Qdrant validation with health check failure"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response
        
        config = Mock()
        config.qdrant.url = "http://localhost:6333"
        config.qdrant.timeout = 60.0
        
        result = self.loader.validate_qdrant_connection(config)
        
        assert result is False
    
    @patch('requests.get')
    def test_validate_qdrant_connection_collections_fail(self, mock_get):
        """Test Qdrant validation with collections check failure"""
        mock_health = Mock()
        mock_health.status_code = 200
        mock_collections = Mock()
        mock_collections.status_code = 404
        
        mock_get.side_effect = [mock_health, mock_collections]
        
        config = Mock()
        config.qdrant.url = "http://localhost:6333"
        config.qdrant.timeout = 60.0
        
        result = self.loader.validate_qdrant_connection(config)
        
        assert result is False
    
    @patch('requests.get')
    def test_validate_qdrant_connection_exception(self, mock_get):
        """Test Qdrant validation with network exception"""
        mock_get.side_effect = Exception("Connection failed")
        
        config = Mock()
        config.qdrant.url = "http://localhost:6333"
        config.qdrant.timeout = 60.0
        
        result = self.loader.validate_qdrant_connection(config)
        
        assert result is False
    
    def test_list_projects_empty_registry(self):
        """Test listing projects with empty registry"""
        with patch.object(self.loader.global_settings, 'global_config_dir', self.temp_path):
            projects = self.loader.list_projects()
            
            assert projects == []
    
    def test_list_projects_with_registry(self):
        """Test listing projects with existing registry"""
        registry_data = {
            "projects": [
                {"name": "project1", "path": "/path/1"},
                {"name": "project2", "path": "/path/2"}
            ]
        }
        
        # Create actual registry file
        registry_file = self.temp_path / "projects.json"
        with open(registry_file, 'w') as f:
            json.dump(registry_data, f)
        
        with patch.object(self.loader.global_settings, 'global_config_dir', self.temp_path):
            projects = self.loader.list_projects()
        
        assert len(projects) == 2
        assert projects[0]["name"] == "project1"
    
    def test_list_projects_invalid_registry(self):
        """Test listing projects with invalid registry file"""
        # Create invalid registry file
        registry_file = self.temp_path / "projects.json"
        with open(registry_file, 'w') as f:
            f.write("invalid json")
        
        with patch.object(self.loader.global_settings, 'global_config_dir', self.temp_path):
            projects = self.loader.list_projects()
        
        assert projects == []
    
    def test_register_project_new_registry(self):
        """Test registering project in new registry"""
        config = Mock()
        config.name = "test-project"
        config.path = Path(self.temp_path / "test_project")  # Use temp path
        config.collection_prefix = "test"
        config.get_collection_names.return_value = {"code": "test-code", "relations": "test-relations"}
        # Mock hasattr check for last_modified
        with patch('builtins.hasattr', return_value=False):
            with patch.object(self.loader.global_settings, 'global_config_dir', self.temp_path):
                result = self.loader.register_project(config)
        
        assert result is True
        
        # Check that registry file was created
        registry_file = self.temp_path / "projects.json"
        assert registry_file.exists()
    
    def test_register_project_existing_registry(self):
        """Test registering project in existing registry"""
        existing_registry = {
            "projects": [
                {"name": "existing", "path": "/existing/path"}
            ]
        }
        
        # Create existing registry file
        registry_file = self.temp_path / "projects.json"
        with open(registry_file, 'w') as f:
            json.dump(existing_registry, f)
        
        config = Mock()
        config.name = "new-project"
        config.path = Path(self.temp_path / "new_project")  # Use temp path
        config.collection_prefix = "new"
        config.get_collection_names.return_value = {"code": "new-code"}
        
        with patch('builtins.hasattr', return_value=False):
            with patch.object(self.loader.global_settings, 'global_config_dir', self.temp_path):
                result = self.loader.register_project(config)
        
        assert result is True
        
        # Check that registry was updated
        with open(registry_file) as f:
            updated_registry = json.load(f)
        assert len(updated_registry["projects"]) == 2
    
    def test_register_project_update_existing(self):
        """Test updating existing project in registry"""
        test_project_path = str(self.temp_path / "test_project")
        
        existing_registry = {
            "projects": [
                {"name": "existing", "path": test_project_path},
                {"name": "other", "path": "/other/path"}
            ]
        }
        
        # Create existing registry file
        registry_file = self.temp_path / "projects.json"
        with open(registry_file, 'w') as f:
            json.dump(existing_registry, f)
        
        config = Mock()
        config.name = "updated-project"
        config.path = Path(test_project_path)  # Same path as existing
        config.collection_prefix = "updated"
        config.get_collection_names.return_value = {"code": "updated-code"}
        
        with patch('builtins.hasattr', return_value=False):
            with patch.object(self.loader.global_settings, 'global_config_dir', self.temp_path):
                result = self.loader.register_project(config)
        
        assert result is True
        
        # Check that existing project was updated
        with open(registry_file) as f:
            updated_registry = json.load(f)
        assert len(updated_registry["projects"]) == 2  # Should still be 2, one updated
    
    def test_register_project_failure(self):
        """Test project registration failure"""
        config = Mock()
        config.name = "failing-project"
        config.path = Path("/test/path")
        config.collection_prefix = "failing"
        config.get_collection_names.return_value = ["failing-code"]
        
        # Use a read-only directory to simulate permission error
        read_only_dir = self.temp_path / "readonly"
        read_only_dir.mkdir()
        read_only_dir.chmod(0o444)  # Read-only
        
        try:
            with patch.object(self.loader.global_settings, 'global_config_dir', read_only_dir):
                result = self.loader.register_project(config)
            
            assert result is False
        finally:
            # Restore permissions for cleanup
            read_only_dir.chmod(0o755)
    
    def test_get_project_by_name_found(self):
        """Test getting project configuration by name when found"""
        projects_data = [
            {"name": "found-project", "path": str(self.temp_path / "found")}
        ]
        
        # Create the project directory and config
        project_path = self.temp_path / "found"
        project_path.mkdir()
        
        with patch.object(self.loader, 'list_projects', return_value=projects_data):
            with patch.object(self.loader, 'load_project_config') as mock_load:
                mock_config = Mock()
                mock_load.return_value = mock_config
                
                result = self.loader.get_project_by_name("found-project")
        
        assert result is mock_config
        mock_load.assert_called_once_with(project_path, "found-project")
    
    def test_get_project_by_name_not_found(self):
        """Test getting project configuration by name when not found"""
        projects_data = [
            {"name": "other-project", "path": "/other/path"}
        ]
        
        with patch.object(self.loader, 'list_projects', return_value=projects_data):
            result = self.loader.get_project_by_name("nonexistent-project")
        
        assert result is None
    
    def test_clear_cache(self):
        """Test clearing configuration cache"""
        # Add some items to cache
        self.loader.config_cache["key1"] = "value1"
        self.loader.config_cache["key2"] = "value2"
        
        assert len(self.loader.config_cache) == 2
        
        self.loader.clear_cache()
        
        assert len(self.loader.config_cache) == 0


class TestTemplateSubstitution:
    """Test template substitution edge cases"""
    
    def setup_method(self):
        self.loader = ConfigurationLoader()
    
    def test_substitute_complex_nested_structure(self):
        """Test substitution in complex nested structure"""
        data = {
            "project": {
                "name": "${project_name}",
                "collections": [
                    "${project_name}-code",
                    "${project_name}-relations"
                ],
                "settings": {
                    "cache_dir": "/cache/${project_name}",
                    "enabled": True,
                    "count": 42
                }
            }
        }
        
        substitutions = {"project_name": "test-project"}
        
        result = self.loader._substitute_template_vars(data, substitutions)
        
        assert result["project"]["name"] == "test-project"
        assert result["project"]["collections"][0] == "test-project-code"
        assert result["project"]["collections"][1] == "test-project-relations"
        assert result["project"]["settings"]["cache_dir"] == "/cache/test-project"
        assert result["project"]["settings"]["enabled"] is True
        assert result["project"]["settings"]["count"] == 42
    
    def test_substitute_partial_substitution(self):
        """Test partial template substitution"""
        data = {"text": "Hello ${name}, welcome to ${place}!"}
        substitutions = {"name": "John"}  # Missing 'place'
        
        result = self.loader._substitute_template_vars(data, substitutions)
        
        # Should perform safe substitution
        assert result["text"] == "Hello John, welcome to ${place}!"
    
    def test_substitute_non_string_values(self):
        """Test substitution preserves non-string values"""
        data = {
            "string": "${project_name}",
            "number": 42,
            "boolean": True,
            "none": None,
            "list": [1, 2, 3],
            "dict": {"nested": "value"}
        }
        
        substitutions = {"project_name": "test"}
        
        result = self.loader._substitute_template_vars(data, substitutions)
        
        assert result["string"] == "test"
        assert result["number"] == 42
        assert result["boolean"] is True
        assert result["none"] is None
        assert result["list"] == [1, 2, 3]
        assert result["dict"]["nested"] == "value"


class TestEnvironmentOverrides:
    """Test environment variable override functionality"""
    
    def setup_method(self):
        self.loader = ConfigurationLoader()
    
    def test_all_env_var_mappings(self):
        """Test all defined environment variable mappings"""
        config_data = {
            "qdrant": {"url": "default", "timeout": 30},
            "stella": {"model_name": "default", "cache_dir": "/default", "device": "cpu"},
            "indexing": {"max_file_size_mb": 5},
            "logging": {"level": "DEBUG", "enable_telemetry": False}
        }
        
        env_overrides = {
            "CLAUDE_INDEXER_QDRANT_URL": "http://override:6333",
            "CLAUDE_INDEXER_QDRANT_TIMEOUT": "120.5",
            "CLAUDE_INDEXER_STELLA_MODEL": "custom_model",
            "CLAUDE_INDEXER_STELLA_CACHE_DIR": "/custom/cache",
            "CLAUDE_INDEXER_STELLA_DEVICE": "cuda",
            "CLAUDE_INDEXER_BATCH_SIZE": "64",
            "CLAUDE_INDEXER_MAX_FILE_SIZE_MB": "20",
            "CLAUDE_INDEXER_LOG_LEVEL": "ERROR",
            "CLAUDE_INDEXER_ENABLE_TELEMETRY": "true"
        }
        
        with patch.dict(os.environ, env_overrides):
            result = self.loader._apply_env_overrides(config_data)
        
        assert result["qdrant"]["url"] == "http://override:6333"
        assert result["qdrant"]["timeout"] == 120.5
        assert result["stella"]["model_name"] == "custom_model"
        assert result["stella"]["cache_dir"] == "/custom/cache"
        assert result["stella"]["device"] == "cuda"
        assert result["qdrant"]["batch_size"] == 64
        assert result["indexing"]["max_file_size_mb"] == 20
        assert result["logging"]["level"] == "ERROR"
        assert result["logging"]["enable_telemetry"] is True
    
    def test_env_override_creates_missing_keys(self):
        """Test environment overrides create missing nested keys"""
        config_data = {}
        
        with patch.dict(os.environ, {"CLAUDE_INDEXER_QDRANT_URL": "http://new:6333"}):
            result = self.loader._apply_env_overrides(config_data)
        
        assert result["qdrant"]["url"] == "http://new:6333"


class TestIntegrationScenarios:
    """Integration tests for configuration loader"""
    
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.loader = ConfigurationLoader()
    
    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_project_lifecycle(self):
        """Test complete project setup and management lifecycle"""
        project_path = self.temp_path / "lifecycle_project"
        project_path.mkdir()
        
        # Setup project
        config = self.loader.setup_project(project_path, "lifecycle-test")
        
        assert config.name == "lifecycle-test"
        assert config.path.resolve() == project_path.resolve()
        
        # Save configuration
        save_result = self.loader.save_project_config(config)
        assert save_result is True
        
        # Clear cache and reload
        self.loader.clear_cache()
        reloaded_config = self.loader.load_project_config(project_path)
        
        assert reloaded_config.name == "lifecycle-test"
        
        # Register project
        with patch.object(self.loader.global_settings, 'global_config_dir', self.temp_path):
            register_result = self.loader.register_project(reloaded_config)
        
        assert register_result is True
    
    def test_project_with_env_overrides(self):
        """Test project setup with environment variable overrides"""
        project_path = self.temp_path / "env_project"
        project_path.mkdir()
        
        env_overrides = {
            "CLAUDE_INDEXER_QDRANT_URL": "http://env-override:6333",
            "CLAUDE_INDEXER_STELLA_MODEL": "stella_en_1.5B_v5",  # Use valid model name
            "CLAUDE_INDEXER_LOG_LEVEL": "DEBUG"
        }
        
        with patch.dict(os.environ, env_overrides):
            config = self.loader.setup_project(project_path, "env-test")
        
        assert config.qdrant.url == "http://env-override:6333"
        assert config.stella.model_name == "stella_en_1.5B_v5"
    
    def test_config_error_recovery(self):
        """Test configuration error recovery scenarios"""
        project_path = self.temp_path / "error_project"
        project_path.mkdir()
        
        # Create corrupted config file
        config_dir = project_path / ".claude-indexer"
        config_dir.mkdir()
        config_file = config_dir / "config.json"
        
        with open(config_file, 'w') as f:
            f.write("{ corrupted json content")
        
        # Should recover by creating new config - uses directory name as fallback
        config = self.loader.load_project_config(project_path, "error-recovery")
        
        assert isinstance(config, ProjectConfig)
        # Error recovery uses directory name as fallback, not the provided project name
        assert config.name == "error-project"