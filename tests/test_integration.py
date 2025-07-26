"""
Basic integration tests for sprint 1 deliverables.

Tests end-to-end workflows for project setup and configuration.
"""

import pytest
import tempfile
import json
import subprocess
from pathlib import Path
from config.loader import ConfigurationLoader
from core.models.config import ProjectConfig


class TestProjectSetupIntegration:
    """Integration tests for project setup workflow"""
    
    def test_configuration_loader_end_to_end(self):
        """Test complete configuration loading workflow"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            project_name = "test-integration-project"
            
            # Test configuration creation
            loader = ConfigurationLoader()
            config = loader.setup_project(project_path, project_name)
            
            # Verify configuration was created
            assert config.name == project_name
            assert config.path == project_path.resolve()
            assert config.collection_prefix == project_name
            
            # Verify files were created
            config_file = project_path / ".claude-indexer" / "config.json"
            assert config_file.exists()
            
            settings_file = project_path / ".claude/settings.json"
            assert settings_file.exists()
            
            # Test configuration loading
            loaded_config = loader.load_project_config(project_path, project_name)
            assert loaded_config.name == config.name
            assert loaded_config.collection_prefix == config.collection_prefix
            
            # Test configuration persistence
            assert loader.save_project_config(config) is True
            
            # Verify saved configuration is valid JSON
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                assert config_data["name"] == project_name
                assert config_data["collection_prefix"] == project_name
    
    def test_project_setup_script_integration(self):
        """Test project setup script execution (if available)"""
        setup_script = Path(__file__).parent.parent / "setup-project.sh"
        
        if not setup_script.exists():
            pytest.skip("setup-project.sh not found")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            project_name = "test-script-project"
            
            # Run setup script with skip flags to avoid external dependencies
            cmd = [
                str(setup_script),
                project_name,
                "--path", str(project_path),
                "--skip-qdrant",
                "--overwrite"
            ]
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                # Script should complete successfully or with minor warnings
                assert result.returncode in [0, 1], f"Setup script failed: {result.stderr}"
                
                # Verify expected files were created
                expected_files = [
                    project_path / ".claude-indexer" / "config.json",
                    project_path / ".claude" / "settings.json"
                ]
                
                for file_path in expected_files:
                    assert file_path.exists(), f"Expected file not created: {file_path}"
                
                # Verify configuration is valid
                config_file = project_path / ".claude-indexer" / "config.json"
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                    assert config_data["name"] == project_name
                
            except subprocess.TimeoutExpired:
                pytest.fail("Setup script timed out")
            except FileNotFoundError:
                pytest.skip("Setup script not executable")
    
    def test_stella_installation_script_info(self):
        """Test Stella installation script info command"""
        stella_script = Path(__file__).parent.parent / "scripts" / "install_stella.py"
        
        if not stella_script.exists():
            pytest.skip("install_stella.py not found")
        
        try:
            # Test info command (should not require model download)
            result = subprocess.run(
                ["python", str(stella_script), "--info"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Should return JSON info
            assert result.returncode == 0, f"Stella script failed: {result.stderr}"
            
            # Try to parse output as JSON
            try:
                info_data = json.loads(result.stdout)
                assert "model_name" in info_data
                assert "cached" in info_data
                assert "platform" in info_data
            except json.JSONDecodeError:
                pytest.fail("Stella script did not return valid JSON")
                
        except subprocess.TimeoutExpired:
            pytest.fail("Stella script timed out")
        except FileNotFoundError:
            pytest.skip("Python not available for script execution")
    
    def test_qdrant_setup_script_help(self):
        """Test Qdrant setup script help command"""
        qdrant_script = Path(__file__).parent.parent / "scripts" / "setup-qdrant.sh"
        
        if not qdrant_script.exists():
            pytest.skip("setup-qdrant.sh not found")
        
        try:
            # Test help command
            result = subprocess.run(
                [str(qdrant_script), "help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            assert result.returncode == 0, f"Qdrant script help failed: {result.stderr}"
            assert "Usage:" in result.stdout
            assert "Commands:" in result.stdout
            
        except subprocess.TimeoutExpired:
            pytest.fail("Qdrant script timed out")
        except FileNotFoundError:
            pytest.skip("Bash not available for script execution")
    
    def test_project_validation_script(self):
        """Test project validation script"""
        test_script = Path(__file__).parent.parent / "scripts" / "test-project.sh"
        
        if not test_script.exists():
            pytest.skip("test-project.sh not found")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            
            # Create minimal project structure
            config_dir = project_path / ".claude-indexer"
            config_dir.mkdir()
            
            claude_dir = project_path / ".claude"
            claude_dir.mkdir()
            
            # Create minimal config files
            config_file = config_dir / "config.json"
            config_data = {
                "name": "test-project",
                "path": str(project_path),
                "collection_prefix": "test-project",
                "qdrant": {"url": "http://localhost:6333"},
                "stella": {"model_name": "stella_en_400M_v5"},
                "indexing": {"include_patterns": ["*.py"]}
            }
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f)
            
            settings_file = claude_dir / "settings.json"
            settings_data = {
                "hooks": {},
                "project_info": {"name": "test-project"}
            }
            
            with open(settings_file, 'w') as f:
                json.dump(settings_data, f)
            
            try:
                # Run validation script with skip flags
                result = subprocess.run(
                    [
                        str(test_script),
                        str(project_path),
                        "--skip-qdrant",
                        "--quick"
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                # Should pass basic validation (0=perfect, 1=minor issues, 2=major issues)
                assert result.returncode in [0, 1, 2], f"Validation failed: {result.stderr}"
                assert "File system structure" in result.stdout
                assert "Configuration validity" in result.stdout
                
            except subprocess.TimeoutExpired:
                pytest.fail("Validation script timed out")
            except FileNotFoundError:
                pytest.skip("Bash not available for script execution")


class TestConfigurationValidation:
    """Integration tests for configuration validation"""
    
    def test_project_config_validation_end_to_end(self):
        """Test complete project configuration validation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            
            # Test valid configuration
            config = ProjectConfig(
                name="valid-project",
                path=project_path,
                collection_prefix="valid-project"
            )
            
            # Test serialization round-trip
            config_dict = config.to_dict()
            restored_config = ProjectConfig.from_dict(config_dict)
            
            assert restored_config.name == config.name
            assert restored_config.path == config.path
            assert restored_config.collection_prefix == config.collection_prefix
            
            # Test collection names generation
            collections = config.get_collection_names()
            expected_collections = ["code", "relations", "embeddings"]
            
            for coll_type in expected_collections:
                assert coll_type in collections
                assert collections[coll_type].startswith("valid-project-")
            
            # Test configuration directory creation
            config_dir = config.get_config_dir()
            assert config_dir.exists()
            assert config_dir.is_dir()
    
    def test_template_substitution_integration(self):
        """Test template substitution with real templates"""
        template_dir = Path(__file__).parent.parent / "templates"
        
        if not template_dir.exists():
            pytest.skip("Templates directory not found")
        
        config_template = template_dir / ".claude-indexer" / "config.json.template"
        
        if config_template.exists():
            # Read template
            template_content = config_template.read_text()
            
            # Verify template has expected placeholders
            assert "${PROJECT_NAME}" in template_content
            assert "${PROJECT_PATH}" in template_content
            assert "${COLLECTION_PREFIX}" in template_content
            
            # Test manual substitution
            substituted = template_content.replace("${PROJECT_NAME}", "test-project")
            substituted = substituted.replace("${PROJECT_PATH}", "/test/path")
            substituted = substituted.replace("${COLLECTION_PREFIX}", "test-project")
            
            # Should be valid JSON after substitution
            try:
                config_data = json.loads(substituted)
                assert config_data["name"] == "test-project"
                assert config_data["collection_prefix"] == "test-project"
            except json.JSONDecodeError:
                pytest.fail("Template does not produce valid JSON after substitution")


class TestScriptIntegration:
    """Integration tests for shell scripts"""
    
    def test_global_installation_script_help(self):
        """Test global installation script help"""
        install_script = Path(__file__).parent.parent / "install-global.sh"
        
        if not install_script.exists():
            pytest.skip("install-global.sh not found")
        
        try:
            result = subprocess.run(
                [str(install_script), "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            assert result.returncode == 0, f"Install script help failed: {result.stderr}"
            assert "Usage:" in result.stdout
            assert "Options:" in result.stdout
            
        except subprocess.TimeoutExpired:
            pytest.fail("Install script timed out")
        except FileNotFoundError:
            pytest.skip("Bash not available")
    
    def test_script_executability(self):
        """Test that all shell scripts are executable"""
        script_files = [
            "setup-project.sh",
            "install-global.sh",
            "scripts/setup-qdrant.sh",
            "scripts/test-project.sh"
        ]
        
        project_root = Path(__file__).parent.parent
        
        for script_file in script_files:
            script_path = project_root / script_file
            
            if script_path.exists():
                # Check if file is executable
                import stat
                file_stat = script_path.stat()
                is_executable = bool(file_stat.st_mode & stat.S_IEXEC)
                
                assert is_executable, f"Script {script_file} is not executable"
    
    def test_python_script_syntax(self):
        """Test that Python scripts have valid syntax"""
        python_scripts = [
            "scripts/install_stella.py"
        ]
        
        project_root = Path(__file__).parent.parent
        
        for script_file in python_scripts:
            script_path = project_root / script_file
            
            if script_path.exists():
                try:
                    # Test syntax by compiling
                    with open(script_path, 'r') as f:
                        script_content = f.read()
                    
                    compile(script_content, str(script_path), 'exec')
                    
                except SyntaxError as e:
                    pytest.fail(f"Python script {script_file} has syntax error: {e}")


@pytest.mark.slow
class TestPerformanceIntegration:
    """Integration tests for performance characteristics"""
    
    def test_configuration_loading_performance(self):
        """Test configuration loading performance"""
        import time
        
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            
            loader = ConfigurationLoader()
            
            # Test configuration creation performance
            start_time = time.time()
            config = loader.setup_project(project_path, "perf-test-project")
            creation_time = time.time() - start_time
            
            # Should complete quickly (< 1 second)
            assert creation_time < 1.0, f"Configuration creation too slow: {creation_time:.2f}s"
            
            # Test configuration loading performance
            start_time = time.time()
            loaded_config = loader.load_project_config(project_path, "perf-test-project")
            loading_time = time.time() - start_time
            
            # Should load very quickly (< 0.1 seconds)
            assert loading_time < 0.1, f"Configuration loading too slow: {loading_time:.2f}s"
            
            # Verify loaded config is correct
            assert loaded_config.name == config.name
    
    def test_multiple_project_setup_performance(self):
        """Test setting up multiple projects"""
        import time
        
        num_projects = 5
        loader = ConfigurationLoader()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            
            start_time = time.time()
            
            for i in range(num_projects):
                project_path = base_path / f"project_{i}"
                project_path.mkdir()
                
                config = loader.setup_project(project_path, f"test-project-{i}")
                assert config.name == f"test-project-{i}"
            
            total_time = time.time() - start_time
            avg_time = total_time / num_projects
            
            # Should average less than 0.5 seconds per project
            assert avg_time < 0.5, f"Average project setup too slow: {avg_time:.2f}s"


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])