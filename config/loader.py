"""
Configuration loading and management with template support.

Handles project setup, template substitution, and multi-project configuration.
"""

import json
import os
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional, Union
import logging

from core.models.config import ProjectConfig, GlobalSettings
from .defaults import DEFAULT_SETTINGS, get_default_project_config, ENV_VAR_MAPPING

logger = logging.getLogger(__name__)


class ConfigurationLoader:
    """Load and manage project configurations with template support"""
    
    def __init__(self):
        self.global_settings = GlobalSettings()
        self.template_dir = Path(__file__).parent.parent / "templates"
        self.config_cache: Dict[str, ProjectConfig] = {}
        
        # Ensure template directory exists
        self.template_dir.mkdir(exist_ok=True)
    
    def load_project_config(
        self, 
        project_path: Union[str, Path],
        project_name: Optional[str] = None
    ) -> ProjectConfig:
        """Load or create project configuration"""
        project_path = Path(project_path).resolve()
        
        if not project_name:
            project_name = project_path.name.lower().replace(' ', '-')
        
        # Check cache first
        cache_key = str(project_path)
        if cache_key in self.config_cache:
            return self.config_cache[cache_key]
        
        config_file = project_path / ".claude-indexer" / "config.json"
        
        if config_file.exists():
            config = self._load_existing_config(config_file)
        else:
            config = self._create_project_config(project_path, project_name)
        
        # Cache the configuration
        self.config_cache[cache_key] = config
        return config
    
    def _load_existing_config(self, config_file: Path) -> ProjectConfig:
        """Load existing configuration file with validation"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Apply environment variable overrides
            data = self._apply_env_overrides(data)
            
            # Convert paths
            if 'path' in data:
                data['path'] = Path(data['path'])
            if 'stella' in data and 'cache_dir' in data['stella']:
                data['stella']['cache_dir'] = Path(data['stella']['cache_dir'])
            
            return ProjectConfig(**data)
        
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to load config from {config_file}: {e}")
            # Fall back to creating new config
            return self._create_project_config(
                config_file.parent.parent,
                config_file.parent.parent.name
            )
    
    def _create_project_config(self, project_path: Path, project_name: str) -> ProjectConfig:
        """Create new project configuration from template"""
        # Sanitize project name for collection naming
        safe_project_name = project_name.lower().replace(' ', '-').replace('_', '-')
        
        # Start with default template
        config_data = get_default_project_config()
        
        # Apply template substitution
        substitutions = {
            'project_name': safe_project_name,
            'project_path': str(project_path),
            'collection_prefix': safe_project_name
        }
        
        config_data = self._substitute_template_vars(config_data, substitutions)
        
        # Apply environment overrides
        config_data = self._apply_env_overrides(config_data)
        
        # Convert paths
        config_data['path'] = project_path
        config_data['stella']['cache_dir'] = Path(config_data['stella']['cache_dir'])
        
        # Create and validate config
        config = ProjectConfig(**config_data)
        
        # Ensure Qdrant collections are properly configured
        config.qdrant.collections = config.get_collection_names()
        
        return config
    
    def _substitute_template_vars(
        self, 
        data: Any, 
        substitutions: Dict[str, str]
    ) -> Any:
        """Recursively substitute template variables in configuration"""
        if isinstance(data, dict):
            return {
                key: self._substitute_template_vars(value, substitutions)
                for key, value in data.items()
            }
        elif isinstance(data, list):
            return [
                self._substitute_template_vars(item, substitutions)
                for item in data
            ]
        elif isinstance(data, str):
            try:
                template = Template(data)
                return template.safe_substitute(substitutions)
            except (ValueError, KeyError):
                return data
        else:
            return data
    
    def _apply_env_overrides(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration"""
        for env_var, config_path in ENV_VAR_MAPPING.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                self._set_nested_value(config_data, config_path, env_value)
        
        return config_data
    
    def _set_nested_value(self, data: Dict[str, Any], path: str, value: str) -> None:
        """Set nested dictionary value using dot notation path"""
        keys = path.split('.')
        current = data
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Convert value to appropriate type
        final_key = keys[-1]
        converted_value = self._convert_env_value(value)
        current[final_key] = converted_value
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type"""
        # Boolean conversion
        if value.lower() in ('true', 'yes', '1', 'on'):
            return True
        elif value.lower() in ('false', 'no', '0', 'off'):
            return False
        
        # Numeric conversion
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def save_project_config(self, config: ProjectConfig) -> bool:
        """Save project configuration to disk"""
        try:
            config_dir = config.get_config_dir()
            config_file = config.get_config_file()
            
            # Ensure directory exists
            config_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert to dict and save
            config_data = config.to_dict()
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved configuration to {config_file}")
            
            # Update cache
            cache_key = str(config.path)
            self.config_cache[cache_key] = config
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save config for {config.path}: {e}")
            return False
    
    def setup_project(
        self,
        project_path: Union[str, Path],
        project_name: Optional[str] = None,
        overwrite: bool = False
    ) -> ProjectConfig:
        """Complete project setup with configuration and templates"""
        project_path = Path(project_path).resolve()
        
        if not project_path.exists():
            raise ValueError(f"Project path does not exist: {project_path}")
        
        if not project_name:
            project_name = project_path.name.lower().replace(' ', '-')
        
        logger.info(f"Setting up project '{project_name}' at {project_path}")
        
        # Create configuration
        config = self.load_project_config(project_path, project_name)
        
        # Check if already initialized
        if config.is_initialized and not overwrite:
            logger.info(f"Project already initialized at {project_path}")
            return config
        
        # Create project structure
        self._create_project_structure(project_path, config)
        
        # Save configuration
        self.save_project_config(config)
        
        # Generate Claude Code hook configuration
        self._setup_claude_hooks(project_path, config)
        
        logger.info(f"Project setup complete for '{project_name}'")
        return config
    
    def _create_project_structure(self, project_path: Path, config: ProjectConfig) -> None:
        """Create project directory structure"""
        # Create .claude-indexer directory
        indexer_dir = project_path / ".claude-indexer"
        indexer_dir.mkdir(exist_ok=True)
        
        # Create .claude directory for hooks
        claude_dir = project_path / ".claude"
        claude_dir.mkdir(exist_ok=True)
        
        hooks_dir = claude_dir / "hooks"
        hooks_dir.mkdir(exist_ok=True)
    
    def _setup_claude_hooks(self, project_path: Path, config: ProjectConfig) -> None:
        """Setup Claude Code hooks configuration"""
        # Create hooks directory
        hooks_dir = project_path / ".claude" / "hooks"
        hooks_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate Claude Code settings.json
        settings_file = project_path / ".claude" / "settings.json"
        claude_settings = {
            "hooks": {
                "user_prompt_submit": [
                    {
                        "command": ["python", "-m", "hooks.user_prompt_submit"],
                        "working_directory": str(project_path),
                        "environment": {
                            "PROJECT_NAME": config.name,
                            "COLLECTION_PREFIX": config.collection_prefix,
                            "QDRANT_URL": config.qdrant.url
                        }
                    }
                ]
            }
        }
        
        with open(settings_file, 'w', encoding='utf-8') as f:
            json.dump(claude_settings, f, indent=2)
        
        logger.info(f"Created Claude Code settings at {settings_file}")
    
    def validate_qdrant_connection(self, config: ProjectConfig) -> bool:
        """Validate Qdrant connection and collections"""
        try:
            import requests
            
            # Test connection
            response = requests.get(
                f"{config.qdrant.url}/health",
                timeout=config.qdrant.timeout
            )
            
            if response.status_code != 200:
                logger.error(f"Qdrant health check failed: {response.status_code}")
                return False
            
            # Test collections endpoint
            collections_response = requests.get(
                f"{config.qdrant.url}/collections",
                timeout=config.qdrant.timeout
            )
            
            if collections_response.status_code != 200:
                logger.error(f"Qdrant collections check failed: {collections_response.status_code}")
                return False
            
            logger.info("Qdrant connection validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Qdrant connection validation failed: {e}")
            return False
    
    def list_projects(self) -> List[Dict[str, Any]]:
        """List all configured projects"""
        projects = []
        
        # Check global registry
        registry_file = self.global_settings.projects_registry_file
        if registry_file.exists():
            try:
                with open(registry_file, 'r', encoding='utf-8') as f:
                    registry_data = json.load(f)
                    projects.extend(registry_data.get('projects', []))
            except Exception as e:
                logger.error(f"Failed to load projects registry: {e}")
        
        return projects
    
    def register_project(self, config: ProjectConfig) -> bool:
        """Register project in global registry"""
        try:
            registry_file = self.global_settings.projects_registry_file
            
            # Load existing registry
            projects = []
            if registry_file.exists():
                with open(registry_file, 'r', encoding='utf-8') as f:
                    registry_data = json.load(f)
                    projects = registry_data.get('projects', [])
            
            # Update or add project
            project_info = {
                'name': config.name,
                'path': str(config.path),
                'collection_prefix': config.collection_prefix,
                'collections': config.get_collection_names(),
                'last_updated': config.last_modified.isoformat() if hasattr(config, 'last_modified') else None
            }
            
            # Remove existing entry for same path
            projects = [p for p in projects if p.get('path') != str(config.path)]
            projects.append(project_info)
            
            # Save registry
            registry_data = {
                'version': '1.0.0',
                'projects': projects
            }
            
            registry_file.parent.mkdir(parents=True, exist_ok=True)
            with open(registry_file, 'w', encoding='utf-8') as f:
                json.dump(registry_data, f, indent=2)
            
            logger.info(f"Registered project '{config.name}' in global registry")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register project: {e}")
            return False
    
    def get_project_by_name(self, project_name: str) -> Optional[ProjectConfig]:
        """Get project configuration by name"""
        projects = self.list_projects()
        
        for project_info in projects:
            if project_info.get('name') == project_name:
                project_path = Path(project_info['path'])
                return self.load_project_config(project_path, project_name)
        
        return None
    
    def clear_cache(self) -> None:
        """Clear configuration cache"""
        self.config_cache.clear()
        logger.info("Configuration cache cleared")