"""
Configuration models for claude-code-context.

Handles project settings, Qdrant configuration, and Stella embeddings setup.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from pydantic_settings import BaseSettings
import os


class QdrantConfig(BaseModel):
    """Qdrant vector database configuration"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )
    
    # Connection settings
    url: str = "http://localhost:6333"
    api_key: Optional[str] = None
    timeout: float = 60.0
    
    # Collection settings (project-specific)
    collections: Dict[str, str] = Field(default_factory=dict)
    
    # Performance settings
    batch_size: int = Field(default=100, ge=1, le=1000)
    parallel_requests: int = Field(default=4, ge=1, le=16)
    
    # Storage settings
    vector_size: int = 1024  # Stella embedding dimensions
    distance_metric: str = "cosine"
    
    @field_validator('url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate Qdrant URL format"""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('Qdrant URL must start with http:// or https://')
        return v.rstrip('/')
    
    @field_validator('distance_metric')
    @classmethod
    def validate_distance_metric(cls, v: str) -> str:
        """Validate distance metric"""
        valid_metrics = {'cosine', 'euclidean', 'dot'}
        if v.lower() not in valid_metrics:
            raise ValueError(f'Distance metric must be one of: {valid_metrics}')
        return v.lower()
    
    def get_collection_name(self, project_name: str, collection_type: str) -> str:
        """Generate collection name for project and type"""
        safe_project = project_name.lower().replace(' ', '-').replace('_', '-')
        return f"{safe_project}-{collection_type}"
    
    def get_all_collections(self, project_name: str) -> Dict[str, str]:
        """Get all collection names for a project"""
        return {
            'code': self.get_collection_name(project_name, 'code'),
            'relations': self.get_collection_name(project_name, 'relations'),
            'embeddings': self.get_collection_name(project_name, 'embeddings')
        }


class StellaConfig(BaseModel):
    """Stella embeddings configuration"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )
    
    # Model settings
    model_name: str = "stella_en_400M_v5"
    dimensions: int = 1024
    
    # Local storage
    cache_dir: Path = Field(default_factory=lambda: Path.home() / ".cache" / "claude-indexer" / "stella")
    
    # Hardware settings
    device: Optional[str] = None  # Auto-detect if None (cuda, mps, cpu)
    batch_size: int = Field(default=32, ge=1, le=128)
    max_length: int = Field(default=512, ge=1, le=8192)
    
    # Performance settings
    normalize_embeddings: bool = True
    use_fp16: bool = True  # Use half precision for speed
    
    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate Stella model name"""
        valid_models = {
            'stella_en_400M_v5',
            'stella_en_1.5B_v5', 
            'stella_base_en_v2'
        }
        if v not in valid_models:
            raise ValueError(f'Model must be one of: {valid_models}')
        return v
    
    @field_validator('cache_dir')
    @classmethod
    def validate_cache_dir(cls, v: Path) -> Path:
        """Ensure cache directory exists"""
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @property
    def model_path(self) -> Path:
        """Get full path to cached model"""
        return self.cache_dir / self.model_name
    
    @property
    def is_model_cached(self) -> bool:
        """Check if model is already downloaded"""
        return self.model_path.exists() and any(self.model_path.iterdir())
    
    def get_device(self) -> str:
        """Auto-detect or return configured device"""
        if self.device:
            return self.device
        
        # Auto-detection logic
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"  # Apple Silicon
            else:
                return "cpu"
        except ImportError:
            return "cpu"


class IndexingConfig(BaseModel):
    """File indexing and processing configuration"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True
    )
    
    # File patterns
    include_patterns: List[str] = Field(
        default_factory=lambda: [
            "*.py", "*.js", "*.ts", "*.jsx", "*.tsx",
            "*.go", "*.rs", "*.java", "*.cpp", "*.c", "*.h", "*.hpp",
            "*.cs", "*.rb", "*.php", "*.swift", "*.kt", "*.scala",
            "*.md", "*.txt", "*.json", "*.yaml", "*.yml", "*.toml"
        ]
    )
    
    exclude_patterns: List[str] = Field(
        default_factory=lambda: [
            "node_modules/*", ".git/*", "__pycache__/*", "*.pyc",
            "venv/*", ".venv/*", "env/*", ".env/*",
            "dist/*", "build/*", "target/*", ".next/*",
            "*.log", "*.tmp", "*.cache", ".DS_Store",
            "*.min.js", "*.min.css", "coverage/*"
        ]
    )
    
    # Processing limits
    max_file_size_mb: int = Field(default=10, ge=1, le=100)
    max_files_per_batch: int = Field(default=50, ge=1, le=500)
    
    # Content processing
    extract_docstrings: bool = True
    extract_comments: bool = True
    include_test_files: bool = True
    
    # Language-specific settings
    language_settings: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    @field_validator('include_patterns', 'exclude_patterns')
    @classmethod
    def validate_patterns(cls, v: List[str]) -> List[str]:
        """Validate glob patterns"""
        if not v:
            raise ValueError('Pattern list cannot be empty')
        return [pattern.strip() for pattern in v if pattern.strip()]
    
    def should_index_file(self, file_path: Path) -> bool:
        """Check if file should be indexed based on patterns"""
        from fnmatch import fnmatch
        
        file_str = str(file_path)
        
        # Check exclude patterns first
        for pattern in self.exclude_patterns:
            if fnmatch(file_str, pattern):
                return False
        
        # Check include patterns
        for pattern in self.include_patterns:
            if fnmatch(file_str, pattern):
                return True
        
        return False
    
    @property
    def max_file_size_bytes(self) -> int:
        """Get max file size in bytes"""
        return self.max_file_size_mb * 1024 * 1024


class ProjectConfig(BaseModel):
    """Project-specific configuration with validation"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=False
    )
    
    # Project identification
    name: str
    path: Path
    collection_prefix: str
    
    # Component configurations
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    stella: StellaConfig = Field(default_factory=StellaConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    
    # Project metadata
    description: Optional[str] = None
    version: str = "1.0.0"
    
    # Runtime settings
    enable_watch_mode: bool = False
    auto_index_on_change: bool = True
    
    # Performance settings
    max_concurrent_operations: int = Field(default=4, ge=1, le=16)
    cache_embeddings: bool = True
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate project name"""
        if not v or not v.replace('-', '').replace('_', '').replace(' ', '').isalnum():
            raise ValueError('Project name must be alphanumeric with dashes, underscores, or spaces')
        return v.strip()
    
    @field_validator('collection_prefix')
    @classmethod
    def validate_collection_prefix(cls, v: str) -> str:
        """Validate collection prefix"""
        if not v or not v.replace('-', '').replace('_', '').isalnum():
            raise ValueError('Collection prefix must be alphanumeric with dashes/underscores')
        return v.lower()
    
    @field_validator('path')
    @classmethod
    def validate_path(cls, v: Path) -> Path:
        """Validate project path exists"""
        if not v.exists():
            raise ValueError(f'Project path does not exist: {v}')
        if not v.is_dir():
            raise ValueError(f'Project path is not a directory: {v}')
        return v.resolve()
    
    
    def get_collection_names(self) -> Dict[str, str]:
        """Generate all collection names for this project"""
        return self.qdrant.get_all_collections(self.collection_prefix)
    
    def get_config_dir(self) -> Path:
        """Get project configuration directory"""
        config_dir = self.path / ".claude-indexer"
        config_dir.mkdir(exist_ok=True)
        return config_dir
    
    def get_config_file(self) -> Path:
        """Get project configuration file path"""
        return self.get_config_dir() / "config.json"
    
    @property
    def is_initialized(self) -> bool:
        """Check if project is properly initialized"""
        return self.get_config_file().exists()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = self.model_dump()
        # Convert Path objects to strings
        data['path'] = str(data['path'])
        data['stella']['cache_dir'] = str(data['stella']['cache_dir'])
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProjectConfig':
        """Create from dictionary"""
        # Convert string paths back to Path objects
        if 'path' in data:
            data['path'] = Path(data['path'])
        if 'stella' in data and 'cache_dir' in data['stella']:
            data['stella']['cache_dir'] = Path(data['stella']['cache_dir'])
        return cls(**data)


class GlobalSettings(BaseSettings):
    """Global application settings with environment variable support"""
    model_config = ConfigDict(
        env_prefix="CLAUDE_INDEXER_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # Default configurations
    default_qdrant_url: str = "http://localhost:6333"
    default_stella_model: str = "stella_en_400M_v5"
    default_batch_size: int = 100
    
    # Global directories
    global_cache_dir: Path = Field(
        default_factory=lambda: Path.home() / ".cache" / "claude-indexer"
    )
    global_config_dir: Path = Field(
        default_factory=lambda: Path.home() / ".claude-indexer"
    )
    
    # Performance defaults
    max_concurrent_projects: int = Field(default=10, ge=1, le=50)
    default_timeout: float = Field(default=60.0, ge=1.0, le=300.0)
    
    # Logging
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    log_to_file: bool = False
    
    # Feature flags
    enable_telemetry: bool = False
    enable_auto_updates: bool = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self.global_cache_dir.mkdir(parents=True, exist_ok=True)
        self.global_config_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def stella_cache_dir(self) -> Path:
        """Get Stella model cache directory"""
        stella_dir = self.global_cache_dir / "stella"
        stella_dir.mkdir(exist_ok=True)
        return stella_dir
    
    @property
    def projects_registry_file(self) -> Path:
        """Get projects registry file path"""
        return self.global_config_dir / "projects.json"
    
    def get_log_file(self) -> Optional[Path]:
        """Get log file path if logging to file is enabled"""
        if not self.log_to_file:
            return None
        log_dir = self.global_config_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        return log_dir / "claude-indexer.log"


# Default settings instance
DEFAULT_SETTINGS = GlobalSettings()