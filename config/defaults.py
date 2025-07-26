"""
Default configuration values for claude-code-context.

Centralized defaults that can be overridden by environment variables or config files.
"""

from pathlib import Path
from typing import Dict, List, Any

# Global default settings
DEFAULT_SETTINGS = {
    # Qdrant configuration
    "qdrant": {
        "url": "http://localhost:6333",
        "timeout": 60.0,
        "batch_size": 100,
        "parallel_requests": 4,
        "vector_size": 1024,
        "distance_metric": "cosine"
    },
    
    # Stella embeddings
    "stella": {
        "model_name": "stella_en_400M_v5",
        "dimensions": 1024,
        "cache_dir": str(Path.home() / ".cache" / "claude-indexer" / "stella"),
        "device": None,  # Auto-detect
        "batch_size": 32,
        "max_length": 512,
        "normalize_embeddings": True,
        "use_fp16": True
    },
    
    # File indexing
    "indexing": {
        "include_patterns": [
            "*.py", "*.js", "*.ts", "*.jsx", "*.tsx",
            "*.go", "*.rs", "*.java", "*.cpp", "*.c", "*.h", "*.hpp",
            "*.cs", "*.rb", "*.php", "*.swift", "*.kt", "*.scala",
            "*.md", "*.txt", "*.json", "*.yaml", "*.yml", "*.toml"
        ],
        "exclude_patterns": [
            "node_modules/*", ".git/*", "__pycache__/*", "*.pyc",
            "venv/*", ".venv/*", "env/*", ".env/*",
            "dist/*", "build/*", "target/*", ".next/*",
            "*.log", "*.tmp", "*.cache", ".DS_Store",
            "*.min.js", "*.min.css", "coverage/*"
        ],
        "max_file_size_mb": 10,
        "max_files_per_batch": 50,
        "extract_docstrings": True,
        "extract_comments": True,
        "include_test_files": True
    },
    
    # Project defaults
    "project": {
        "version": "1.0.0",
        "enable_watch_mode": False,
        "auto_index_on_change": True,
        "max_concurrent_operations": 4,
        "cache_embeddings": True
    },
    
    # Global directories
    "directories": {
        "global_cache_dir": str(Path.home() / ".cache" / "claude-indexer"),
        "global_config_dir": str(Path.home() / ".claude-indexer"),
        "stella_cache_dir": str(Path.home() / ".cache" / "claude-indexer" / "stella")
    },
    
    # Performance settings
    "performance": {
        "max_concurrent_projects": 10,
        "default_timeout": 60.0,
        "max_processing_time_ms": 5000
    },
    
    # Hook settings
    "hooks": {
        "max_results_per_query": 5,
        "min_relevance_score": 0.3,
        "enable_hybrid_search": True,
        "qdrant_timeout": 10.0
    },
    
    # Logging
    "logging": {
        "level": "INFO",
        "log_to_file": False,
        "enable_telemetry": False,
        "enable_auto_updates": False
    }
}

# Supported programming languages for tree-sitter
SUPPORTED_LANGUAGES = {
    'python': {
        'extensions': ['.py', '.pyi'],
        'tree_sitter_name': 'python',
        'comment_style': '#'
    },
    'javascript': {
        'extensions': ['.js', '.jsx', '.mjs'],
        'tree_sitter_name': 'javascript', 
        'comment_style': '//'
    },
    'typescript': {
        'extensions': ['.ts', '.tsx', '.d.ts'],
        'tree_sitter_name': 'typescript',
        'comment_style': '//'
    },
    'go': {
        'extensions': ['.go'],
        'tree_sitter_name': 'go',
        'comment_style': '//'
    },
    'rust': {
        'extensions': ['.rs'],
        'tree_sitter_name': 'rust',
        'comment_style': '//'
    },
    'java': {
        'extensions': ['.java'],
        'tree_sitter_name': 'java',
        'comment_style': '//'
    },
    'cpp': {
        'extensions': ['.cpp', '.cc', '.cxx', '.c++', '.hpp', '.hh', '.hxx', '.h++'],
        'tree_sitter_name': 'cpp',
        'comment_style': '//'
    },
    'c': {
        'extensions': ['.c', '.h'],
        'tree_sitter_name': 'c',
        'comment_style': '//'
    },
    'csharp': {
        'extensions': ['.cs'],
        'tree_sitter_name': 'c_sharp',
        'comment_style': '//'
    },
    'ruby': {
        'extensions': ['.rb', '.rbw'],
        'tree_sitter_name': 'ruby',
        'comment_style': '#'
    },
    'php': {
        'extensions': ['.php', '.phtml'],
        'tree_sitter_name': 'php',
        'comment_style': '//'
    }
}

# Tree-sitter entity types mapping
TREE_SITTER_ENTITY_MAPPING = {
    'python': {
        'function_definition': 'function',
        'class_definition': 'class',
        'assignment': 'variable',
        'import_statement': 'import',
        'import_from_statement': 'import',
        'decorated_definition': 'decorator'
    },
    'javascript': {
        'function_declaration': 'function',
        'function_expression': 'function',
        'arrow_function': 'function',
        'class_declaration': 'class',
        'variable_declaration': 'variable',
        'import_statement': 'import',
        'export_statement': 'export'
    },
    'typescript': {
        'function_declaration': 'function',
        'function_expression': 'function', 
        'arrow_function': 'function',
        'class_declaration': 'class',
        'interface_declaration': 'interface',
        'type_alias_declaration': 'type_alias',
        'variable_declaration': 'variable',
        'import_statement': 'import',
        'export_statement': 'export'
    },
    'go': {
        'function_declaration': 'function',
        'method_declaration': 'method',
        'type_declaration': 'type_alias',
        'var_declaration': 'variable',
        'const_declaration': 'constant',
        'import_declaration': 'import'
    },
    'rust': {
        'function_item': 'function',
        'impl_item': 'class',
        'struct_item': 'struct',
        'enum_item': 'enum',
        'const_item': 'constant',
        'static_item': 'variable',
        'use_declaration': 'import'
    },
    'java': {
        'method_declaration': 'method',
        'class_declaration': 'class',
        'interface_declaration': 'interface',
        'field_declaration': 'variable',
        'import_declaration': 'import'
    }
}

# Collection name templates
COLLECTION_TEMPLATES = {
    'code': '{project_name}-code',
    'relations': '{project_name}-relations', 
    'embeddings': '{project_name}-embeddings'
}

# Environment variable mappings
ENV_VAR_MAPPING = {
    'CLAUDE_INDEXER_QDRANT_URL': 'qdrant.url',
    'CLAUDE_INDEXER_QDRANT_TIMEOUT': 'qdrant.timeout',
    'CLAUDE_INDEXER_STELLA_MODEL': 'stella.model_name',
    'CLAUDE_INDEXER_STELLA_CACHE_DIR': 'stella.cache_dir',
    'CLAUDE_INDEXER_STELLA_DEVICE': 'stella.device',
    'CLAUDE_INDEXER_BATCH_SIZE': 'qdrant.batch_size',
    'CLAUDE_INDEXER_MAX_FILE_SIZE_MB': 'indexing.max_file_size_mb',
    'CLAUDE_INDEXER_LOG_LEVEL': 'logging.level',
    'CLAUDE_INDEXER_ENABLE_TELEMETRY': 'logging.enable_telemetry'
}

def get_default_project_config() -> Dict[str, Any]:
    """Get default project configuration template"""
    return {
        'name': '${project_name}',
        'path': '${project_path}',
        'collection_prefix': '${project_name}',
        'qdrant': DEFAULT_SETTINGS['qdrant'],
        'stella': DEFAULT_SETTINGS['stella'],
        'indexing': DEFAULT_SETTINGS['indexing'],
        'description': None,
        'version': DEFAULT_SETTINGS['project']['version'],
        'enable_watch_mode': DEFAULT_SETTINGS['project']['enable_watch_mode'],
        'auto_index_on_change': DEFAULT_SETTINGS['project']['auto_index_on_change'],
        'max_concurrent_operations': DEFAULT_SETTINGS['project']['max_concurrent_operations'],
        'cache_embeddings': DEFAULT_SETTINGS['project']['cache_embeddings']
    }

def get_language_config(file_extension: str) -> Dict[str, Any]:
    """Get language configuration for file extension"""
    for lang, config in SUPPORTED_LANGUAGES.items():
        if file_extension.lower() in config['extensions']:
            return {
                'language': lang,
                'tree_sitter_name': config['tree_sitter_name'],
                'comment_style': config['comment_style'],
                'entity_mapping': TREE_SITTER_ENTITY_MAPPING.get(lang, {})
            }
    
    return {
        'language': 'unknown',
        'tree_sitter_name': 'text',
        'comment_style': '#',
        'entity_mapping': {}
    }