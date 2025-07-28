"""
Tree-sitter based multi-language parsers for code analysis.

This module provides parsers for extracting entities and relationships from
source code using Tree-sitter AST parsing. Supports Python, JavaScript/TypeScript,
Go, Rust, Java, and configuration files (JSON/YAML).

Key Components:
- ParserProtocol: Abstract interface for all parsers
- TreeSitterBase: Common Tree-sitter functionality
- ParserRegistry: Dynamic parser factory and discovery
- Language-specific parsers for each supported language

Example:
    from core.parser.registry import parser_registry
    from pathlib import Path
    
    # Get parser for a Python file
    parser = parser_registry.get_parser_for_file(Path("example.py"))
    if parser:
        result = parser.parse_file(Path("example.py"))
        print(f"Found {len(result.entities)} entities")
"""

from .base import ParserProtocol, ParseResult
from .registry import ParserRegistry, parser_registry

__all__ = [
    "ParserProtocol",
    "ParseResult", 
    "ParserRegistry",
    "parser_registry"
]

# Version info for Tree-sitter integration
__version__ = "1.0.0"
__tree_sitter_version__ = ">=0.21.3"