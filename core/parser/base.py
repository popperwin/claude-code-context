"""
Abstract base classes and protocols for Tree-sitter parsers.

Defines the standard interface that all language parsers must implement,
along with common data structures for parse results and error handling.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import time

from ..models.entities import Entity, ASTNode, Relation
from ..models.storage import StorageResult


@dataclass
class ParseResult:
    """
    Result of parsing operation with comprehensive metadata.
    
    Contains all entities, AST nodes, and relations extracted from a file,
    along with performance metrics and error information.
    """
    # Source information
    file_path: Path
    language: str
    
    # Extracted data
    entities: List[Entity] = field(default_factory=list)
    ast_nodes: List[ASTNode] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    
    # Performance metrics
    parse_time: float = 0.0  # Seconds
    file_size: int = 0  # Bytes
    file_hash: str = ""
    
    # Version information
    tree_sitter_version: str = ""
    parser_version: str = ""
    
    # Error tracking
    syntax_errors: List[Dict[str, Any]] = field(default_factory=list)
    partial_parse: bool = False
    error_recovery_applied: bool = False
    warnings: List[str] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def success(self) -> bool:
        """Check if parsing completed without syntax errors"""
        return len(self.syntax_errors) == 0
    
    @property
    def entity_count(self) -> int:
        """Total number of entities extracted"""
        return len(self.entities)
    
    @property
    def relation_count(self) -> int:
        """Total number of relations extracted"""
        return len(self.relations)
    
    @property
    def performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        entities_per_ms = (
            self.entity_count / (self.parse_time * 1000) 
            if self.parse_time > 0 else 0
        )
        
        return {
            "parse_time_ms": self.parse_time * 1000,
            "file_size_kb": self.file_size / 1024,
            "entities_extracted": self.entity_count,
            "relations_extracted": self.relation_count,
            "entities_per_ms": entities_per_ms,
            "success": self.success,
            "errors": len(self.syntax_errors),
            "warnings": len(self.warnings)
        }
    
    def add_warning(self, message: str) -> None:
        """Add a warning message"""
        self.warnings.append(message)
    
    def add_syntax_error(self, error: Dict[str, Any]) -> None:
        """Add a syntax error"""
        self.syntax_errors.append(error)
        self.partial_parse = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "file_path": str(self.file_path),
            "language": self.language,
            "entity_count": self.entity_count,
            "relation_count": self.relation_count,
            "performance": self.performance_summary,
            "success": self.success,
            "created_at": self.created_at.isoformat()
        }


class ParserProtocol(ABC):
    """
    Abstract protocol for language parsers.
    
    All language-specific parsers must implement this interface to ensure
    consistent behavior and integration with the parsing pipeline.
    """
    
    @abstractmethod
    def get_language_name(self) -> str:
        """Return the language name this parser handles"""
        pass
    
    @abstractmethod 
    def get_supported_extensions(self) -> List[str]:
        """Return list of file extensions this parser supports"""
        pass
    
    @abstractmethod
    def can_parse(self, file_path: Path) -> bool:
        """
        Check if this parser can handle the given file.
        
        Args:
            file_path: Path to file to check
            
        Returns:
            True if parser can handle this file
        """
        pass
    
    @abstractmethod
    def parse_file(self, file_path: Path) -> ParseResult:
        """
        Parse file and extract entities, AST nodes, and relations.
        
        Args:
            file_path: Path to file to parse
            
        Returns:
            ParseResult with extracted data and metadata
            
        Raises:
            FileNotFoundError: If file doesn't exist
            PermissionError: If file cannot be read
            UnicodeDecodeError: If file encoding is unsupported
        """
        pass
    
    @abstractmethod
    def extract_entities(
        self, 
        tree: Any,  # tree_sitter.Tree
        content: str,
        file_path: Path
    ) -> List[Entity]:
        """
        Extract semantic entities from AST.
        
        Args:
            tree: Tree-sitter AST
            content: Source code content
            file_path: Path to source file
            
        Returns:
            List of extracted entities
        """
        pass
    
    @abstractmethod
    def extract_relations(
        self,
        tree: Any,  # tree_sitter.Tree
        content: str,
        entities: List[Entity],
        file_path: Path
    ) -> List[Relation]:
        """
        Extract relationships between entities.
        
        Args:
            tree: Tree-sitter AST
            content: Source code content
            entities: Previously extracted entities
            file_path: Path to source file
            
        Returns:
            List of extracted relations
        """
        pass
    
    def get_parser_info(self) -> Dict[str, Any]:
        """Get parser metadata and capabilities"""
        return {
            "language": self.get_language_name(),
            "extensions": self.get_supported_extensions(),
            "version": getattr(self, "__version__", "1.0.0"),
            "features": getattr(self, "SUPPORTED_FEATURES", [])
        }
    
    def validate_file(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Validate file before parsing.
        
        Args:
            file_path: Path to validate
            
        Returns:
            (is_valid, error_message)
        """
        if not file_path.exists():
            return False, f"File does not exist: {file_path}"
        
        if not file_path.is_file():
            return False, f"Path is not a file: {file_path}"
        
        if not self.can_parse(file_path):
            return False, f"Parser cannot handle file: {file_path}"
        
        # Check file size (default 10MB limit)
        max_size = getattr(self, "MAX_FILE_SIZE", 10 * 1024 * 1024)
        if file_path.stat().st_size > max_size:
            return False, f"File too large: {file_path.stat().st_size} bytes"
        
        return True, None


class BaseParser(ParserProtocol):
    """
    Base implementation with common functionality.
    
    Provides default implementations for validation, error handling,
    and utility methods that most parsers can use.
    """
    
    # Parser configuration
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    PARSE_TIMEOUT = 30.0  # 30 seconds
    
    def __init__(self, language: str):
        self.language = language
        self._parser_start_time = 0.0
    
    def get_language_name(self) -> str:
        """Return language name"""
        return self.language
    
    def _start_timing(self) -> None:
        """Start timing for performance measurement"""
        self._parser_start_time = time.perf_counter()
    
    def _get_elapsed_time(self) -> float:
        """Get elapsed time since timing started"""
        return time.perf_counter() - self._parser_start_time
    
    def _read_file_safe(self, file_path: Path) -> Tuple[str, str, int]:
        """
        Safely read file with encoding detection.
        
        Returns:
            (content, file_hash, file_size)
        """
        import hashlib
        
        # Try different encodings
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                content = file_path.read_text(encoding=encoding)
                file_size = len(content.encode('utf-8'))
                file_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
                return content, file_hash, file_size
            except UnicodeDecodeError:
                continue
        
        # Fallback: read as binary and replace errors
        raw_content = file_path.read_bytes()
        content = raw_content.decode('utf-8', errors='replace')
        file_size = len(raw_content)
        file_hash = hashlib.sha256(raw_content).hexdigest()[:16]
        
        return content, file_hash, file_size
    
    def _create_error_result(
        self,
        file_path: Path,
        error_message: str,
        content: str = "",
        file_hash: str = "",
        file_size: int = 0
    ) -> ParseResult:
        """Create ParseResult for error cases"""
        result = ParseResult(
            file_path=file_path,
            language=self.language,
            parse_time=self._get_elapsed_time(),
            file_size=file_size,
            file_hash=file_hash,
            partial_parse=True,
            error_recovery_applied=True
        )
        
        result.add_syntax_error({
            "type": "PARSER_ERROR",
            "message": error_message,
            "line": 0,
            "column": 0
        })
        
        return result


# Error types for parser exceptions
class ParseError(Exception):
    """Base class for parsing errors"""
    pass


class UnsupportedLanguageError(ParseError):
    """Raised when language is not supported"""
    pass


class TreeSitterError(ParseError):
    """Raised when Tree-sitter parsing fails"""
    pass


class EntityExtractionError(ParseError):
    """Raised when entity extraction fails"""
    pass