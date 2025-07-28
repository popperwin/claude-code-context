"""
Error recovery strategies for robust parsing.

Provides multiple strategies to handle parsing errors gracefully,
including encoding detection, syntax error recovery, and memory management.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import chardet

from .base import ParseResult, ParseError

logger = logging.getLogger(__name__)


class ParseErrorRecovery:
    """
    Comprehensive error recovery strategies for parsing operations.
    
    Handles various failure modes including encoding issues, syntax errors,
    memory constraints, and file corruption.
    """
    
    @staticmethod
    def detect_encoding(file_path: Path) -> str:
        """
        Detect file encoding using multiple strategies.
        
        Args:
            file_path: Path to file
            
        Returns:
            Detected encoding name
        """
        # Try to read small sample for detection
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(min(10000, file_path.stat().st_size))
            
            # Use chardet for detection
            detection = chardet.detect(raw_data)
            if detection and detection['confidence'] > 0.7:
                return detection['encoding']
        except Exception as e:
            logger.warning(f"Encoding detection failed for {file_path}: {e}")
        
        # Fallback encoding order
        return 'utf-8'
    
    @staticmethod
    def read_with_encoding_fallback(file_path: Path) -> Tuple[str, str]:
        """
        Read file with multiple encoding attempts.
        
        Args:
            file_path: Path to file
            
        Returns:
            (content, actual_encoding_used)
        """
        # Detect encoding first
        detected_encoding = ParseErrorRecovery.detect_encoding(file_path)
        
        # Try encodings in order of preference
        encodings = [
            detected_encoding,
            'utf-8',
            'utf-8-sig',  # UTF-8 with BOM
            'latin1',     # Very permissive
            'cp1252',     # Windows
            'iso-8859-1'  # Fallback
        ]
        
        # Remove duplicates while preserving order
        encodings = list(dict.fromkeys(encodings))
        
        for encoding in encodings:
            try:
                content = file_path.read_text(encoding=encoding)
                logger.debug(f"Successfully read {file_path} with encoding: {encoding}")
                return content, encoding
            except UnicodeDecodeError as e:
                logger.debug(f"Failed to read {file_path} with {encoding}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Unexpected error reading {file_path} with {encoding}: {e}")
                continue
        
        # Final fallback: binary read with error replacement
        logger.warning(f"Using binary fallback for {file_path}")
        raw_content = file_path.read_bytes()
        content = raw_content.decode('utf-8', errors='replace')
        return content, 'utf-8-binary-fallback'
    
    @staticmethod
    def handle_large_file(
        file_path: Path, 
        max_size: int = 10 * 1024 * 1024
    ) -> Tuple[str, bool]:
        """
        Handle large files by chunking or truncation.
        
        Args:
            file_path: Path to large file
            max_size: Maximum size to process in bytes
            
        Returns:
            (content, was_truncated)
        """
        file_size = file_path.stat().st_size
        
        if file_size <= max_size:
            content, _ = ParseErrorRecovery.read_with_encoding_fallback(file_path)
            return content, False
        
        logger.warning(f"File {file_path} is large ({file_size} bytes), truncating to {max_size}")
        
        # Read only the first portion
        with open(file_path, 'rb') as f:
            raw_data = f.read(max_size)
        
        # Try to decode truncated content
        try:
            content = raw_data.decode('utf-8', errors='replace')
        except Exception:
            # If all else fails, use latin1 which accepts any byte
            content = raw_data.decode('latin1')
        
        # Add truncation marker
        content += "\n\n# ... [FILE TRUNCATED] ..."
        
        return content, True
    
    @staticmethod
    def recover_from_syntax_errors(
        syntax_errors: List[Dict[str, Any]],
        content: str,
        file_path: Path
    ) -> Tuple[str, List[str]]:
        """
        Attempt to recover from syntax errors by cleaning content.
        
        Args:
            syntax_errors: List of syntax errors
            content: Original content
            file_path: Path to file
            
        Returns:
            (cleaned_content, recovery_actions_taken)
        """
        recovery_actions = []
        cleaned_content = content
        
        # Strategy 1: Remove obviously problematic lines
        if len(syntax_errors) > 0:
            lines = cleaned_content.split('\n')
            error_lines = set()
            
            for error in syntax_errors:
                if 'line' in error and error['line'] > 0:
                    error_lines.add(error['line'] - 1)  # Convert to 0-based
            
            if error_lines and len(error_lines) < len(lines) * 0.1:  # Don't remove too many lines
                original_line_count = len(lines)
                lines = [line for i, line in enumerate(lines) if i not in error_lines]
                cleaned_content = '\n'.join(lines)
                recovery_actions.append(f"Removed {original_line_count - len(lines)} problematic lines")
        
        # Strategy 2: Fix common encoding issues
        if '�' in cleaned_content:  # Unicode replacement character
            original_length = len(cleaned_content)
            cleaned_content = cleaned_content.replace('�', ' ')
            if len(cleaned_content) != original_length:
                recovery_actions.append("Replaced unicode replacement characters")
        
        # Strategy 3: Remove null bytes
        if '\x00' in cleaned_content:
            cleaned_content = cleaned_content.replace('\x00', '')
            recovery_actions.append("Removed null bytes")
        
        # Strategy 4: Fix line ending issues
        if '\r\n' in cleaned_content or '\r' in cleaned_content:
            original_content = cleaned_content
            cleaned_content = cleaned_content.replace('\r\n', '\n').replace('\r', '\n')
            if cleaned_content != original_content:
                recovery_actions.append("Normalized line endings")
        
        return cleaned_content, recovery_actions
    
    @staticmethod
    def create_minimal_result(
        file_path: Path,
        language: str,
        error_message: str,
        recovery_info: Optional[Dict[str, Any]] = None
    ) -> ParseResult:
        """
        Create minimal ParseResult for severely corrupted files.
        
        Args:
            file_path: Path to file
            language: Language name
            error_message: Description of the error
            recovery_info: Additional recovery information
            
        Returns:
            Minimal ParseResult with error information
        """
        result = ParseResult(
            file_path=file_path,
            language=language,
            entities=[],
            ast_nodes=[],
            relations=[],
            parse_time=0.0,
            file_size=file_path.stat().st_size if file_path.exists() else 0,
            file_hash="",
            partial_parse=True,
            error_recovery_applied=True
        )
        
        result.add_syntax_error({
            "type": "RECOVERY_ERROR",
            "message": error_message,
            "line": 0,
            "column": 0,
            "recovery_info": recovery_info or {}
        })
        
        return result
    
    @staticmethod
    def analyze_parse_failure(
        file_path: Path,
        exception: Exception,
        language: str
    ) -> Dict[str, Any]:
        """
        Analyze parse failure to determine recovery strategy.
        
        Args:
            file_path: Path to failed file
            exception: Exception that occurred
            language: Language being parsed
            
        Returns:
            Analysis results with suggested recovery actions
        """
        analysis = {
            "file_path": str(file_path),
            "error_type": type(exception).__name__,
            "error_message": str(exception),
            "language": language,
            "recovery_suggestions": []
        }
        
        # File-level analysis
        if file_path.exists():
            file_size = file_path.stat().st_size
            analysis["file_size"] = file_size
            
            if file_size == 0:
                analysis["recovery_suggestions"].append("File is empty")
            elif file_size > 50 * 1024 * 1024:  # 50MB
                analysis["recovery_suggestions"].append("File is very large, consider chunking")
        else:
            analysis["recovery_suggestions"].append("File does not exist")
        
        # Exception-specific analysis
        if isinstance(exception, UnicodeDecodeError):
            analysis["recovery_suggestions"].extend([
                "Try alternative encodings",
                "Use binary fallback with error replacement"
            ])
        elif isinstance(exception, MemoryError):
            analysis["recovery_suggestions"].extend([
                "File too large for memory",
                "Use chunked processing",
                "Increase available memory"
            ])
        elif isinstance(exception, PermissionError):
            analysis["recovery_suggestions"].append("Check file permissions")
        elif "timeout" in str(exception).lower():
            analysis["recovery_suggestions"].extend([
                "Parsing timeout occurred",
                "Simplify file or increase timeout",
                "Check for infinite loops in grammar"
            ])
        
        return analysis


class RobustParsingContext:
    """
    Context manager for robust parsing with automatic error recovery.
    
    Provides a clean interface for parsing with comprehensive error handling
    and recovery strategies applied automatically.
    """
    
    def __init__(
        self,
        file_path: Path,
        language: str,
        max_file_size: int = 10 * 1024 * 1024,
        enable_recovery: bool = True
    ):
        self.file_path = file_path
        self.language = language
        self.max_file_size = max_file_size
        self.enable_recovery = enable_recovery
        self.recovery_actions: List[str] = []
        self.content: Optional[str] = None
        self.encoding_used: Optional[str] = None
    
    def __enter__(self) -> 'RobustParsingContext':
        """Initialize parsing context with error recovery"""
        try:
            # Step 1: Handle large files
            if self.file_path.stat().st_size > self.max_file_size:
                self.content, was_truncated = ParseErrorRecovery.handle_large_file(
                    self.file_path, self.max_file_size
                )
                if was_truncated:
                    self.recovery_actions.append("File was truncated due to size")
                self.encoding_used = "truncated"
            else:
                # Step 2: Read with encoding fallback
                self.content, self.encoding_used = ParseErrorRecovery.read_with_encoding_fallback(
                    self.file_path
                )
            
            return self
            
        except Exception as e:
            logger.error(f"Failed to initialize parsing context for {self.file_path}: {e}")
            raise ParseError(f"Cannot prepare file for parsing: {e}") from e
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up parsing context"""
        if exc_type is not None:
            logger.debug(f"Parsing context exited with exception: {exc_type.__name__}")
        
        # Log recovery actions taken
        if self.recovery_actions:
            logger.info(f"Recovery actions for {self.file_path}: {self.recovery_actions}")
    
    def get_content(self) -> str:
        """Get file content with recovery applied"""
        if self.content is None:
            raise ParseError("Content not available - context not properly initialized")
        return self.content
    
    def apply_syntax_error_recovery(self, syntax_errors: List[Dict[str, Any]]) -> str:
        """Apply syntax error recovery to content"""
        if not self.enable_recovery or not syntax_errors:
            return self.get_content()
        
        recovered_content, actions = ParseErrorRecovery.recover_from_syntax_errors(
            syntax_errors, self.get_content(), self.file_path
        )
        
        self.recovery_actions.extend(actions)
        return recovered_content
    
    def get_recovery_info(self) -> Dict[str, Any]:
        """Get information about recovery actions taken"""
        return {
            "encoding_used": self.encoding_used,
            "recovery_actions": self.recovery_actions,
            "recovery_applied": len(self.recovery_actions) > 0
        }