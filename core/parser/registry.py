"""
Parser registry for dynamic parser discovery and instantiation.

Provides a centralized registry for all language parsers with automatic
discovery by file extension and language detection capabilities.
"""

import logging
from pathlib import Path
from typing import Dict, Type, Optional, List, Callable, Any
from concurrent.futures import ThreadPoolExecutor
import threading

from .base import ParserProtocol, ParseResult

logger = logging.getLogger(__name__)


class ParserRegistry:
    """
    Registry for language parsers with dynamic discovery and instantiation.
    
    Manages all available parsers, provides file-to-parser mapping,
    and supports plugin-style parser registration.
    """
    
    def __init__(self):
        self._parsers: Dict[str, Type[ParserProtocol]] = {}
        self._extension_map: Dict[str, str] = {}
        self._parser_instances: Dict[str, ParserProtocol] = {}
        self._lock = threading.RLock()
        
        # Register default parsers (will be implemented in subsequent phases)
        self._register_default_parsers()
    
    def _register_default_parsers(self) -> None:
        """Register default parsers for supported languages"""
        # Will be populated as parsers are implemented in subsequent phases
        # For now, just prepare the structure
        logger.debug("Parser registry initialized - parsers will be registered as they are implemented")
    
    def register(
        self,
        language: str,
        parser_class: Type[ParserProtocol],
        extensions: List[str],
        override: bool = False
    ) -> None:
        """
        Register a parser for a language.
        
        Args:
            language: Language name (e.g., 'python', 'javascript')
            parser_class: Parser class that implements ParserProtocol
            extensions: List of file extensions (e.g., ['.py', '.pyi'])
            override: Whether to override existing registration
        """
        with self._lock:
            if language in self._parsers and not override:
                raise ValueError(f"Parser for {language} already registered. Use override=True to replace.")
            
            # Register parser class
            self._parsers[language] = parser_class
            
            # Register extensions
            for ext in extensions:
                ext_lower = ext.lower()
                if ext_lower in self._extension_map and not override:
                    existing_lang = self._extension_map[ext_lower]
                    logger.warning(f"Extension {ext} already mapped to {existing_lang}, overriding with {language}")
                
                self._extension_map[ext_lower] = language
            
            # Clear any cached instance
            if language in self._parser_instances:
                del self._parser_instances[language]
            
            logger.info(f"Registered {language} parser with extensions: {extensions}")
    
    def register_factory(
        self,
        language: str,
        parser_factory: Callable[[], ParserProtocol],
        extensions: List[str],
        override: bool = False
    ) -> None:
        """
        Register a parser factory function.
        
        Args:
            language: Language name
            parser_factory: Factory function that returns parser instance
            extensions: List of file extensions
            override: Whether to override existing registration
        """
        # Wrap factory in a class-like interface
        class FactoryWrapper:
            def __init__(self):
                pass
            
            def __call__(self):
                return parser_factory()
        
        self.register(language, FactoryWrapper, extensions, override)
    
    def unregister(self, language: str) -> bool:
        """
        Unregister a parser.
        
        Args:
            language: Language to unregister
            
        Returns:
            True if parser was unregistered, False if not found
        """
        with self._lock:
            if language not in self._parsers:
                return False
            
            # Remove parser
            del self._parsers[language]
            
            # Remove extensions
            extensions_to_remove = [
                ext for ext, lang in self._extension_map.items()
                if lang == language
            ]
            for ext in extensions_to_remove:
                del self._extension_map[ext]
            
            # Remove cached instance
            if language in self._parser_instances:
                del self._parser_instances[language]
            
            logger.info(f"Unregistered {language} parser")
            return True
    
    def get_parser_for_file(self, file_path: Path) -> Optional[ParserProtocol]:
        """
        Get appropriate parser for a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Parser instance or None if no parser available
        """
        if not file_path or not file_path.suffix:
            return None
        
        extension = file_path.suffix.lower()
        
        with self._lock:
            if extension not in self._extension_map:
                return None
            
            language = self._extension_map[extension]
            return self.get_parser(language)
    
    def get_parser(self, language: str) -> Optional[ParserProtocol]:
        """
        Get parser instance for a language.
        
        Args:
            language: Language name
            
        Returns:
            Parser instance or None if not available
        """
        with self._lock:
            if language not in self._parsers:
                return None
            
            # Return cached instance if available
            if language in self._parser_instances:
                return self._parser_instances[language]
            
            # Create new instance
            try:
                parser_class = self._parsers[language]
                
                # Handle factory functions or wrapper classes
                if hasattr(parser_class, '__call__') and not isinstance(parser_class, type):
                    # This is a factory wrapper
                    instance = parser_class()
                else:
                    # This is a regular class
                    instance = parser_class()
                
                # Cache instance for reuse
                self._parser_instances[language] = instance
                logger.debug(f"Created parser instance for {language}")
                
                return instance
                
            except Exception as e:
                logger.error(f"Failed to create parser for {language}: {e}")
                return None
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        with self._lock:
            return list(self._parsers.keys())
    
    def get_supported_extensions(self) -> List[str]:
        """Get all supported file extensions"""
        with self._lock:
            return list(self._extension_map.keys())
    
    def get_extension_mapping(self) -> Dict[str, str]:
        """Get mapping of extensions to languages"""
        with self._lock:
            return self._extension_map.copy()
    
    def can_parse(self, file_path: Path) -> bool:
        """
        Check if any parser can handle the file.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if a parser is available
        """
        return self.get_parser_for_file(file_path) is not None
    
    def get_parser_info(self, language: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a parser.
        
        Args:
            language: Language name
            
        Returns:
            Parser information or None if not found
        """
        parser = self.get_parser(language)
        if parser is None:
            return None
        
        try:
            return parser.get_parser_info()
        except Exception as e:
            logger.warning(f"Failed to get info for {language} parser: {e}")
            return {
                "language": language,
                "extensions": [ext for ext, lang in self._extension_map.items() if lang == language],
                "error": str(e)
            }
    
    def discover_files(
        self,
        directory: Path,
        recursive: bool = True,
        follow_symlinks: bool = False
    ) -> List[Path]:
        """
        Discover parseable files in a directory.
        
        Args:
            directory: Directory to search
            recursive: Whether to search recursively
            follow_symlinks: Whether to follow symbolic links
            
        Returns:
            List of files that can be parsed
        """
        if not directory.exists() or not directory.is_dir():
            return []
        
        supported_extensions = set(self.get_supported_extensions())
        files = []
        
        try:
            if recursive:
                pattern = "**/*"
            else:
                pattern = "*"
            
            for path in directory.glob(pattern):
                if not path.is_file():
                    continue
                
                if not follow_symlinks and path.is_symlink():
                    continue
                
                if path.suffix.lower() in supported_extensions:
                    files.append(path)
        
        except Exception as e:
            logger.error(f"Error discovering files in {directory}: {e}")
        
        return sorted(files)
    
    def parse_files_parallel(
        self,
        file_paths: List[Path],
        max_workers: Optional[int] = None,
        progress_callback: Optional[Callable[[Path, bool], None]] = None
    ) -> List[ParseResult]:
        """
        Parse multiple files in parallel.
        
        Args:
            file_paths: List of files to parse
            max_workers: Maximum number of worker threads
            progress_callback: Callback function called for each file
            
        Returns:
            List of parse results
        """
        if not file_paths:
            return []
        
        results = []
        
        def parse_single_file(file_path: Path) -> Optional[ParseResult]:
            """Parse a single file with error handling"""
            try:
                parser = self.get_parser_for_file(file_path)
                if parser is None:
                    logger.warning(f"No parser available for {file_path}")
                    if progress_callback:
                        progress_callback(file_path, False)
                    return None
                
                result = parser.parse_file(file_path)
                if progress_callback:
                    progress_callback(file_path, result.success)
                
                return result
                
            except Exception as e:
                logger.error(f"Failed to parse {file_path}: {e}")
                if progress_callback:
                    progress_callback(file_path, False)
                return None
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(parse_single_file, file_path): file_path
                for file_path in file_paths
            }
            
            for future in future_to_file:
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    file_path = future_to_file[future]
                    logger.error(f"Parsing task failed for {file_path}: {e}")
        
        return results
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        with self._lock:
            return {
                "total_parsers": len(self._parsers),
                "total_extensions": len(self._extension_map),
                "cached_instances": len(self._parser_instances),
                "languages": list(self._parsers.keys()),
                "extensions": list(self._extension_map.keys())
            }
    
    def clear_cache(self) -> None:
        """Clear cached parser instances"""
        with self._lock:
            self._parser_instances.clear()
            logger.info("Cleared parser instance cache")


# Global registry instance
parser_registry = ParserRegistry()


def register_parser(
    language: str,
    extensions: List[str],
    override: bool = False
):
    """
    Decorator for registering parser classes.
    
    Args:
        language: Language name
        extensions: Supported file extensions
        override: Whether to override existing registration
    
    Example:
        @register_parser("python", [".py", ".pyi"])
        class PythonParser(ParserProtocol):
            pass
    """
    def decorator(parser_class: Type[ParserProtocol]):
        parser_registry.register(language, parser_class, extensions, override)
        return parser_class
    
    return decorator