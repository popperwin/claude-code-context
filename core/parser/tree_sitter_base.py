"""
Base Tree-sitter functionality for all language parsers.

Provides common Tree-sitter operations, language loading, query management,
and AST traversal utilities that can be shared across all language parsers.
"""

import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Iterator
from abc import ABC

try:
    import tree_sitter
except ImportError:
    tree_sitter = None

from .base import BaseParser, ParseResult, ParseError, TreeSitterError
from ..models.entities import Entity, ASTNode, SourceLocation

logger = logging.getLogger(__name__)


class TreeSitterBase(BaseParser, ABC):
    """
    Base class for Tree-sitter parsers with common functionality.
    
    Provides language loading, query compilation, AST traversal,
    and error recovery for all Tree-sitter-based parsers.
    """
    
    # Language module mapping - will be expanded as parsers are implemented
    LANGUAGE_MODULES = {
        "python": "tree_sitter_python",
        "javascript": "tree_sitter_javascript",
        "typescript": "tree_sitter_typescript", 
        "html": "tree_sitter_html",
        "css": "tree_sitter_css",
        "go": "tree_sitter_go",
        "rust": "tree_sitter_rust",
        "java": "tree_sitter_java",
        "c": "tree_sitter_c",
        "cpp": "tree_sitter_cpp",
        "json": "tree_sitter_json",
        "yaml": "tree_sitter_yaml"
    }
    
    def __init__(self, language: str):
        super().__init__(language)
        
        if tree_sitter is None:
            raise TreeSitterError("tree-sitter package not installed")
        
        self.parser = tree_sitter.Parser()
        self._query_cache: Dict[str, tree_sitter.Query] = {}
        self._language_loaded = False
        
        # Initialize Tree-sitter language
        try:
            self._setup_language()
            self._language_loaded = True
        except Exception as e:
            logger.error(f"Failed to setup {language} parser: {e}")
            raise TreeSitterError(f"Cannot initialize {language} parser: {e}")
    
    def _setup_language(self) -> None:
        """Initialize Tree-sitter language for this parser"""
        if self.language not in self.LANGUAGE_MODULES:
            raise TreeSitterError(f"Unsupported language: {self.language}")
        
        module_name = self.LANGUAGE_MODULES[self.language]
        
        try:
            # Dynamic import of language module
            language_module = __import__(module_name)
            
            # Get language object - handle different module APIs
            if self.language == "javascript":
                # JavaScript module uses 'language' function
                self.tree_sitter_language = tree_sitter.Language(language_module.language())
            elif self.language == "typescript":
                # TypeScript module - prefer TSX for better JSX support
                if hasattr(language_module, 'language_tsx'):
                    # Use TSX language for better JSX/TSX support
                    self.tree_sitter_language = tree_sitter.Language(language_module.language_tsx())
                elif hasattr(language_module, 'language_typescript'):
                    self.tree_sitter_language = tree_sitter.Language(language_module.language_typescript())
                else:
                    raise TreeSitterError(f"No suitable language function found in {module_name}")
            elif hasattr(language_module, 'language'):
                # Modern API: module.language() returns PyCapsule that needs wrapping
                self.tree_sitter_language = tree_sitter.Language(language_module.language())
            elif self.language == "cpp":
                # C++ uses different function name
                self.tree_sitter_language = tree_sitter.Language(language_module.CPP())
            else:
                # Legacy API: try uppercase then lowercase function names
                lang_func = getattr(language_module, self.language.upper(), None)
                if lang_func is None:
                    lang_func = getattr(language_module, self.language.lower())
                
                self.tree_sitter_language = tree_sitter.Language(lang_func())
            
            self.parser.language = self.tree_sitter_language
            logger.debug(f"Successfully loaded {self.language} Tree-sitter language")
            
        except ImportError as e:
            raise TreeSitterError(
                f"Tree-sitter language module '{module_name}' not installed. "
                f"Install with: pip install {module_name}"
            ) from e
        except Exception as e:
            raise TreeSitterError(f"Failed to initialize {self.language} language: {e}") from e
    
    def get_or_compile_query(self, query_name: str) -> Optional[tree_sitter.Query]:
        """
        Get compiled query with caching.
        
        Args:
            query_name: Name of query (usually same as language)
            
        Returns:
            Compiled Tree-sitter query or None if not found
        """
        cache_key = f"{self.language}:{query_name}"
        
        if cache_key not in self._query_cache:
            query_path = Path(__file__).parent / "queries" / f"{self.language}.scm"
            
            if not query_path.exists():
                logger.warning(f"Query file not found: {query_path}")
                return None
            
            try:
                with open(query_path, 'r', encoding='utf-8') as f:
                    query_text = f.read()
                
                # Compile query
                compiled_query = self.tree_sitter_language.query(query_text)
                self._query_cache[cache_key] = compiled_query
                logger.debug(f"Compiled query for {self.language}: {query_name}")
                
            except Exception as e:
                logger.error(f"Failed to compile query {query_name} for {self.language}: {e}")
                return None
        
        return self._query_cache[cache_key]
    
    def parse_file(self, file_path: Path) -> ParseResult:
        """
        Parse file using Tree-sitter with comprehensive error handling.
        
        Args:
            file_path: Path to file to parse
            
        Returns:
            ParseResult with extracted entities and relations
        """
        self._start_timing()
        
        # Validate file
        is_valid, error = self.validate_file(file_path)
        if not is_valid:
            return self._create_error_result(file_path, error or "Validation failed")
        
        try:
            # Read file safely
            content, file_hash, file_size = self._read_file_safe(file_path)
            
            # Parse with Tree-sitter
            tree = self.parser.parse(content.encode('utf-8'))
            
            if tree is None:
                return self._create_error_result(
                    file_path, "Tree-sitter parsing failed", content, file_hash, file_size
                )
            
            # Check for syntax errors
            syntax_errors = self._extract_syntax_errors(tree, content)
            has_errors = len(syntax_errors) > 0
            
            # Extract entities
            entities = []
            try:
                entities = self.extract_entities(tree, content, file_path)
            except Exception as e:
                logger.warning(f"Entity extraction failed for {file_path}: {e}")
            
            # Extract AST nodes (optional, for detailed analysis)
            ast_nodes = []
            try:
                ast_nodes = self._extract_ast_nodes(tree, content, file_path)
            except Exception as e:
                logger.warning(f"AST node extraction failed for {file_path}: {e}")
            
            # Extract relations
            relations = []
            try:
                relations = self.extract_relations(tree, content, entities, file_path)
            except Exception as e:
                logger.warning(f"Relation extraction failed for {file_path}: {e}")
            
            # Create result
            result = ParseResult(
                file_path=file_path,
                language=self.language,
                entities=entities,
                ast_nodes=ast_nodes,
                relations=relations,
                parse_time=self._get_elapsed_time(),
                file_size=file_size,
                file_hash=file_hash,
                tree_sitter_version=getattr(tree_sitter, '__version__', 'unknown'),
                parser_version=getattr(self, "__version__", "1.0.0"),
                syntax_errors=syntax_errors,
                partial_parse=has_errors,
                error_recovery_applied=has_errors
            )
            
            logger.debug(
                f"Parsed {file_path}: {len(entities)} entities, "
                f"{len(relations)} relations in {result.parse_time*1000:.1f}ms"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Parsing failed for {file_path}: {e}")
            return self._create_error_result(file_path, str(e))
    
    def _extract_syntax_errors(
        self, 
        tree: tree_sitter.Tree, 
        content: str
    ) -> List[Dict[str, Any]]:
        """
        Extract syntax errors from Tree-sitter AST with enhanced context.
        
        Args:
            tree: Tree-sitter AST
            content: Source code content
            
        Returns:
            List of syntax error dictionaries with detailed context
        """
        errors = []
        error_count = 0
        max_errors = 50  # Limit errors to prevent memory issues
        
        def find_errors(node: tree_sitter.Node, depth: int = 0) -> None:
            """Recursively find error nodes with context"""
            nonlocal error_count
            
            if error_count >= max_errors:
                return
            
            if node.type == "ERROR" or node.is_missing:
                try:
                    # Extract error text safely
                    error_text = self._safe_extract_text(node, content)
                    
                    # Get surrounding context for better error reporting
                    line_start = max(0, node.start_point[0] - 2)
                    line_end = min(len(content.split('\n')), node.start_point[0] + 3)
                    lines = content.split('\n')[line_start:line_end]
                    context = '\n'.join(lines)
                    
                    # Determine error severity
                    severity = self._classify_error_severity(node, error_text)
                    
                    errors.append({
                        "type": "SYNTAX_ERROR" if node.type == "ERROR" else "MISSING_NODE",
                        "severity": severity,
                        "message": self._generate_error_message(node, error_text),
                        "line": node.start_point[0] + 1,
                        "column": node.start_point[1],
                        "start_byte": node.start_byte,
                        "end_byte": node.end_byte,
                        "text": error_text,
                        "context": context[:200] + "..." if len(context) > 200 else context,
                        "depth": depth,
                        "parent_type": node.parent.type if node.parent else None,
                        "expected_type": self._infer_expected_type(node)
                    })
                    error_count += 1
                    
                except Exception as e:
                    # Fallback error entry if context extraction fails
                    logger.warning(f"Error context extraction failed: {e}")
                    errors.append({
                        "type": "EXTRACTION_ERROR",
                        "severity": "low",
                        "message": f"Failed to extract error context: {e}",
                        "line": node.start_point[0] + 1 if hasattr(node, 'start_point') else 0,
                        "column": node.start_point[1] if hasattr(node, 'start_point') else 0
                    })
            
            # Continue searching children with depth limit
            if depth < 20:  # Prevent infinite recursion
                for child in node.children:
                    find_errors(child, depth + 1)
        
        if tree.root_node and tree.root_node.has_error:
            find_errors(tree.root_node)
        
        return errors
    
    def _extract_ast_nodes(
        self,
        tree: tree_sitter.Tree,
        content: str,
        file_path: Path,
        max_nodes: int = 1000
    ) -> List[ASTNode]:
        """
        Extract AST nodes for detailed analysis with memory optimization.
        
        Args:
            tree: Tree-sitter AST
            content: Source code content  
            file_path: Path to source file
            max_nodes: Maximum nodes to extract (performance limit)
            
        Returns:
            List of AST nodes
        """
        ast_nodes = []
        node_count = 0
        max_depth = 15  # Prevent excessive recursion
        
        def extract_node(
            node: tree_sitter.Node, 
            parent_id: Optional[str] = None,
            depth: int = 0
        ) -> None:
            nonlocal node_count
            
            # Multiple termination conditions for safety
            if node_count >= max_nodes or depth >= max_depth:
                return
            
            try:
                # Generate safe node ID
                node_id = f"ast_{node.start_byte}_{node.end_byte}_{node_count}"
                
                # Safe text extraction with fallback
                node_text = self._safe_extract_text(node, content, max_length=200)
                
                # Validate node bounds
                if not self._validate_node_bounds(node, content):
                    logger.warning(f"Invalid node bounds: {node.start_byte}-{node.end_byte}")
                    return
                
                location = SourceLocation(
                    file_path=file_path,
                    start_line=max(1, node.start_point[0] + 1),
                    end_line=max(1, node.end_point[0] + 1),
                    start_column=max(0, node.start_point[1]),
                    end_column=max(0, node.end_point[1]),
                    start_byte=node.start_byte,
                    end_byte=node.end_byte
                )
                
                ast_node = ASTNode(
                    node_id=node_id,
                    node_type=node.type or "unknown",
                    language=self.language,
                    location=location,
                    text=node_text,
                    parent_id=parent_id,
                    children_ids=[],  # Will be filled in later
                    is_named=getattr(node, 'is_named', False),
                    is_error=node.type == "ERROR" or getattr(node, 'is_missing', False)
                )
                
                ast_nodes.append(ast_node)
                node_count += 1
                
                # Extract children with depth tracking
                if len(node.children) > 0 and node_count < max_nodes:
                    # Limit children per node to prevent memory explosion
                    children_to_process = node.children[:20]  # Max 20 children per node
                    for child in children_to_process:
                        extract_node(child, node_id, depth + 1)
                        
            except Exception as e:
                logger.warning(f"Failed to extract AST node at {node.start_byte}: {e}")
                # Continue processing other nodes instead of failing completely
        
        # Start extraction from root with error handling
        try:
            if tree.root_node:
                extract_node(tree.root_node)
        except Exception as e:
            logger.error(f"AST extraction failed: {e}")
            return []  # Return empty list instead of crashing
        
        # Fill in children_ids with error handling
        try:
            node_map = {node.node_id: node for node in ast_nodes}
            for node in ast_nodes:
                if node.parent_id and node.parent_id in node_map:
                    parent = node_map[node.parent_id]
                    parent.children_ids.append(node.node_id)
        except Exception as e:
            logger.warning(f"Failed to build parent-child relationships: {e}")
        
        logger.debug(f"Extracted {len(ast_nodes)} AST nodes (limit: {max_nodes})")
        return ast_nodes
    
    def walk_tree(self, tree: tree_sitter.Tree) -> Iterator[tree_sitter.Node]:
        """
        Walk Tree-sitter AST depth-first.
        
        Args:
            tree: Tree-sitter AST
            
        Yields:
            AST nodes in depth-first order
        """
        def walk_node(node: tree_sitter.Node) -> Iterator[tree_sitter.Node]:
            yield node
            for child in node.children:
                yield from walk_node(child)
        
        if tree.root_node:
            yield from walk_node(tree.root_node)
    
    def find_nodes_by_type(
        self, 
        tree: tree_sitter.Tree, 
        node_types: List[str]
    ) -> List[tree_sitter.Node]:
        """
        Find all nodes of specified types.
        
        Args:
            tree: Tree-sitter AST
            node_types: List of node type names to find
            
        Returns:
            List of matching nodes
        """
        matching_nodes = []
        
        for node in self.walk_tree(tree):
            if node.type in node_types:
                matching_nodes.append(node)
        
        return matching_nodes
    
    def get_node_text(self, node: tree_sitter.Node, content: str) -> str:
        """
        Get text content of a Tree-sitter node.
        
        Args:
            node: Tree-sitter node
            content: Source code content
            
        Returns:
            Text content of the node
        """
        # Tree-sitter works with byte offsets, but we need to convert to character offsets
        # for proper Unicode handling
        try:
            content_bytes = content.encode('utf-8')
            node_bytes = content_bytes[node.start_byte:node.end_byte]
            return node_bytes.decode('utf-8')
        except (UnicodeDecodeError, UnicodeEncodeError):
            # Fallback to simple string slicing if encoding fails
            logger.warning(f"Unicode handling failed for node at {node.start_byte}-{node.end_byte}")
            return content[node.start_byte:node.end_byte]
    
    def find_child_by_type(
        self, 
        node: tree_sitter.Node, 
        child_type: str
    ) -> Optional[tree_sitter.Node]:
        """
        Find first child node of specified type.
        
        Args:
            node: Parent node
            child_type: Type of child to find
            
        Returns:
            First matching child node or None
        """
        for child in node.children:
            if child.type == child_type:
                return child
        return None
    
    def find_children_by_type(
        self, 
        node: tree_sitter.Node, 
        child_type: str
    ) -> List[tree_sitter.Node]:
        """
        Find all child nodes of specified type.
        
        Args:
            node: Parent node
            child_type: Type of children to find
            
        Returns:
            List of matching child nodes
        """
        return [child for child in node.children if child.type == child_type]
    
    def _safe_extract_text(
        self, 
        node: tree_sitter.Node, 
        content: str, 
        max_length: int = 50
    ) -> str:
        """
        Safely extract text from a Tree-sitter node with bounds checking.
        
        Args:
            node: Tree-sitter node
            content: Source code content
            max_length: Maximum text length to extract
            
        Returns:
            Safely extracted text
        """
        try:
            # Validate bounds
            if not self._validate_node_bounds(node, content):
                return "<invalid_bounds>"
            
            # Extract text with Unicode handling
            node_text = self.get_node_text(node, content)
            
            # Truncate if too long
            if len(node_text) > max_length:
                node_text = node_text[:max_length] + "..."
            
            # Replace problematic characters
            node_text = node_text.replace('\x00', '<null>').replace('\r\n', '\\n').replace('\n', '\\n')
            
            return node_text or "<empty>"
            
        except Exception as e:
            logger.warning(f"Text extraction failed for node at {getattr(node, 'start_byte', '?')}: {e}")
            return f"<extraction_error: {e}>"
    
    def _validate_node_bounds(self, node: tree_sitter.Node, content: str) -> bool:
        """
        Validate that node bounds are within content limits.
        
        Args:
            node: Tree-sitter node
            content: Source code content
            
        Returns:
            True if bounds are valid
        """
        try:
            content_bytes = content.encode('utf-8')
            return (
                hasattr(node, 'start_byte') and
                hasattr(node, 'end_byte') and
                node.start_byte >= 0 and
                node.end_byte >= node.start_byte and
                node.end_byte <= len(content_bytes)
            )
        except Exception:
            return False
    
    def _classify_error_severity(self, node: tree_sitter.Node, error_text: str) -> str:
        """
        Classify syntax error severity based on context.
        
        Args:
            node: Error node
            error_text: Error text content
            
        Returns:
            Severity level: 'critical', 'high', 'medium', 'low'
        """
        # Critical errors that prevent parsing
        if node.type == "ERROR" and not error_text.strip():
            return "critical"
        
        # High severity for structural errors
        if node.parent and node.parent.type in ["program", "source_file", "module"]:
            return "high"
        
        # Medium severity for statement-level errors
        if error_text and any(keyword in error_text.lower() for keyword in 
                            ["function", "class", "def", "struct", "interface"]):
            return "medium"
        
        # Low severity for minor syntax issues
        return "low"
    
    def _generate_error_message(self, node: tree_sitter.Node, error_text: str) -> str:
        """
        Generate descriptive error message based on node context.
        
        Args:
            node: Error node
            error_text: Error text content
            
        Returns:
            Descriptive error message
        """
        if node.type == "ERROR":
            if not error_text.strip():
                return "Unexpected empty syntax error"
            return f"Syntax error in {self.language}: '{error_text}'"
        
        if node.is_missing:
            parent_type = node.parent.type if node.parent else "unknown"
            return f"Missing {node.type} in {parent_type} context"
        
        return f"Parse error: {error_text}"
    
    def _infer_expected_type(self, node: tree_sitter.Node) -> Optional[str]:
        """
        Infer expected node type based on context.
        
        Args:
            node: Error node
            
        Returns:
            Expected node type or None
        """
        if not node.parent:
            return None
        
        parent_type = node.parent.type
        
        # Language-agnostic patterns
        expected_patterns = {
            "function_definition": "identifier",
            "class_definition": "identifier", 
            "method_definition": "identifier",
            "parameter_list": "parameter",
            "argument_list": "argument",
            "block": "statement",
            "expression_statement": "expression"
        }
        
        return expected_patterns.get(parent_type)
    
    def _safe_encode_content(self, content: str) -> bytes:
        """
        Safely encode content for Tree-sitter parsing with fallback handling.
        
        Args:
            content: Source code content
            
        Returns:
            Encoded content bytes
        """
        try:
            return content.encode('utf-8')
        except UnicodeEncodeError as e:
            logger.warning(f"UTF-8 encoding failed: {e}, using fallback")
            # Replace problematic characters and retry
            content_clean = content.encode('utf-8', errors='replace').decode('utf-8')
            return content_clean.encode('utf-8')