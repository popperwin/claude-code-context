"""
Real code sample generator for Sprint 2 integration testing.

Extracts actual entities from the project codebase to create realistic
test data for Entity → Embedding → Search pipeline validation.
"""

import ast
import hashlib
import inspect
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import importlib.util
import sys
from datetime import datetime

from core.models.entities import Entity, EntityType, SourceLocation, Visibility


class RealCodeExtractor:
    """Extracts real code entities from the project for testing"""
    
    def __init__(self, project_root: Path):
        """
        Initialize extractor with project root.
        
        Args:
            project_root: Path to project root directory
        """
        self.project_root = project_root
        self.core_path = project_root / "core"
        self.config_path = project_root / "config"
        self.scripts_path = project_root / "scripts"
        
    def extract_all_entities(self) -> List[Entity]:
        """
        Extract all entities from the project codebase.
        
        Returns:
            List of Entity objects with real code data
        """
        entities = []
        
        # Extract from core modules
        entities.extend(self._extract_from_directory(self.core_path))
        
        # Extract from config modules  
        entities.extend(self._extract_from_directory(self.config_path))
        
        # Extract from scripts
        entities.extend(self._extract_from_directory(self.scripts_path))
        
        return entities
    
    def _extract_from_directory(self, directory: Path) -> List[Entity]:
        """Extract entities from all Python files in directory"""
        entities = []
        
        for py_file in directory.rglob("*.py"):
            if py_file.name.startswith("__"):
                continue
                
            try:
                file_entities = self._extract_from_file(py_file)
                entities.extend(file_entities)
            except Exception as e:
                print(f"Warning: Could not extract from {py_file}: {e}")
                continue
                
        return entities
    
    def _extract_from_file(self, file_path: Path) -> List[Entity]:
        """Extract entities from a single Python file"""
        entities = []
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            # Parse AST
            tree = ast.parse(source_code)
            
            # Extract file entity
            entities.append(self._create_file_entity(file_path, source_code))
            
            # Extract code entities
            for node in ast.walk(tree):
                entity = self._extract_entity_from_ast_node(node, file_path, source_code)
                if entity:
                    entities.append(entity)
                    
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            
        return entities
    
    def _create_file_entity(self, file_path: Path, source_code: str) -> Entity:
        """Create Entity for the file itself"""
        relative_path = file_path.relative_to(self.project_root)
        
        return Entity(
            id=f"file::{relative_path}",
            name=file_path.name,
            qualified_name=str(relative_path),
            entity_type=EntityType.FILE,
            location=SourceLocation(
                file_path=relative_path,
                start_line=1,
                end_line=len(source_code.splitlines()),
                start_column=0,
                end_column=0,
                start_byte=0,
                end_byte=len(source_code.encode('utf-8'))
            ),
            source_code=source_code[:500] + "..." if len(source_code) > 500 else source_code,
            visibility=Visibility.PUBLIC,
            signature=f"# File: {relative_path}",
            docstring=self._extract_module_docstring(source_code),
            source_hash=hashlib.sha256(source_code.encode('utf-8')).hexdigest()[:16],
            metadata={
                "file_size": len(source_code),
                "line_count": len(source_code.splitlines()),
                "language": "python"
            }
        )
    
    def _extract_entity_from_ast_node(
        self, 
        node: ast.AST, 
        file_path: Path, 
        source_code: str
    ) -> Optional[Entity]:
        """Extract Entity from AST node"""
        
        if isinstance(node, ast.FunctionDef):
            return self._create_function_entity(node, file_path, source_code)
        elif isinstance(node, ast.AsyncFunctionDef):
            return self._create_function_entity(node, file_path, source_code, is_async=True)
        elif isinstance(node, ast.ClassDef):
            return self._create_class_entity(node, file_path, source_code)
        elif isinstance(node, ast.Assign):
            return self._create_variable_entity(node, file_path, source_code)
        elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            return self._create_import_entity(node, file_path, source_code)
            
        return None
    
    def _create_function_entity(
        self, 
        node: ast.FunctionDef, 
        file_path: Path, 
        source_code: str,
        is_async: bool = False
    ) -> Entity:
        """Create Entity for function/method"""
        relative_path = file_path.relative_to(self.project_root)
        
        # Determine if this is a method (inside a class)
        entity_type = EntityType.METHOD if self._is_inside_class(node) else EntityType.FUNCTION
        
        # Build qualified name
        module_name = str(relative_path).replace('/', '.').replace('.py', '')
        qualified_name = f"{module_name}::{node.name}"
        
        # Extract function source
        func_source = self._get_node_source(node, source_code)
        
        # Build signature
        signature = self._build_function_signature(node, is_async)
        
        # Extract docstring
        docstring = ast.get_docstring(node) or ""
        
        # Determine visibility
        visibility = Visibility.PRIVATE if node.name.startswith('_') else Visibility.PUBLIC
        
        return Entity(
            id=f"{relative_path}::{node.name}",
            name=node.name,
            qualified_name=qualified_name,
            entity_type=entity_type,
            location=self._get_ast_location(node, relative_path, source_code),
            source_code=func_source,
            visibility=visibility,
            signature=signature,
            docstring=docstring,
            source_hash=hashlib.sha256(func_source.encode('utf-8')).hexdigest()[:16],
            metadata={
                "is_async": is_async,
                "arg_count": len(node.args.args),
                "decorator_count": len(node.decorator_list),
                "returns_annotation": bool(node.returns),
                "line_count": node.end_lineno - node.lineno + 1 if node.end_lineno else 1
            }
        )
    
    def _create_class_entity(
        self, 
        node: ast.ClassDef, 
        file_path: Path, 
        source_code: str
    ) -> Entity:
        """Create Entity for class"""
        relative_path = file_path.relative_to(self.project_root)
        
        # Build qualified name
        module_name = str(relative_path).replace('/', '.').replace('.py', '')
        qualified_name = f"{module_name}::{node.name}"
        
        # Extract class source
        class_source = self._get_node_source(node, source_code)
        
        # Build signature
        bases = [ast.unparse(base) for base in node.bases] if node.bases else []
        signature = f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}"
        
        # Extract docstring
        docstring = ast.get_docstring(node) or ""
        
        # Determine visibility
        visibility = Visibility.PRIVATE if node.name.startswith('_') else Visibility.PUBLIC
        
        return Entity(
            id=f"{relative_path}::{node.name}",
            name=node.name,
            qualified_name=qualified_name,
            entity_type=EntityType.CLASS,
            location=self._get_ast_location(node, relative_path, source_code),
            source_code=class_source,
            visibility=visibility,
            signature=signature,
            docstring=docstring,
            source_hash=hashlib.sha256(class_source.encode('utf-8')).hexdigest()[:16],
            metadata={
                "base_count": len(node.bases),
                "decorator_count": len(node.decorator_list),
                "method_count": len([n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]),
                "line_count": node.end_lineno - node.lineno + 1 if node.end_lineno else 1
            }
        )
    
    def _create_variable_entity(
        self, 
        node: ast.Assign, 
        file_path: Path, 
        source_code: str
    ) -> Optional[Entity]:
        """Create Entity for module-level variable/constant"""
        # Only extract simple assignments at module level
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return None
            
        target = node.targets[0]
        relative_path = file_path.relative_to(self.project_root)
        
        # Skip complex assignments
        if not isinstance(target.ctx, ast.Store):
            return None
            
        # Determine if constant (all caps) or variable
        entity_type = EntityType.CONSTANT if target.id.isupper() else EntityType.VARIABLE
        
        # Build qualified name
        module_name = str(relative_path).replace('/', '.').replace('.py', '')
        qualified_name = f"{module_name}::{target.id}"
        
        # Extract assignment source
        var_source = self._get_node_source(node, source_code)
        
        # Build signature
        signature = f"{target.id} = {ast.unparse(node.value)}"
        
        # Determine visibility
        visibility = Visibility.PRIVATE if target.id.startswith('_') else Visibility.PUBLIC
        
        return Entity(
            id=f"{relative_path}::{target.id}",
            name=target.id,
            qualified_name=qualified_name,
            entity_type=entity_type,
            location=self._get_ast_location(node, relative_path, source_code),
            source_code=var_source,
            visibility=visibility,
            signature=signature,
            docstring="",
            source_hash=hashlib.sha256(var_source.encode('utf-8')).hexdigest()[:16],
            metadata={
                "value_type": type(node.value).__name__,
                "is_constant": entity_type == EntityType.CONSTANT
            }
        )
    
    def _create_import_entity(
        self, 
        node: ast.Import | ast.ImportFrom, 
        file_path: Path, 
        source_code: str
    ) -> Entity:
        """Create Entity for import statement"""
        relative_path = file_path.relative_to(self.project_root)
        
        # Extract import source
        import_source = self._get_node_source(node, source_code)
        
        # Build name and signature
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
            name = f"import_{names[0]}" if names else "import"
            signature = f"import {', '.join(names)}"
        else:  # ImportFrom
            module = node.module or ""
            names = [alias.name for alias in node.names]
            name = f"from_{module}_{names[0]}" if names else f"from_{module}"
            signature = f"from {module} import {', '.join(names)}"
        
        # Build qualified name
        module_name = str(relative_path).replace('/', '.').replace('.py', '')
        qualified_name = f"{module_name}::{name}"
        
        return Entity(
            id=f"{relative_path}::{name}",
            name=name,
            qualified_name=qualified_name,
            entity_type=EntityType.IMPORT,
            location=self._get_ast_location(node, relative_path, source_code),
            source_code=import_source,
            visibility=Visibility.PUBLIC,
            signature=signature,
            docstring="",
            source_hash=hashlib.sha256(import_source.encode('utf-8')).hexdigest()[:16],
            metadata={
                "import_type": "from" if isinstance(node, ast.ImportFrom) else "direct",
                "module": getattr(node, 'module', None),
                "names_count": len(node.names)
            }
        )
    
    def _is_inside_class(self, node: ast.AST) -> bool:
        """Check if AST node is inside a class definition"""
        # This is a simplified check - in a real implementation,
        # we'd need to walk up the AST tree
        return False  # Simplified for now
    
    def _get_node_source(self, node: ast.AST, source_code: str) -> str:
        """Extract source code for AST node"""
        lines = source_code.splitlines()
        
        start_line = getattr(node, 'lineno', 1) - 1
        end_line = getattr(node, 'end_lineno', start_line + 1) - 1
        
        if start_line < 0 or start_line >= len(lines):
            return ""
            
        if end_line >= len(lines):
            end_line = len(lines) - 1
            
        node_lines = lines[start_line:end_line + 1]
        return '\n'.join(node_lines)
    
    def _build_function_signature(self, node: ast.FunctionDef, is_async: bool = False) -> str:
        """Build function signature string"""
        args = []
        
        # Regular arguments
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)
        
        # Keyword-only arguments
        for arg in node.args.kwonlyargs:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)
        
        # *args
        if node.args.vararg:
            arg_str = f"*{node.args.vararg.arg}"
            if node.args.vararg.annotation:
                arg_str += f": {ast.unparse(node.args.vararg.annotation)}"
            args.append(arg_str)
        
        # **kwargs
        if node.args.kwarg:
            arg_str = f"**{node.args.kwarg.arg}"
            if node.args.kwarg.annotation:
                arg_str += f": {ast.unparse(node.args.kwarg.annotation)}"
            args.append(arg_str)
        
        signature = f"{'async ' if is_async else ''}def {node.name}({', '.join(args)})"
        
        if node.returns:
            signature += f" -> {ast.unparse(node.returns)}"
            
        return signature
    
    def _get_ast_location(
        self, 
        node: ast.AST, 
        relative_path: Path, 
        source_code: str
    ) -> SourceLocation:
        """Create SourceLocation from AST node"""
        lines = source_code.splitlines()
        
        start_line = getattr(node, 'lineno', 1)
        end_line = getattr(node, 'end_lineno', start_line)
        start_column = getattr(node, 'col_offset', 0)
        end_column = getattr(node, 'end_col_offset', 0)
        
        # Calculate byte offsets (simplified)
        start_byte = sum(len(line.encode('utf-8')) + 1 for line in lines[:start_line-1])
        end_byte = start_byte + len('\n'.join(lines[start_line-1:end_line]).encode('utf-8'))
        
        return SourceLocation(
            file_path=relative_path,
            start_line=start_line,
            end_line=end_line,
            start_column=start_column,
            end_column=end_column,
            start_byte=start_byte,
            end_byte=end_byte
        )
    
    def _extract_module_docstring(self, source_code: str) -> str:
        """Extract module-level docstring"""
        try:
            tree = ast.parse(source_code)
            return ast.get_docstring(tree) or ""
        except:
            return ""


def generate_real_entities_from_project(project_root: Optional[Path] = None) -> List[Entity]:
    """
    Generate real Entity objects from the project codebase.
    
    Args:
        project_root: Path to project root (auto-detected if None)
        
    Returns:
        List of real Entity objects for testing
    """
    if project_root is None:
        # Auto-detect project root
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent
    
    extractor = RealCodeExtractor(project_root)
    entities = extractor.extract_all_entities()
    
    # Filter and prioritize interesting entities for testing
    filtered_entities = []
    
    for entity in entities:
        # Skip very short or trivial entities
        if len(entity.source_code) < 20:
            continue
            
        # Skip test files and __pycache__
        if "test_" in str(entity.location.file_path) or "__pycache__" in str(entity.location.file_path):
            continue
            
        # Prioritize public functions and classes
        if entity.entity_type in [EntityType.FUNCTION, EntityType.CLASS, EntityType.METHOD]:
            if entity.visibility == Visibility.PUBLIC:
                filtered_entities.append(entity)
        # Include some variables and imports
        elif entity.entity_type in [EntityType.CONSTANT, EntityType.IMPORT]:
            filtered_entities.append(entity)
        # Include file entities
        elif entity.entity_type == EntityType.FILE:
            filtered_entities.append(entity)
    
    # Prioritize critical entities for search quality testing
    critical_entity_names = [
        "HybridQdrantClient", "StellaEmbedder", "BatchIndexer", 
        "search_semantic", "search_payload", "search_hybrid",
        "embed_texts", "embed_single", "CollectionConfig", "QdrantPoint"
    ]
    
    # Separate critical entities from others
    critical_entities = []
    other_entities = []
    
    for entity in filtered_entities:
        if any(critical_name in entity.name for critical_name in critical_entity_names):
            critical_entities.append(entity)
        else:
            other_entities.append(entity)
    
    # Combine: all critical entities + fill remaining slots with others
    final_entities = critical_entities + other_entities
    
    # Increase limit to ensure critical entities are included
    return final_entities[:200]  # Increased limit for better search quality testing


def get_sample_entities_for_search_testing() -> List[Entity]:
    """
    Get a curated set of entities specifically for search testing.
    
    Returns:
        List of entities with known searchable characteristics
    """
    all_entities = generate_real_entities_from_project()
    
    # Curate entities that are good for search testing
    search_entities = []
    
    for entity in all_entities:
        # Functions with descriptive names
        if entity.entity_type == EntityType.FUNCTION and len(entity.name) > 5:
            search_entities.append(entity)
        
        # Classes with docstrings
        if entity.entity_type == EntityType.CLASS and entity.docstring:
            search_entities.append(entity)
        
        # Constants that might be searched for
        if entity.entity_type == EntityType.CONSTANT:
            search_entities.append(entity)
    
    return search_entities[:50]  # Return 50 entities for search testing


if __name__ == "__main__":
    # Test the extractor
    entities = generate_real_entities_from_project()
    print(f"Extracted {len(entities)} entities from project")
    
    for entity_type in EntityType:
        count = sum(1 for e in entities if e.entity_type == entity_type)
        if count > 0:
            print(f"  {entity_type.value}: {count}")
    
    # Show some examples
    print("\nExample entities:")
    for entity in entities[:5]:
        print(f"  {entity.entity_type.value}: {entity.qualified_name}")
        print(f"    Location: {entity.location.file_path}:{entity.location.start_line}")
        print(f"    Signature: {entity.signature}")
        print()